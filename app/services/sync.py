# sync.py
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pyrubberband as rb
import soundfile as sf
from pydub import AudioSegment

try:
    from app.configs import get_job_paths
except ModuleNotFoundError as exc:
    if exc.name != "app":
        raise
    from configs import get_job_paths
from services.segment_quality import build_segment_issues, sync_percent_from_durations
from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    load_compact_transcript,
    segment_views,
)

MAX_SLOW_RATIO = float(os.getenv("SYNC_MAX_SLOW_RATIO", "1.1"))


def _resolve_audio_path(path_str: str, fallback_dir: Path) -> Path:
    path = Path(path_str)
    if path.is_file():
        return path
    candidate = fallback_dir / path.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"TTS 오디오 파일을 찾을 수 없습니다: {path_str}")


def _time_stretch(input_path: Path, rate: float) -> AudioSegment:
    """pyrubberband를 사용해 고품질로 시간 조절. rate>1이면 빨라져 길이가 줄고, rate<1이면 느려져 길이가 늘어납니다."""
    if rate <= 0:
        raise ValueError("재생 속도(rate)는 0보다 커야 합니다.")
    if abs(rate - 1.0) < 0.01:
        return AudioSegment.from_file(str(input_path))
    y, sr = sf.read(str(input_path))
    # pyrubberband의 time_stretch는 길이 배율이 아닌 재생 속도를 인자로 받습니다.
    stretched_y = rb.time_stretch(y, sr, rate)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        temp_output = Path(tmp.name)

    try:
        sf.write(str(temp_output), stretched_y, sr)
        return AudioSegment.from_wav(str(temp_output))

    finally:
        if temp_output.exists():
            temp_output.unlink()


def _sync_single_segment(
    audio_path: Path,
    target_ms: int,
    allow_ratio: float,
) -> tuple[AudioSegment, float, int, int]:
    audio = AudioSegment.from_file(str(audio_path))
    current_ms = len(audio)
    if current_ms <= 0:
        raise RuntimeError(f"빈 오디오 파일입니다: {audio_path}")

    desired_ratio = target_ms / current_ms
    if desired_ratio <= 0:
        raise RuntimeError("목표 길이가 잘못되었습니다.")
    # 길이 배율(ratio)과 pyrubberband의 재생 속도(rate) 방향이 반대이므로
    # ratio를 clamp한 뒤 역수를 전달한다.
    hit_slow_cap = False
    ratio_to_apply = desired_ratio
    if desired_ratio > 1.0:
        if desired_ratio > allow_ratio:
            ratio_to_apply = allow_ratio
            hit_slow_cap = True
    rate = 1.0 / ratio_to_apply
    stretched = _time_stretch(audio_path, rate)
    if len(stretched) <= 0:
        raise RuntimeError("시간 조정 결과가 비정상입니다.")

    # 무음 패딩
    padding_ms = 0
    # if hit_slow_cap and len(stretched) < target_ms:
    #     padding_ms = target_ms - len(stretched)
    #     stretched += AudioSegment.silent(duration=padding_ms)
    return stretched, ratio_to_apply, padding_ms, current_ms


def sync_segment_to_range(
    input_path: Path, target_duration_ms: int, output_path: Path
) -> Path:
    """
    segment_tts의 fixed 모드에서 사용할 길이 보정:
    - sync.py와 동일하게 pyrubberband로 tempo 조절
    - 너무 많이 느려지는 건 MAX_SLOW_RATIO까지만 허용
    """
    if target_duration_ms <= 0:
        raise ValueError("target_duration_ms must be positive")

    # sync.py의 시간 보정 로직 재사용
    synced_audio, ratio_applied, padding_ms, original_ms = _sync_single_segment(
        input_path,
        target_ms=target_duration_ms,
        allow_ratio=MAX_SLOW_RATIO,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    synced_audio.export(str(output_path), format="wav")
    return output_path


def sync_segments(job_id: str) -> List[Dict]:
    """
    번역/TTS 후 구간별 오디오를 원본 화자 길이에 맞게 보정합니다.
    - 길이가 더 길면 배속(tempo up)으로만 맞추고 자르지 않음
    - 길이가 짧으면 최대 MAX_SLOW_RATIO까지만 감속(tempo down)하고 남은 구간은 비워둠
    """
    paths = get_job_paths(job_id)
    transcript_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
    tts_meta_path = paths.vid_tts_dir / "segments.json"
    if not transcript_path.is_file():
        raise FileNotFoundError(
            f"원본 전사({COMPACT_ARCHIVE_NAME})를 찾을 수 없습니다."
        )
    if not tts_meta_path.is_file():
        raise FileNotFoundError(
            "TTS 세그먼트 메타데이터가 없습니다. /tts 단계를 먼저 실행하세요."
        )

    bundle = load_compact_transcript(transcript_path)
    base_segments = segment_views(bundle)
    src_lookup = {seg.segment_id(): seg for seg in base_segments}

    with open(tts_meta_path, "r", encoding="utf-8") as f:
        tts_segments = json.load(f)

    synced_dir = paths.vid_tts_dir / "synced"
    synced_dir.mkdir(parents=True, exist_ok=True)

    synced_metadata: List[Dict] = []
    for entry in tts_segments:
        seg_id = entry.get("segment_id")
        if not seg_id or seg_id not in src_lookup:
            raise KeyError(f"segment_id {seg_id} 에 해당하는 원본 구간이 없습니다.")
        source_seg = src_lookup[seg_id]
        target_duration = float(
            entry.get("target_duration") or source_seg.duration_seconds
        )
        target_ms = max(1, int(target_duration * 1000))

        audio_path = _resolve_audio_path(entry["audio_file"], paths.vid_tts_dir)
        synced_audio, ratio_applied, padding_ms, original_ms = _sync_single_segment(
            audio_path,
            target_ms,
            MAX_SLOW_RATIO,
        )
        output_path = synced_dir / Path(audio_path).name
        synced_audio.export(str(output_path), format="wav")

        source_duration_sec = round(target_ms / 1000, 3)
        orig_duration_sec = round(original_ms / 1000, 3)
        synced_duration_sec = round(len(synced_audio) / 1000, 3)
        synced_entry = {
            "segment_id": seg_id,
            "start": round(source_seg.start_seconds, 3),
            "source_duration": source_duration_sec,
            "tts_duration": orig_duration_sec,
            "synced_duration": synced_duration_sec,
            "ratio_target": round(target_ms / original_ms, 4),
            "ratio_applied": round(ratio_applied, 4),
            "padding_ms": padding_ms,
            "audio_file": str(output_path),
        }
        # tts_segments에서 source_text와 기타 필드 보존
        if "source_text" in entry:
            synced_entry["source_text"] = entry["source_text"]
        elif source_seg.text:
            synced_entry["source_text"] = source_seg.text

        # 기타 필드들도 보존 (speaker, seg_idx, start, end, prompt_text 등)
        for key in [
            "speaker",
            "seg_idx",
            "start",
            "end",
            "prompt_text",
            "voice_sample",
        ]:
            if key in entry:
                synced_entry[key] = entry[key]

        sync_percent = sync_percent_from_durations(
            source_duration_sec, synced_duration_sec
        )
        synced_entry["issues"] = build_segment_issues(
            stt_score_q=getattr(source_seg, "score_q", None),
            base=entry.get("issues"),
            sync_percent=sync_percent,
            speaker_unknown=getattr(source_seg, "speaker_unknown", None),
        )

        synced_metadata.append(synced_entry)

    meta_path = synced_dir / "segments_synced.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(synced_metadata, f, ensure_ascii=False, indent=2)
    return synced_metadata
