# tts.py
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import torch
import torchaudio
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from services.segment_quality import build_segment_issues
from services.self_reference import (
    SpeakerReferenceSample,
    deserialize_reference_mapping,
)

try:
    from app.configs import get_job_paths
except ModuleNotFoundError as exc:
    if exc.name != "app":
        raise
    from configs import get_job_paths
from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    load_compact_transcript,
    segment_views,
)

logger = logging.getLogger(__name__)


def _inject_cosyvoice_paths() -> None:
    """Ensure CosyVoice's source tree (and Matcha-TTS) are importable."""
    cosy_root = Path(os.getenv("COSYVOICE_DIR", "/opt/CosyVoice"))
    candidates = (
        cosy_root,
        cosy_root / "third_party" / "Matcha-TTS",
    )
    for path in candidates:
        try:
            if path.is_dir():
                path_str = str(path)
                if path_str not in sys.path:
                    sys.path.append(path_str)
        except Exception:
            continue


_inject_cosyvoice_paths()
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore
    from cosyvoice.utils.file_utils import load_wav  # type: ignore

    COSYVOICE_AVAILABLE = True
except Exception as exc:  # noqa: F841
    CosyVoice2 = None  # type: ignore
    load_wav = None  # type: ignore
    COSYVOICE_AVAILABLE = False


PROMPT_STT_MODEL_ID = os.getenv("COSYVOICE_PROMPT_STT_MODEL", "large-v3-turbo")
DEFAULT_TTS_DEVICE = (
    os.getenv("TTS_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
).lower()
PROMPT_STT_DEVICE = (
    os.getenv("COSYVOICE_PROMPT_STT_DEVICE") or DEFAULT_TTS_DEVICE
).lower()
PROMPT_STT_COMPUTE = os.getenv("COSYVOICE_PROMPT_STT_COMPUTE")


# ms초 이하 클립은 아예 안 자름
TRIM_MIN_CLIP_MS = 1000

# ms 이상 지속되면 '실제 침묵 구간'으로 본다
TRIM_MIN_SILENCE_MS = 100

# 평균 볼륨보다 dB 이상 작으면 침묵 후보
TRIM_SILENCE_DB_DROP = 20.0

# 실제 잘라낼 때 여유로 남겨주는 여백
TRIM_EDGE_GUARD_MS = 100

# ms 이하이면 '짧은 잡소리 조각'으로 간주
TRIM_EDGE_ARTIFACT_MS = 320

# 시작/끝에서 ms 범위만 '잡소리 후보'로 본다
TRIM_EDGE_WINDOW_MS = 1200

# VAD에서 ms 이내로 끊기면 한 덩어리로 묶음
TRIM_VAD_BRIDGE_MS = 240

# VAD 프레임 길이 (30ms)
TRIM_VAD_FRAME_MS = 30

_PROMPT_CACHE: Dict[str, str] = {}


def _resolve_cosyvoice_model_dir() -> Path:
    """Pick the CosyVoice2 model directory from env or default location."""
    raw = (
        os.getenv("COSYVOICE2_MODEL_DIR") or os.getenv("COSYVOICE_MODEL_DIR") or ""
    ).strip()
    if raw:
        return Path(raw).expanduser()
    cosy_root = Path(os.getenv("COSYVOICE_DIR", "/opt/CosyVoice"))
    return cosy_root / "pretrained_models" / "CosyVoice2-0.5B"


@lru_cache(maxsize=1)
def _get_cosyvoice2():
    """Lazily load CosyVoice2 (if available)."""
    if not COSYVOICE_AVAILABLE or CosyVoice2 is None or load_wav is None:
        raise RuntimeError("CosyVoice2 backend is not installed/configured.")
    model_dir = _resolve_cosyvoice_model_dir()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"CosyVoice2 model directory not found: {model_dir}")
    fp16 = DEFAULT_TTS_DEVICE == "cuda" and torch.cuda.is_available()
    cv = CosyVoice2(
        str(model_dir),
        load_jit=False,
        load_trt=False,
        load_vllm=False,
        fp16=fp16,
    )
    return cv, load_wav


@lru_cache(maxsize=1)
def _get_prompt_stt_model():
    """Load the fast-whisper model (large-v3-turbo by default) for prompt extraction."""
    from faster_whisper import WhisperModel

    device = PROMPT_STT_DEVICE if torch.cuda.is_available() else "cpu"
    if device not in {"cuda", "cpu"}:
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    compute_type = PROMPT_STT_COMPUTE
    if not compute_type:
        compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(
        PROMPT_STT_MODEL_ID,
        device=device,
        compute_type=compute_type,
    )


def _strip_background_from_sample(sample_path: Path, work_dir: Path) -> Path:
    """Use Demucs two-stem separation to isolate vocals from a custom voice sample."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="demucs_user_", dir=str(work_dir)
        ) as tmpdir:
            output_dir = Path(tmpdir)
            cmd = [
                "python3",
                "-m",
                "demucs.separate",
                "-d",
                device,
                "-n",
                "htdemucs",
                "--two-stems",
                "vocals",
                "-o",
                str(output_dir),
                str(sample_path),
            ]
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            demucs_dir = output_dir / "htdemucs" / sample_path.stem
            vocals_path = demucs_dir / "vocals.wav"
            if vocals_path.is_file():
                cleaned = work_dir / "user_voice_sample_vocals.wav"
                shutil.copyfile(vocals_path, cleaned)
                return cleaned
            logger.warning(
                "Demucs 결과에서 보컬 트랙을 찾지 못했습니다: %s", vocals_path
            )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        logger.warning(
            "사용자 보이스 샘플 배경 제거 실패 (%s, device=%s): %s",
            sample_path,
            device,
            exc,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "보이스 샘플 정제 중 알 수 없는 오류 발생 (%s): %s", sample_path, exc
        )
    return sample_path


def _transcribe_prompt_text(sample_path: Path, language: str | None = None) -> str:
    """ASR the reference sample to obtain a light-weight prompt text."""
    lang_value = (language or "").strip()
    lang_key = lang_value.lower()
    cache_key = f"{sample_path.resolve()}::{lang_key or 'auto'}"
    if cache_key in _PROMPT_CACHE:
        return _PROMPT_CACHE[cache_key]

    text = ""
    try:
        model = _get_prompt_stt_model()
        lang_arg = None
        if lang_key and lang_key not in {"auto"}:
            lang_arg = lang_value or None
        segments, _ = model.transcribe(
            str(sample_path),
            beam_size=1,
            best_of=1,
            language=lang_arg,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 200},
        )
        collected = []
        for seg in segments:
            piece = getattr(seg, "text", "").strip()
            if piece:
                collected.append(piece)
        text = " ".join(collected).strip()
    except Exception as exc:
        logger.warning("Prompt STT fallback failed for %s: %s", sample_path, exc)
    _PROMPT_CACHE[cache_key] = text
    return text


def _resolve_path(candidate: str, paths) -> Path:
    """Resolve a user-provided path relative to the job directories."""
    path_obj = Path(candidate).expanduser()
    if path_obj.is_absolute():
        return path_obj
    search_roots = (
        paths.interim_dir,
        paths.outputs_dir,
        paths.input_dir,
        Path("."),
    )
    for root in search_roots:
        resolved = (root / path_obj).resolve()
        if resolved.exists():
            return resolved
    return (paths.interim_dir / path_obj).resolve()


def _select_voice_sample(
    seg: dict,
    speaker_refs: Dict[str, SpeakerReferenceSample],
    paths,
    global_sample: Path | None = None,
    speaker_override_refs: Dict[str, SpeakerReferenceSample] | None = None,
) -> tuple[Path, SpeakerReferenceSample | None]:
    """Pick the best voice-sample path (and metadata) for the segment."""
    override = seg.get("voice_sample_path") or seg.get("voice_sample")
    if isinstance(override, str) and override.strip():
        resolved = _resolve_path(override.strip(), paths)
        if not resolved.is_file():
            raise FileNotFoundError(f"Voice sample override not found: {resolved}")
        return resolved, None
    speaker = seg.get("speaker")
    if not speaker:
        raise ValueError("Segment is missing speaker information.")
    if global_sample and global_sample.is_file():
        return global_sample, None
    if speaker_override_refs:
        override_ref = speaker_override_refs.get(speaker)
        if override_ref:
            path = override_ref.audio_path
            if not path.is_file():
                raise FileNotFoundError(
                    f"Override voice sample missing for speaker {speaker}: {path}"
                )
            return path, override_ref
    ref = speaker_refs.get(speaker)
    if not ref:
        raise FileNotFoundError(
            f"No self-reference audio prepared for speaker {speaker}."
        )
    path = ref.audio_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Self-reference audio missing for speaker {speaker}: {path}"
        )
    return path, ref


def _resolve_prompt_text(
    seg: dict,
    sample_path: Path,
    prompt_text_override: str | None = None,
    ref_prompt_text: str | None = None,
) -> str:
    """Return prompt text, generating it via ASR if missing."""
    if prompt_text_override and prompt_text_override.strip():
        return prompt_text_override.strip()
    prompt = (
        seg.get("prompt_text") or seg.get("prompt") or seg.get("reference_text") or ""
    )
    prompt = prompt.strip()
    if prompt:
        return prompt
    if ref_prompt_text and ref_prompt_text.strip():
        return ref_prompt_text.strip()
    prompt = _transcribe_prompt_text(sample_path)
    if prompt:
        return prompt
    return (seg.get("text") or "").strip()


def _synthesize_with_cosyvoice2(
    text: str, prompt_text: str, sample_path: Path, output_path: Path
) -> None:
    """Run CosyVoice2 zero-shot inference; avoid duplicate audio within one segment."""
    cv, load_wav_fn = _get_cosyvoice2()
    prompt_speech_16k = load_wav_fn(str(sample_path), 16000)

    generator = cv.inference_zero_shot(
        text,
        prompt_text,
        prompt_speech_16k,
        stream=False,
        text_frontend=False,
    )

    chunks: list[torch.Tensor] = []
    for i, item in enumerate(generator):
        speech = item.get("tts_speech")
        if isinstance(speech, torch.Tensor):
            chunks.append(speech)
        else:
            logger.warning("CosyVoice2 chunk %s has no valid 'tts_speech'", i)

    if not chunks:
        raise RuntimeError("CosyVoice2 zero-shot 클로닝 결과가 비어 있습니다.")

    if len(chunks) == 1:
        waveform = chunks[0]
    else:
        # 여러 chunk가 나오는 경우: 가장 긴 chunk 하나만 사용 (보통 전체 문장을 포함)
        lengths = [c.shape[-1] for c in chunks]
        max_idx = max(range(len(lengths)), key=lambda i: lengths[i])
        waveform = chunks[max_idx]
        logger.warning(
            "CosyVoice2 returned %d chunks for one segment; "
            "using longest chunk index=%d (len=%d, total=%d).",
            len(chunks),
            max_idx,
            lengths[max_idx],
            sum(lengths),
        )

    sample_rate = getattr(cv, "sample_rate", 24000)
    torchaudio.save(str(output_path), waveform, sample_rate)


def _detect_speech_bounds_vad(segment: AudioSegment) -> tuple[int, int] | None:
    """
    Use webrtcvad (if available) to locate the dominant speech window.
    Returns (start_ms, end_ms) relative to segment start.
    """
    try:
        import webrtcvad  # type: ignore
    except Exception:
        return None

    normalized = segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    raw = normalized.raw_data
    if not raw:
        return None

    frame_ms = TRIM_VAD_FRAME_MS
    if frame_ms not in (10, 20, 30):
        frame_ms = 30
    frame_samples = int(normalized.frame_rate * frame_ms / 1000)
    frame_bytes = frame_samples * normalized.sample_width
    if frame_bytes <= 0:
        return None

    vad = webrtcvad.Vad(3)
    speech_ranges: list[list[float]] = []
    ts = 0.0
    step_s = frame_ms / 1000.0
    in_speech = False
    start_t = 0.0

    for offset in range(0, len(raw), frame_bytes):
        frame = raw[offset : offset + frame_bytes]
        if len(frame) < frame_bytes:
            break
        try:
            is_speech = vad.is_speech(frame, normalized.frame_rate)
        except Exception:
            break
        if is_speech and not in_speech:
            in_speech = True
            start_t = ts
        elif (not is_speech) and in_speech:
            in_speech = False
            speech_ranges.append([start_t, ts + step_s])
        ts += step_s
    if in_speech:
        speech_ranges.append([start_t, ts])
    if not speech_ranges:
        return None

    bridge = max(0.0, TRIM_VAD_BRIDGE_MS / 1000.0)
    merged: list[list[float]] = []
    for start, end in speech_ranges:
        if not merged:
            merged.append([start, end])
            continue
        if start <= merged[-1][1] + bridge:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    start_ms = max(0, int(merged[0][0] * 1000))
    end_ms = min(len(normalized), int(merged[-1][1] * 1000))
    if end_ms <= start_ms:
        return None
    return start_ms, end_ms


def _trim_tts_artifacts(audio_path: Path) -> None:
    """
    Remove spurious silence or very short vocal artifacts at clip edges.
    Uses a simple amplitude-based detector so we do not need extra models.
    """
    try:
        audio = AudioSegment.from_file(str(audio_path))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to load synthesized clip for trimming (%s): %s", audio_path, exc
        )
        return

    duration_ms = len(audio)
    if duration_ms <= TRIM_MIN_CLIP_MS:
        return

    base_db = audio.dBFS if audio.dBFS != float("-inf") else -60.0
    silence_thresh = max(base_db - TRIM_SILENCE_DB_DROP, -75.0)
    non_silence_spans = detect_nonsilent(
        audio,
        min_silence_len=TRIM_MIN_SILENCE_MS,
        silence_thresh=silence_thresh,
        seek_step=15,
    )
    if not non_silence_spans:
        return

    trimmed = False
    spans = non_silence_spans

    # Drop very short leading artifacts that hug the start of the clip.
    while (
        len(spans) > 1
        and spans[0][0] <= TRIM_EDGE_WINDOW_MS
        and (spans[0][1] - spans[0][0]) <= TRIM_EDGE_ARTIFACT_MS
    ):
        spans = spans[1:]
        trimmed = True

    # Drop very short trailing artifacts stuck to the end.
    while (
        len(spans) > 1
        and (duration_ms - spans[-1][1]) <= TRIM_EDGE_WINDOW_MS
        and (spans[-1][1] - spans[-1][0]) <= TRIM_EDGE_ARTIFACT_MS
    ):
        spans = spans[:-1]
        trimmed = True

    if not spans:
        return

    keep_start = max(0, spans[0][0] - TRIM_EDGE_GUARD_MS)
    keep_end = min(duration_ms, spans[-1][1] + TRIM_EDGE_GUARD_MS)
    if keep_end - keep_start <= 0:
        return

    trimmed = trimmed or keep_start > 0 or keep_end < duration_ms

    # Secondary, speech-aware clamp using VAD (when available)
    candidate = audio[keep_start:keep_end]
    speech_bounds = _detect_speech_bounds_vad(candidate)
    if speech_bounds:
        local_start = max(0, speech_bounds[0] - TRIM_EDGE_GUARD_MS)
        local_end = min(len(candidate), speech_bounds[1] + TRIM_EDGE_GUARD_MS)
        if local_end > local_start:
            abs_start = keep_start + local_start
            abs_end = keep_start + (local_end)
            trimmed = trimmed or abs_start > keep_start or abs_end < keep_end
            keep_start, keep_end = abs_start, abs_end

    if not trimmed:
        return

    cleaned = audio[keep_start:keep_end]
    cleaned.export(str(audio_path), format="wav")


def generate_tts(
    job_id: str,
    target_lang: str,
    voice_sample_path: Path | None = None,
    prompt_text_override: str | None = None,
    speaker_voice_overrides: dict[str, dict[str, Any]] | None = None,
):
    """Use CosyVoice2 (if available) to synthesize translated segments."""
    if not COSYVOICE_AVAILABLE:
        raise RuntimeError(
            "CosyVoice2 백엔드가 설치되지 않아 보이스 클로닝을 진행할 수 없습니다."
        )
    paths = get_job_paths(job_id)
    trans_path = paths.trg_sentence_dir / "translated.json"
    if not trans_path.is_file():
        raise FileNotFoundError(
            "Translated text not found. Run translation stage first."
        )
    with open(trans_path, "r", encoding="utf-8") as f:
        translation_entries = json.load(f)
    translation_map: Dict[int, dict] = {}
    for entry in translation_entries:
        seg_idx = entry.get("seg_idx")
        try:
            idx = int(seg_idx)
        except (TypeError, ValueError):
            continue
        translation_map[idx] = entry

    transcript_archive = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
    bundle = load_compact_transcript(transcript_archive)
    base_segments = segment_views(bundle)
    if not base_segments:
        raise RuntimeError("No source segments found. Run ASR stage first.")

    tts_dir = paths.vid_tts_dir
    tts_dir.mkdir(parents=True, exist_ok=True)

    speaker_refs: Dict[str, SpeakerReferenceSample] = {}
    speaker_refs_path = tts_dir / "speaker_refs.json"

    if speaker_refs_path.is_file():
        # STT 단계에서 미리 생성해 둔 매핑 사용
        with open(speaker_refs_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if isinstance(mapping, dict):
            speaker_refs = deserialize_reference_mapping(mapping, tts_dir)
            logger.info(
                "Loaded %d precomputed self-reference samples from %s",
                len(speaker_refs),
                speaker_refs_path,
            )
        else:
            logger.warning(
                "speaker_refs.json has unexpected format (%s); ignoring",
                type(mapping).__name__,
            )

    normalized_user_sample = None
    if voice_sample_path:
        candidate = Path(voice_sample_path)
        if candidate.is_file():
            cleaned_candidate = _strip_background_from_sample(candidate, tts_dir)
            try:
                user_audio = AudioSegment.from_file(str(cleaned_candidate))
                normalized_user_sample = tts_dir / "user_voice_sample.wav"
                user_audio.set_frame_rate(16000).set_channels(1).export(
                    normalized_user_sample, format="wav"
                )
            except Exception as exc:
                logger.warning(
                    "사용자 보이스 샘플 변환 실패 (%s), 원본을 그대로 사용합니다: %s",
                    cleaned_candidate,
                    exc,
                )
                normalized_user_sample = cleaned_candidate
        else:
            logger.warning("제공된 보이스 샘플을 찾을 수 없습니다: %s", candidate)

    replacement_meta = speaker_voice_overrides or {}
    override_refs: Dict[str, SpeakerReferenceSample] = {}
    if replacement_meta:
        replacements_path = tts_dir / "voice_replacements.json"
        try:
            with open(replacements_path, "w", encoding="utf-8") as f:
                json.dump(replacement_meta, f, ensure_ascii=False, indent=2)
        except OSError as exc:
            logger.warning("Failed to persist voice replacement metadata: %s", exc)
        for speaker, info in replacement_meta.items():
            audio_value = info.get("audio_path")
            if not audio_value:
                continue
            resolved = _resolve_path(str(audio_value), paths)
            if not resolved.is_file():
                logger.warning(
                    "Voice replacement sample missing for %s at %s", speaker, resolved
                )
                continue
            similarity_val = info.get("similarity")
            try:
                similarity_score = (
                    float(similarity_val) if similarity_val is not None else None
                )
            except (TypeError, ValueError):
                similarity_score = None
            override_refs[speaker] = SpeakerReferenceSample(
                speaker=speaker,
                audio_path=resolved,
                text=(info.get("prompt_text") or "").strip(),
                segment_idx=-1,
                segment_id=f"voice_replacement_{speaker}",
                start_ms=0,
                end_ms=0,
                audio_duration_ms=0,
                score=similarity_score,
            )

    segment_lookup = {seg.idx: seg for seg in base_segments}

    def _synthesize_segment(seg) -> dict:
        override = translation_map.get(seg.idx, {})
        text = override.get("translation") or seg.text
        speaker = override.get("speaker") or seg.speaker
        segment_start = seg.start_seconds
        segment_end = seg.end_seconds
        duration = max(0.0, seg.duration_seconds)
        output_file = tts_dir / f"{speaker}_{segment_start:.2f}.wav"

        seg_payload = {
            "speaker": speaker,
            "text": text,
            "start": segment_start,
            "end": segment_end,
            "voice_sample_path": override.get("voice_sample_path")
            or override.get("voice_sample"),
            "voice_sample": override.get("voice_sample"),
            "prompt_text": override.get("prompt_text") or override.get("prompt"),
            "reference_text": override.get("reference_text"),
        }
        effective_sample, ref_info = _select_voice_sample(
            seg_payload,
            speaker_refs,
            paths,
            normalized_user_sample,
            override_refs if override_refs else None,
        )

        # 실제로 '글로벌 유저 샘플'을 쓰는 경우에만 override 사용
        use_user_sample = (
            normalized_user_sample is not None
            and effective_sample.resolve() == normalized_user_sample.resolve()
        )

        final_prompt_text_override = prompt_text_override if use_user_sample else None

        ref_prompt_text = ref_info.text if (ref_info and ref_info.text) else None

        prompt_text = _resolve_prompt_text(
            seg_payload,
            effective_sample,
            final_prompt_text_override,
            ref_prompt_text=ref_prompt_text,
        )

        _synthesize_with_cosyvoice2(
            text=text,
            prompt_text=prompt_text,
            sample_path=effective_sample,
            output_path=output_file,
        )
        _trim_tts_artifacts(output_file)

        # --- 길이 기반 품질 체크 ---
        tts_status = "ok"
        quality_note = None
        duration_ratio: float | None = None
        try:
            clip = AudioSegment.from_file(str(output_file))
            dur_ms = len(clip)
            target_ms = int(duration * 1000)
            if target_ms > 0:
                duration_ratio = dur_ms / target_ms
            # 타겟 대비 비정상적으로 짧거나 긴 경우
            if target_ms > 0 and (
                duration_ratio is not None
                and (duration_ratio < 0.4 or duration_ratio > 2.5)
            ):
                tts_status = "suspect_duration"
                quality_note = (
                    f"ratio={duration_ratio:.2f}, dur={dur_ms}ms, target={target_ms}ms"
                )
                logger.warning(
                    "TTS duration looks off for seg_idx=%s (%s)",
                    seg.idx,
                    quality_note,
                )
        except Exception as exc:
            logger.warning(
                "Failed to inspect TTS duration for %s: %s", output_file, exc
            )

        entry = {
            "segment_id": seg.segment_id(),
            "seg_idx": seg.idx,
            "speaker": speaker,
            "start": segment_start,
            "end": segment_end,
            "target_duration": duration,
            "audio_file": str(output_file),
            "voice_sample": str(effective_sample),
            "prompt_text": prompt_text,
            "tts_backend": "cosyvoice2",
            "source_text": seg.text,
            "tts_status": tts_status,
            "quality_note": quality_note,
        }
        # 보이스 유사도/강제 대체 정보 계산
        replacement_info = replacement_meta.get(speaker, {}) if replacement_meta else {}
        voice_similarity: float | None = None
        voice_low_sim_forced: bool | None = None

        sim_raw = replacement_info.get("similarity")
        if sim_raw is not None:
            try:
                voice_similarity = float(sim_raw)
            except (TypeError, ValueError):
                voice_similarity = None

        # “라이브러리에서 골라온 대체 보이스”인 경우에만 강제 여부 판단
        if speaker in override_refs and voice_similarity is not None:
            LOW_SIM_THRESHOLD = 0.45  # 코사인 0.45 이하면 낮은 편으로 봄 (조절 가능)
            if voice_similarity < LOW_SIM_THRESHOLD:
                voice_low_sim_forced = True

        # 기존 issues 호출을 voice 인자까지 넘기도록 확장
        entry["issues"] = build_segment_issues(
            stt_score_q=seg.score_q,
            tts_ratio=duration_ratio,
            speaker_unknown=seg.speaker_unknown,
            voice_similarity=voice_similarity,
            voice_low_similarity_forced=voice_low_sim_forced,
        )
        if speaker in override_refs:
            entry["voice_replacement"] = {
                "voice_id": replacement_info.get("voice_id"),
                "similarity": voice_similarity,
                "sample_key": replacement_info.get("sample_key"),
                "sample_bucket": replacement_info.get("sample_bucket"),
                "language": replacement_info.get("language"),
            }
        return entry

    synthesized_segments: list[dict] = []
    suspect_indices: list[int] = []
    index_by_seg_idx: Dict[int, int] = {}
    for seg in base_segments:
        entry = _synthesize_segment(seg)
        index = len(synthesized_segments)
        synthesized_segments.append(entry)
        index_by_seg_idx[seg.idx] = index
        if entry.get("tts_status") == "suspect_duration":
            suspect_indices.append(seg.idx)

    if suspect_indices:
        remaining = suspect_indices
        max_retry = 1
        for attempt in range(max_retry):
            next_round: list[int] = []
            logger.info(
                "Retrying TTS for %d suspect segments (attempt %d/%d): %s",
                len(remaining),
                attempt + 1,
                max_retry,
                remaining,
            )
            for seg_idx in remaining:
                seg_obj = segment_lookup.get(seg_idx)
                if not seg_obj:
                    continue
                updated_entry = _synthesize_segment(seg_obj)
                pos = index_by_seg_idx.get(seg_idx)
                if pos is not None:
                    synthesized_segments[pos] = updated_entry
                if updated_entry.get("tts_status") == "suspect_duration":
                    next_round.append(seg_idx)
            if not next_round:
                break
            remaining = next_round
        if remaining:
            logger.warning(
                "Segments still flagged as suspect after retry: %s", remaining
            )
    meta_path = tts_dir / "segments.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(synthesized_segments, f, ensure_ascii=False, indent=2)
    return synthesized_segments
