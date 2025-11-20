# services/self_reference.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

from pydub import AudioSegment
from pydub.silence import detect_silence

from .transcript_store import SegmentView

logger = logging.getLogger(__name__)

# self-reference 프롬프트 오디오 길이 제한 및 가중치 정의
MIN_REF_DURATION_MS = 2_000
IDEAL_REF_DURATION_MS = 5_000
MAX_REF_DURATION_MS = 15_000

CLARITY_WEIGHT = 0.7
SILENCE_WEIGHT = 0.3
QUALITY_WEIGHT = 0.65
LENGTH_WEIGHT = 0.35
MIN_CONTENT_SCORE = 0.2


@dataclass
class SpeakerReferenceSample:
    speaker: str
    audio_path: Path
    text: str
    segment_idx: int
    segment_id: str
    start_ms: int
    end_ms: int
    audio_duration_ms: int
    score: float | None = None

    @property
    def segment_duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    def to_payload(self, base_dir: Path) -> dict:
        try:
            audio_rel = self.audio_path.relative_to(base_dir)
            audio_value = str(audio_rel)
        except ValueError:
            audio_value = str(self.audio_path)
        return {
            "audio": audio_value,
            "text": self.text,
            "segment_idx": self.segment_idx,
            "segment_id": self.segment_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "segment_duration_ms": self.segment_duration_ms,
            "audio_duration_ms": self.audio_duration_ms,
            "score": self.score,
        }

    @classmethod
    def from_payload(
        cls, speaker: str, payload: Any, base_dir: Path
    ) -> "SpeakerReferenceSample":
        if isinstance(payload, str):
            audio_value = payload
            meta: dict[str, Any] = {}
        elif isinstance(payload, dict):
            audio_value = payload.get("audio") or payload.get("path") or ""
            meta = payload
        else:
            audio_value = ""
            meta = {}

        if not audio_value:
            audio_value = f"{speaker}_self_ref.wav"

        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = (base_dir / audio_path).resolve()

        def _int_value(key: str, default: int = 0) -> int:
            value = meta.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        segment_idx = _int_value("segment_idx", -1)
        start_ms = _int_value("start_ms", 0)
        end_ms = _int_value("end_ms", start_ms)
        seg_duration = _int_value("segment_duration_ms", max(0, end_ms - start_ms))
        if end_ms <= start_ms:
            end_ms = start_ms + seg_duration
        audio_duration = _int_value("audio_duration_ms", seg_duration)

        score_val = meta.get("score")
        try:
            score = float(score_val) if score_val is not None else None
        except (TypeError, ValueError):
            score = None

        text = (meta.get("text") or "").strip()
        segment_id = meta.get("segment_id")
        if not segment_id:
            segment_suffix = f"{segment_idx:04d}" if segment_idx >= 0 else "unknown"
            segment_id = f"segment_{segment_suffix}"

        return cls(
            speaker=speaker,
            audio_path=audio_path,
            text=text,
            segment_idx=segment_idx,
            segment_id=segment_id,
            start_ms=start_ms,
            end_ms=end_ms,
            audio_duration_ms=audio_duration,
            score=score,
        )


def serialize_reference_mapping(
    references: Dict[str, SpeakerReferenceSample], base_dir: Path
) -> Dict[str, Any]:
    return {speaker: ref.to_payload(base_dir) for speaker, ref in references.items()}


def deserialize_reference_mapping(
    payload: Dict[str, Any], base_dir: Path
) -> Dict[str, SpeakerReferenceSample]:
    mapping: Dict[str, SpeakerReferenceSample] = {}
    for speaker, entry in payload.items():
        try:
            mapping[speaker] = SpeakerReferenceSample.from_payload(
                speaker, entry, base_dir
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to parse speaker reference for %s: %s", speaker, exc)
    return mapping


def _calculate_length_score(duration_ms: int) -> float:
    """Triangular length score with its peak at IDEAL_REF_DURATION_MS."""
    if duration_ms < MIN_REF_DURATION_MS or duration_ms > MAX_REF_DURATION_MS:
        return 0.0
    if duration_ms == IDEAL_REF_DURATION_MS:
        return 1.0
    if duration_ms < IDEAL_REF_DURATION_MS:
        span = IDEAL_REF_DURATION_MS - MIN_REF_DURATION_MS
        return (duration_ms - MIN_REF_DURATION_MS) / span if span else 0.0
    span = MAX_REF_DURATION_MS - IDEAL_REF_DURATION_MS
    return (MAX_REF_DURATION_MS - duration_ms) / span if span else 0.0


def _speech_density_score(audio: AudioSegment) -> float:
    """Estimate how much of the clip contains speech vs. silence."""
    duration_ms = len(audio)
    if duration_ms <= 0:
        return 0.0
    base_db = audio.dBFS if audio.dBFS != float("-inf") else -60.0
    silence_thresh = max(base_db - 16.0, -70.0)
    silence_spans = detect_silence(
        audio,
        min_silence_len=120,
        silence_thresh=silence_thresh,
        seek_step=15,
    )
    silent_ms = sum(end - start for start, end in silence_spans)
    speech_ratio = 1.0 - min(1.0, max(0.0, silent_ms / duration_ms))
    return max(0.0, min(1.0, speech_ratio))


def prepare_self_reference_samples(
    vocals_audio: AudioSegment, segments: Sequence[SegmentView], out_dir: Path
) -> Dict[str, SpeakerReferenceSample]:
    """
    화자별 self-reference 샘플을 선택한다.

    - 3~15초 범위 밖의 세그먼트는 제외한다.
    - 길이 6초에서 최고점을 갖는 삼각형 길이 점수를 계산한다.
    - 발음 명확도(=WhisperX score) 70%와 무음 비율(30%)을 결합해 컨텐츠 점수를 만든다.
    - 컨텐츠:길이 = 35%:65% 가중합으로 최종 점수를 만든다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    total_length = len(vocals_audio)
    best_candidates: Dict[str, dict[str, Any]] = {}

    for seg in segments:
        speaker = getattr(seg, "speaker", None)
        text = (getattr(seg, "text", "") or "").strip()
        if not speaker or not text:
            continue

        seg_start = getattr(seg, "start_ms", 0) or 0
        seg_end = getattr(seg, "end_ms", seg_start)
        start_ms = max(0, min(seg_start, total_length))
        end_ms = max(start_ms, min(seg_end, total_length))
        duration_ms = end_ms - start_ms

        if (
            duration_ms < MIN_REF_DURATION_MS
            or duration_ms > MAX_REF_DURATION_MS
            or duration_ms <= 0
        ):
            continue

        # pydub 슬라이스는 끝점이 전체 길이를 넘어가면 자동으로 자르므로 재확인
        sample_audio = vocals_audio[start_ms:end_ms]
        if len(sample_audio) < MIN_REF_DURATION_MS:
            continue

        clarity_val = getattr(seg, "score", None)
        try:
            clarity_score = max(
                0.0,
                min(1.0, float(clarity_val)),
            )
        except (TypeError, ValueError):
            clarity_score = 0.0

        speech_density = _speech_density_score(sample_audio)
        content_score = CLARITY_WEIGHT * clarity_score + SILENCE_WEIGHT * speech_density
        length_score = _calculate_length_score(duration_ms)
        final_score = QUALITY_WEIGHT * content_score + LENGTH_WEIGHT * length_score

        # 컨텐츠 품질이 너무 낮으면 스킵
        if content_score < MIN_CONTENT_SCORE:
            continue

        current = best_candidates.get(speaker)
        if current is None or final_score > current["score"]:
            best_candidates[speaker] = {
                "segment": seg,
                "score": final_score,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }

    references: Dict[str, SpeakerReferenceSample] = {}

    for speaker, candidate in best_candidates.items():
        seg: SegmentView = candidate["segment"]
        score = candidate["score"]
        start_ms = candidate["start_ms"]
        end_ms = candidate["end_ms"]

        sample_audio = vocals_audio[start_ms:end_ms]
        if len(sample_audio) == 0:
            continue

        final_audio = sample_audio.set_frame_rate(16000).set_channels(1)
        if len(final_audio) == 0:
            continue

        ref_path = out_dir / f"{speaker}_self_ref.wav"
        final_audio.export(ref_path, format="wav")

        references[speaker] = SpeakerReferenceSample(
            speaker=speaker,
            audio_path=ref_path,
            text=seg.text.strip(),
            segment_idx=seg.idx,
            segment_id=seg.segment_id(),
            start_ms=start_ms,
            end_ms=end_ms,
            audio_duration_ms=len(final_audio),
            score=round(score, 4),
        )

    logger.info(
        "Prepared %d self-reference samples (max %d ms each)",
        len(references),
        MAX_REF_DURATION_MS,
    )
    return references
