"""Helpers for computing per-segment quality diagnostics."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def stt_quality_from_quantized(score_q: Optional[int]) -> Optional[int]:
    """Convert WhisperX quantized confidence (0-255) to a 0-100 score."""
    if not isinstance(score_q, int):
        return None
    normalized = _clamp(score_q, 0, 255) / 255.0
    return int(round(normalized * 100))


def tts_quality_from_ratio(ratio: Optional[float]) -> Optional[int]:
    """
    Derive a 0-100 score from duration ratio (tts_duration / target_duration).

    Small deviations keep the score in the 80-100 range while large gaps drop it.
    """
    if ratio is None or ratio <= 0:
        return None
    deviation = abs(1.0 - ratio)
    normalized = _clamp(1.0 - _clamp(deviation, 0.0, 1.0), 0.0, 1.0)
    return int(round(normalized * 100))


def voice_quality_from_similarity(similarity: Optional[float]) -> Optional[int]:
    """
    0.0~1.0 코사인 유사도를 0~100 점수로 변환.
    0.5 이상이면 대략 정상 영역, 그 이하는 점점 나쁜 쪽으로 본다.
    """
    if similarity is None:
        return None
    try:
        sim = float(similarity)
    except (TypeError, ValueError):
        return None
    sim = _clamp(sim, 0.0, 1.0)
    # 0.0 -> 0점, 1.0 -> 100점
    return int(round(sim * 100))


def sync_percent_from_durations(
    source_seconds: Optional[float], synced_seconds: Optional[float]
) -> Optional[int]:
    """Return % difference between synced audio and source duration."""
    if (
        source_seconds is None
        or synced_seconds is None
        or source_seconds <= 0
        or synced_seconds <= 0
    ):
        return None
    delta = (synced_seconds / source_seconds - 1.0) * 100.0
    return int(round(delta))


def ensure_issue_payload(existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ensure the issue payload has a `q` dict with `stt/tts/sync` keys.

    Existing entries are shallow-copied so callers can freely mutate.
    """
    issues: Dict[str, Any] = dict(existing or {})
    q_payload = dict(issues.get("q") or {})
    issues["q"] = q_payload
    q_payload.setdefault("stt", None)
    q_payload.setdefault("tts", None)
    q_payload.setdefault("sync", None)
    q_payload.setdefault("voice", None)
    if "spk" not in issues:
        issues["spk"] = issues.get("spk")
    return issues


def build_segment_issues(
    *,
    stt_score_q: Optional[int] = None,
    tts_ratio: Optional[float] = None,
    sync_percent: Optional[int] = None,
    speaker_unknown: Optional[bool] = None,
    voice_similarity: Optional[float] = None,
    voice_low_similarity_forced: Optional[bool] = None,
    base: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Attach the requested metrics (if provided) to the issue payload."""
    issues = ensure_issue_payload(base)
    q_payload = issues["q"]
    if stt_score_q is not None:
        q_payload["stt"] = stt_quality_from_quantized(stt_score_q)
    if tts_ratio is not None:
        q_payload["tts"] = tts_quality_from_ratio(tts_ratio)
    if sync_percent is not None:
        q_payload["sync"] = sync_percent
    if voice_similarity is not None:
        q_payload["voice"] = voice_quality_from_similarity(voice_similarity)
    if speaker_unknown is not None:
        issues["spk"] = bool(speaker_unknown)
    elif "spk" not in issues or issues["spk"] is None:
        issues["spk"] = False
    if voice_low_similarity_forced is not None:
        issues["voice_low_sim_forced"] = bool(voice_low_similarity_forced)
    return issues
