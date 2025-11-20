from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from services.lang import normalize_lang_code

SCHEMA_VERSION = 1
# 기본 저장 포맷을 gzip(.gz)에서 평문 JSON(.json)으로 변경
# 기존 .gz 파일과의 호환을 위해 로드 시 자동 폴백을 지원합니다.
COMPACT_ARCHIVE_NAME = "transcript.comp.json"

DEFAULT_SPEAKER_NAME = "SPEAKER_00"
UNKNOWN_SPEAKER_NAME = "unknown_speaker"


def _to_ms(value) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value) * 1000))
    except (TypeError, ValueError):
        return None


def _ms_to_seconds(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / 1000, 3)


def _quantize_score(value) -> int:
    if value is None:
        return 0
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    return int(round(score * 255))


def _normalize_speaker(raw) -> str:
    if raw is None:
        return DEFAULT_SPEAKER_NAME
    if isinstance(raw, int):
        return f"SPEAKER_{raw:02d}"
    value = str(raw).strip()
    if not value:
        return DEFAULT_SPEAKER_NAME
    if value == UNKNOWN_SPEAKER_NAME:
        return DEFAULT_SPEAKER_NAME
    return value


def _ensure_speaker_index(name: str, table: dict[str, int], ordered: list[str]) -> int:
    if name in table:
        return table[name]
    idx = len(ordered)
    table[name] = idx
    ordered.append(name)
    return idx


def _ensure_vocab_index(
    token: str, table: dict[str, int], ordered: list[str]
) -> int:
    if token in table:
        return table[token]
    idx = len(ordered)
    table[token] = idx
    ordered.append(token)
    return idx


@dataclass(frozen=True)
class SegmentView:
    idx: int
    start_ms: int
    end_ms: int
    speaker: str
    speaker_unknown: bool
    text: str
    gap_after_ms: int | None
    gap_after_vad_ms: int | None
    word_start: int
    word_count: int
    overlap: bool
    orig: int | None
    score_q: int | None = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def start_seconds(self) -> float:
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        return self.end_ms / 1000.0

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

    @property
    def score(self) -> float | None:
        if self.score_q is None:
            return None
        return round(self.score_q / 255.0, 4)

    def segment_id(self) -> str:
        return f"segment_{self.idx:04d}"

    def to_public_dict(self) -> dict:
        return {
            "idx": self.idx,
            "segment_id": self.segment_id(),
            "speaker": self.speaker,
            "speaker_unknown": self.speaker_unknown,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start": _ms_to_seconds(self.start_ms),
            "end": _ms_to_seconds(self.end_ms),
            "duration_ms": self.duration_ms,
            "duration": _ms_to_seconds(self.duration_ms),
            "text": self.text,
            "gap_after_ms": self.gap_after_ms,
            "gap_after_vad_ms": self.gap_after_vad_ms,
            "gap_after": _ms_to_seconds(self.gap_after_ms),
            "gap_after_vad": _ms_to_seconds(self.gap_after_vad_ms),
            "word_count": self.word_count,
            "overlap": self.overlap,
            "orig_segment_id": self.orig,
            "score": self.score,
        }


def build_compact_transcript(
    aligned_segments: Sequence[dict], language: str | None = None
) -> dict:
    """Convert WhisperX-aligned segments into the compact schema."""
    speakers: list[str] = []
    speaker_index: dict[str, int] = {}
    vocab: list[str] = []
    vocab_index: dict[str, int] = {}
    compact_segments: list[dict] = []
    compact_words: list[list[int]] = []
    prev_end_ms: int | None = None

    for idx, seg in enumerate(aligned_segments):
        start_ms = _to_ms(seg.get("start"))
        end_ms = _to_ms(seg.get("end"))
        if start_ms is None or end_ms is None:
            continue
        text = (seg.get("text") or "").strip()
        speaker_name = _normalize_speaker(seg.get("speaker"))
        speaker_idx = _ensure_speaker_index(speaker_name, speaker_index, speakers)

        w_start = len(compact_words)
        word_scores: list[float] = []
        words = seg.get("words") or []
        for word in words:
            token = (word.get("word") or "").strip()
            if not token:
                continue
            w_abs_start = _to_ms(word.get("start"))
            w_abs_end = _to_ms(word.get("end"))
            if w_abs_start is None or w_abs_end is None:
                continue
            offset_start = max(0, w_abs_start - start_ms)
            offset_end = max(offset_start, w_abs_end - start_ms)
            vocab_idx = _ensure_vocab_index(token, vocab_index, vocab)
            score_val = word.get("score")
            if score_val is not None:
                try:
                    word_scores.append(float(score_val))
                except (TypeError, ValueError):
                    pass
            score_q = _quantize_score(score_val)
            compact_words.append(
                [idx, offset_start, offset_end, vocab_idx, score_q]
            )
        w_count = len(compact_words) - w_start
        segment_score_q: int | None = None
        if word_scores:
            avg_score = sum(word_scores) / len(word_scores)
            segment_score_q = _quantize_score(avg_score)
        else:
            fallback_score = seg.get("score")
            if fallback_score is not None:
                segment_score_q = _quantize_score(fallback_score)

        next_start_ms = (
            _to_ms(aligned_segments[idx + 1].get("start"))
            if idx + 1 < len(aligned_segments)
            else None
        )
        gap_after = (
            next_start_ms - end_ms if next_start_ms is not None else None
        )
        gap_after_vad = (
            max(gap_after, 0) if gap_after is not None else None
        )
        overlap = bool(prev_end_ms is not None and start_ms < prev_end_ms)
        prev_end_ms = end_ms if prev_end_ms is None else max(prev_end_ms, end_ms)

        segment_entry = {
            "s": start_ms,
            "e": end_ms,
            "sp": speaker_idx,
            "txt": text,
            "gap": [gap_after, gap_after_vad],
            "w_off": [w_start, w_count],
            "o": seg.get("id", idx),
            "ov": overlap,
        }
        if segment_score_q is not None:
            segment_entry["sc"] = segment_score_q

        compact_segments.append(segment_entry)

    return {
        "v": SCHEMA_VERSION,
        "unit": "ms",
        "lang": language,
        "speakers": speakers,
        "segments": compact_segments,
        "vocab": vocab,
        "words": compact_words,
    }


def save_compact_transcript(bundle: dict, path: Path) -> None:
    """Compact transcript를 평문 JSON으로 저장합니다.

    기존에는 gzip(.gz) 압축으로 저장했으나, I/O 성능 이슈를 고려해
    이제 기본은 `.json` 평문 파일로 저장합니다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")


def load_compact_transcript(path: Path) -> dict:
    """Compact transcript 로드.

    우선 지정된 경로를 시도하고, 없으면 `.gz` ↔ `.json` 상호 폴백을 시도합니다.
    둘 다 없으면 FileNotFoundError.
    """
    candidate = path
    if not candidate.is_file():
        # 확장자 폴백(.json ↔ .gz)
        if candidate.suffix == ".gz":
            alt = candidate.with_suffix("")  # .gz 제거 → .json일 가능성
        else:
            alt = candidate.with_suffix(candidate.suffix + ".gz")
        if alt.is_file():
            candidate = alt
        else:
            raise FileNotFoundError(f"Transcript archive not found: {path}")

    data: bytes
    if candidate.suffix == ".gz":
        with gzip.open(candidate, "rb") as fh:
            data = fh.read()
    else:
        data = candidate.read_bytes()
    return json.loads(data.decode("utf-8"))


def segment_views(bundle: dict) -> List[SegmentView]:
    speakers = bundle.get("speakers") or []
    views: list[SegmentView] = []
    for idx, seg in enumerate(bundle.get("segments") or []):
        start_ms = int(seg.get("s") or 0)
        end_ms = int(seg.get("e") or start_ms)
        sp_idx = seg.get("sp")
        speaker_unknown = False
        if isinstance(sp_idx, int) and 0 <= sp_idx < len(speakers):
            resolved_speaker = speakers[sp_idx]
        else:
            resolved_speaker = UNKNOWN_SPEAKER_NAME
            speaker_unknown = True
        if resolved_speaker == UNKNOWN_SPEAKER_NAME:
            speaker_unknown = True
            speaker = DEFAULT_SPEAKER_NAME
        else:
            speaker = resolved_speaker
        gap = seg.get("gap") or [None, None]
        w_off = seg.get("w_off") or [0, 0]
        score_q = seg.get("sc")
        score_val = None
        if isinstance(score_q, int):
            score_val = max(0, min(255, score_q))
        views.append(
            SegmentView(
                idx=idx,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=speaker,
                speaker_unknown=speaker_unknown,
                text=seg.get("txt") or "",
                gap_after_ms=gap[0],
                gap_after_vad_ms=gap[1],
                word_start=w_off[0],
                word_count=w_off[1],
                overlap=bool(seg.get("ov")),
                orig=seg.get("o"),
                score_q=score_val,
            )
        )
    return views


def segment_preview(bundle: dict) -> list[dict]:
    return [view.to_public_dict() for view in segment_views(bundle)]


def read_transcript_language(transcript_path: Path) -> str | None:
    """Load a compact transcript and return its normalized language code."""

    try:
        bundle = load_compact_transcript(transcript_path)
    except FileNotFoundError:
        return None
    return normalize_lang_code(bundle.get("lang"))
