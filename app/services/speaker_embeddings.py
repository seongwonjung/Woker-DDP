# services/speaker_embeddings.py
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from .self_reference import SpeakerReferenceSample

_VOICE_ENCODER: VoiceEncoder | None = None


def _get_encoder() -> VoiceEncoder:
    """Lazy-load the global Resemblyzer encoder."""
    global _VOICE_ENCODER
    if _VOICE_ENCODER is None:
        _VOICE_ENCODER = VoiceEncoder()
    return _VOICE_ENCODER


def _relative_str(path: Path, base_dir: Path) -> str:
    try:
        return path.relative_to(base_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _serialize_embedding_payload(
    *,
    label: str,
    embedding: list[float],
    audio_path: Path,
    base_dir: Path,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": label,
        "embedding": embedding,
        "dim": len(embedding),
        "audio": _relative_str(audio_path, base_dir),
    }
    if meta:
        payload.update(meta)
    return payload


def embed_audio_file(audio_path: Path) -> list[float]:
    """
    Compute a Resemblyzer embedding for the provided wav file.

    The file is kept local for now, but the resulting JSON payload can be
    mirrored to `s3://<bucket>/voice-samples/...` once remote storage is ready.
    """
    wav = preprocess_wav(str(audio_path))
    if wav.size == 0:
        raise ValueError(f"Audio at {audio_path} is empty after preprocessing.")
    encoder = _get_encoder()
    vector = encoder.embed_utterance(wav)
    return vector.astype(float).tolist() if isinstance(vector, np.ndarray) else list(vector)


def save_embedding_payload(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_reference_embeddings(
    speaker_refs: Dict[str, "SpeakerReferenceSample"],
    output_dir: Path,
    *,
    base_dir: Path,
) -> dict[str, dict[str, Any]]:
    """
    Compute embeddings for prepared self-reference clips and persist them to disk.
    Returns mapping speaker -> payload for downstream use.
    """
    from .self_reference import SpeakerReferenceSample  # Local import to avoid cycles

    output_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, dict[str, Any]] = {}

    for speaker, ref in speaker_refs.items():
        audio_path = ref.audio_path
        if not audio_path.is_file():
            logger.warning("Speaker %s reference audio missing at %s", speaker, audio_path)
            continue
        try:
            embedding = embed_audio_file(audio_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to embed speaker %s reference: %s", speaker, exc)
            continue

        meta = {
            "speaker": speaker,
            "segment_id": ref.segment_id,
            "start_ms": ref.start_ms,
            "end_ms": ref.end_ms,
            "duration_ms": ref.audio_duration_ms,
            "score": ref.score,
            "source": "self_reference",
        }
        payload = _serialize_embedding_payload(
            label=speaker,
            embedding=embedding,
            audio_path=audio_path,
            base_dir=base_dir,
            meta=meta,
        )
        payloads[speaker] = payload
        save_embedding_payload(payload, output_dir / f"{speaker}.json")

    index_path = output_dir / "speaker_embeddings.json"
    save_embedding_payload(payloads, index_path)
    logger.info("Saved %d speaker embeddings to %s", len(payloads), index_path)
    return payloads


def save_audio_embedding(
    audio_path: Path,
    output_path: Path,
    *,
    label: str,
    base_dir: Path | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Embed an arbitrary audio file and write the payload to JSON."""
    base_dir = base_dir or output_path.parent
    embedding = embed_audio_file(audio_path)
    payload = _serialize_embedding_payload(
        label=label,
        embedding=embedding,
        audio_path=audio_path,
        base_dir=base_dir,
        meta=meta,
    )
    save_embedding_payload(payload, output_path)
    return payload


def load_embedding_index(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.is_file():
        return {}
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse speaker embedding index %s: %s", index_path, exc)
        return {}
    if isinstance(payload, dict):
        return payload
    logger.warning("Unexpected embedding index format at %s (%s)", index_path, type(payload).__name__)
    return {}
