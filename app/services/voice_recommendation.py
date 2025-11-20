# services/voice_recommendation.py
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import numpy as np

logger = logging.getLogger(__name__)

_S3_CLIENT: Any | None = None


def _default_library_index() -> Path:
    raw = os.getenv("VOICE_LIBRARY_INDEX")
    if raw:
        return Path(raw).expanduser()
    return Path("/data/voice-samples/embedding/default.json")


def _default_library_root() -> Path:
    raw = os.getenv("VOICE_LIBRARY_DIR")
    if raw:
        return Path(raw).expanduser()
    return Path("/data/voice-samples/embedding")


def _normalize_lang(value: str | None) -> str:
    return (value or "").strip().lower()


def _resolve_local_library_path(
    language: str | None, index_path: Path | None
) -> Path:
    root = index_path or _default_library_root()
    if root.is_dir():
        lang_slug = _normalize_lang(language) or "default"
        candidate = root / lang_slug / f"{lang_slug}.json"
        if not candidate.is_file():
            alt_candidate = root / f"{lang_slug}.json"
            return alt_candidate if alt_candidate.is_file() else candidate
        return candidate
    return root if root.is_file() else _default_library_index()


def _get_s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        region = os.getenv("AWS_REGION")
        kwargs = {"region_name": region} if region else {}
        _S3_CLIENT = boto3.client("s3", **kwargs)
    return _S3_CLIENT


def _library_s3_candidates(language: str | None) -> list[str]:
    lang_slug = _normalize_lang(language) or "default"
    candidates = [
        f"voice-samples/embedding/{lang_slug}/{lang_slug}.json",
        f"voice-samples/embedding/{lang_slug}.json",
    ]
    if lang_slug != "default":
        candidates.append("voice-samples/embedding/default/default.json")
        candidates.append("voice-samples/embedding/default.json")
    # Preserve order but remove duplicates
    seen = set()
    ordered: list[str] = []
    for key in candidates:
        if key not in seen:
            ordered.append(key)
            seen.add(key)
    return ordered


def _load_library_payload_from_s3(
    language: str | None,
) -> tuple[Any | None, str | None]:
    bucket = os.getenv("VOICE_LIBRARY_BUCKET") or os.getenv("AWS_S3_BUCKET")
    if not bucket:
        return None, None
    candidates = _library_s3_candidates(language)
    if not candidates:
        return None, None
    try:
        client = _get_s3_client()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Unable to create S3 client for voice library: %s", exc)
        return None, None

    for key in candidates:
        try:
            response = client.get_object(Bucket=bucket, Key=key)
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code in {"NoSuchKey", "404"}:
                continue
            logger.warning(
                "Failed to fetch voice library from s3://%s/%s: %s",
                bucket,
                key,
                exc,
            )
            return None, None
        except BotoCoreError as exc:
            logger.warning(
                "Failed to fetch voice library from s3://%s/%s: %s", bucket, key, exc
            )
            return None, None

        body = response.get("Body")
        if body is None:
            continue
        try:
            raw = body.read()
        finally:
            body.close()

        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Voice library payload at s3://%s/%s is invalid JSON: %s",
                bucket,
                key,
                exc,
            )
            continue

        logger.info("Loaded voice library from s3://%s/%s", bucket, key)
        return payload, f"s3://{bucket}/{key}"

    logger.info("Voice library not found in S3 bucket %s for %s", bucket, language)
    return None, None


def _to_vector(values: Any) -> np.ndarray | None:
    if isinstance(values, np.ndarray):
        vec = values.astype(float)
        return vec if vec.size else None
    if isinstance(values, (list, tuple)):
        try:
            vec = np.asarray(values, dtype=float)
        except ValueError:
            return None
        return vec if vec.size else None
    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class VoiceLibraryEntry:
    voice_id: str
    language: str
    embedding: list[float]
    sample_key: str | None = None
    sample_bucket: str | None = None
    sample_path: str | None = None
    prompt_text: str | None = None
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_payload(
        cls, payload: dict[str, Any], fallback_language: str | None = None
    ) -> "VoiceLibraryEntry":
        voice_id = payload.get("voice_id") or payload.get("id")
        language = payload.get("language") or payload.get("lang") or fallback_language
        embedding = payload.get("embedding")
        if not voice_id or not language or not embedding:
            raise ValueError("Voice library entry missing required keys.")
        vec = [float(x) for x in embedding]
        sample = payload.get("sample") or {}
        if isinstance(sample, str):
            sample_key = sample
            sample_bucket = None
            sample_path = None
        else:
            sample_key = sample.get("key") or sample.get("s3_key")
            sample_bucket = sample.get("bucket")
            sample_path = sample.get("path")
        default_bucket = (
            payload.get("sample_bucket")
            or sample_bucket
            or os.getenv("VOICE_LIBRARY_BUCKET")
            or os.getenv("AWS_S3_BUCKET")
        )
        return cls(
            voice_id=str(voice_id),
            language=_normalize_lang(str(language)),
            embedding=vec,
            sample_key=payload.get("sample_key") or sample_key,
            sample_bucket=default_bucket,
            sample_path=payload.get("sample_path") or sample_path,
            prompt_text=payload.get("prompt_text") or sample.get("prompt_text"),
            metadata=payload.get("metadata"),
        )


@dataclass
class VoiceReplacement:
    speaker: str
    similarity: float
    entry: VoiceLibraryEntry

    def summary(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "voice_id": self.entry.voice_id,
            "similarity": self.similarity,
            "language": self.entry.language,
            "sample_key": self.entry.sample_key,
        }


def load_voice_library(
    language: str | None = None, index_path: Path | None = None
) -> list[VoiceLibraryEntry]:
    """Load the target-language voice library metadata."""
    payload, source = _load_library_payload_from_s3(language)
    path: Path | None = None
    if payload is None:
        path = _resolve_local_library_path(language, index_path)
        if not path.is_file():
            logger.info("Voice library index not found at %s", path)
            return []

        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read voice library %s: %s", path, exc)
            return []
        source = str(path)

    entries: list[VoiceLibraryEntry] = []
    raw_entries: Iterable[Any]
    if isinstance(payload, dict):
        raw_entries = payload.get("voices") or payload.get("entries") or []
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        logger.warning(
            "Voice library format must be list/dict, found %s", type(payload).__name__
        )
        return []

    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        try:
            entries.append(
                VoiceLibraryEntry.from_payload(item, fallback_language=language)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Skipping invalid voice entry: %s", exc)
    location = source or (str(path) if path else "unknown")
    logger.info("Loaded %d voice library entries from %s", len(entries), location)
    return entries


def update_voice_library_entry(
    language: str,
    entry: Dict[str, Any],
    base_dir: Path | None = None,
) -> Path:
    """Insert or update a voice entry inside the per-language library file."""
    lang_slug = _normalize_lang(language) or "default"
    root = base_dir or _default_library_root()
    lang_dir = root / lang_slug
    lang_dir.mkdir(parents=True, exist_ok=True)
    path = lang_dir / f"{lang_slug}.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        payload = []
    if not isinstance(payload, list):
        payload = []

    updated = False
    for idx, existing in enumerate(payload):
        if isinstance(existing, dict) and existing.get("voice_id") == entry.get(
            "voice_id"
        ):
            payload[idx] = entry
            updated = True
            break
    if not updated:
        payload.append(entry)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


def recommend_voice_replacements(
    job_embeddings: Dict[str, dict[str, Any]],
    library: Sequence[VoiceLibraryEntry],
    target_lang: str | None = None,
    min_similarity: float = 0.0,
) -> dict[str, VoiceReplacement]:
    """
    Recommend target-language voices based on cosine similarity between
    speaker embeddings and the shared voice library.
    """
    if not job_embeddings or not library:
        return {}

    target_norm = _normalize_lang(target_lang)
    candidates = [entry for entry in library if not target_norm or entry.language == target_norm]
    if not candidates:
        candidates = list(library)

    matches: dict[str, VoiceReplacement] = {}
    for speaker, payload in job_embeddings.items():
        vector = _to_vector(payload.get("embedding"))
        if vector is None:
            continue
        best_entry = None
        best_score = 0.0
        for entry in candidates:
            entry_vec = _to_vector(entry.embedding)
            if entry_vec is None:
                continue
            score = _cosine_similarity(vector, entry_vec)
            if score > best_score:
                best_score = score
                best_entry = entry
        if best_entry:
            # 항상 최고 유사도 항목을 사용 (임계치 미달이어도 선택)
            matches[speaker] = VoiceReplacement(
                speaker=speaker,
                similarity=float(best_score),
                entry=best_entry,
            )
    return matches
