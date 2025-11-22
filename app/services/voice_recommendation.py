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

from services.speaker_embeddings import load_embedding_index

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


def _resolve_local_library_path(language: str | None, index_path: Path | None) -> Path:
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
    """Load the target-language voice library metadata.

    먼저 로컬 캐시를 확인하고, S3의 LastModified와 비교하여
    필요할 때만 다운로드합니다.
    """
    from datetime import datetime, timezone
    from utils.s3 import download_from_s3

    path = _resolve_local_library_path(language, index_path)
    payload = None
    source = None

    # S3 정보 준비
    bucket = os.getenv("VOICE_LIBRARY_BUCKET") or os.getenv("AWS_S3_BUCKET")
    lang_slug = _normalize_lang(language) or "default"
    candidates = _library_s3_candidates(language)
    remote_key = None

    # 로컬 파일이 있는 경우
    if path.is_file():
        # S3의 LastModified 확인
        should_download = False
        if bucket and candidates:
            try:
                client = _get_s3_client()
                for key in candidates:
                    try:
                        response = client.head_object(Bucket=bucket, Key=key)
                        s3_last_modified = response.get("LastModified")
                        if s3_last_modified:
                            # 로컬 파일의 수정 시간
                            local_modified = datetime.fromtimestamp(
                                path.stat().st_mtime, tz=timezone.utc
                            )
                            # S3가 더 최신이면 다운로드 필요
                            if s3_last_modified > local_modified:
                                logger.info(
                                    "S3 voice library is newer (S3: %s, Local: %s), "
                                    "downloading...",
                                    s3_last_modified,
                                    local_modified,
                                )
                                should_download = True
                                # 다운로드할 키 저장
                                remote_key = key
                                break
                            else:
                                logger.debug(
                                    "Local voice library is up to date (S3: %s, Local: %s)",
                                    s3_last_modified,
                                    local_modified,
                                )
                                break
                    except ClientError as exc:
                        error_code = exc.response.get("Error", {}).get("Code")
                        if error_code in {"NoSuchKey", "404"}:
                            continue
                        logger.warning(
                            "Failed to check S3 voice library metadata: %s", exc
                        )
                        break
                    except BotoCoreError as exc:
                        logger.warning(
                            "Failed to check S3 voice library metadata: %s", exc
                        )
                        break
            except Exception as exc:
                logger.warning("Failed to check S3 voice library metadata: %s", exc)

        # 로컬 파일이 최신이면 로컬 파일 사용
        if not should_download:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                source = str(path)
                logger.info("Using cached voice library from %s", path)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Failed to read voice library %s: %s", path, exc)
                # 로컬 파일 읽기 실패 시 S3에서 다운로드 시도
                should_download = True

        # S3에서 다운로드 필요하면 다운로드
        if should_download and bucket and candidates:
            # remote_key가 설정되지 않았으면 첫 번째 candidate 사용
            if remote_key is None:
                remote_key = candidates[0]

            # 로컬 디렉토리 생성
            path.parent.mkdir(parents=True, exist_ok=True)

            if download_from_s3(bucket, remote_key, path, force=True):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    source = f"s3://{bucket}/{remote_key} (downloaded)"
                    logger.info("Downloaded and loaded voice library from S3")
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Failed to read downloaded voice library %s: %s", path, exc
                    )
            else:
                logger.warning("Failed to download voice library from S3")
                # 다운로드 실패 시 기존 로컬 파일 사용 시도
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    source = str(path) + " (fallback)"
                    logger.info("Using existing local file after download failure")
                except (OSError, json.JSONDecodeError):
                    pass

    # 로컬 파일이 없는 경우
    else:
        logger.info("Voice library index not found locally at %s", path)

        # S3에서 다운로드 시도
        if bucket and candidates:
            remote_key = candidates[0]
            path.parent.mkdir(parents=True, exist_ok=True)

            if download_from_s3(bucket, remote_key, path, force=True):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    source = f"s3://{bucket}/{remote_key} (downloaded)"
                    logger.info("Downloaded and loaded voice library from S3")
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Failed to read downloaded voice library %s: %s", path, exc
                    )
            else:
                # 다운로드 실패 시 S3에서 직접 읽기 시도 (기존 방식)
                payload, source = _load_library_payload_from_s3(language)
                if payload:
                    # 다운로드는 실패했지만 메모리에서 읽었으므로 로컬에 저장 시도
                    try:
                        with open(path, "w", encoding="utf-8") as f:
                            json.dump(payload, f, ensure_ascii=False, indent=2)
                        logger.info("Saved voice library to local cache: %s", path)
                    except Exception as exc:
                        logger.warning("Failed to save voice library to cache: %s", exc)

    # payload가 없으면 빈 리스트 반환
    if payload is None:
        logger.info("Voice library index not found")
        return []

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
    candidates = [
        entry for entry in library if not target_norm or entry.language == target_norm
    ]
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


def strip_voice_samples_prefix(value: str) -> str:
    """voice-samples/ prefix를 제거합니다."""
    marker = "voice-samples/"
    key = value
    if key.startswith("s3://"):
        remainder = key.split("://", 1)[1]
        if "/" in remainder:
            key = remainder.split("/", 1)[1]
        else:
            key = ""
    if marker in key:
        key = key.split(marker, 1)[1]
    return key.lstrip("/")


def ensure_voice_library_index(
    language: str,
    force_refresh: bool = False,
    voice_samples_embed_dir: Path | None = None,
    voice_library_bucket: str | None = None,
) -> Path | None:
    """Voice sample metadata is mirrored from S3; refresh before touching the local cache."""
    from services.lang import normalize_lang_code
    from utils.s3 import download_from_s3

    lang_slug = normalize_lang_code(language) or "misc"
    if voice_samples_embed_dir is None:
        voice_samples_embed_dir = _default_library_root()
    lang_dir = voice_samples_embed_dir / lang_slug
    lang_dir.mkdir(parents=True, exist_ok=True)
    local_path = lang_dir / f"{lang_slug}.json"
    remote_key = f"voice-samples/embedding/{lang_slug}/{lang_slug}.json"
    if voice_library_bucket is None:
        voice_library_bucket = os.getenv("VOICE_LIBRARY_BUCKET") or os.getenv(
            "AWS_S3_BUCKET"
        )
    if force_refresh or not local_path.is_file():
        if voice_library_bucket:
            download_from_s3(voice_library_bucket, remote_key, local_path)
    return local_path if local_path.is_file() else None


def resolve_s3_location(raw: str, default_bucket: str) -> tuple[str, str]:
    """
    Parse strings like 's3://bucket/key' or bare keys into (bucket, key).
    Falls back to default_bucket when explicit bucket is missing.
    """
    value = (raw or "").strip()
    if not value:
        raise ValueError("빈 S3 위치 문자열입니다.")
    if value.startswith("s3://"):
        remainder = value[5:]
        if "/" not in remainder:
            raise ValueError(f"Invalid S3 URI: {raw}")
        bucket, key = remainder.split("/", 1)
        return bucket, key
    key = value.lstrip("/")
    return default_bucket, key


def materialize_voice_replacements(
    paths,
    replacements: dict[str, VoiceReplacement],
    default_bucket: str,
) -> dict[str, dict]:
    """Voice replacement 샘플을 로컬에 다운로드하고 준비합니다."""
    from utils.s3 import download_from_s3

    asset_dir = paths.interim_dir / "voice_replacements"
    asset_dir.mkdir(parents=True, exist_ok=True)
    prepared: dict[str, dict] = {}
    for speaker, plan in replacements.items():
        entry = plan.entry
        local_path: Path | None = None
        if entry.sample_path:
            candidate = Path(entry.sample_path)
            if not candidate.is_absolute():
                candidate = (paths.interim_dir / candidate).resolve()
            if candidate.is_file():
                local_path = candidate
            else:
                logger.warning(
                    "Voice replacement sample for %s not found at %s",
                    entry.voice_id,
                    candidate,
                )
                continue
        elif entry.sample_key:
            bucket = entry.sample_bucket or default_bucket
            local_path = asset_dir / f"{speaker}_{entry.voice_id}.wav"
            # Voice replacement clips live in S3; download locally when preparing overrides.
            if not download_from_s3(bucket, entry.sample_key, local_path):
                logger.warning(
                    "Failed to download voice replacement sample %s from s3://%s/%s",
                    entry.voice_id,
                    bucket,
                    entry.sample_key,
                )
                continue
        else:
            logger.warning(
                "Voice library entry %s lacks sample reference.", entry.voice_id
            )
            continue

        prepared[speaker] = {
            "audio_path": str(local_path),
            "prompt_text": entry.prompt_text,
            "voice_id": entry.voice_id,
            "similarity": plan.similarity,
            "sample_key": entry.sample_key,
            "sample_bucket": entry.sample_bucket or default_bucket,
            "metadata": entry.metadata or {},
            "language": entry.language,
        }
    return prepared


def maybe_prepare_voice_replacements(
    paths,
    target_lang: str,
    default_bucket: str,
    voice_samples_embed_dir: Path | None = None,
    voice_library_bucket: str | None = None,
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Voice replacement를 준비합니다."""
    diagnostics: dict[str, Any] = {
        "enabled": False,
        "target_lang": target_lang,
    }
    index_path = paths.vid_tts_dir / "speaker_embeddings" / "speaker_embeddings.json"
    embeddings = load_embedding_index(index_path)
    if not embeddings:
        diagnostics["reason"] = "missing_embeddings"
        return {}, diagnostics

    ensure_voice_library_index(
        target_lang,
        force_refresh=True,
        voice_samples_embed_dir=voice_samples_embed_dir,
        voice_library_bucket=voice_library_bucket,
    )
    library = load_voice_library(
        target_lang, voice_samples_embed_dir or _default_library_root()
    )
    if not library:
        diagnostics["reason"] = "library_unavailable"
        return {}, diagnostics

    replacements = recommend_voice_replacements(
        embeddings,
        library,
        target_lang=target_lang,
    )
    if not replacements:
        diagnostics["reason"] = "no_matches"
        return {}, diagnostics

    prepared = materialize_voice_replacements(paths, replacements, default_bucket)
    if not prepared:
        diagnostics["reason"] = "materialization_failed"
        return {}, diagnostics

    diagnostics["enabled"] = True
    diagnostics["reason"] = "ok"
    diagnostics["matches"] = {
        speaker: repl.summary() for speaker, repl in replacements.items()
    }
    diagnostics["prepared_speakers"] = sorted(prepared.keys())
    return prepared, diagnostics
