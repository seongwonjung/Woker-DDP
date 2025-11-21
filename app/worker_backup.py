import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import boto3
import requests
from botocore.exceptions import ClientError, BotoCoreError

# 워커의 서비스 모듈들
from services.lang import normalize_lang_code
from services.stt import run_asr
from services.translate import translate_transcript
from services.tts import generate_tts
from services.sync import sync_segments, _sync_single_segment, MAX_SLOW_RATIO
from services.mux import mux_audio_video
from configs import JobPaths, ensure_job_dirs
from services.demucs_split import split_vocals
from services.tts import (
    _transcribe_prompt_text,
    _trim_tts_artifacts,
    _synthesize_with_cosyvoice2,
)
from services.transcript_store import COMPACT_ARCHIVE_NAME, read_transcript_language
from services.speaker_embeddings import (
    load_embedding_index,
    save_audio_embedding,
)
from services.voice_recommendation import (
    VoiceReplacement,
    load_voice_library,
    recommend_voice_replacements,
    update_voice_library_entry,
)
from pydub import AudioSegment
import shutil

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
noisy_loggers = ["boto3", "botocore", "s3transfer", "urllib3"]
for logger_name in noisy_loggers:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# Numba 디버그 로그 억제
for name in [
    "numba",
    "numba.core",
    "numba.core.ssa",
    "numba.core.byteflow",
    "numba.core.typeinfer",
]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    logger.propagate = False  # 부모(root)로 안 올리게
    logger.handlers.clear()  # 혹시 자기 handler 갖고 있으면 날려버리기

# AWS 설정
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
JOB_QUEUE_URL = os.getenv("JOB_QUEUE_URL")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
VOICE_LIBRARY_BUCKET = os.getenv("VOICE_LIBRARY_BUCKET") or AWS_S3_BUCKET

# FIFO 큐 자동 감지: 환경변수 또는 큐 URL로 판단
JOB_QUEUE_FIFO_ENV = os.getenv("JOB_QUEUE_FIFO", "false").lower() == "true"
JOB_QUEUE_FIFO_URL = JOB_QUEUE_URL and JOB_QUEUE_URL.endswith(".fifo")
JOB_QUEUE_FIFO = JOB_QUEUE_FIFO_ENV or JOB_QUEUE_FIFO_URL
JOB_QUEUE_MESSAGE_GROUP_ID = os.getenv("JOB_QUEUE_MESSAGE_GROUP_ID")

# FIFO 큐 설정 로깅
logging.info(
    f"SQS Queue Configuration: FIFO={JOB_QUEUE_FIFO} "
    f"(env={JOB_QUEUE_FIFO_ENV}, url_detected={JOB_QUEUE_FIFO_URL}), "
    f"MessageGroupId={JOB_QUEUE_MESSAGE_GROUP_ID or 'will use project_id'}, "
    f"QueueURL={JOB_QUEUE_URL}"
)

# 청킹 설정
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "5"))  # 기본값: 5개 세그먼트당 1 청크

VOICE_SAMPLES_ROOT = Path(os.getenv("VOICE_SAMPLES_ROOT", "/data/voice-samples"))
VOICE_SAMPLES_SAMPLES_DIR = VOICE_SAMPLES_ROOT / "samples"
VOICE_SAMPLES_TTS_DIR = VOICE_SAMPLES_ROOT / "tts"
VOICE_SAMPLES_EMBED_DIR = VOICE_SAMPLES_ROOT / "embedding"
for directory in (
    VOICE_SAMPLES_ROOT,
    VOICE_SAMPLES_SAMPLES_DIR,
    VOICE_SAMPLES_TTS_DIR,
    VOICE_SAMPLES_EMBED_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

if not all([JOB_QUEUE_URL, AWS_S3_BUCKET]):
    raise ValueError(
        "JOB_QUEUE_URL and AWS_S3_BUCKET environment variables must be set."
    )

sqs_client = boto3.client("sqs", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)


def send_callback(
    callback_url: str,
    status: str,
    message: str,
    stage: str | None = None,
    metadata: dict | None = None,
):
    """백엔드로 진행 상황 콜백을 보냅니다."""
    try:
        payload = {"status": status, "message": message}

        # 메타데이터 구성
        callback_metadata = metadata or {}
        if stage:
            callback_metadata["stage"] = stage

        if callback_metadata:
            payload["metadata"] = callback_metadata

        response = requests.post(callback_url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info(
            f"Sent callback to {callback_url} with status: {status}, stage: {stage or 'N/A'}"
        )
    except requests.RequestException as e:
        logging.error(f"Failed to send callback to {callback_url}: {e}")


def download_from_s3(bucket: str, key: str, local_path: Path) -> bool:
    """S3에서 파일을 다운로드합니다. 파일이 이미 존재하면 건너뜁니다."""
    # 파일이 이미 존재하면 다운로드 건너뛰기
    if local_path.exists() and local_path.is_file():
        logging.debug(
            f"File already exists at {local_path}, skipping download from s3://{bucket}/{key}"
        )
        return True

    try:
        logging.info(f"Downloading s3://{bucket}/{key} to {local_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        logging.info(f"Successfully downloaded s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to download from S3: {e}")
        return False


def upload_to_s3(bucket: str, key: str, local_path: Path) -> bool:
    """S3로 파일을 업로드합니다."""
    try:
        logging.info(f"Uploading {local_path} to s3://{bucket}/{key}...")
        s3_client.upload_file(str(local_path), bucket, key)
        logging.info(f"Successfully uploaded to s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to upload to S3: {e}")
        return False
    except FileNotFoundError:
        logging.error(f"Local file not found for upload: {local_path}")
        return False


def resolve_output_prefix(
    project_id: str | None, job_id: str, override: str | None
) -> str:
    """결과물을 저장할 기본 경로를 계산합니다."""
    if override:
        return override.rstrip("/")
    if project_id:
        return f"projects/{project_id}/outputs/{job_id}"
    return f"jobs/{job_id}/outputs"


def _build_sqs_message_kwargs(
    message_body: str,
    project_id: str | None = None,
    deduplication_id: str | None = None,
    group_id: str | None = None,
) -> dict:
    """FIFO 큐를 위한 SQS 메시지 파라미터를 구성합니다."""
    kwargs = {
        "QueueUrl": JOB_QUEUE_URL,
        "MessageBody": message_body,
    }

    if JOB_QUEUE_FIFO:
        # FIFO 큐를 위한 파라미터 추가
        # group_id가 제공되면 사용, 아니면 기본값 사용
        if group_id:
            final_group_id = group_id
        else:
            final_group_id = JOB_QUEUE_MESSAGE_GROUP_ID or project_id or "default"
        kwargs["MessageGroupId"] = final_group_id

        # DeduplicationId는 메시지 내용의 해시 또는 제공된 ID 사용
        if deduplication_id:
            kwargs["MessageDeduplicationId"] = deduplication_id
        else:
            # 메시지 본문의 해시를 사용 (간단한 방법)
            import hashlib

            kwargs["MessageDeduplicationId"] = hashlib.md5(
                message_body.encode("utf-8")
            ).hexdigest()

        logging.debug(
            f"SQS FIFO message: GroupId={final_group_id}, "
            f"DeduplicationId={kwargs.get('MessageDeduplicationId')}"
        )

    return kwargs


def upload_metadata_to_s3(bucket: str, key: str, metadata: dict) -> bool:
    """파이프라인 메타데이터를 JSON으로 직렬화해 S3에 업로드합니다."""
    try:
        body = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        logging.info(f"Uploading metadata to s3://{bucket}/{key}...")
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )
        logging.info(f"Successfully uploaded metadata to s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to upload metadata to S3: {e}")
        return False


def _strip_voice_samples_prefix(value: str) -> str:
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


# Voice sample metadata is mirrored from S3; refresh before touching the local cache
# so future DB migrations have a single integration point.
def _ensure_voice_library_index(
    language: str, force_refresh: bool = False
) -> Path | None:
    lang_slug = normalize_lang_code(language) or "misc"
    lang_dir = VOICE_SAMPLES_EMBED_DIR / lang_slug
    lang_dir.mkdir(parents=True, exist_ok=True)
    local_path = lang_dir / f"{lang_slug}.json"
    remote_key = f"voice-samples/embedding/{lang_slug}/{lang_slug}.json"
    if force_refresh or not local_path.is_file():
        download_from_s3(VOICE_LIBRARY_BUCKET, remote_key, local_path)
    return local_path if local_path.is_file() else None


def _resolve_s3_location(raw: str, default_bucket: str) -> tuple[str, str]:
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


def _sync_segment_to_range(
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


def _segments_with_remote_audio_paths(
    segments: list[dict],
    project_prefix: str,
    job_id: str,
    paths: JobPaths,
) -> list[dict]:
    """
    Copy segment dicts while rewriting local `/data/interim/<job_id>` audio paths
    to remote keys that mirror the uploaded layout.
    """
    if not segments:
        return []
    base_dir = paths.interim_dir
    remote_prefix = f"{project_prefix}/interim/{job_id}"
    normalized: list[dict] = []
    for segment in segments:
        updated = dict(segment)
        audio_value = updated.get("audio_file")
        if isinstance(audio_value, str):
            if audio_value.startswith("s3://") or audio_value.startswith(remote_prefix):
                normalized.append(updated)
                continue
            candidate = Path(audio_value)
            try:
                relative_path = candidate.relative_to(base_dir)
            except ValueError:
                logging.debug(
                    "audio_file 경로 %s 가 %s 기준 상대 경로가 아닙니다. 원본 값을 유지합니다.",
                    candidate,
                    base_dir,
                )
            else:
                updated["audio_file"] = f"{remote_prefix}/{relative_path.as_posix()}"
        normalized.append(updated)
    return normalized


def _build_speaker_metadata(
    paths: JobPaths, project_prefix: str, job_id: str
) -> list[dict]:
    """
    Collect speaker metadata consisting of speaker name, uploaded sample key,
    and optional prompt text.
    Returns list format for TTS completion callback.
    """
    speaker_refs_json_path = paths.vid_tts_dir / "speaker_refs.json"
    if not speaker_refs_json_path.is_file():
        return []

    try:
        refs = json.loads(speaker_refs_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to parse %s: %s", speaker_refs_json_path, exc)
        return []

    remote_prefix = f"{project_prefix}/interim/{job_id}"
    base_dir = paths.interim_dir.resolve()
    metadata: list[dict] = []

    for speaker, payload in refs.items():
        if isinstance(payload, str):
            audio_value = payload
            prompt_text = ""
        elif isinstance(payload, dict):
            audio_value = payload.get("audio") or payload.get("path") or ""
            prompt_text = (payload.get("text") or "").strip()
        else:
            continue

        if not audio_value:
            continue

        sample_path = Path(audio_value)
        if not sample_path.is_absolute():
            sample_path = (paths.vid_tts_dir / sample_path).resolve()

        try:
            rel_path = sample_path.relative_to(base_dir)
            voice_sample_key = f"{remote_prefix}/{rel_path.as_posix()}"
        except ValueError:
            logging.warning(
                "Voice sample %s is outside interim dir; using absolute path.",
                sample_path,
            )
            voice_sample_key = str(sample_path)

        entry = {
            "speaker": speaker,
            "voice_sample_key": voice_sample_key,
        }
        if prompt_text:
            entry["prompt_text"] = prompt_text
        metadata.append(entry)

    return metadata


def _build_speaker_refs_metadata(
    paths: JobPaths,
    project_prefix: str,
    job_id: str,
    output_bucket: str,
) -> dict:
    """
    Upload speaker reference samples to S3 and return metadata in dict format.
    Returns dict format for final pipeline callback: {"speaker0": {"ref_wav_key": "s3://...", "prompt_text": "..."}}
    """
    speaker_refs_metadata = {}
    tts_dir = paths.vid_tts_dir
    speaker_ref_dir = tts_dir / "self_refs"
    speaker_refs_json_path = tts_dir / "speaker_refs.json"

    if not speaker_refs_json_path.is_file():
        return speaker_refs_metadata

    try:
        # speaker_refs.json 읽기
        with open(speaker_refs_json_path, "r", encoding="utf-8") as f:
            speaker_refs_mapping = json.load(f)

        # self_refs 디렉토리의 모든 wav 파일을 S3에 업로드
        if speaker_ref_dir.is_dir():
            for ref_file in speaker_ref_dir.glob("*.wav"):
                try:
                    relative_path = ref_file.relative_to(paths.interim_dir)
                except ValueError:
                    relative_path = ref_file.relative_to(speaker_ref_dir)
                    logging.warning(
                        "Speaker ref 경로 %s 를 interim 디렉터리 기준으로 계산하지 못했습니다. "
                        "self_refs 디렉터리 상대 경로를 사용합니다.",
                        ref_file,
                    )
                ref_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
                if upload_to_s3(output_bucket, str(ref_key), ref_file):
                    logging.info(
                        f"Speaker ref uploaded to s3://{output_bucket}/{ref_key}"
                    )

        # speaker_refs.json도 S3에 업로드
        refs_json_key = f"{project_prefix}/interim/{job_id}/tts/speaker_refs.json"
        if upload_to_s3(output_bucket, refs_json_key, speaker_refs_json_path):
            logging.info(
                f"Speaker refs JSON uploaded to s3://{output_bucket}/{refs_json_key}"
            )

        # 각 스피커별 ref_wav의 S3 키와 prompt_text를 매핑
        for speaker, ref_data in speaker_refs_mapping.items():
            if isinstance(ref_data, dict):
                audio_path = ref_data.get("audio", "")
                prompt_text = ref_data.get("text", "")
                # 상대 경로를 절대 경로로 변환
                if audio_path and not Path(audio_path).is_absolute():
                    ref_audio_path = tts_dir / audio_path
                else:
                    ref_audio_path = Path(audio_path) if audio_path else None

                if ref_audio_path and ref_audio_path.exists():
                    try:
                        relative_path = ref_audio_path.relative_to(paths.interim_dir)
                    except ValueError:
                        relative_path = ref_audio_path.relative_to(tts_dir)
                    ref_s3_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
                    speaker_refs_metadata[speaker] = {
                        "ref_wav_key": f"s3://{output_bucket}/{ref_s3_key}",
                        "prompt_text": prompt_text,
                    }
                else:
                    # 파일이 없으면 speaker_refs.json의 audio 경로를 기반으로 S3 키 생성
                    if audio_path:
                        ref_s3_key = (
                            f"{project_prefix}/interim/{job_id}/tts/{audio_path}"
                        )
                        speaker_refs_metadata[speaker] = {
                            "ref_wav_key": f"s3://{output_bucket}/{ref_s3_key}",
                            "prompt_text": prompt_text,
                        }
            else:
                logging.warning(
                    f"Unexpected format for speaker {speaker} in speaker_refs.json"
                )

        if speaker_refs_metadata:
            logging.info(
                f"Prepared speaker_refs metadata for {len(speaker_refs_metadata)} speakers"
            )
    except Exception as exc:
        logging.warning(
            f"Failed to upload speaker_refs for job {job_id}: {exc}", exc_info=True
        )

    return speaker_refs_metadata


def _upload_speaker_embeddings(
    paths: JobPaths,
    job_id: str,
    bucket: str,
) -> dict:
    """
    Upload locally cached speaker embeddings to the shared voice-samples prefix.
    Returns metadata describing where the embeddings were stored.
    """
    embedding_dir = paths.vid_tts_dir / "speaker_embeddings"
    if not embedding_dir.is_dir():
        return {}
    uploaded: dict[str, str] = {}
    base_prefix = f"voice-samples/jobs/{job_id}/embeddings"
    for file in embedding_dir.glob("*.json"):
        if not file.is_file():
            continue
        s3_key = f"{base_prefix}/{file.name}"
        if upload_to_s3(bucket, s3_key, file):
            uploaded[file.name] = f"s3://{bucket}/{s3_key}"
    if not uploaded:
        return {}
    return {"prefix": f"s3://{bucket}/{base_prefix}", "files": uploaded}


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _materialize_voice_replacements(
    paths: JobPaths,
    replacements: dict[str, VoiceReplacement],
    default_bucket: str,
) -> dict[str, dict]:
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
                logging.warning(
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
                logging.warning(
                    "Failed to download voice replacement sample %s from s3://%s/%s",
                    entry.voice_id,
                    bucket,
                    entry.sample_key,
                )
                continue
        else:
            logging.warning(
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


def _maybe_prepare_voice_replacements(
    paths: JobPaths,
    target_lang: str,
    default_bucket: str,
) -> tuple[dict[str, dict], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "enabled": False,
        "target_lang": target_lang,
    }
    index_path = paths.vid_tts_dir / "speaker_embeddings" / "speaker_embeddings.json"
    embeddings = load_embedding_index(index_path)
    if not embeddings:
        diagnostics["reason"] = "missing_embeddings"
        return {}, diagnostics

    _ensure_voice_library_index(target_lang, force_refresh=True)
    library = load_voice_library(target_lang, VOICE_SAMPLES_EMBED_DIR)
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

    prepared = _materialize_voice_replacements(paths, replacements, default_bucket)
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


def _parse_positive_int(value, field_name: str) -> int | None:
    """Optional int parser that tolerates strings and invalid inputs."""
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logging.warning(
            "Ignoring %s=%r because it is not an integer", field_name, value
        )
        return None
    if parsed < 1:
        logging.warning("Ignoring %s=%r because it must be >= 1", field_name, value)
        return None
    return parsed


def check_and_trigger_mux_if_complete(
    job_id: str,
    project_id: str | None,
    total_segments: int,
    output_bucket: str,
    project_prefix: str,
    callback_url: str | None,
    job_details: dict,
) -> bool:
    """S3의 synced .wav 파일 개수를 확인하고, 모두 완료되면 mux 큐에 메시지 전송.

    Returns:
        True if mux was triggered, False otherwise
    """
    synced_prefix = f"{project_prefix}/interim/{job_id}/tts/synced/"

    # S3에서 .wav 파일 개수 확인 (pagination 처리)
    wav_count = 0
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=output_bucket, Prefix=synced_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".wav"):
                    wav_count += 1
    except ClientError as e:
        logging.error(f"Job {job_id}: Failed to list S3 objects for mux check: {e}")
        return False

    logging.info(
        f"Job {job_id}: Found {wav_count} synced .wav files, "
        f"expected {total_segments} segments"
    )

    # 모든 세그먼트가 완료되었는지 확인
    if wav_count >= total_segments:
        # 중복 실행 방지를 위한 락 파일 확인
        mux_lock_key = f"{project_prefix}/interim/{job_id}/mux_lock.json"

        try:
            # 락 파일이 이미 있으면 다른 워커가 이미 트리거했음
            s3_client.head_object(Bucket=output_bucket, Key=mux_lock_key)
            logging.info(
                f"Job {job_id}: Mux already triggered by another worker, skipping"
            )
            return False
        except ClientError:
            # 락 파일이 없으면 생성 (원자적 연산 시도)
            try:
                lock_data = {
                    "job_id": job_id,
                    "project_id": project_id,
                    "triggered_at": time.time(),
                    "wav_count": wav_count,
                    "total_segments": total_segments,
                }
                upload_metadata_to_s3(output_bucket, mux_lock_key, lock_data)
                logging.info(f"Job {job_id}: Created mux lock file at {mux_lock_key}")
            except Exception as e:
                logging.warning(
                    f"Job {job_id}: Failed to create mux lock, "
                    f"another worker may have triggered: {e}"
                )
                return False

        # manifest에서 input_key와 input_bucket 가져오기
        manifest_key = job_details.get("manifest_key")
        input_key = job_details.get("input_key")
        input_bucket = job_details.get("input_bucket")

        # manifest에서 input_key가 없으면 manifest를 다운로드해서 읽기
        if not input_key and manifest_key:
            try:
                import tempfile

                with tempfile.NamedTemporaryFile(
                    mode="w+", delete=False, suffix=".json"
                ) as tmp_file:
                    tmp_path = Path(tmp_file.name)

                if download_from_s3(output_bucket, manifest_key, tmp_path):
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    input_key = manifest.get("input_key")
                    input_bucket = manifest.get("input_bucket") or output_bucket
                    tmp_path.unlink()  # 임시 파일 삭제
            except Exception as e:
                logging.warning(
                    f"Job {job_id}: Failed to read input_key from manifest: {e}"
                )

        # Mux 큐에 메시지 전송
        mux_message = {
            "task": "mux",
            "job_id": job_id,
            "project_id": project_id,
            "total_segments": total_segments,
            "output_bucket": output_bucket,
            "project_prefix": project_prefix,
            "callback_url": callback_url,
            "manifest_key": manifest_key,
            "input_key": input_key,
            "input_bucket": input_bucket or output_bucket,
            # 기타 필요한 정보 전달
            "target_lang": job_details.get("target_lang"),
            "source_lang": job_details.get("source_lang"),
        }

        try:
            message_body = json.dumps(mux_message)
            message_kwargs = _build_sqs_message_kwargs(
                message_body,
                project_id=project_id,
                deduplication_id=f"{job_id}_mux",
            )
            response = sqs_client.send_message(**message_kwargs)
            logging.debug(
                f"Job {job_id}: SQS send_message response: {response.get('MessageId')}"
            )
            logging.info(
                f"Job {job_id}: Triggered mux task. "
                f"Found {wav_count} synced .wav files (expected {total_segments})"
            )

            if callback_url:
                send_callback(
                    callback_url,
                    "in_progress",
                    "All chunks completed. Starting mux...",
                    stage="mux_triggered",
                    metadata={
                        "job_id": job_id,
                        "project_id": project_id,
                        "total_segments": total_segments,
                        "wav_files_found": wav_count,
                    },
                )

            return True
        except Exception as e:
            logging.error(f"Job {job_id}: Failed to send mux message to queue: {e}")
            return False

    return False


def create_chunks(total_segments: int, chunk_size: int) -> list[dict]:
    """세그먼트를 청크로 분할합니다.

    Args:
        total_segments: 전체 세그먼트 개수
        chunk_size: 각 청크에 포함될 세그먼트 개수

    Returns:
        청크 정보 리스트. 각 청크는 다음 정보를 포함:
        - chunk_index: 청크 인덱스 (0부터 시작)
        - start_segment_index: 시작 세그먼트 인덱스 (inclusive)
        - end_segment_index: 끝 세그먼트 인덱스 (inclusive)
        - segment_count: 이 청크에 포함된 세그먼트 개수
    """
    if total_segments <= 0:
        return []
    if chunk_size <= 0:
        chunk_size = CHUNK_SIZE

    chunks = []
    for i in range(0, total_segments, chunk_size):
        end_idx = min(i + chunk_size - 1, total_segments - 1)
        chunks.append(
            {
                "chunk_index": i // chunk_size,
                "start_segment_index": i,
                "end_segment_index": end_idx,
                "segment_count": end_idx - i + 1,
            }
        )

    return chunks


def full_pipeline(job_details: dict):
    """전체 더빙 파이프라인을 실행합니다."""
    job_id = job_details["job_id"]
    project_id = job_details.get("project_id")
    input_key = job_details["input_key"]
    callback_url = job_details["callback_url"]

    target_lang = job_details.get("target_lang", "en")
    source_lang = normalize_lang_code(job_details.get("source_lang"))
    speaker_count = _parse_positive_int(
        job_details.get("speaker_count"), "speaker_count"
    )
    voice_config = job_details.get("voice_config")
    voice_replacement_flag = (
        job_details.get("replace_voice_samples")
        or job_details.get("is_replace_voice_samples")
        or job_details.get("voice_sample_substitution")
    )
    replace_voice_samples = _parse_bool(voice_replacement_flag)
    voice_library_bucket = (
        job_details.get("voice_library_bucket") or VOICE_LIBRARY_BUCKET
    )
    input_bucket = job_details.get("input_bucket") or AWS_S3_BUCKET
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET
    project_prefix = f"projects/{project_id}" if project_id else "jobs"
    output_prefix = resolve_output_prefix(
        project_id, job_id, job_details.get("output_prefix")
    )
    metadata_key = (
        job_details.get("metadata_key") or f"{output_prefix}/metadata/{job_id}.json"
    )

    send_callback(
        callback_url,
        "in_progress",
        f"Starting full pipeline for job {job_id}",
        stage="starting",
        metadata={
            "job_id": job_id,
            "project_id": project_id,
            "target_lang": target_lang,
        },
    )

    # 1. 로컬 작업 디렉토리 설정
    paths = ensure_job_dirs(job_id)
    source_video_path = paths.input_dir / Path(input_key).name

    # 파이프라인 시간 측정 초기화
    timing_info = {
        "job_id": job_id,
        "project_id": project_id,
        "pipeline_start_time": time.time(),
        "stages": {},
    }

    # 2. S3에서 원본 영상 다운로드
    stage_name = "download"
    stage_start = time.time()
    if not download_from_s3(input_bucket, input_key, source_video_path):
        send_callback(
            callback_url,
            "failed",
            "Failed to download source video from S3.",
            stage="download_failed",
            metadata={"job_id": job_id, "project_id": project_id},
        )
        return
    stage_end = time.time()
    timing_info["stages"][stage_name] = {
        "start_time": stage_start,
        "end_time": stage_end,
        "duration_seconds": stage_end - stage_start,
    }

    # 3. voice_config에서 사용자 음성 샘플 다운로드 (필요 시)
    user_voice_sample_path = None
    if voice_config and voice_config.get("kind") == "s3" and voice_config.get("key"):
        voice_key = voice_config["key"]
        voice_bucket = (
            voice_config.get("bucket")
            or voice_config.get("bucket_name")
            or input_bucket
        )
        user_voice_sample_path = paths.interim_dir / Path(voice_key).name
        if not download_from_s3(voice_bucket, voice_key, user_voice_sample_path):
            send_callback(
                callback_url,
                "failed",
                f"Failed to download voice sample from S3 key: {voice_key}",
                stage="download_failed",
                metadata={"job_id": job_id, "project_id": project_id},
            )
            return
        send_callback(
            callback_url,
            "in_progress",
            "Custom voice sample downloaded.",
            stage="downloaded",
            metadata={"job_id": job_id, "project_id": project_id},
        )

    translations: list[dict] = []
    segments_payload: list[dict] = []
    speaker_metadata: list[dict] = []
    final_audio_path: Path | None = None
    detected_source_lang: str | None = None
    effective_source_lang: str | None = source_lang
    speaker_voice_overrides: dict[str, dict] = {}
    voice_replacement_meta: dict[str, Any] = {
        "requested": replace_voice_samples,
        "enabled": False,
        "target_lang": target_lang,
    }

    try:
        # 4. ASR (STT)
        stage_name = "asr"
        stage_start = time.time()
        send_callback(
            callback_url, "in_progress", "Starting ASR...", stage="asr_started"
        )
        run_asr(
            job_id,
            source_video_path,
            source_lang=source_lang,
            speaker_count=speaker_count,
        )
        stage_end = time.time()
        timing_info["stages"][stage_name] = {
            "start_time": stage_start,
            "end_time": stage_end,
            "duration_seconds": stage_end - stage_start,
        }
        # ASR 결과물(compact transcript)을 S3에 업로드
        asr_result_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
        upload_to_s3(
            output_bucket,
            f"{project_prefix}/interim/{job_id}/{COMPACT_ARCHIVE_NAME}",
            asr_result_path,
        )
        detected_source_lang = read_transcript_language(asr_result_path)
        if effective_source_lang is None and detected_source_lang:
            effective_source_lang = detected_source_lang
        # 원본 오디오(audio.wav)를 S3에 업로드
        raw_audio_path = paths.vid_speaks_dir / "audio.wav"
        audio_key = None
        if raw_audio_path.is_file():
            audio_key = f"{project_prefix}/interim/{job_id}/audio/audio.wav"
            if upload_to_s3(output_bucket, audio_key, raw_audio_path):
                logging.info(f"Raw audio uploaded to s3://{output_bucket}/{audio_key}")
            else:
                logging.warning("Failed to upload audio.wav to S3")

        # 발화 음성(vocals.wav)과 배경음(background.wav)을 S3에 업로드
        vocals_path = paths.vid_speaks_dir / "vocals.wav"
        background_path = paths.vid_bgm_dir / "background.wav"

        vocals_key = None
        background_key = None

        if vocals_path.is_file():
            vocals_key = f"{project_prefix}/interim/{job_id}/audio/vocals.wav"
            if upload_to_s3(output_bucket, vocals_key, vocals_path):
                logging.info(f"Vocals uploaded to s3://{output_bucket}/{vocals_key}")
            else:
                logging.warning("Failed to upload vocals.wav to S3")

        if background_path.is_file():
            background_key = f"{project_prefix}/interim/{job_id}/audio/background.wav"
            if upload_to_s3(output_bucket, background_key, background_path):
                logging.info(
                    f"Background uploaded to s3://{output_bucket}/{background_key}"
                )
            else:
                logging.warning("Failed to upload background.wav to S3")

        send_callback(
            callback_url,
            "in_progress",
            "ASR completed.",
            stage="asr_completed",
            metadata=(
                {
                    "audio_key": audio_key,
                    "vocals_key": vocals_key,
                    "background_key": background_key,
                }
                if (audio_key or vocals_key or background_key)
                else None
            ),
        )

        # 5. 번역
        stage_name = "translation"
        stage_start = time.time()
        send_callback(
            callback_url,
            "in_progress",
            "Starting translation...",
            stage="translation_started",
        )
        translations = translate_transcript(
            job_id, target_lang, src_lang=effective_source_lang
        )
        stage_end = time.time()
        timing_info["stages"][stage_name] = {
            "start_time": stage_start,
            "end_time": stage_end,
            "duration_seconds": stage_end - stage_start,
        }
        # 번역 결과물(translated.json)을 S3에 업로드
        trans_result_path = paths.trg_sentence_dir / "translated.json"
        upload_to_s3(
            output_bucket,
            f"{project_prefix}/interim/{job_id}/translated.json",
            trans_result_path,
        )
        send_callback(
            callback_url,
            "in_progress",
            "Translation completed.",
            stage="translation_completed",
        )
        if replace_voice_samples:
            overrides, diagnostics = _maybe_prepare_voice_replacements(
                paths, target_lang, voice_library_bucket or output_bucket
            )
            speaker_voice_overrides = overrides
            voice_replacement_meta.update(diagnostics)
        else:
            voice_replacement_meta.setdefault("reason", "not_requested")

        # 6. TTS
        stage_name = "tts"
        stage_start = time.time()
        send_callback(
            callback_url, "in_progress", "Starting TTS...", stage="tts_started"
        )
        segments_payload = generate_tts(
            job_id,
            target_lang,
            voice_sample_path=user_voice_sample_path,
            speaker_voice_overrides=(
                speaker_voice_overrides if speaker_voice_overrides else None
            ),
        )
        stage_end = time.time()
        timing_info["stages"][stage_name] = {
            "start_time": stage_start,
            "end_time": stage_end,
            "duration_seconds": stage_end - stage_start,
        }
        # TTS 결과물(개별 wav 파일 및 segments.json)을 S3에 업로드
        tts_dir = paths.vid_tts_dir
        for tts_file in tts_dir.glob("**/*"):
            if not tts_file.is_file():
                continue
            try:
                relative_path = tts_file.relative_to(paths.interim_dir)
            except ValueError:
                relative_path = tts_file.relative_to(tts_dir)
                logging.warning(
                    "TTS 경로 %s 를 interim 디렉터리 기준으로 계산하지 못했습니다. "
                    "tts 디렉터리 상대 경로를 사용합니다.",
                    tts_file,
                )
            tts_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
            upload_to_s3(output_bucket, str(tts_key), tts_file)
        speaker_metadata = _build_speaker_metadata(paths, project_prefix, job_id)
        send_callback(
            callback_url,
            "in_progress",
            "TTS completed.",
            stage="tts_completed",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "speakers": speaker_metadata,
                "speaker_count": len(speaker_metadata),
                "voice_replacement": voice_replacement_meta,
            },
        )

        # 7. Sync
        stage_name = "sync"
        stage_start = time.time()
        send_callback(
            callback_url, "in_progress", "Starting sync...", stage="sync_started"
        )
        try:
            synced_segments = sync_segments(job_id)
        except FileNotFoundError as exc:
            logging.info(
                "Sync artifacts not found, skipping segment alignment: %s", exc
            )
            synced_segments = []
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Sync step failed, proceeding without alignment: %s", exc)
            synced_segments = []
        else:
            if synced_segments:
                segments_payload = synced_segments
        # Sync 결과물(synced 디렉토리)을 S3에 업로드
        synced_dir = paths.vid_tts_dir / "synced"
        for sync_file in synced_dir.glob("**/*"):
            if not sync_file.is_file():
                continue
            try:
                relative_path = sync_file.relative_to(paths.interim_dir)
            except ValueError:
                relative_path = sync_file.relative_to(synced_dir)
                logging.warning(
                    "Synced 경로 %s 를 interim 디렉터리 기준으로 계산하지 못했습니다. "
                    "synced 디렉터리 상대 경로를 사용합니다.",
                    sync_file,
                )
            sync_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
            upload_to_s3(output_bucket, str(sync_key), sync_file)
        stage_end = time.time()
        timing_info["stages"][stage_name] = {
            "start_time": stage_start,
            "end_time": stage_end,
            "duration_seconds": stage_end - stage_start,
        }
        send_callback(
            callback_url, "in_progress", "Sync completed.", stage="sync_completed"
        )

        # 8. Mux
        stage_name = "mux"
        stage_start = time.time()
        send_callback(
            callback_url, "in_progress", "Starting mux...", stage="mux_started"
        )
        mux_results = mux_audio_video(job_id, source_video_path)
        stage_end = time.time()
        timing_info["stages"][stage_name] = {
            "start_time": stage_start,
            "end_time": stage_end,
            "duration_seconds": stage_end - stage_start,
        }
        output_video_path = Path(mux_results["output_video"])
        final_audio_path = Path(mux_results["output_audio"])

        # 9. 최종 결과물 S3에 업로드
        result_key = f"projects/{project_id}/outputs/dubbed_video.mp4"
        if not upload_to_s3(output_bucket, result_key, output_video_path):
            raise Exception("Failed to upload final video to S3")

        metadata_segments = _segments_with_remote_audio_paths(
            segments_payload,
            project_prefix,
            job_id,
            paths,
        )

        metadata_payload = {
            "job_id": job_id,
            "project_id": project_id,
            "target_lang": target_lang,
            "source_lang": effective_source_lang,
            "input_bucket": input_bucket,
            "input_key": input_key,
            "result_bucket": output_bucket,
            "result_key": result_key,
            "metadata_key": metadata_key,
            "segments": metadata_segments,
            "segment_count": len(metadata_segments),
            "translations": translations,
            "speakers": speaker_metadata,
            "speaker_count": len(speaker_metadata),
            "voice_replacement": voice_replacement_meta,
        }
        if detected_source_lang:
            metadata_payload["detected_source_lang"] = detected_source_lang
        if final_audio_path:
            metadata_payload["audio_artifact"] = str(final_audio_path)

        # 10. Speaker reference samples를 S3에 업로드하고 콜백으로 전달
        speaker_refs_metadata = _build_speaker_refs_metadata(
            paths, project_prefix, job_id, output_bucket
        )
        speaker_embeddings_metadata = _upload_speaker_embeddings(
            paths, job_id, output_bucket
        )

        final_metadata = {
            "job_id": job_id,
            "project_id": project_id,
            "result_bucket": output_bucket,
            "result_key": result_key,
            "metadata_key": metadata_key,
            "segment_count": len(metadata_segments),
            "speaker_count": len(speaker_metadata),
            "target_lang": target_lang,
        }
        if speaker_embeddings_metadata:
            metadata_payload["speaker_embeddings"] = speaker_embeddings_metadata

        if not upload_metadata_to_s3(output_bucket, metadata_key, metadata_payload):
            raise Exception("Failed to upload metadata to S3")

        final_metadata["voice_replacement"] = voice_replacement_meta
        if effective_source_lang:
            final_metadata["source_lang"] = effective_source_lang
        if detected_source_lang:
            final_metadata["detected_source_lang"] = detected_source_lang
        if speaker_refs_metadata:
            final_metadata["speaker_refs"] = speaker_refs_metadata
        if speaker_embeddings_metadata:
            final_metadata["speaker_embeddings"] = speaker_embeddings_metadata

        # 파이프라인 종료 시간 계산 및 요약
        pipeline_end_time = time.time()
        timing_info["pipeline_end_time"] = pipeline_end_time
        timing_info["total_duration_seconds"] = (
            pipeline_end_time - timing_info["pipeline_start_time"]
        )

        # 각 단계별 소요 시간 요약 추가
        timing_summary = {}
        for stage, info in timing_info["stages"].items():
            timing_summary[stage] = {
                "duration_seconds": round(info["duration_seconds"], 2),
                "duration_minutes": round(info["duration_seconds"] / 60, 2),
            }
        timing_info["summary"] = timing_summary
        timing_info["total_duration_minutes"] = round(
            timing_info["total_duration_seconds"] / 60, 2
        )

        # 시간 정보를 S3에 업로드
        timing_key = f"{project_prefix}/full_pipeline_timing.json"
        if not upload_metadata_to_s3(output_bucket, timing_key, timing_info):
            logging.warning(f"Failed to upload timing info to S3: {timing_key}")

        send_callback(
            callback_url,
            "done",
            "Pipeline completed successfully.",
            stage="done",
            metadata=final_metadata,
        )

    except Exception as e:
        logging.error(f"Pipeline failed for job {job_id}: {e}", exc_info=True)
        send_callback(
            callback_url,
            "failed",
            str(e),
            stage="failed",
            metadata={"job_id": job_id, "project_id": project_id},
        )


def split_up(job_details: dict):
    """ASR 수행 후 청킹하고 각 청크를 큐에 전송합니다."""
    from services.transcript_store import load_compact_transcript, segment_views

    job_id = job_details["job_id"]
    project_id = job_details.get("project_id")
    input_key = job_details["input_key"]
    callback_url = job_details["callback_url"]

    target_lang = job_details.get("target_lang", "en")
    source_lang = normalize_lang_code(job_details.get("source_lang"))
    speaker_count = _parse_positive_int(
        job_details.get("speaker_count"), "speaker_count"
    )
    input_bucket = job_details.get("input_bucket") or AWS_S3_BUCKET
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET
    project_prefix = f"projects/{project_id}" if project_id else "jobs"

    # 청크 크기 설정 (job_details에서 오버라이드 가능)
    chunk_size = (
        _parse_positive_int(job_details.get("chunk_size"), "chunk_size") or CHUNK_SIZE
    )

    send_callback(
        callback_url,
        "in_progress",
        f"Starting split_up for job {job_id}",
        stage="split_up_started",
        metadata={
            "job_id": job_id,
            "project_id": project_id,
            "target_lang": target_lang,
            "chunk_size": chunk_size,
        },
    )

    # 1. 로컬 작업 디렉토리 설정
    paths = ensure_job_dirs(job_id)
    source_video_path = paths.input_dir / Path(input_key).name

    # 2. S3에서 원본 영상 다운로드
    if not download_from_s3(input_bucket, input_key, source_video_path):
        send_callback(
            callback_url,
            "failed",
            "Failed to download source video from S3.",
            stage="download_failed",
            metadata={"job_id": job_id, "project_id": project_id},
        )
        return

    detected_source_lang: str | None = None
    effective_source_lang: str | None = source_lang

    try:
        # 3. ASR (STT) 수행
        send_callback(
            callback_url, "in_progress", "Starting ASR...", stage="asr_started"
        )
        run_asr(
            job_id,
            source_video_path,
            source_lang=source_lang,
            speaker_count=speaker_count,
        )

        # ASR 결과물 로드
        asr_result_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
        detected_source_lang = read_transcript_language(asr_result_path)
        if effective_source_lang is None and detected_source_lang:
            effective_source_lang = detected_source_lang

        # compact transcript 로드하여 세그먼트 개수 확인
        bundle = load_compact_transcript(asr_result_path)
        segments = segment_views(bundle)
        total_segments = len(segments)

        logging.info(f"Job {job_id}: ASR completed. Total segments: {total_segments}")

        # 4. ASR 결과물을 S3에 업로드
        compact_transcript_key = (
            f"{project_prefix}/interim/{job_id}/{COMPACT_ARCHIVE_NAME}"
        )
        if not upload_to_s3(output_bucket, compact_transcript_key, asr_result_path):
            raise Exception("Failed to upload compact transcript to S3")

        # 원본 오디오(audio.wav)를 S3에 업로드
        raw_audio_path = paths.vid_speaks_dir / "audio.wav"
        audio_key = None
        if raw_audio_path.is_file():
            audio_key = f"{project_prefix}/interim/{job_id}/audio/audio.wav"
            if not upload_to_s3(output_bucket, audio_key, raw_audio_path):
                logging.warning("Failed to upload audio.wav to S3")

        # 발화 음성(vocals.wav)과 배경음(background.wav)을 S3에 업로드
        vocals_path = paths.vid_speaks_dir / "vocals.wav"
        background_path = paths.vid_bgm_dir / "background.wav"

        vocals_key = None
        background_key = None

        if vocals_path.is_file():
            vocals_key = f"{project_prefix}/interim/{job_id}/audio/vocals.wav"
            if not upload_to_s3(output_bucket, vocals_key, vocals_path):
                logging.warning("Failed to upload vocals.wav to S3")

        if background_path.is_file():
            background_key = f"{project_prefix}/interim/{job_id}/audio/background.wav"
            if not upload_to_s3(output_bucket, background_key, background_path):
                logging.warning("Failed to upload background.wav to S3")

        send_callback(
            callback_url,
            "in_progress",
            "ASR completed. Creating chunks...",
            stage="asr_completed",
            metadata={
                "total_segments": total_segments,
                "chunk_size": chunk_size,
                "audio_key": audio_key,
                "vocals_key": vocals_key,
                "background_key": background_key,
            },
        )

        # 4.5. speaker_refs.json과 self_refs 디렉토리를 S3에 업로드
        tts_dir = paths.vid_tts_dir
        speaker_refs_json_path = tts_dir / "speaker_refs.json"
        speaker_ref_dir = tts_dir / "self_refs"

        if speaker_refs_json_path.is_file():
            # speaker_refs.json 업로드
            refs_json_key = f"{project_prefix}/interim/{job_id}/tts/speaker_refs.json"
            if upload_to_s3(output_bucket, refs_json_key, speaker_refs_json_path):
                logging.info(f"Job {job_id}: Uploaded speaker_refs.json to S3")
            else:
                logging.warning(
                    f"Job {job_id}: Failed to upload speaker_refs.json to S3"
                )

        # self_refs 디렉토리의 모든 wav 파일 업로드
        if speaker_ref_dir.is_dir():
            for ref_file in speaker_ref_dir.glob("*.wav"):
                try:
                    relative_path = ref_file.relative_to(paths.interim_dir)
                except ValueError:
                    relative_path = ref_file.relative_to(speaker_ref_dir)
                    logging.warning(
                        "Speaker ref 경로 %s 를 interim 디렉터리 기준으로 계산하지 못했습니다. "
                        "self_refs 디렉터리 상대 경로를 사용합니다.",
                        ref_file,
                    )
                ref_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
                if upload_to_s3(output_bucket, str(ref_key), ref_file):
                    logging.info(
                        f"Job {job_id}: Uploaded speaker ref {ref_file.name} to S3"
                    )
                else:
                    logging.warning(
                        f"Job {job_id}: Failed to upload speaker ref {ref_file.name} to S3"
                    )

        # 5. 청크 생성
        chunks = create_chunks(total_segments, chunk_size)
        total_chunks = len(chunks)

        logging.info(
            f"Job {job_id}: Created {total_chunks} chunks from {total_segments} segments"
        )

        # 6. manifest.json 생성
        manifest = {
            "job_id": job_id,
            "project_id": project_id,
            "total_segments": total_segments,
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
            "chunks": chunks,
            "audio_files": {
                "compact_transcript": f"s3://{output_bucket}/{compact_transcript_key}",
            },
            "input_key": input_key,
            "input_bucket": input_bucket,
            "source_lang": effective_source_lang,
            "target_lang": target_lang,
            "output_bucket": output_bucket,
            "project_prefix": project_prefix,
            "callback_url": callback_url,
        }

        # audio 파일 키 추가 (있는 경우만)
        if audio_key:
            manifest["audio_files"]["audio_wav"] = f"s3://{output_bucket}/{audio_key}"
        if vocals_key:
            manifest["audio_files"]["vocals_wav"] = f"s3://{output_bucket}/{vocals_key}"
        if background_key:
            manifest["audio_files"][
                "background_wav"
            ] = f"s3://{output_bucket}/{background_key}"

        # manifest.json을 S3에 업로드
        manifest_key = f"{project_prefix}/interim/{job_id}/manifest.json"
        if not upload_metadata_to_s3(output_bucket, manifest_key, manifest):
            raise Exception("Failed to upload manifest.json to S3")

        logging.info(
            f"Job {job_id}: Manifest uploaded to s3://{output_bucket}/{manifest_key}"
        )

        # 7. 각 청크를 chunk_work 큐에 전송
        chunk_messages_sent = 0
        for chunk in chunks:
            chunk_message = {
                "task": "chunk_work",
                "job_id": job_id,
                "project_id": project_id,
                "chunk_index": chunk["chunk_index"],
                "start_segment_index": chunk["start_segment_index"],
                "end_segment_index": chunk["end_segment_index"],
                "total_chunks": total_chunks,
                "total_segments": total_segments,
                "manifest_key": manifest_key,
                "target_lang": target_lang,
                "source_lang": effective_source_lang,
                "output_bucket": output_bucket,
                "project_prefix": project_prefix,
                "callback_url": callback_url,
                # 기타 필요한 정보 전달
                "voice_config": job_details.get("voice_config"),
                "replace_voice_samples": job_details.get("replace_voice_samples"),
                "is_replace_voice_samples": job_details.get("is_replace_voice_samples"),
                "voice_sample_substitution": job_details.get(
                    "voice_sample_substitution"
                ),
                "voice_library_bucket": job_details.get("voice_library_bucket"),
            }

            try:
                message_body = json.dumps(chunk_message)
                # 청크별로 다른 그룹 ID 사용 (병렬 처리 가능)
                chunk_group_id = f"{project_id}_chunk_{chunk['chunk_index']}"
                message_kwargs = _build_sqs_message_kwargs(
                    message_body,
                    project_id=project_id,
                    deduplication_id=f"{job_id}_chunk_{chunk['chunk_index']}",
                    group_id=chunk_group_id,
                )
                response = sqs_client.send_message(**message_kwargs)
                logging.debug(
                    f"Job {job_id}: SQS send_message response: {response.get('MessageId')}"
                )
                chunk_messages_sent += 1
                logging.info(
                    f"Job {job_id}: Sent chunk {chunk['chunk_index']} to queue "
                    f"(segments {chunk['start_segment_index']}-{chunk['end_segment_index']})"
                )
            except Exception as e:
                logging.error(
                    f"Job {job_id}: Failed to send chunk {chunk['chunk_index']} to queue: {e}",
                    exc_info=True,
                )
                raise

        send_callback(
            callback_url,
            "in_progress",
            f"Split completed. Created {total_chunks} chunks.",
            stage="split_completed",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "total_segments": total_segments,
                "total_chunks": total_chunks,
                "chunk_size": chunk_size,
                "chunks_sent": chunk_messages_sent,
                "manifest_key": manifest_key,
            },
        )

        logging.info(
            f"Job {job_id}: split_up completed. "
            f"Sent {chunk_messages_sent}/{total_chunks} chunks to queue"
        )

    except Exception as e:
        logging.error(f"split_up failed for job {job_id}: {e}", exc_info=True)
        send_callback(
            callback_url,
            "failed",
            str(e),
            stage="split_up_failed",
            metadata={"job_id": job_id, "project_id": project_id},
        )
        raise


def chunk_work(job_details: dict):
    """청크별로 번역, TTS, Sync를 수행합니다."""
    from services.transcript_store import (
        load_compact_transcript,
        segment_views,
        save_compact_transcript,
    )
    import shutil

    job_id = job_details["job_id"]
    project_id = job_details.get("project_id")
    chunk_index = job_details["chunk_index"]
    start_segment_index = job_details["start_segment_index"]
    end_segment_index = job_details["end_segment_index"]
    total_chunks = job_details["total_chunks"]
    total_segments = job_details["total_segments"]
    manifest_key = job_details["manifest_key"]
    target_lang = job_details.get("target_lang", "en")
    source_lang = job_details.get("source_lang")
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET
    project_prefix = job_details.get("project_prefix") or (
        f"projects/{project_id}" if project_id else "jobs"
    )
    callback_url = job_details.get("callback_url")

    voice_config = job_details.get("voice_config")
    voice_replacement_flag = (
        job_details.get("replace_voice_samples")
        or job_details.get("is_replace_voice_samples")
        or job_details.get("voice_sample_substitution")
    )
    replace_voice_samples = _parse_bool(voice_replacement_flag)
    voice_library_bucket = (
        job_details.get("voice_library_bucket") or VOICE_LIBRARY_BUCKET
    )

    logging.info(
        f"Job {job_id}: Starting chunk_work for chunk {chunk_index} "
        f"(segments {start_segment_index}-{end_segment_index})"
    )

    if callback_url:
        send_callback(
            callback_url,
            "in_progress",
            f"Processing chunk {chunk_index + 1}/{total_chunks}...",
            stage="chunk_work_started",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
            },
        )

    # 1. 로컬 작업 디렉토리 설정
    paths = ensure_job_dirs(job_id)

    try:
        # 2. manifest.json 다운로드
        manifest_path = paths.interim_dir / "manifest.json"
        if not download_from_s3(output_bucket, manifest_key, manifest_path):
            raise Exception(f"Failed to download manifest.json from {manifest_key}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # 3. compact_transcript 다운로드
        compact_transcript_s3_key = manifest["audio_files"]["compact_transcript"]
        # s3://bucket/key 형식 처리
        if compact_transcript_s3_key.startswith("s3://"):
            _, compact_transcript_key = compact_transcript_s3_key.replace(
                "s3://", ""
            ).split("/", 1)
        else:
            compact_transcript_key = compact_transcript_s3_key

        compact_transcript_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
        compact_transcript_path.parent.mkdir(parents=True, exist_ok=True)

        if not download_from_s3(
            output_bucket, compact_transcript_key, compact_transcript_path
        ):
            raise Exception(
                f"Failed to download compact_transcript from {compact_transcript_key}"
            )

        # 3.5. speaker_refs.json과 self_refs 디렉토리를 S3에서 다운로드
        tts_dir = paths.vid_tts_dir
        speaker_refs_json_path = tts_dir / "speaker_refs.json"
        speaker_ref_dir = tts_dir / "self_refs"

        # speaker_refs.json 다운로드 (이미 있으면 건너뛰기)
        refs_json_key = f"{project_prefix}/interim/{job_id}/tts/speaker_refs.json"
        speaker_refs_json_path.parent.mkdir(parents=True, exist_ok=True)

        speaker_refs_downloaded = False
        if speaker_refs_json_path.is_file():
            logging.debug(
                f"Job {job_id} chunk {chunk_index}: "
                f"speaker_refs.json already exists, skipping download"
            )
            speaker_refs_downloaded = True
        elif download_from_s3(output_bucket, refs_json_key, speaker_refs_json_path):
            logging.info(
                f"Job {job_id} chunk {chunk_index}: Downloaded speaker_refs.json from S3"
            )
            speaker_refs_downloaded = True
        else:
            logging.warning(
                f"Job {job_id} chunk {chunk_index}: "
                f"speaker_refs.json not found in S3, skipping speaker refs download"
            )

        # speaker_refs.json을 읽어서 self_refs 파일 다운로드 (이미 있으면 건너뛰기)
        if speaker_refs_downloaded:
            try:
                with open(speaker_refs_json_path, "r", encoding="utf-8") as f:
                    speaker_refs_mapping = json.load(f)

                speaker_ref_dir.mkdir(parents=True, exist_ok=True)
                for speaker, ref_data in speaker_refs_mapping.items():
                    if isinstance(ref_data, dict) and "audio" in ref_data:
                        ref_audio_path_str = ref_data["audio"]
                        ref_audio_path = Path(ref_audio_path_str)
                        if not ref_audio_path.is_absolute():
                            ref_audio_path = speaker_ref_dir / ref_audio_path.name

                        # 이미 파일이 있으면 건너뛰기
                        if ref_audio_path.is_file():
                            logging.debug(
                                f"Job {job_id} chunk {chunk_index}: "
                                f"Speaker ref {ref_audio_path.name} already exists, skipping download"
                            )
                            continue

                        # S3에서 다운로드
                        try:
                            relative_ref_path = ref_audio_path.relative_to(
                                paths.interim_dir
                            )
                        except ValueError:
                            relative_ref_path = (
                                Path("text/vid/tts/self_refs") / ref_audio_path.name
                            )

                        ref_s3_key = (
                            f"{project_prefix}/interim/{job_id}/{relative_ref_path}"
                        )
                        if download_from_s3(output_bucket, ref_s3_key, ref_audio_path):
                            logging.debug(
                                f"Job {job_id} chunk {chunk_index}: "
                                f"Downloaded speaker ref {ref_audio_path.name} from S3"
                            )
                        else:
                            logging.warning(
                                f"Job {job_id} chunk {chunk_index}: "
                                f"Failed to download speaker ref {ref_audio_path.name} from S3"
                            )
                    elif isinstance(ref_data, str):
                        # 문자열로 직접 경로가 지정된 경우
                        ref_audio_path = Path(ref_data)
                        if not ref_audio_path.is_absolute():
                            ref_audio_path = speaker_ref_dir / ref_audio_path.name

                        # 이미 파일이 있으면 건너뛰기
                        if ref_audio_path.is_file():
                            logging.debug(
                                f"Job {job_id} chunk {chunk_index}: "
                                f"Speaker ref {ref_audio_path.name} already exists, skipping download"
                            )
                            continue

                        try:
                            relative_ref_path = ref_audio_path.relative_to(
                                paths.interim_dir
                            )
                        except ValueError:
                            relative_ref_path = (
                                Path("text/vid/tts/self_refs") / ref_audio_path.name
                            )

                        ref_s3_key = (
                            f"{project_prefix}/interim/{job_id}/{relative_ref_path}"
                        )
                        if download_from_s3(output_bucket, ref_s3_key, ref_audio_path):
                            logging.debug(
                                f"Job {job_id} chunk {chunk_index}: "
                                f"Downloaded speaker ref {ref_audio_path.name} from S3"
                            )
                        else:
                            logging.warning(
                                f"Job {job_id} chunk {chunk_index}: "
                                f"Failed to download speaker ref {ref_audio_path.name} from S3"
                            )
            except Exception as e:
                logging.warning(
                    f"Job {job_id} chunk {chunk_index}: "
                    f"Failed to download speaker refs: {e}",
                    exc_info=True,
                )

        # 4. 청크 범위의 세그먼트만 필터링
        bundle = load_compact_transcript(compact_transcript_path)
        all_segments = segment_views(bundle)

        # 청크 범위 필터링
        chunk_segments = [
            seg
            for seg in all_segments
            if start_segment_index <= seg.idx <= end_segment_index
        ]

        if not chunk_segments:
            raise ValueError(
                f"No segments found in range {start_segment_index}-{end_segment_index}"
            )

        logging.info(
            f"Job {job_id} chunk {chunk_index}: "
            f"Filtered {len(chunk_segments)} segments from {len(all_segments)} total"
        )

        # 필터링된 세그먼트로 compact_transcript 재구성
        # 원본 bundle의 구조를 유지하면서 segments만 필터링
        # 중요: 필터링된 배열 인덱스 -> 원본 인덱스 매핑 생성
        filtered_bundle = dict(bundle)
        original_segments = bundle.get("segments", [])
        filtered_segments = []
        filtered_to_original_idx_map = {}  # 필터링된 인덱스 -> 원본 인덱스 매핑

        for i, seg in enumerate(original_segments):
            if start_segment_index <= i <= end_segment_index:
                filtered_idx = len(filtered_segments)  # 필터링된 배열에서의 새 인덱스
                filtered_to_original_idx_map[filtered_idx] = i  # 원본 인덱스 저장
                filtered_segments.append(seg)

        filtered_bundle["segments"] = filtered_segments

        # 필터링된 compact_transcript를 임시로 저장 (번역/TTS/Sync에서 사용)
        chunk_transcript_path = (
            paths.src_sentence_dir / f"chunk_{chunk_index}_transcript.comp.json"
        )
        from services.transcript_store import save_compact_transcript

        save_compact_transcript(filtered_bundle, chunk_transcript_path)

        # 원본 transcript를 백업하고 청크 버전으로 교체 (임시)
        original_transcript_backup = (
            paths.src_sentence_dir / f"original_{COMPACT_ARCHIVE_NAME}"
        )
        if compact_transcript_path.exists():
            shutil.copy(compact_transcript_path, original_transcript_backup)

        # 청크 버전을 원본 위치에 복사 (번역/TTS/Sync 함수들이 이 경로를 사용)
        shutil.copy(chunk_transcript_path, compact_transcript_path)

        # 5. 번역 수행 (청크 범위만)
        logging.info(f"Job {job_id} chunk {chunk_index}: Starting translation...")
        from services.translate import translate_transcript

        translations = translate_transcript(job_id, target_lang, src_lang=source_lang)

        # 번역 결과의 seg_idx를 원본 인덱스로 변환
        for trans in translations:
            filtered_idx = trans.get("seg_idx")
            if (
                filtered_idx is not None
                and filtered_idx in filtered_to_original_idx_map
            ):
                original_idx = filtered_to_original_idx_map[filtered_idx]
                trans["seg_idx"] = original_idx
                logging.debug(
                    f"Job {job_id} chunk {chunk_index}: "
                    f"Mapped filtered seg_idx {filtered_idx} -> original seg_idx {original_idx}"
                )

        # 번역 결과를 청크별로 저장
        trans_result_path = (
            paths.trg_sentence_dir / f"chunk_{chunk_index}_translated.json"
        )
        trans_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trans_result_path, "w", encoding="utf-8") as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)

        # S3에 업로드
        trans_s3_key = f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_index}/translated.json"
        upload_to_s3(output_bucket, trans_s3_key, trans_result_path)

        logging.info(f"Job {job_id} chunk {chunk_index}: Translation completed")

        # 6. Voice replacement 준비 (필요한 경우)
        speaker_voice_overrides: dict[str, dict] = {}
        if replace_voice_samples:
            overrides, diagnostics = _maybe_prepare_voice_replacements(
                paths, target_lang, voice_library_bucket or output_bucket
            )
            speaker_voice_overrides = overrides

        # 7. TTS 수행 (청크 범위만)
        logging.info(f"Job {job_id} chunk {chunk_index}: Starting TTS...")
        from services.tts import generate_tts

        user_voice_sample_path = None
        if (
            voice_config
            and voice_config.get("kind") == "s3"
            and voice_config.get("key")
        ):
            voice_key = voice_config["key"]
            voice_bucket = (
                voice_config.get("bucket")
                or voice_config.get("bucket_name")
                or output_bucket
            )
            user_voice_sample_path = paths.interim_dir / Path(voice_key).name
            download_from_s3(voice_bucket, voice_key, user_voice_sample_path)

        segments_payload = generate_tts(
            job_id,
            target_lang,
            voice_sample_path=user_voice_sample_path,
            speaker_voice_overrides=(
                speaker_voice_overrides if speaker_voice_overrides else None
            ),
        )

        # TTS 결과물을 청크별 디렉토리에 업로드
        tts_dir = paths.vid_tts_dir
        for tts_file in tts_dir.glob("**/*"):
            if not tts_file.is_file():
                continue
            try:
                relative_path = tts_file.relative_to(paths.interim_dir)
            except ValueError:
                relative_path = tts_file.relative_to(tts_dir)
            tts_key = (
                f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_index}/"
                f"{relative_path}"
            )
            upload_to_s3(output_bucket, str(tts_key), tts_file)

        logging.info(f"Job {job_id} chunk {chunk_index}: TTS completed")

        # 8. Sync 수행 (청크 범위만)
        logging.info(f"Job {job_id} chunk {chunk_index}: Starting sync...")
        from services.sync import sync_segments

        try:
            synced_segments = sync_segments(job_id)
        except FileNotFoundError as exc:
            logging.warning(
                f"Job {job_id} chunk {chunk_index}: Sync artifacts not found: {exc}"
            )
            synced_segments = []
        except Exception as exc:
            logging.warning(f"Job {job_id} chunk {chunk_index}: Sync failed: {exc}")
            synced_segments = []

        # Global Index 복원: 청크 내부 인덱스를 전체 영상 기준 인덱스로 변환
        if synced_segments and start_segment_index is not None:
            logging.info(
                f"Job {job_id} chunk {chunk_index}: Restoring global indices "
                f"(start_segment_index={start_segment_index})"
            )
            for i, seg in enumerate(synced_segments):
                # Global Index 계산: 배열 인덱스(i)를 기준으로 계산
                # (sync_segments가 순서대로 반환하므로 배열 인덱스가 Local Index)
                global_idx = start_segment_index + i

                # seg_idx 업데이트 (Global Index로 변환)
                seg["seg_idx"] = global_idx

                # segment_index 업데이트 (Global Index로 변환)
                seg["segment_index"] = global_idx

                # segment_id 업데이트 ("segment_XXXX" 형식, Global Index 사용)
                seg["segment_id"] = f"segment_{global_idx:04d}"

            logging.info(
                f"Job {job_id} chunk {chunk_index}: Restored global indices for "
                f"{len(synced_segments)} segments"
            )

        # Sync 결과물을 S3에 업로드 (중요: synced 디렉토리의 .wav 파일들)
        synced_dir = paths.vid_tts_dir / "synced"
        if synced_dir.exists():
            for sync_file in synced_dir.glob("**/*"):
                if not sync_file.is_file():
                    continue
                try:
                    relative_path = sync_file.relative_to(paths.interim_dir)
                except ValueError:
                    relative_path = sync_file.relative_to(synced_dir)
                    logging.warning(
                        "Synced 경로 %s 를 interim 디렉터리 기준으로 계산하지 못했습니다. "
                        "synced 디렉터리 상대 경로를 사용합니다.",
                        sync_file,
                    )
                # full_pipeline과 동일하게 relative_path 사용 (프론트엔드가 기대하는 경로)
                sync_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
                upload_to_s3(output_bucket, str(sync_key), sync_file)

                # mux를 위한 경로에도 업로드 (기존 경로 유지)
                mux_sync_key = (
                    f"{project_prefix}/interim/{job_id}/tts/synced/{sync_file.name}"
                )
                upload_to_s3(output_bucket, mux_sync_key, sync_file)

        # segments_synced.json도 업로드
        synced_meta_path = synced_dir / "segments_synced.json"
        
        # [FIX] 수정된 인덱스가 적용된 synced_segments를 파일에 다시 씀
        if synced_segments:
            with open(synced_meta_path, "w", encoding="utf-8") as f:
                json.dump(synced_segments, f, ensure_ascii=False, indent=2)
            logging.info(f"Job {job_id} chunk {chunk_index}: Overwrote segments_synced.json with corrected indices")

        if synced_meta_path.exists():
            synced_meta_key = (
                f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_index}/"
                f"segments_synced.json"
            )
            upload_to_s3(output_bucket, synced_meta_key, synced_meta_path)

        # 9. 원본 transcript 복원
        if original_transcript_backup.exists():
            shutil.copy(original_transcript_backup, compact_transcript_path)

        # 10. 완료 체크 및 Mux 트리거
        check_and_trigger_mux_if_complete(
            job_id,
            project_id,
            total_segments,
            output_bucket,
            project_prefix,
            callback_url,
            job_details,
        )

        if callback_url:
            send_callback(
                callback_url,
                "in_progress",
                f"Chunk {chunk_index + 1}/{total_chunks} completed.",
                stage="chunk_work_completed",
                metadata={
                    "job_id": job_id,
                    "project_id": project_id,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "segments_processed": len(chunk_segments),
                },
            )

        logging.info(
            f"Job {job_id} chunk {chunk_index}: chunk_work completed successfully"
        )

    except Exception as e:
        logging.error(
            f"chunk_work failed for job {job_id} chunk {chunk_index}: {e}",
            exc_info=True,
        )
        if callback_url:
            send_callback(
                callback_url,
                "failed",
                f"Chunk {chunk_index + 1} failed: {str(e)}",
                stage="chunk_work_failed",
                metadata={
                    "job_id": job_id,
                    "project_id": project_id,
                    "chunk_index": chunk_index,
                },
            )
        raise


def handle_mux_task(job_details: dict):
    """모든 청크의 결과를 병합하여 최종 영상을 생성합니다."""
    job_id = job_details["job_id"]
    project_id = job_details.get("project_id")
    total_segments = job_details["total_segments"]
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET
    project_prefix = job_details.get("project_prefix") or (
        f"projects/{project_id}" if project_id else "jobs"
    )
    callback_url = job_details.get("callback_url")
    manifest_key = job_details.get("manifest_key")

    target_lang = job_details.get("target_lang", "en")
    source_lang = job_details.get("source_lang")

    logging.info(f"Job {job_id}: Starting mux task")

    if callback_url:
        send_callback(
            callback_url,
            "in_progress",
            "Starting mux...",
            stage="mux_started",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
            },
        )

    # 1. 로컬 작업 디렉토리 설정
    paths = ensure_job_dirs(job_id)

    try:
        # 2. manifest.json 다운로드
        manifest_path = paths.interim_dir / "manifest.json"
        if manifest_key:
            if not download_from_s3(output_bucket, manifest_key, manifest_path):
                raise Exception(f"Failed to download manifest.json from {manifest_key}")
        else:
            # manifest_key가 없으면 기본 경로 시도
            default_manifest_key = f"{project_prefix}/interim/{job_id}/manifest.json"
            if not download_from_s3(output_bucket, default_manifest_key, manifest_path):
                raise Exception("manifest_key not provided and default path not found")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        total_chunks = manifest.get("total_chunks", 1)

        # 3. 모든 청크의 segments_synced.json 다운로드 및 병합
        all_synced_segments = []

        for chunk_idx in range(total_chunks):
            chunk_synced_key = (
                f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_idx}/"
                f"segments_synced.json"
            )

            chunk_synced_path = paths.interim_dir / f"chunk_{chunk_idx}_synced.json"

            if download_from_s3(output_bucket, chunk_synced_key, chunk_synced_path):
                with open(chunk_synced_path, "r", encoding="utf-8") as f:
                    chunk_segments = json.load(f)
                    all_synced_segments.extend(chunk_segments)
                logging.info(
                    f"Job {job_id}: Loaded {len(chunk_segments)} segments from chunk {chunk_idx}"
                )
            else:
                logging.warning(
                    f"Job {job_id}: Failed to download segments_synced.json "
                    f"for chunk {chunk_idx}, skipping"
                )

        if not all_synced_segments:
            # 청크별 segments_synced.json이 없으면 S3의 synced 디렉토리에서 직접 로드 시도
            logging.info(
                f"Job {job_id}: No chunk segments found, trying to load from synced directory"
            )
            # S3에서 직접 segments_synced.json 다운로드 시도
            synced_meta_key = (
                f"{project_prefix}/interim/{job_id}/tts/synced/segments_synced.json"
            )
            synced_meta_path = paths.vid_tts_dir / "synced" / "segments_synced.json"
            synced_meta_path.parent.mkdir(parents=True, exist_ok=True)

            if download_from_s3(output_bucket, synced_meta_key, synced_meta_path):
                with open(synced_meta_path, "r", encoding="utf-8") as f:
                    all_synced_segments = json.load(f)
            else:
                raise Exception("No synced segments found in S3")

        # 세그먼트를 segment_index 또는 start 시간 기준으로 정렬
        # segment_id가 "segment_XXXX" 형식이면 인덱스 추출, 아니면 start 시간으로 정렬
        def get_sort_key(seg):
            seg_id = seg.get("segment_id", "")
            if isinstance(seg_id, str) and seg_id.startswith("segment_"):
                try:
                    idx_str = seg_id.replace("segment_", "")
                    return int(idx_str)
                except ValueError:
                    pass
            # segment_index가 있으면 사용
            if "segment_index" in seg:
                return seg.get("segment_index", 0)
            # 아니면 start 시간 사용
            return seg.get("start", 0)

        all_synced_segments.sort(key=get_sort_key)

        logging.info(
            f"Job {job_id}: Merged {len(all_synced_segments)} segments from {total_chunks} chunks"
        )

        # 통합된 segments_synced.json 저장
        synced_dir = paths.vid_tts_dir / "synced"
        synced_dir.mkdir(parents=True, exist_ok=True)
        merged_synced_path = synced_dir / "segments_synced.json"
        with open(merged_synced_path, "w", encoding="utf-8") as f:
            json.dump(all_synced_segments, f, ensure_ascii=False, indent=2)

        # 4. 원본 영상 다운로드
        input_key = job_details.get("input_key")
        input_bucket = job_details.get("input_bucket") or output_bucket

        if not input_key:
            # manifest에서 찾기
            input_key = manifest.get("input_key")
            if not input_key:
                raise Exception("input_key not found in job_details or manifest")

        source_video_path = paths.input_dir / Path(input_key).name
        if not download_from_s3(input_bucket, input_key, source_video_path):
            raise Exception(f"Failed to download source video from {input_key}")

        # 5. background.wav 다운로드 (manifest에서 경로 확인)
        background_key = None
        if "audio_files" in manifest:
            bgm_s3_path = manifest["audio_files"].get("background_wav")
            if bgm_s3_path:
                if bgm_s3_path.startswith("s3://"):
                    _, background_key = bgm_s3_path.replace("s3://", "").split("/", 1)
                else:
                    background_key = bgm_s3_path

        if not background_key:
            # 기본 경로 시도
            background_key = f"{project_prefix}/interim/{job_id}/audio/background.wav"

        background_path = paths.vid_bgm_dir / "background.wav"
        background_path.parent.mkdir(parents=True, exist_ok=True)

        if not download_from_s3(output_bucket, background_key, background_path):
            logging.warning(
                f"Job {job_id}: Failed to download background.wav, "
                f"mux may fail if background is required"
            )

        # 6. synced 디렉토리의 .wav 파일들이 로컬에 있는지 확인
        # 없으면 S3에서 다운로드
        synced_wav_dir = synced_dir
        for seg in all_synced_segments:
            audio_file = seg.get("audio_file")
            if not audio_file:
                continue

            audio_path = Path(audio_file)
            # 절대 경로가 아니면 synced_dir 기준으로 해석
            if not audio_path.is_absolute():
                audio_path = synced_dir / audio_path.name

            # 파일이 없으면 S3에서 다운로드 시도
            if not audio_path.exists():
                # S3 경로 추정
                audio_s3_key = (
                    f"{project_prefix}/interim/{job_id}/tts/synced/{audio_path.name}"
                )
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                download_from_s3(output_bucket, audio_s3_key, audio_path)

        # 7. Mux 수행
        logging.info(f"Job {job_id}: Running mux_audio_video...")
        mux_results = mux_audio_video(job_id, source_video_path)
        output_video_path = Path(mux_results["output_video"])
        final_audio_path = Path(mux_results["output_audio"])

        # 8. 최종 결과물 S3에 업로드
        output_prefix = resolve_output_prefix(
            project_id, job_id, job_details.get("output_prefix")
        )
        result_key = f"{output_prefix}/dubbed_video.mp4"
        if not upload_to_s3(output_bucket, result_key, output_video_path):
            raise Exception("Failed to upload final video to S3")

        # 9. 메타데이터 생성 및 업로드
        metadata_key = (
            job_details.get("metadata_key") or f"{output_prefix}/metadata/{job_id}.json"
        )

        # segments를 S3 경로로 변환
        metadata_segments = _segments_with_remote_audio_paths(
            all_synced_segments,
            project_prefix,
            job_id,
            paths,
        )

        # segment_index 기준으로 정렬 (번역과 매칭을 위해)
        def get_segment_index(seg):
            """세그먼트의 인덱스를 추출"""
            if "segment_index" in seg:
                return seg.get("segment_index", 0)
            elif "seg_idx" in seg:
                return int(seg.get("seg_idx", 0))
            elif "segment_id" in seg:
                seg_id = seg.get("segment_id", "")
                # "segment_XXXX" 형식에서 인덱스 추출 시도
                try:
                    if isinstance(seg_id, str) and seg_id.startswith("segment_"):
                        return int(seg_id.replace("segment_", ""))
                except (ValueError, TypeError):
                    pass
            return 0

        metadata_segments.sort(key=get_segment_index)

        # 번역 결과 병합 (모든 청크의 번역 결과)
        all_translations = []
        for chunk_idx in range(total_chunks):
            chunk_trans_key = (
                f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_idx}/"
                f"translated.json"
            )
            chunk_trans_path = paths.interim_dir / f"chunk_{chunk_idx}_translated.json"

            if download_from_s3(output_bucket, chunk_trans_key, chunk_trans_path):
                with open(chunk_trans_path, "r", encoding="utf-8") as f:
                    chunk_trans = json.load(f)
                    all_translations.extend(chunk_trans)

        # translations를 seg_idx로 딕셔너리로 매핑
        translations_map = {}
        for trans in all_translations:
            seg_idx = trans.get("seg_idx", 0)
            translations_map[seg_idx] = trans

        # segments의 순서에 맞춰 translations 재정렬
        # segments의 segment_index (또는 seg_idx)를 기준으로 translations 매핑
        sorted_translations = []
        for seg in metadata_segments:
            seg_idx = get_segment_index(seg)
            if seg_idx in translations_map:
                sorted_translations.append(translations_map[seg_idx])
            else:
                # 매칭되는 번역이 없으면 빈 번역 추가 (segments와 translations 순서 일치 보장)
                logging.warning(
                    f"Job {job_id}: No translation found for segment_index {seg_idx}"
                )
                sorted_translations.append({"seg_idx": seg_idx, "translation": ""})

        all_translations = sorted_translations

        metadata_payload = {
            "job_id": job_id,
            "project_id": project_id,
            "target_lang": target_lang,
            "source_lang": source_lang,
            "input_bucket": input_bucket,
            "input_key": input_key,
            "result_bucket": output_bucket,
            "result_key": result_key,
            "metadata_key": metadata_key,
            "segments": metadata_segments,
            "segment_count": len(metadata_segments),
            "translations": all_translations,
        }

        if final_audio_path:
            metadata_payload["audio_artifact"] = str(final_audio_path)

        if not upload_metadata_to_s3(output_bucket, metadata_key, metadata_payload):
            raise Exception("Failed to upload metadata to S3")

        # 9.5. speaker_refs 읽기 (S3에서)
        speaker_refs_metadata = None
        refs_json_key = f"{project_prefix}/interim/{job_id}/tts/speaker_refs.json"
        refs_json_path = paths.interim_dir / "speaker_refs.json"
        if download_from_s3(output_bucket, refs_json_key, refs_json_path):
            try:
                with open(refs_json_path, "r", encoding="utf-8") as f:
                    speaker_refs_mapping = json.load(f)

                # _build_speaker_refs_metadata와 동일한 형식으로 변환
                speaker_refs_metadata = {}
                for speaker, ref_data in speaker_refs_mapping.items():
                    if isinstance(ref_data, dict):
                        audio_path = ref_data.get("audio", "")
                        prompt_text = ref_data.get("text", "")

                        # S3 경로로 변환
                        if audio_path:
                            if not Path(audio_path).is_absolute():
                                # 상대 경로인 경우 S3 키 생성
                                ref_s3_key = (
                                    f"{project_prefix}/interim/{job_id}/text/vid/tts/self_refs/"
                                    f"{Path(audio_path).name}"
                                )
                            else:
                                # 절대 경로인 경우 relative_path 계산 시도
                                try:
                                    relative_path = Path(audio_path).relative_to(
                                        paths.interim_dir
                                    )
                                    ref_s3_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
                                except ValueError:
                                    ref_s3_key = (
                                        f"{project_prefix}/interim/{job_id}/text/vid/tts/self_refs/"
                                        f"{Path(audio_path).name}"
                                    )

                            speaker_refs_metadata[speaker] = {
                                "ref_wav_key": f"s3://{output_bucket}/{ref_s3_key}",
                                "prompt_text": prompt_text,
                            }
            except Exception as e:
                logging.warning(
                    f"Job {job_id}: Failed to load speaker_refs from S3: {e}",
                    exc_info=True,
                )

        # 10. 최종 콜백
        final_metadata = {
            "job_id": job_id,
            "project_id": project_id,
            "result_bucket": output_bucket,
            "result_key": result_key,
            "metadata_key": metadata_key,
            "segment_count": len(metadata_segments),
            "target_lang": target_lang,
            "translations": all_translations,
        }
        if source_lang:
            final_metadata["source_lang"] = source_lang
        if speaker_refs_metadata:
            final_metadata["speaker_refs"] = speaker_refs_metadata

        if callback_url:
            send_callback(
                callback_url,
                "done",
                "Pipeline completed successfully.",
                stage="done",
                metadata=final_metadata,
            )

        logging.info(f"Job {job_id}: Mux completed successfully")

    except Exception as e:
        logging.error(f"Mux failed for job {job_id}: {e}", exc_info=True)
        if callback_url:
            send_callback(
                callback_url,
                "failed",
                str(e),
                stage="mux_failed",
                metadata={"job_id": job_id, "project_id": project_id},
            )
        raise


def _handle_tts_segments(job_details: dict) -> None:
    """segment_tts / tts 작업을 처리합니다."""
    callback_url = job_details.get("callback_url")
    segments_req = job_details.get("segments") or []

    if not callback_url:
        raise ValueError("segment_tts requires callback_url.")
    if not segments_req:
        raise ValueError("segment_tts requires at least one segment entry.")

    project_id = job_details.get("project_id")
    target_lang = job_details.get("target_lang", "ko")
    mod_raw = (job_details.get("mod") or "dynamic").strip().lower()
    mod = mod_raw if mod_raw in {"fixed", "dynamic"} else "dynamic"
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET

    # original_job_id가 있으면 사용 (full_pipeline job_id), 없으면 현재 job_id 사용
    original_job_id = job_details.get("original_job_id") or job_details.get("job_id")
    if not original_job_id:
        raise ValueError("segment_tts requires either original_job_id or job_id.")

    # 파일명에 사용할 job_id (현재 job_id)
    job_id = job_details.get("job_id")
    if not job_id:
        raise ValueError("segment_tts requires job_id.")

    project_prefix = f"projects/{project_id}" if project_id else "jobs"
    remote_interim_prefix = f"{project_prefix}/interim/{original_job_id}"

    send_callback(
        callback_url,
        "in_progress",
        f"Starting segment TTS for job {original_job_id}",
        stage="segment_tts_started",
        metadata={"job_id": original_job_id, "project_id": project_id, "mod": mod},
    )

    try:
        paths = ensure_job_dirs(original_job_id)
        resynth_dir = paths.vid_tts_dir / "resynth"
        resynth_dir.mkdir(parents=True, exist_ok=True)

        # --- 보이스 샘플 준비 ---
        speaker_spec = job_details.get("speaker_voices") or {}
        voice_key = speaker_spec.get("key")
        if not voice_key:
            raise ValueError("speaker_voices.key is required for segment_tts.")
        speaker_bucket = (
            speaker_spec.get("bucket")
            or job_details.get("input_bucket")
            or output_bucket
        )

        speaker_asset_dir = paths.interim_dir / "speaker_assets"
        speaker_asset_dir.mkdir(parents=True, exist_ok=True)

        voice_bucket, resolved_voice_key = _resolve_s3_location(
            voice_key, speaker_bucket
        )
        sample_path = speaker_asset_dir / Path(resolved_voice_key).name
        if not download_from_s3(voice_bucket, resolved_voice_key, sample_path):
            raise RuntimeError(
                f"Failed to download speaker sample from {resolved_voice_key}"
            )

        # CosyVoice는 30초를 초과하는 오디오에서 speech token을 추출할 수 없으므로 30초로 자름
        try:
            audio = AudioSegment.from_file(str(sample_path))
            max_duration_ms = 30 * 1000  # 30초 = 30000ms

            if len(audio) > max_duration_ms:
                logging.info(
                    f"Voice sample is {len(audio)/1000:.2f}s, trimming to 30s for CosyVoice compatibility"
                )
                audio = audio[:max_duration_ms]
                audio.export(str(sample_path), format="wav")
                logging.info("Voice sample trimmed to 30 seconds")
            else:
                logging.debug(
                    f"Voice sample is {len(audio)/1000:.2f}s, no trimming needed"
                )
        except Exception as e:
            logging.warning(
                f"Failed to check/trim voice sample duration: {e}, continuing with original file"
            )

        # --- 텍스트 프롬프트 준비 ---
        prompt_text = (speaker_spec.get("text_prompt_value") or "").strip()
        text_prompt_key = speaker_spec.get("text_prompt")
        if not prompt_text and text_prompt_key:
            prompt_bucket = (
                speaker_spec.get("text_prompt_bucket")
                or speaker_spec.get("bucket")
                or job_details.get("input_bucket")
                or output_bucket
            )
            prompt_bucket, prompt_key = _resolve_s3_location(
                text_prompt_key, prompt_bucket
            )
            prompt_path = speaker_asset_dir / Path(prompt_key).name
            if not download_from_s3(prompt_bucket, prompt_key, prompt_path):
                raise RuntimeError(
                    f"Failed to download speaker text prompt from {prompt_key}"
                )
            prompt_text = prompt_path.read_text(encoding="utf-8").strip()

        if not prompt_text:
            fallback_prompt = (job_details.get("prompt_text") or "").strip()
            prompt_text = fallback_prompt or _transcribe_prompt_text(sample_path)

        if not prompt_text:
            raise ValueError("Unable to resolve prompt text for TTS segments.")

        # --- 세그먼트별 TTS ---
        results: list[dict] = []

        def _to_seconds(value) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        for idx, seg_req in enumerate(segments_req):
            text = (seg_req.get("text") or "").strip()
            if not text:
                raise ValueError("Segment is missing 'text'.")

            s_val = seg_req.get("s", seg_req.get("start"))
            e_val = seg_req.get("e", seg_req.get("end"))

            s_sec = _to_seconds(s_val) or 0.0
            e_sec = _to_seconds(e_val)

            s_ms = max(0, int(s_sec * 1000))
            e_ms = int(e_sec * 1000) if e_sec is not None else None

            # 각 세그먼트별로 고유한 파일명 사용
            local_tts = resynth_dir / f"{job_id}_{idx}.wav"
            synced_tts = resynth_dir / f"{job_id}_{idx}_synced.wav"

            # 1) CosyVoice2로 합성
            _synthesize_with_cosyvoice2(
                text=text,
                prompt_text=prompt_text,
                sample_path=sample_path,
                output_path=local_tts,
            )

            # 2) full_pipeline과 동일하게 TTS 아티팩트 / 침묵 트리밍
            _trim_tts_artifacts(local_tts)

            # 3) fixed 모드면 원래 [s, e] 길이에 맞춰 sync
            synced_path = local_tts
            if mod == "fixed":
                if e_ms is None or e_ms <= s_ms:
                    raise ValueError("Segment missing valid s/e for fixed mode.")
                target_duration_ms = max(1, e_ms - s_ms)
                synced_path = _sync_segment_to_range(
                    local_tts,
                    target_duration_ms,
                    synced_tts,
                )

            # 4) S3 업로드
            try:
                relative = synced_path.relative_to(paths.interim_dir)
            except ValueError:
                raise RuntimeError(
                    f"TTS artifact {synced_path} is outside interim dir"
                ) from None

            s3_key = f"{remote_interim_prefix}/{relative.as_posix()}"
            if not upload_to_s3(output_bucket, s3_key, synced_path):
                raise RuntimeError("Failed to upload segment to S3.")

            results.append(
                {
                    "text": text,
                    "s": s_sec,
                    "e": e_sec,
                    "audio_key": s3_key,
                    "bucket": output_bucket,
                    "mod": mod,
                }
            )

        # segment_id를 metadata에 포함 (백엔드에서 콜백 처리 시 사용)
        callback_metadata = {
            "job_id": original_job_id,
            "project_id": project_id,
            "target_lang": target_lang,
            "mod": mod,
            "segments": results,
        }
        # job_details에서 segment_id가 있으면 metadata에 포함
        segment_id = job_details.get("segment_id")
        if segment_id:
            callback_metadata["segment_id"] = segment_id

        send_callback(
            callback_url,
            "done",
            "Segment TTS completed.",
            stage="segment_tts_completed",
            metadata=callback_metadata,
        )
    except Exception as exc:
        logging.error(
            "segment_tts failed for job %s: %s", original_job_id, exc, exc_info=True
        )
        send_callback(
            callback_url,
            "failed",
            f"Segment TTS failed: {exc}",
            stage="segment_tts_failed",
            metadata={"job_id": original_job_id, "project_id": project_id, "mod": mod},
        )
        raise


def _handle_test_synthesis(job_details: dict):
    """test_synthesis 작업을 처리합니다.
    처리 순서:
    1. s3에서 보이스 샘플 다운로드
    2. Demucs로 보컬 분리 (전처리)
    3. STT로 프롬프트 텍스트 추출
    4. CosyVoice2로 TTS 생성
    5. S3에 결과 업로드
    6. 콜백 전송
    """
    job_id = job_details.get("job_id")
    callback_url = job_details.get("callback_url")
    file_path = job_details.get("file_path") or job_details.get("input_key")
    text = job_details.get("text")
    target_lang = job_details.get("target_lang", "ko")
    voice_sample_id = job_details.get("voice_sample_id")
    sample_lang_input = job_details.get("sample_lang") or target_lang
    sample_lang_code = normalize_lang_code(sample_lang_input) or "misc"

    if not all([job_id, callback_url, file_path, text]):
        raise ValueError(
            "Missing required fields: job_id, callback_url, file_path, text"
        )

    logging.info(f"Processing test_synthesis job {job_id}")
    send_callback(
        callback_url,
        "in_progress",
        f"Starting test_synthesis for job {job_id}",
        stage="test_synthesis_started",
    )

    # 1. 작업 디렉토리 생성
    paths = ensure_job_dirs(job_id)

    # 2. S3에서 보이스 샘플 다운로드
    local_voice_sample = paths.input_dir / "voice_sample.wav"
    logging.info(
        f"Downloading voice sample from s3://{AWS_S3_BUCKET}/{file_path} to {local_voice_sample}"
    )
    if not download_from_s3(AWS_S3_BUCKET, file_path, local_voice_sample):
        send_callback(
            callback_url,
            "failed",
            f"Failed to download voice sample from S3: {file_path}",
            stage="download_failed",
        )
        return

    send_callback(
        callback_url,
        "in_progress",
        "Voice sample downloaded, starting preprocessing...",
        stage="downloaded",
    )

    # 2-1. 오디오 파일이 15초를 넘으면 15초로 자르기
    trimmed = False
    try:
        audio = AudioSegment.from_file(str(local_voice_sample))
        max_duration_ms = 15 * 1000  # 15초 = 15000ms

        if len(audio) > max_duration_ms:
            logging.info(f"Audio file is {len(audio)/1000:.2f}s, trimming to 15s")
            audio = audio[:max_duration_ms]
            audio.export(str(local_voice_sample), format="wav")
            logging.info("Audio file trimmed to 30 seconds")
            trimmed = True

            # 30초로 자른 경우 원본 S3 파일도 업데이트
            if upload_to_s3(AWS_S3_BUCKET, file_path, local_voice_sample):
                logging.info(
                    f"Updated original S3 file at s3://{AWS_S3_BUCKET}/{file_path} with trimmed version"
                )
            else:
                logging.warning(
                    f"Failed to update original S3 file at s3://{AWS_S3_BUCKET}/{file_path}"
                )
        else:
            logging.info(f"Audio file is {len(audio)/1000:.2f}s, no trimming needed")
    except Exception as e:
        logging.warning(
            f"Failed to check/trim audio duration: {e}, continuing with original file"
        )

    embedding_s3_key: str | None = None
    sample_s3_key: str | None = None
    base_library_entry: dict[str, Any] | None = None  # ← 임베딩만 담은 1차 엔트리

    try:
        # 3. Demucs 전처리: 보이스 샘플을 paths.vid_speaks_dir/audio.wav에 복사
        # split_vocals는 paths.vid_speaks_dir/audio.wav를 찾으므로 복사 필요
        audio_path_for_demucs = paths.vid_speaks_dir / "audio.wav"
        audio_path_for_demucs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_voice_sample, audio_path_for_demucs)

        logging.info("Running Demucs for vocal separation...")
        demucs_result = split_vocals(job_id)
        vocals_path = Path(demucs_result["vocals"])

        sample_label = voice_sample_id or f"voice_{uuid.uuid4().hex[:10]}"
        # Voice sample audio itself is persisted to S3 so other workers can reuse it.
        sample_s3_key = f"voice-samples/samples/{sample_lang_code}/{sample_label}.wav"
        if not upload_to_s3(VOICE_LIBRARY_BUCKET, sample_s3_key, vocals_path):
            logging.warning(
                "Failed to upload processed voice sample to s3://%s/%s",
                VOICE_LIBRARY_BUCKET,
                sample_s3_key,
            )
            sample_s3_key = None

        # 3-1. 임베딩 계산 및 라이브러리 인덱스에 1차 등록 (prompt_text는 비워둠)
        try:
            embedding_path = paths.outputs_dir / "voice_sample_embedding.json"
            embedding_payload = save_audio_embedding(
                vocals_path,
                embedding_path,
                label=sample_label,
                base_dir=paths.interim_dir,
                meta={
                    "job_id": job_id,
                    "source": "voice_sample",
                    "voice_sample_id": voice_sample_id,
                },
            )
            embedding_vector = embedding_payload.get("embedding") or []

            base_library_entry = {
                "voice_id": sample_label,
                "sample_key": sample_s3_key,
                "embedding": embedding_vector,
                # prompt_text는 아직 없으므로 빈 문자열로 1차 저장
                "prompt_text": "",
            }
            # Always pull the freshest S3-backed library index before mutating it.
            _ensure_voice_library_index(sample_lang_code, force_refresh=True)
            local_index = update_voice_library_entry(
                sample_lang_code,
                base_library_entry,
                base_dir=VOICE_SAMPLES_EMBED_DIR,
            )
            embedding_s3_key = (
                f"voice-samples/embedding/{sample_lang_code}/{sample_lang_code}.json"
            )
            # Voice sample metadata (JSON index) is synced back to S3 for sharing.
            if upload_to_s3(VOICE_LIBRARY_BUCKET, embedding_s3_key, local_index):
                logging.info(
                    "Voice library index uploaded to s3://%s/%s",
                    VOICE_LIBRARY_BUCKET,
                    embedding_s3_key,
                )
            else:
                embedding_s3_key = None
        except Exception as exc:
            logging.warning("Failed to persist voice sample embedding: %s", exc)

        send_callback(
            callback_url,
            "in_progress",
            "Vocal separation completed, extracting prompt text...",
            stage="preprocessing_completed",
        )

        # 4. STT로 프롬프트 텍스트 추출
        logging.info("Extracting prompt text using STT...")
        prompt_text = _transcribe_prompt_text(vocals_path)

        if not prompt_text:
            logging.warning("Failed to extract prompt text, using empty string")
            prompt_text = ""

        logging.info(f"Extracted prompt text: {prompt_text[:100]}...")

        # 4-1. prompt_text를 포함해 라이브러리 엔트리 2차 업데이트
        try:
            if base_library_entry is not None:
                full_entry = dict(base_library_entry)
                full_entry["prompt_text"] = prompt_text
                # Reload S3 metadata again to merge with any concurrent updates.
                _ensure_voice_library_index(sample_lang_code, force_refresh=True)
                local_index = update_voice_library_entry(
                    sample_lang_code,
                    full_entry,
                    base_dir=VOICE_SAMPLES_EMBED_DIR,
                )
                if embedding_s3_key:
                    # Persist the enriched metadata to S3 so other runtimes stay in sync.
                    if upload_to_s3(
                        VOICE_LIBRARY_BUCKET, embedding_s3_key, local_index
                    ):
                        logging.info(
                            "Updated voice library index with prompt_text at s3://%s/%s",
                            VOICE_LIBRARY_BUCKET,
                            embedding_s3_key,
                        )
        except Exception as exc:
            logging.warning(
                "Failed to update voice library entry with prompt_text: %s", exc
            )

        send_callback(
            callback_url,
            "in_progress",
            "Prompt text extracted, generating TTS...",
            stage="prompt_extracted",
        )

        # 5. TTS 생성
        output_path = paths.outputs_dir / "tts_output.wav"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logging.info("Generating TTS with CosyVoice2...")
        _synthesize_with_cosyvoice2(
            text=text,
            prompt_text=prompt_text,
            sample_path=vocals_path,
            output_path=output_path,
        )

        send_callback(
            callback_url,
            "in_progress",
            "TTS generated, uploading to S3...",
            stage="tts_completed",
        )

        # 6. S3에 업로드
        s3_key = f"voice-samples/tts/{sample_label}.wav"
        if not upload_to_s3(VOICE_LIBRARY_BUCKET, s3_key, output_path):
            raise Exception("Failed to upload TTS result to S3")

        # 7. 콜백: 완료
        audio_sample_url = f"/api/storage/media/{s3_key}"
        send_callback(
            callback_url,
            "done",
            "Test synthesis completed successfully.",
            stage="test_synthesis_completed",
            metadata={
                "result_key": s3_key,
                "audio_sample_url": audio_sample_url,
                "voice_sample_id": voice_sample_id,
                "prompt_text": prompt_text,
                "sample_key": sample_s3_key,
                "embedding_key": embedding_s3_key,
            },
        )
        logging.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logging.error(f"Test synthesis failed for job {job_id}: {e}", exc_info=True)
        send_callback(
            callback_url,
            "failed",
            f"Test synthesis failed: {str(e)}",
            stage="test_synthesis_failed",
        )
        raise


async def poll_sqs():
    """SQS 큐를 폴링하여 메시지를 처리합니다."""
    logging.info("Starting SQS poller...")
    while True:
        try:
            response = sqs_client.receive_message(
                QueueUrl=JOB_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                MessageAttributeNames=["All"],
            )

            messages = response.get("Messages", [])
            if not messages:
                continue

            for message in messages:
                receipt_handle = message["ReceiptHandle"]
                try:
                    logging.info(f"Received message: {message['MessageId']}")
                    job_details = json.loads(message["Body"])

                    task = (job_details.get("task") or "full_pipeline").lower()
                    job_id = job_details.get("job_id", "unknown")

                    logging.info(f"Received task: {task} for job {job_id}")

                    if task == "test_synthesis":
                        _handle_test_synthesis(job_details)
                    elif task in ("segment_tts", "tts"):
                        _handle_tts_segments(job_details)
                    elif task == "split_up":
                        split_up(job_details)
                    elif task == "chunk_work":
                        chunk_work(job_details)
                    elif task == "mux":
                        handle_mux_task(job_details)
                    else:
                        # 기본값: full_pipeline (하위 호환성)
                        full_pipeline(job_details)

                    # 파이프라인 실행
                    # full_pipeline(job_details)

                    # 처리 완료 후 메시지 삭제
                    sqs_client.delete_message(
                        QueueUrl=JOB_QUEUE_URL, ReceiptHandle=receipt_handle
                    )
                    logging.info(f"Deleted message: {message['MessageId']}")

                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON in message body: {e}")
                    # 잘못된 형식의 메시지는 큐에서 삭제
                    sqs_client.delete_message(
                        QueueUrl=JOB_QUEUE_URL, ReceiptHandle=receipt_handle
                    )
                except Exception as e:
                    logging.error(
                        f"Error processing message {message.get('MessageId', 'N/A')}: {e}"
                    )
                    # 처리 중 에러 발생 시, 메시지를 바로 삭제하지 않고 SQS의 Visibility Timeout에 따라 재처리되도록 둡니다.
                    # 또는 Dead Letter Queue로 보내는 정책을 사용할 수 있습니다.
                    # 여기서는 일단 로깅만 하고 넘어갑니다.
                    pass

        except (BotoCoreError, ClientError) as e:
            logging.error(f"Error polling SQS: {e}")
            await asyncio.sleep(10)  # 에러 발생 시 잠시 대기 후 재시도
        except Exception as e:
            logging.error(f"An unexpected error occurred in the poller: {e}")
            await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(poll_sqs())
