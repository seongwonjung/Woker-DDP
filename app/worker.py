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
    """S3에서 파일을 다운로드합니다."""
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
                    else:
                        # 파이프라인 실행
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
