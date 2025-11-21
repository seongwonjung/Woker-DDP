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
from services.sync import (
    sync_segments,
    _sync_single_segment,
    MAX_SLOW_RATIO,
    sync_segment_to_range,
)
from services.mux import mux_audio_video
from configs import JobPaths, ensure_job_dirs, CHUNK_SIZE
from configs.utils import (
    send_callback,
    resolve_output_prefix,
    parse_bool,
    parse_positive_int,
)
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
    strip_voice_samples_prefix,
    ensure_voice_library_index,
    resolve_s3_location,
    materialize_voice_replacements,
    maybe_prepare_voice_replacements,
)
from services.metadata import (
    build_speaker_metadata,
    build_speaker_refs_metadata,
    segments_with_remote_audio_paths,
)
from services.speaker import (
    upload_speaker_refs,
    download_speaker_refs,
    upload_speaker_embeddings,
)
from utils.s3 import (
    init_s3_client,
    download_from_s3,
    upload_to_s3,
    upload_metadata_to_s3,
    upload_audio_artifacts,
)
from utils.sqs import (
    init_sqs_client,
    build_sqs_message_kwargs,
    check_and_trigger_mux_if_complete,
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

# S3 및 SQS 클라이언트 초기화
init_s3_client(AWS_REGION)
init_sqs_client(
    JOB_QUEUE_URL,
    AWS_REGION,
    JOB_QUEUE_FIFO_ENV,
    JOB_QUEUE_FIFO_URL,
    JOB_QUEUE_MESSAGE_GROUP_ID,
)

# 하위 호환성을 위한 별칭
sqs_client = boto3.client("sqs", region_name=AWS_REGION)
s3_client = boto3.client("s3", region_name=AWS_REGION)


# 하위 호환성을 위한 별칭 (함수명 변경 및 재export)
_parse_bool = parse_bool
_parse_positive_int = parse_positive_int
_sync_segment_to_range = sync_segment_to_range
_segments_with_remote_audio_paths = segments_with_remote_audio_paths
_build_speaker_metadata = build_speaker_metadata
_build_speaker_refs_metadata = build_speaker_refs_metadata
_upload_speaker_embeddings = upload_speaker_embeddings
_materialize_voice_replacements = materialize_voice_replacements
_maybe_prepare_voice_replacements = (
    lambda paths, target_lang, default_bucket: maybe_prepare_voice_replacements(
        paths,
        target_lang,
        default_bucket,
        VOICE_SAMPLES_EMBED_DIR,
        VOICE_LIBRARY_BUCKET,
    )
)
_ensure_voice_library_index = (
    lambda language, force_refresh=False: ensure_voice_library_index(
        language, force_refresh, VOICE_SAMPLES_EMBED_DIR, VOICE_LIBRARY_BUCKET
    )
)
_resolve_s3_location = resolve_s3_location
_strip_voice_samples_prefix = strip_voice_samples_prefix
_build_sqs_message_kwargs = build_sqs_message_kwargs


# check_and_trigger_mux_if_complete를 래퍼로 감싸서 send_callback을 자동으로 전달
def check_and_trigger_mux_if_complete_wrapper(
    job_id: str,
    project_id: str | None,
    total_segments: int,
    output_bucket: str,
    project_prefix: str,
    callback_url: str | None,
    job_details: dict,
) -> bool:
    """check_and_trigger_mux_if_complete의 래퍼 함수 (하위 호환성)"""
    return check_and_trigger_mux_if_complete(
        job_id,
        project_id,
        total_segments,
        output_bucket,
        project_prefix,
        callback_url,
        job_details,
        send_callback_func=send_callback,
    )


# 하위 호환성을 위해 원래 이름으로도 export
check_and_trigger_mux_if_complete = check_and_trigger_mux_if_complete_wrapper


def full_pipeline(job_details: dict):
    """전체 더빙 파이프라인을 실행합니다."""
    from pipelines.full_pipeline import full_pipeline as pipeline_func

    pipeline_func(job_details)


def _handle_tts_segments(job_details: dict) -> None:
    """segment_tts / tts 작업을 처리합니다."""
    from pipelines.tts_segments import handle_tts_segments

    handle_tts_segments(job_details)


def _handle_test_synthesis(job_details: dict):
    """test_synthesis 작업을 처리합니다."""
    from pipelines.test_synthesis import handle_test_synthesis

    handle_test_synthesis(job_details)


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

                    # Lazy import to avoid circular import
                    if task == "test_synthesis":
                        from pipelines import handle_test_synthesis

                        handle_test_synthesis(job_details)
                    elif task in ("segment_tts", "tts"):
                        from pipelines import handle_tts_segments

                        handle_tts_segments(job_details)
                    elif task == "split_up":
                        from pipelines import split_up

                        split_up(job_details)
                    elif task == "chunk_work":
                        from pipelines import chunk_work

                        chunk_work(job_details)
                    elif task == "mux":
                        from pipelines import handle_mux_task

                        handle_mux_task(job_details)
                    else:
                        # 파이프라인 실행
                        from pipelines import full_pipeline

                        full_pipeline(job_details)

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
