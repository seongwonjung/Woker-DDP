"""SQS utility functions for message queue operations."""

import hashlib
import json
import logging
import time
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from utils.s3 import get_s3_client, upload_metadata_to_s3, download_from_s3

# SQS 설정
JOB_QUEUE_URL = None
JOB_QUEUE_FIFO = False
JOB_QUEUE_MESSAGE_GROUP_ID = None
sqs_client = None


def init_sqs_client(
    queue_url: str,
    region: str = "ap-northeast-2",
    fifo_env: bool = False,
    fifo_url: bool = False,
    message_group_id: str | None = None,
):
    """SQS 클라이언트를 초기화합니다."""
    global JOB_QUEUE_URL, JOB_QUEUE_FIFO, JOB_QUEUE_MESSAGE_GROUP_ID, sqs_client
    JOB_QUEUE_URL = queue_url
    JOB_QUEUE_FIFO = fifo_env or fifo_url
    JOB_QUEUE_MESSAGE_GROUP_ID = message_group_id
    sqs_client = boto3.client("sqs", region_name=region)
    return sqs_client


def get_sqs_client():
    """SQS 클라이언트를 반환합니다."""
    global sqs_client
    if sqs_client is None:
        raise ValueError("SQS client not initialized. Call init_sqs_client() first.")
    return sqs_client


def build_sqs_message_kwargs(
    message_body: str,
    project_id: str | None = None,
    deduplication_id: str | None = None,
    group_id: str | None = None,
) -> dict:
    """FIFO 큐를 위한 SQS 메시지 파라미터를 구성합니다."""
    if JOB_QUEUE_URL is None:
        raise ValueError("JOB_QUEUE_URL not initialized. Call init_sqs_client() first.")

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
            kwargs["MessageDeduplicationId"] = hashlib.md5(
                message_body.encode("utf-8")
            ).hexdigest()

        logging.debug(
            f"SQS FIFO message: GroupId={final_group_id}, "
            f"DeduplicationId={kwargs.get('MessageDeduplicationId')}"
        )

    return kwargs


def check_and_trigger_mux_if_complete(
    job_id: str,
    project_id: str | None,
    total_segments: int,
    output_bucket: str,
    project_prefix: str,
    callback_url: str | None,
    job_details: dict,
    send_callback_func,
) -> bool:
    """S3의 synced .wav 파일 개수를 확인하고, 모두 완료되면 mux 큐에 메시지 전송.

    Args:
        send_callback_func: 콜백 전송 함수 (send_callback)

    Returns:
        True if mux was triggered, False otherwise
    """
    synced_prefix = f"{project_prefix}/interim/{job_id}/text/vid/tts/synced/"

    # S3에서 .wav 파일 개수 확인 (pagination 처리)
    wav_count = 0
    try:
        paginator = get_s3_client().get_paginator("list_objects_v2")
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
            get_s3_client().head_object(Bucket=output_bucket, Key=mux_lock_key)
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
            message_kwargs = build_sqs_message_kwargs(
                message_body,
                project_id=project_id,
                deduplication_id=f"{job_id}_mux",
            )
            response = get_sqs_client().send_message(**message_kwargs)
            logging.debug(
                f"Job {job_id}: SQS send_message response: {response.get('MessageId')}"
            )
            logging.info(
                f"Job {job_id}: Triggered mux task. "
                f"Found {wav_count} synced .wav files (expected {total_segments})"
            )

            if callback_url and send_callback_func:
                send_callback_func(
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
