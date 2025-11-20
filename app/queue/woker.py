import json
import logging
import time
from typing import Any, Dict, Optional

import boto3
import requests
from botocore.exceptions import BotoCoreError, ClientError

try:
    from app.configs.utils import JobProcessingError, post_status
    from app.configs.env import (
        JOB_QUEUE_URL,
        AWS_REGION,
        AWS_S3_BUCKET,
        LOG_LEVEL,
        POLL_WAIT,
        PROFILE,
        VISIBILITY_TIMEOUT,
    )
except ModuleNotFoundError as exc:
    if exc.name != "app":
        raise
    from configs.utils import JobProcessingError, post_status
    from configs.env import (
        JOB_QUEUE_URL,
        AWS_REGION,
        AWS_S3_BUCKET,
        LOG_LEVEL,
        POLL_WAIT,
        PROFILE,
        VISIBILITY_TIMEOUT,
    )
from .pipeline.full_pipeline import FullPipeline

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
else:
    logger.setLevel(LOG_LEVEL)


class QueueWorker:
    def __init__(self):
        # 큐 위치 및 처리 파라미터를 미리 보관해 둔다.
        self.queue_url = JOB_QUEUE_URL
        self.bucket = AWS_S3_BUCKET
        self.visibility_timeout = VISIBILITY_TIMEOUT
        self.poll_wait = POLL_WAIT

        session_kwargs: dict = {}
        if PROFILE:
            session_kwargs["profile_name"] = PROFILE
        boto_session = boto3.Session(region_name=AWS_REGION, **session_kwargs)
        self.sqs_client = boto_session.client("sqs", region_name=AWS_REGION)
        self.s3_client = boto_session.client("s3", region_name=AWS_REGION)
        self.http = requests.Session()

    def poll_forever(self) -> None:
        """SQS에서 작업을 영구적으로 끌어와 처리한다."""
        while True:
            try:
                messages = self.sqs_client.receive_message(
                    QueueUrl=self.queue_url,
                    MaxNumberOfMessages=1,
                    WaitTimeSeconds=self.poll_wait,
                    VisibilityTimeout=self.visibility_timeout,
                    MessageAttributeNames=["All"],
                    AttributeNames=["All"],
                )
            except (BotoCoreError, ClientError) as exc:
                logger.error("SQS 폴링에 실패했습니다: %s", exc)
                time.sleep(5)
                continue

            for message in messages.get("Messages", []):
                receipt = message["ReceiptHandle"]
                receive_count = int(
                    message.get("Attributes", {}).get("ApproximateReceiveCount", "1")
                )
                message_id = message.get("MessageId")
                logger.info(
                    "SQS 메시지를 수신했습니다 (수신횟수=%s, MessageId=%s)",
                    receive_count,
                    message_id,
                )

                payload = self._decode_payload(message)
                if payload is None:
                    self._delete_message(receipt, message_id)
                    continue

                try:
                    result = self.__handle_job(payload)
                except JobProcessingError as exc:
                    logger.error(
                        "작업 %s 이(가) 실패했습니다: %s",
                        payload.get("job_id") or "unknown",
                        exc,
                    )
                    self._handle_failure(payload, exc, receive_count)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception(
                        "메시지 처리 중 알 수 없는 오류가 발생했습니다: %s", exc
                    )
                    wrapped = JobProcessingError(str(exc))
                    self._handle_failure(payload, wrapped, receive_count)
                else:
                    self._handle_success(payload, result)
                finally:
                    self._delete_message(receipt, message_id)

    def __handle_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        task = (payload.get("task") or "full_pipeline").lower()
        if task != "full_pipeline":
            raise JobProcessingError(f"지원하지 않는 작업 유형입니다: {task}")

        pipeline = FullPipeline(
            payload=payload,
            s3_client=self.s3_client,
            http=self.http,
            input_bucket=payload.get("input_bucket") or self.bucket,
            output_bucket=payload.get("output_bucket") or self.bucket,
        )
        return pipeline.process()

    # ------------------------------------------------------------------
    def _decode_payload(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """SQS 메시지 body 를 JSON 딕셔너리로 변환한다."""
        try:
            body = json.loads(message.get("Body") or "{}")
        except json.JSONDecodeError:
            logger.error("메시지 본문이 잘못되어 삭제합니다: %s", message.get("Body"))
            return None

        if (
            isinstance(body, dict)
            and "Message" in body
            and isinstance(body["Message"], str)
        ):
            try:
                return json.loads(body["Message"])
            except json.JSONDecodeError:
                logger.error("중첩된 Message 필드가 올바른 JSON 이 아닙니다: %s", body)
                return None
        if isinstance(body, dict):
            return body
        logger.error("예상하지 못한 메시지 구조입니다: %s", body)
        return None

    def _handle_success(self, payload: Dict[str, Any], result: Dict[str, Any]) -> None:
        """파이프라인 성공 시 콜백 API 로 결과를 알린다."""
        callback_url = payload.get("callback_url")
        if not callback_url:
            logger.info(
                "작업 %s 성공 완료, 그러나 callback_url 이 없어 콜백을 생략합니다",
                result.get("job_id"),
            )
            return
        metadata = {
            "job_id": result.get("job_id"),
            "project_id": result.get("project_id"),
            "result_bucket": result.get("result_bucket"),
            "result_key": result.get("result_key"),
            "metadata_key": result.get("metadata_key"),
            "segment_count": result.get("segment_count"),
            "target_lang": payload.get("target_lang"),
            "source_lang": result.get("source_lang"),
            "detected_source_lang": result.get("detected_source_lang"),
        }
        try:
            post_status(
                self.http,
                callback_url,
                "done",
                result_key=result.get("result_key"),
                metadata=metadata,
                project_id=payload.get("project_id"),
            )
            logger.info("작업 %s 성공 콜백을 발송했습니다", metadata["job_id"])
        except JobProcessingError as exc:
            logger.error("성공 콜백 전송에 실패했습니다: %s", exc)

    def _handle_failure(
        self, payload: Dict[str, Any], error: JobProcessingError, receive_count: int
    ) -> None:
        """파이프라인 실패 시 콜백 API 로 오류를 알린다."""
        callback_url = payload.get("callback_url")
        if not callback_url:
            logger.error(
                "작업 %s 이(가) 실패했지만 callback_url 이 없습니다",
                payload.get("job_id") or "unknown",
            )
            return
        metadata = {
            "job_id": payload.get("job_id"),
            "project_id": payload.get("project_id"),
            "receive_count": receive_count,
        }
        try:
            post_status(
                self.http,
                callback_url,
                "failed",
                error=str(error),
                metadata=metadata,
                project_id=payload.get("project_id"),
            )
            logger.info(
                "작업 %s 실패 콜백을 발송했습니다",
                payload.get("job_id") or "unknown",
            )
        except JobProcessingError as exc:
            logger.error("실패 콜백 전송에 실패했습니다: %s", exc)

    def _delete_message(self, receipt: str, message_id: Optional[str]) -> None:
        """현재 처리한 메시지를 큐에서 제거한다."""
        try:
            self.sqs_client.delete_message(
                QueueUrl=self.queue_url, ReceiptHandle=receipt
            )
            logger.info("SQS 메시지를 삭제했습니다: %s", message_id)
        except (BotoCoreError, ClientError) as exc:
            logger.error("SQS 메시지 %s 삭제에 실패했습니다: %s", message_id, exc)


if __name__ == "__main__":
    worker = QueueWorker()
    worker.poll_forever()
