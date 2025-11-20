import os
from dotenv import load_dotenv

load_dotenv()

JOB_QUEUE_URL = os.getenv(
    "JOB_QUEUE_URL",
    "https://sqs.ap-northeast-2.amazonaws.com/148761638563/test-dupilot-queue.fifo",
)
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "dupilot-dev-media")
AWS_REGION = os.getenv("AWS_REGION", "ap-northeast-2")
VISIBILITY_TIMEOUT = int(os.getenv("JOB_VISIBILITY_TIMEOUT", "300"))
POLL_WAIT = int(os.getenv("JOB_QUEUE_WAIT", "20"))
CALLBACK_LOCALHOST_HOST = os.getenv(
    "JOB_CALLBACK_LOCALHOST_HOST", "host.docker.internal"
)
PROFILE = os.getenv("AWS_PROFILE", "dev")
LOG_LEVEL = os.getenv("WORKER_LOG_LEVEL", "INFO")
DEFAULT_TARGET_LANG = os.getenv("DEFAULT_TARGET_LANG", "en")
DEFAULT_SOURCE_LANG = os.getenv("DEFAULT_SOURCE_LANG", "auto")
