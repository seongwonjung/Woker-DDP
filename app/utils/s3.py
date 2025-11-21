"""S3 utility functions for file upload/download operations."""

import json
import logging
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

# AWS 설정
AWS_REGION = None
s3_client = None


def init_s3_client(region: str = "ap-northeast-2"):
    """S3 클라이언트를 초기화합니다."""
    global AWS_REGION, s3_client
    AWS_REGION = region
    s3_client = boto3.client("s3", region_name=AWS_REGION)
    return s3_client


def get_s3_client():
    """S3 클라이언트를 반환합니다. 없으면 초기화합니다."""
    global s3_client
    if s3_client is None:
        init_s3_client()
    return s3_client


def download_from_s3(
    bucket: str, key: str, local_path: Path, force: bool = False
) -> bool:
    """S3에서 파일을 다운로드합니다. (캐싱 지원)

    Args:
        bucket: S3 버킷 이름
        key: S3 객체 키
        local_path: 로컬 저장 경로
        force: True면 캐시 무시하고 강제 다운로드
    """
    # 파일이 이미 존재하고 force=False면 스킵
    if not force and local_path.exists() and local_path.is_file():
        logging.debug(
            f"File already exists at {local_path}, skipping download from s3://{bucket}/{key}"
        )
        return True

    try:
        logging.info(f"Downloading s3://{bucket}/{key} to {local_path}...")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        get_s3_client().download_file(bucket, key, str(local_path))
        logging.info(f"Successfully downloaded s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to download from S3: {e}")
        return False


def upload_to_s3(bucket: str, key: str, local_path: Path) -> bool:
    """S3로 파일을 업로드합니다."""
    try:
        logging.info(f"Uploading {local_path} to s3://{bucket}/{key}...")
        get_s3_client().upload_file(str(local_path), bucket, key)
        logging.info(f"Successfully uploaded to s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logging.error(f"Failed to upload to S3: {e}")
        return False
    except FileNotFoundError:
        logging.error(f"Local file not found for upload: {local_path}")
        return False


def upload_metadata_to_s3(bucket: str, key: str, metadata: dict) -> bool:
    """파이프라인 메타데이터를 JSON으로 직렬화해 S3에 업로드합니다."""
    try:
        body = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
        logging.info(f"Uploading metadata to s3://{bucket}/{key}...")
        get_s3_client().put_object(
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


def upload_audio_artifacts(
    paths,
    project_prefix: str,
    job_id: str,
    output_bucket: str,
) -> dict:
    """ASR 후 생성된 오디오 아티팩트들을 S3에 업로드합니다.

    Returns:
        업로드된 파일들의 S3 키를 담은 딕셔너리
    """
    audio_files = {}

    # 원본 오디오(audio.wav) 업로드
    raw_audio_path = paths.vid_speaks_dir / "audio.wav"
    if raw_audio_path.is_file():
        audio_key = f"{project_prefix}/interim/{job_id}/audio/audio.wav"
        if upload_to_s3(output_bucket, audio_key, raw_audio_path):
            audio_files["audio.wav"] = f"s3://{output_bucket}/{audio_key}"
            logging.info(f"Raw audio uploaded to s3://{output_bucket}/{audio_key}")
        else:
            logging.warning("Failed to upload audio.wav to S3")

    # 발화 음성(vocals.wav) 업로드
    vocals_path = paths.vid_speaks_dir / "vocals.wav"
    if vocals_path.is_file():
        vocals_key = f"{project_prefix}/interim/{job_id}/audio/vocals.wav"
        if upload_to_s3(output_bucket, vocals_key, vocals_path):
            audio_files["vocals.wav"] = f"s3://{output_bucket}/{vocals_key}"
            logging.info(f"Vocals uploaded to s3://{output_bucket}/{vocals_key}")
        else:
            logging.warning("Failed to upload vocals.wav to S3")

    # 배경음(background.wav) 업로드
    background_path = paths.vid_bgm_dir / "background.wav"
    if background_path.is_file():
        background_key = f"{project_prefix}/interim/{job_id}/audio/background.wav"
        if upload_to_s3(output_bucket, background_key, background_path):
            audio_files["background.wav"] = f"s3://{output_bucket}/{background_key}"
            logging.info(
                f"Background uploaded to s3://{output_bucket}/{background_key}"
            )
        else:
            logging.warning("Failed to upload background.wav to S3")

    return audio_files
