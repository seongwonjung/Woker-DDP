"""Speaker-related utility functions."""

import json
import logging
from pathlib import Path

from configs import JobPaths
from utils.s3 import download_from_s3, upload_to_s3


def upload_speaker_refs(
    paths: JobPaths,
    project_prefix: str,
    job_id: str,
    output_bucket: str,
) -> None:
    """Speaker reference 파일들을 S3에 업로드합니다."""
    tts_dir = paths.vid_tts_dir
    speaker_refs_json_path = tts_dir / "speaker_refs.json"
    speaker_ref_dir = tts_dir / "self_refs"

    if speaker_refs_json_path.is_file():
        # speaker_refs.json 업로드
        refs_json_key = f"{project_prefix}/interim/{job_id}/tts/speaker_refs.json"
        if upload_to_s3(output_bucket, refs_json_key, speaker_refs_json_path):
            logging.info(f"Job {job_id}: Uploaded speaker_refs.json to S3")
        else:
            logging.warning(f"Job {job_id}: Failed to upload speaker_refs.json to S3")

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


def download_speaker_refs(
    paths: JobPaths,
    project_prefix: str,
    job_id: str,
    output_bucket: str,
    chunk_index: int | None = None,
) -> None:
    """Speaker reference 파일들을 S3에서 다운로드합니다."""
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


def upload_speaker_embeddings(
    paths: JobPaths,
    job_id: str,
    bucket: str,
) -> dict:
    """
    Upload locally cached speaker embeddings to the shared voice-samples prefix.
    Returns metadata describing where the embeddings were stored.
    """
    from utils.s3 import upload_to_s3

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
