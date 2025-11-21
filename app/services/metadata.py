"""Metadata building functions for pipeline results."""

import json
import logging
from pathlib import Path

from configs import JobPaths


def build_speaker_metadata(
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


def build_speaker_refs_metadata(
    paths: JobPaths,
    project_prefix: str,
    job_id: str,
    output_bucket: str,
) -> dict:
    """
    Upload speaker reference samples to S3 and return metadata in dict format.
    Returns dict format for final pipeline callback: {"speaker0": {"ref_wav_key": "s3://...", "prompt_text": "..."}}
    """
    from utils.s3 import upload_to_s3

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


def segments_with_remote_audio_paths(
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
