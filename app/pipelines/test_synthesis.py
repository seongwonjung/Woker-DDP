"""Test Synthesis Pipeline - 보이스 샘플 테스트 합성"""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker import (
    AWS_S3_BUCKET,
    VOICE_LIBRARY_BUCKET,
    VOICE_SAMPLES_EMBED_DIR,
    send_callback,
    download_from_s3,
    upload_to_s3,
    _ensure_voice_library_index,
)

from services.lang import normalize_lang_code
from services.demucs_split import split_vocals
from services.tts import _transcribe_prompt_text, _synthesize_with_cosyvoice2
from services.speaker_embeddings import save_audio_embedding
from services.voice_recommendation import update_voice_library_entry
from configs import ensure_job_dirs
from pydub import AudioSegment


def handle_test_synthesis(job_details: dict):
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
