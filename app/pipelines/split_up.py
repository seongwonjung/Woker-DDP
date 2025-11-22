"""Split Up Pipeline - ASR 후 청킹 및 큐 전송"""

import json
import logging
import time
from pathlib import Path

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker import (
    AWS_S3_BUCKET,
    CHUNK_SIZE,
    VOICE_LIBRARY_BUCKET,
    send_callback,
    download_from_s3,
    upload_to_s3,
    upload_metadata_to_s3,
    upload_audio_artifacts,
    upload_speaker_refs,
    _parse_positive_int,
    _parse_bool,
    _build_sqs_message_kwargs,
    sqs_client,
)

from services.lang import normalize_lang_code
from services.stt import run_asr
from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    read_transcript_language,
    load_compact_transcript,
    segment_views,
)
from services.voice_recommendation import maybe_prepare_voice_replacements
from configs import ensure_job_dirs


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


def split_up(job_details: dict):
    """ASR 수행 후 청킹하고 각 청크를 큐에 전송합니다."""
    # 시간 측정 시작
    start_time = time.time()

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

    # Voice replacement 플래그 파싱
    voice_replacement_flag = (
        job_details.get("replace_voice_samples")
        or job_details.get("is_replace_voice_samples")
        or job_details.get("voice_sample_substitution")
    )
    replace_voice_samples = _parse_bool(voice_replacement_flag)
    voice_library_bucket = (
        job_details.get("voice_library_bucket") or VOICE_LIBRARY_BUCKET
    )
    # # 청크 크기 설정 (job_details에서 오버라이드 가능)
    # chunk_size = (
    #     _parse_positive_int(job_details.get("chunk_size"), "chunk_size") or CHUNK_SIZE
    # 청크 크기는 ASR 완료 후 total_segments를 알게 되면 동적으로 계산됨
    # 초기값은 job_details에서 오버라이드 가능하거나 기본값 사용
    initial_chunk_size = (
        _parse_positive_int(job_details.get("chunk_size"), "chunk_size") or None
    )
    # chunk_size는 ASR 완료 후 동적으로 계산되므로 초기값 설정
    chunk_size: int = initial_chunk_size or CHUNK_SIZE

    send_callback(
        callback_url,
        "in_progress",
        f"Starting split_up for job {job_id}",
        stage="split_up_started",
        metadata={
            "job_id": job_id,
            "project_id": project_id,
            "target_lang": target_lang,
            "chunk_size": initial_chunk_size or "auto (will be calculated after ASR)",
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
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
        )
        return

    detected_source_lang: str | None = None
    effective_source_lang: str | None = source_lang

    try:
        # 3. ASR (STT) 수행
        send_callback(
            callback_url,
            "in_progress",
            "Starting ASR...",
            stage="asr_started",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
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

        # 청크 크기 동적 계산: job_details에서 명시적으로 제공되지 않으면 total_segments/4 사용
        if initial_chunk_size is not None:
            chunk_size = initial_chunk_size
        else:
            # 세그먼트 개수를 4로 나눈 값으로 청크 크기 설정 (최소 1)
            chunk_size = max(1, total_segments // 4)
            logging.info(
                f"Job {job_id}: Dynamically calculated chunk_size={chunk_size} "
                f"from total_segments={total_segments} (total_segments/4)"
            )

        # 4. ASR 결과물을 S3에 업로드
        compact_transcript_key = (
            f"{project_prefix}/interim/{job_id}/{COMPACT_ARCHIVE_NAME}"
        )
        if not upload_to_s3(output_bucket, compact_transcript_key, asr_result_path):
            raise Exception("Failed to upload compact transcript to S3")

        # Audio artifacts를 S3에 업로드 (헬퍼 함수 사용)
        uploaded_audio = upload_audio_artifacts(
            paths, project_prefix, job_id, output_bucket
        )

        # s3:// 접두사 제거 (upload_audio_artifacts가 s3://bucket/key 형식으로 반환)
        def extract_s3_key(s3_path: str | None) -> str | None:
            """s3://bucket/key 형식에서 key 부분만 추출"""
            if not s3_path:
                return None
            if s3_path.startswith("s3://"):
                # s3://bucket/key 형식에서 key 부분만 추출
                parts = s3_path.replace("s3://", "").split("/", 1)
                if len(parts) > 1:
                    return parts[1]  # key 부분만 반환
                else:
                    return parts[0]  # bucket만 있는 경우 (거의 없음)
            return s3_path  # 이미 key 형식인 경우

        audio_key = extract_s3_key(uploaded_audio.get("audio.wav"))
        vocals_key = extract_s3_key(uploaded_audio.get("vocals.wav"))
        background_key = extract_s3_key(uploaded_audio.get("background.wav"))

        send_callback(
            callback_url,
            "in_progress",
            "ASR completed. Creating chunks...",
            stage="asr_completed",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "total_segments": total_segments,
                "chunk_size": chunk_size,
                "audio_key": audio_key,
                "vocals_key": vocals_key,
                "background_key": background_key,
                "target_lang": target_lang,
            },
        )

        # speaker_refs 업로드 (헬퍼 함수 사용)
        upload_speaker_refs(paths, project_prefix, job_id, output_bucket)

        # 4.5. Voice replacement 결정 및 준비 (replace_voice_samples가 True인 경우)
        voice_replacements = {}
        voice_replacement_diagnostics = {}
        if replace_voice_samples:
            logging.info(
                f"Job {job_id}: Preparing voice replacements for target_lang={target_lang}"
            )
            try:
                overrides, diagnostics = maybe_prepare_voice_replacements(
                    paths, target_lang, voice_library_bucket or output_bucket
                )
                voice_replacements = overrides
                voice_replacement_diagnostics = diagnostics

                # Voice replacement 원본 정보만 manifest에 저장 (S3 업로드 제거)
                # chunk_work에서 Voice Library에서 직접 다운로드하도록 최적화
                if voice_replacements:
                    logging.info(
                        f"Job {job_id}: Prepared {len(voice_replacements)} voice replacements "
                        f"(will be downloaded from Voice Library in chunk_work)"
                    )
                    # manifest에 저장할 때는 원본 Voice Library 경로 정보만 저장
                    # audio_path는 로컬 경로이므로 제거 (chunk_work에서 다운로드할 것)
                    for speaker, replacement_info in voice_replacements.items():
                        # 원본 Voice Library 정보만 유지
                        # sample_key, sample_bucket은 이미 materialize_voice_replacements에서 설정됨
                        if "audio_path" in replacement_info:
                            # 로컬 경로는 manifest에 저장하지 않음
                            del replacement_info["audio_path"]
                        logging.debug(
                            f"Job {job_id}: Voice replacement for {speaker}: "
                            f"sample_key={replacement_info.get('sample_key')}, "
                            f"sample_bucket={replacement_info.get('sample_bucket')}"
                        )
                else:
                    logging.info(
                        f"Job {job_id}: No voice replacements prepared "
                        f"(reason: {voice_replacement_diagnostics.get('reason', 'unknown')})"
                    )
            except Exception as e:
                logging.warning(
                    f"Job {job_id}: Failed to prepare voice replacements: {e}",
                    exc_info=True,
                )
                voice_replacement_diagnostics = {
                    "enabled": False,
                    "reason": f"error: {str(e)}",
                    "target_lang": target_lang,
                }

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
        # audio_key, vocals_key, background_key는 S3 키 형식 (s3:// 접두사 없음)
        if audio_key:
            manifest["audio_files"]["audio_wav"] = audio_key
        if vocals_key:
            manifest["audio_files"]["vocals_wav"] = vocals_key
        if background_key:
            manifest["audio_files"]["background_wav"] = background_key

        # Voice replacement 정보 추가 (있는 경우만)
        if voice_replacements:
            manifest["voice_replacements"] = voice_replacements
        if voice_replacement_diagnostics:
            manifest["voice_replacement_diagnostics"] = voice_replacement_diagnostics

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
                "target_lang": target_lang,
            },
        )

        logging.info(
            f"Job {job_id}: split_up completed. "
            f"Sent {chunk_messages_sent}/{total_chunks} chunks to queue"
        )

        # 시간 측정 및 저장 (끝나기 직전)
        end_time = time.time()
        total_time = end_time - start_time

        timing_data = {
            "job_id": job_id,
            "project_id": project_id,
            "total_time_seconds": round(total_time, 3),
            "total_time_formatted": f"{int(total_time // 60)}m {int(total_time % 60)}s",
            "start_time": start_time,
            "end_time": end_time,
            "total_segments": total_segments,
            "total_chunks": total_chunks,
            "chunk_size": chunk_size,
        }

        # S3에 저장
        if project_id:
            timing_key = f"projects/{project_id}/split_up_time.json"
        else:
            timing_key = f"jobs/{job_id}/split_up_time.json"

        upload_metadata_to_s3(output_bucket, timing_key, timing_data)
        logging.info(
            f"Job {job_id}: Saved timing data to s3://{output_bucket}/{timing_key}"
        )

    except Exception as e:
        logging.error(f"split_up failed for job {job_id}: {e}", exc_info=True)
        send_callback(
            callback_url,
            "failed",
            str(e),
            stage="split_up_failed",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
        )
        raise
