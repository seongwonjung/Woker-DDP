"""Full Pipeline - 전체 더빙 파이프라인"""

import json
import logging
import time
from pathlib import Path
from typing import Any

# 상수 및 설정
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker import (
    AWS_S3_BUCKET,
    VOICE_LIBRARY_BUCKET,
    send_callback,
    download_from_s3,
    upload_to_s3,
    upload_metadata_to_s3,
    upload_audio_artifacts,
    resolve_output_prefix,
    _parse_bool,
    _parse_positive_int,
    _build_speaker_metadata,
    _build_speaker_refs_metadata,
    _upload_speaker_embeddings,
    _segments_with_remote_audio_paths,
    _maybe_prepare_voice_replacements,
)

from services.lang import normalize_lang_code
from services.stt import run_asr
from services.translate import translate_transcript
from services.tts import generate_tts
from services.sync import sync_segments
from services.mux import mux_audio_video
from services.transcript_store import COMPACT_ARCHIVE_NAME, read_transcript_language
from configs import ensure_job_dirs


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
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
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
                metadata={
                    "job_id": job_id,
                    "project_id": project_id,
                    "target_lang": target_lang,
                },
            )
            return
        send_callback(
            callback_url,
            "in_progress",
            "Custom voice sample downloaded.",
            stage="downloaded",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
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
            "ASR completed.",
            stage="asr_completed",
            metadata=(
                {
                    "job_id": job_id,
                    "project_id": project_id,
                    "audio_key": audio_key,
                    "vocals_key": vocals_key,
                    "background_key": background_key,
                    "target_lang": target_lang,
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
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
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
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
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
            callback_url,
            "in_progress",
            "Starting TTS...",
            stage="tts_started",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
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
                "target_lang": target_lang,
            },
        )

        # 7. Sync
        stage_name = "sync"
        stage_start = time.time()
        send_callback(
            callback_url,
            "in_progress",
            "Starting sync...",
            stage="sync_started",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
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
            callback_url,
            "in_progress",
            "Sync completed.",
            stage="sync_completed",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
        )

        # 8. Mux
        stage_name = "mux"
        stage_start = time.time()
        send_callback(
            callback_url,
            "in_progress",
            "Starting mux...",
            stage="mux_started",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
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
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "target_lang": target_lang,
            },
        )
