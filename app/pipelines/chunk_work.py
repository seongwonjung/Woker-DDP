"""Chunk Work Pipeline - 청크별 처리 (번역, TTS, Sync)"""

import json
import logging
import shutil
import time
from pathlib import Path

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
    download_speaker_refs,
    _parse_bool,
)
from utils.sqs import check_and_trigger_mux_if_complete

from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    load_compact_transcript,
    segment_views,
    save_compact_transcript,
)
from services.translate import translate_transcript
from services.tts import generate_tts
from services.sync import sync_segments
from configs import ensure_job_dirs


def chunk_work(job_details: dict):
    """청크별로 번역, TTS, Sync를 수행합니다."""
    # 시간 측정 시작
    start_time = time.time()

    job_id = job_details["job_id"]
    project_id = job_details.get("project_id")
    chunk_index = job_details["chunk_index"]
    start_segment_index = job_details["start_segment_index"]
    end_segment_index = job_details["end_segment_index"]
    total_chunks = job_details["total_chunks"]
    total_segments = job_details["total_segments"]
    manifest_key = job_details["manifest_key"]
    target_lang = job_details.get("target_lang", "en")
    source_lang = job_details.get("source_lang")
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET
    project_prefix = job_details.get("project_prefix") or (
        f"projects/{project_id}" if project_id else "jobs"
    )
    callback_url = job_details.get("callback_url")

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

    logging.info(
        f"Job {job_id}: Starting chunk_work for chunk {chunk_index} "
        f"(segments {start_segment_index}-{end_segment_index})"
    )

    if callback_url:
        send_callback(
            callback_url,
            "in_progress",
            f"Processing chunk {chunk_index + 1}/{total_chunks}...",
            stage="chunk_work_started",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "target_lang": target_lang,
            },
        )

    # 1. 로컬 작업 디렉토리 설정
    paths = ensure_job_dirs(job_id)

    try:
        # 2. manifest.json 다운로드
        manifest_path = paths.interim_dir / "manifest.json"
        if not download_from_s3(output_bucket, manifest_key, manifest_path):
            raise Exception(f"Failed to download manifest.json from {manifest_key}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # 3. compact_transcript 다운로드
        compact_transcript_s3_key = manifest["audio_files"]["compact_transcript"]
        # s3://bucket/key 형식 처리
        if compact_transcript_s3_key.startswith("s3://"):
            _, compact_transcript_key = compact_transcript_s3_key.replace(
                "s3://", ""
            ).split("/", 1)
        else:
            compact_transcript_key = compact_transcript_s3_key

        compact_transcript_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
        compact_transcript_path.parent.mkdir(parents=True, exist_ok=True)

        if not download_from_s3(
            output_bucket, compact_transcript_key, compact_transcript_path
        ):
            raise Exception(
                f"Failed to download compact_transcript from {compact_transcript_key}"
            )

        # 3.5. speaker_refs 다운로드 (헬퍼 함수 사용)
        download_speaker_refs(paths, project_prefix, job_id, output_bucket, chunk_index)

        # 4. 청크 범위의 세그먼트만 필터링
        bundle = load_compact_transcript(compact_transcript_path)
        all_segments = segment_views(bundle)

        # 디버깅: 세그먼트 인덱스 확인
        if all_segments:
            min_idx = min(seg.idx for seg in all_segments)
            max_idx = max(seg.idx for seg in all_segments)
            logging.debug(
                f"Job {job_id} chunk {chunk_index}: "
                f"Total segments: {len(all_segments)}, "
                f"idx range: {min_idx}-{max_idx}, "
                f"looking for: {start_segment_index}-{end_segment_index}"
            )
        else:
            logging.warning(
                f"Job {job_id} chunk {chunk_index}: No segments loaded from transcript"
            )

        # 청크 범위 필터링
        chunk_segments = [
            seg
            for seg in all_segments
            if start_segment_index <= seg.idx <= end_segment_index
        ]

        if not chunk_segments:
            raise ValueError(
                f"No segments found in range {start_segment_index}-{end_segment_index}. "
                f"Total segments: {len(all_segments)}, "
                f"Available idx range: {min(seg.idx for seg in all_segments) if all_segments else 'N/A'}-{max(seg.idx for seg in all_segments) if all_segments else 'N/A'}"
            )

        logging.info(
            f"Job {job_id} chunk {chunk_index}: "
            f"Filtered {len(chunk_segments)} segments from {len(all_segments)} total"
        )

        # 필터링된 세그먼트로 compact_transcript 재구성
        # 원본 bundle의 구조를 유지하면서 segments만 필터링
        # 중요: 필터링된 배열 인덱스 -> 원본 인덱스 매핑 생성
        filtered_bundle = dict(bundle)
        original_segments = bundle.get("segments", [])
        filtered_segments = []
        filtered_to_original_idx_map = {}  # 필터링된 인덱스 -> 원본 인덱스 매핑

        for i, seg in enumerate(original_segments):
            if start_segment_index <= i <= end_segment_index:
                filtered_idx = len(filtered_segments)  # 필터링된 배열에서의 새 인덱스
                filtered_to_original_idx_map[filtered_idx] = i  # 원본 인덱스 저장
                filtered_segments.append(seg)

        filtered_bundle["segments"] = filtered_segments

        # 필터링된 compact_transcript를 임시로 저장 (번역/TTS/Sync에서 사용)
        chunk_transcript_path = (
            paths.src_sentence_dir / f"chunk_{chunk_index}_transcript.comp.json"
        )
        save_compact_transcript(filtered_bundle, chunk_transcript_path)

        # 원본 transcript를 백업하고 청크 버전으로 교체 (임시)
        original_transcript_backup = (
            paths.src_sentence_dir / f"original_{COMPACT_ARCHIVE_NAME}"
        )
        if compact_transcript_path.exists():
            shutil.copy(compact_transcript_path, original_transcript_backup)

        # 청크 버전을 원본 위치에 복사 (번역/TTS/Sync 함수들이 이 경로를 사용)
        shutil.copy(chunk_transcript_path, compact_transcript_path)

        # 5. 번역 수행 (청크 범위만)
        logging.info(f"Job {job_id} chunk {chunk_index}: Starting translation...")
        translations = translate_transcript(job_id, target_lang, src_lang=source_lang)

        # 번역 결과의 seg_idx를 원본 인덱스로 변환
        for trans in translations:
            filtered_idx = trans.get("seg_idx")
            if (
                filtered_idx is not None
                and filtered_idx in filtered_to_original_idx_map
            ):
                original_idx = filtered_to_original_idx_map[filtered_idx]
                trans["seg_idx"] = original_idx
                logging.debug(
                    f"Job {job_id} chunk {chunk_index}: "
                    f"Mapped filtered seg_idx {filtered_idx} -> original seg_idx {original_idx}"
                )

        # 번역 결과를 청크별로 저장
        trans_result_path = (
            paths.trg_sentence_dir / f"chunk_{chunk_index}_translated.json"
        )
        trans_result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trans_result_path, "w", encoding="utf-8") as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)

        # S3에 업로드
        trans_s3_key = f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_index}/translated.json"
        upload_to_s3(output_bucket, trans_s3_key, trans_result_path)

        logging.info(f"Job {job_id} chunk {chunk_index}: Translation completed")

        # 6. Voice replacement 준비 (manifest에서 읽기)
        speaker_voice_overrides: dict[str, dict] = {}
        if replace_voice_samples and "voice_replacements" in manifest:
            voice_replacements = manifest["voice_replacements"]
            logging.info(
                f"Job {job_id} chunk {chunk_index}: "
                f"Loading voice replacements from manifest ({len(voice_replacements)} speakers)"
            )

            # S3에서 voice replacement 파일들 다운로드
            asset_dir = paths.interim_dir / "voice_replacements"
            asset_dir.mkdir(parents=True, exist_ok=True)

            for speaker, replacement_info in voice_replacements.items():
                local_path = asset_dir / f"{speaker}_{replacement_info['voice_id']}.wav"

                # 우선순위: 원본 Voice Library 경로 > job-specific S3 경로 > 로컬 경로
                downloaded = False

                # 1. 원본 Voice Library에서 직접 다운로드 (최적화)
                if "sample_key" in replacement_info and not local_path.exists():
                    sample_bucket = replacement_info.get(
                        "sample_bucket", voice_library_bucket or output_bucket
                    )
                    sample_key = replacement_info["sample_key"]
                    if download_from_s3(sample_bucket, sample_key, local_path):
                        logging.info(
                            f"Job {job_id} chunk {chunk_index}: "
                            f"Downloaded voice replacement for {speaker} from Voice Library "
                            f"(s3://{sample_bucket}/{sample_key})"
                        )
                        downloaded = True
                    else:
                        logging.warning(
                            f"Job {job_id} chunk {chunk_index}: "
                            f"Failed to download voice replacement for {speaker} "
                            f"from Voice Library (s3://{sample_bucket}/{sample_key})"
                        )

                # 2. Fallback: job-specific S3 경로에서 다운로드 (하위 호환성)
                if (
                    not downloaded
                    and "s3_key" in replacement_info
                    and not local_path.exists()
                ):
                    s3_bucket = replacement_info.get("s3_bucket", output_bucket)
                    if download_from_s3(
                        s3_bucket, replacement_info["s3_key"], local_path
                    ):
                        logging.info(
                            f"Job {job_id} chunk {chunk_index}: "
                            f"Downloaded voice replacement for {speaker} from job-specific S3 path"
                        )
                        downloaded = True
                    else:
                        logging.warning(
                            f"Job {job_id} chunk {chunk_index}: "
                            f"Failed to download voice replacement for {speaker} from S3"
                        )

                # 3. Fallback: 이미 로컬에 있는 경우
                if local_path.exists():
                    downloaded = True
                elif "audio_path" in replacement_info:
                    # 이미 로컬 경로가 있는 경우 (fallback)
                    audio_path = Path(replacement_info["audio_path"])
                    if audio_path.exists():
                        # 기존 로컬 경로를 사용
                        local_path = audio_path
                        downloaded = True
                    else:
                        logging.warning(
                            f"Job {job_id} chunk {chunk_index}: "
                            f"Voice replacement file not found for {speaker}: {audio_path}"
                        )

                # 다운로드 성공한 경우에만 사용
                if downloaded and local_path.exists():
                    replacement_info["audio_path"] = str(local_path)
                    speaker_voice_overrides[speaker] = replacement_info
                elif not downloaded:
                    logging.warning(
                        f"Job {job_id} chunk {chunk_index}: "
                        f"Could not download or find voice replacement for {speaker}, skipping"
                    )
                    continue
        elif replace_voice_samples:
            logging.warning(
                f"Job {job_id} chunk {chunk_index}: "
                f"Voice replacement requested but not found in manifest"
            )

        # 7. TTS 수행 (청크 범위만)
        logging.info(f"Job {job_id} chunk {chunk_index}: Starting TTS...")
        user_voice_sample_path = None
        if (
            voice_config
            and voice_config.get("kind") == "s3"
            and voice_config.get("key")
        ):
            voice_key = voice_config["key"]
            voice_bucket = (
                voice_config.get("bucket")
                or voice_config.get("bucket_name")
                or output_bucket
            )
            user_voice_sample_path = paths.interim_dir / Path(voice_key).name
            download_from_s3(voice_bucket, voice_key, user_voice_sample_path)

        segments_payload = generate_tts(
            job_id,
            target_lang,
            voice_sample_path=user_voice_sample_path,
            speaker_voice_overrides=(
                speaker_voice_overrides if speaker_voice_overrides else None
            ),
        )

        # TTS 메타데이터만 업로드 (최적화: 실제 오디오 파일은 sync 후 업로드)
        # segments.json 파일만 업로드
        tts_dir = paths.vid_tts_dir
        segments_json_path = tts_dir / "segments.json"
        if segments_json_path.exists():
            segments_json_key = (
                f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_index}/"
                f"segments.json"
            )
            upload_to_s3(output_bucket, segments_json_key, segments_json_path)

        logging.info(f"Job {job_id} chunk {chunk_index}: TTS completed")

        # 8. Sync 수행 (청크 범위만)
        logging.info(f"Job {job_id} chunk {chunk_index}: Starting sync...")
        try:
            synced_segments = sync_segments(job_id)
        except FileNotFoundError as exc:
            logging.warning(
                f"Job {job_id} chunk {chunk_index}: Sync artifacts not found: {exc}"
            )
            synced_segments = []
        except Exception as exc:
            logging.warning(f"Job {job_id} chunk {chunk_index}: Sync failed: {exc}")
            synced_segments = []

        # Global Index 복원: 청크 내부 인덱스를 전체 영상 기준 인덱스로 변환
        if synced_segments and start_segment_index is not None:
            logging.info(
                f"Job {job_id} chunk {chunk_index}: Restoring global indices "
                f"(start_segment_index={start_segment_index})"
            )
            for i, seg in enumerate(synced_segments):
                # Global Index 계산: 배열 인덱스(i)를 기준으로 계산
                # (sync_segments가 순서대로 반환하므로 배열 인덱스가 Local Index)
                global_idx = start_segment_index + i

                # seg_idx 업데이트 (Global Index로 변환)
                seg["seg_idx"] = global_idx

                # segment_index 업데이트 (Global Index로 변환)
                seg["segment_index"] = global_idx

                # segment_id 업데이트 ("segment_XXXX" 형식, Global Index 사용)
                seg["segment_id"] = f"segment_{global_idx:04d}"

            logging.info(
                f"Job {job_id} chunk {chunk_index}: Restored global indices for "
                f"{len(synced_segments)} segments"
            )

        # Sync 결과물을 S3에 업로드 (최적화: mux가 필요한 표준 경로로만 업로드)
        synced_dir = paths.vid_tts_dir / "synced"
        if synced_dir.exists():
            for sync_file in synced_dir.glob("**/*"):
                if not sync_file.is_file():
                    continue

                # mux를 위한 표준 경로로만 업로드 (중복 제거)
                sync_key = f"{project_prefix}/interim/{job_id}/text/vid/tts/synced/{sync_file.name}"
                upload_to_s3(output_bucket, sync_key, sync_file)

        # segments_synced.json도 업로드
        synced_meta_path = synced_dir / "segments_synced.json"

        # [FIX] 수정된 인덱스가 적용된 synced_segments를 파일에 다시 씀
        if synced_segments:
            with open(synced_meta_path, "w", encoding="utf-8") as f:
                json.dump(synced_segments, f, ensure_ascii=False, indent=2)
            logging.info(
                f"Job {job_id} chunk {chunk_index}: Overwrote segments_synced.json with corrected indices"
            )

        if synced_meta_path.exists():
            synced_meta_key = (
                f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_index}/"
                f"segments_synced.json"
            )
            upload_to_s3(output_bucket, synced_meta_key, synced_meta_path)

        # 9. 원본 transcript 복원
        if original_transcript_backup.exists():
            shutil.copy(original_transcript_backup, compact_transcript_path)

        # 10. 완료 체크 및 Mux 트리거
        check_and_trigger_mux_if_complete(
            job_id,
            project_id,
            total_segments,
            output_bucket,
            project_prefix,
            callback_url,
            job_details,
            send_callback_func=send_callback,
        )

        if callback_url:
            send_callback(
                callback_url,
                "in_progress",
                f"Chunk {chunk_index + 1}/{total_chunks} completed.",
                stage="chunk_work_completed",
                metadata={
                    "job_id": job_id,
                    "project_id": project_id,
                    "chunk_index": chunk_index,
                    "total_chunks": total_chunks,
                    "segments_processed": len(chunk_segments),
                    "target_lang": target_lang,
                },
            )

        logging.info(
            f"Job {job_id} chunk {chunk_index}: chunk_work completed successfully"
        )

        # 시간 측정 및 저장 (끝나기 직전)
        end_time = time.time()
        total_time = end_time - start_time

        segment_count = end_segment_index - start_segment_index + 1

        timing_data = {
            "job_id": job_id,
            "project_id": project_id,
            "chunk_index": chunk_index,
            "total_time_seconds": round(total_time, 3),
            "total_time_formatted": f"{int(total_time // 60)}m {int(total_time % 60)}s",
            "start_time": start_time,
            "end_time": end_time,
            "start_segment_index": start_segment_index,
            "end_segment_index": end_segment_index,
            "segment_count": segment_count,
            "total_segments": total_segments,
            "total_chunks": total_chunks,
        }

        # S3에 저장 (chunk01_work_time.json 형식)
        chunk_index_str = f"{chunk_index + 1:02d}"  # 01, 02, 03 형식
        if project_id:
            timing_key = f"projects/{project_id}/chunk{chunk_index_str}_work_time.json"
        else:
            timing_key = f"jobs/{job_id}/chunk{chunk_index_str}_work_time.json"

        upload_metadata_to_s3(output_bucket, timing_key, timing_data)
        logging.info(
            f"Job {job_id} chunk {chunk_index}: Saved timing data to s3://{output_bucket}/{timing_key}"
        )

    except Exception as e:
        logging.error(
            f"chunk_work failed for job {job_id} chunk {chunk_index}: {e}",
            exc_info=True,
        )
        if callback_url:
            send_callback(
                callback_url,
                "failed",
                f"Chunk {chunk_index + 1} failed: {str(e)}",
                stage="chunk_work_failed",
                metadata={
                    "job_id": job_id,
                    "project_id": project_id,
                    "chunk_index": chunk_index,
                    "target_lang": target_lang,
                },
            )
        raise
