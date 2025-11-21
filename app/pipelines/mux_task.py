"""Mux Task Pipeline - 청크 병합 및 최종 영상 생성"""

import json
import logging
import time
from pathlib import Path

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker import (
    AWS_S3_BUCKET,
    send_callback,
    download_from_s3,
    upload_to_s3,
    upload_metadata_to_s3,
    resolve_output_prefix,
    _segments_with_remote_audio_paths,
)

from services.mux import mux_audio_video
from configs import ensure_job_dirs


def handle_mux_task(job_details: dict):
    """모든 청크의 결과를 병합하여 최종 영상을 생성합니다."""
    # 시간 측정 시작
    start_time = time.time()

    job_id = job_details["job_id"]
    project_id = job_details.get("project_id")
    total_segments = job_details["total_segments"]
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET
    project_prefix = job_details.get("project_prefix") or (
        f"projects/{project_id}" if project_id else "jobs"
    )
    callback_url = job_details.get("callback_url")
    manifest_key = job_details.get("manifest_key")

    target_lang = job_details.get("target_lang", "en")
    source_lang = job_details.get("source_lang")

    logging.info(f"Job {job_id}: Starting mux task")

    if callback_url:
        send_callback(
            callback_url,
            "in_progress",
            "Starting mux...",
            stage="mux_started",
            metadata={
                "job_id": job_id,
                "project_id": project_id,
            },
        )

    # 1. 로컬 작업 디렉토리 설정
    paths = ensure_job_dirs(job_id)

    try:
        # 2. manifest.json 다운로드
        manifest_path = paths.interim_dir / "manifest.json"
        if manifest_key:
            if not download_from_s3(output_bucket, manifest_key, manifest_path):
                raise Exception(f"Failed to download manifest.json from {manifest_key}")
        else:
            # manifest_key가 없으면 기본 경로 시도
            default_manifest_key = f"{project_prefix}/interim/{job_id}/manifest.json"
            if not download_from_s3(output_bucket, default_manifest_key, manifest_path):
                raise Exception("manifest_key not provided and default path not found")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        total_chunks = manifest.get("total_chunks", 1)

        # 3. 모든 청크의 segments_synced.json 다운로드 및 병합
        all_synced_segments = []

        for chunk_idx in range(total_chunks):
            chunk_synced_key = (
                f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_idx}/"
                f"segments_synced.json"
            )

            chunk_synced_path = paths.interim_dir / f"chunk_{chunk_idx}_synced.json"

            if download_from_s3(output_bucket, chunk_synced_key, chunk_synced_path):
                with open(chunk_synced_path, "r", encoding="utf-8") as f:
                    chunk_segments = json.load(f)
                    all_synced_segments.extend(chunk_segments)
                logging.info(
                    f"Job {job_id}: Loaded {len(chunk_segments)} segments from chunk {chunk_idx}"
                )
            else:
                logging.warning(
                    f"Job {job_id}: Failed to download segments_synced.json "
                    f"for chunk {chunk_idx}, skipping"
                )

        if not all_synced_segments:
            # 청크별 segments_synced.json이 없으면 S3의 synced 디렉토리에서 직접 로드 시도
            logging.info(
                f"Job {job_id}: No chunk segments found, trying to load from synced directory"
            )
            # S3에서 직접 segments_synced.json 다운로드 시도
            synced_meta_key = f"{project_prefix}/interim/{job_id}/text/vid/tts/synced/segments_synced.json"
            synced_meta_path = paths.vid_tts_dir / "synced" / "segments_synced.json"
            synced_meta_path.parent.mkdir(parents=True, exist_ok=True)

            if download_from_s3(output_bucket, synced_meta_key, synced_meta_path):
                with open(synced_meta_path, "r", encoding="utf-8") as f:
                    all_synced_segments = json.load(f)
            else:
                raise Exception("No synced segments found in S3")

        # 세그먼트를 segment_index 또는 start 시간 기준으로 정렬
        # segment_id가 "segment_XXXX" 형식이면 인덱스 추출, 아니면 start 시간으로 정렬
        def get_sort_key(seg):
            seg_id = seg.get("segment_id", "")
            if isinstance(seg_id, str) and seg_id.startswith("segment_"):
                try:
                    idx_str = seg_id.replace("segment_", "")
                    return int(idx_str)
                except ValueError:
                    pass
            # segment_index가 있으면 사용
            if "segment_index" in seg:
                return seg.get("segment_index", 0)
            # 아니면 start 시간 사용
            return seg.get("start", 0)

        all_synced_segments.sort(key=get_sort_key)

        logging.info(
            f"Job {job_id}: Merged {len(all_synced_segments)} segments from {total_chunks} chunks"
        )

        # 통합된 segments_synced.json 저장
        synced_dir = paths.vid_tts_dir / "synced"
        synced_dir.mkdir(parents=True, exist_ok=True)
        merged_synced_path = synced_dir / "segments_synced.json"
        with open(merged_synced_path, "w", encoding="utf-8") as f:
            json.dump(all_synced_segments, f, ensure_ascii=False, indent=2)

        # 4. 원본 영상 다운로드
        input_key = job_details.get("input_key")
        input_bucket = job_details.get("input_bucket") or output_bucket

        if not input_key:
            # manifest에서 찾기
            input_key = manifest.get("input_key")
            if not input_key:
                raise Exception("input_key not found in job_details or manifest")

        source_video_path = paths.input_dir / Path(input_key).name
        if not download_from_s3(input_bucket, input_key, source_video_path):
            raise Exception(f"Failed to download source video from {input_key}")

        # 5. background.wav 다운로드 (manifest에서 경로 확인)
        background_key = None
        if "audio_files" in manifest:
            bgm_s3_path = manifest["audio_files"].get("background_wav")
            if bgm_s3_path:
                # s3:// 접두사 제거 (split_up에서 이미 제거했지만, 이전 데이터 호환성 유지)
                if bgm_s3_path.startswith("s3://"):
                    # s3://bucket/key 형식에서 key 부분만 추출
                    parts = bgm_s3_path.replace("s3://", "").split("/", 1)
                    if len(parts) > 1:
                        background_key = parts[1]  # key 부분만 사용
                    else:
                        background_key = parts[0]  # bucket만 있는 경우 (거의 없음)
                else:
                    background_key = bgm_s3_path  # 이미 key 형식

        if not background_key:
            # 기본 경로 시도
            background_key = f"{project_prefix}/interim/{job_id}/audio/background.wav"

        background_path = paths.vid_bgm_dir / "background.wav"
        background_path.parent.mkdir(parents=True, exist_ok=True)

        if not download_from_s3(output_bucket, background_key, background_path):
            logging.warning(
                f"Job {job_id}: Failed to download background.wav, "
                f"mux may fail if background is required"
            )

        # 6. synced 디렉토리의 .wav 파일들이 로컬에 있는지 확인
        # 없으면 S3에서 다운로드
        synced_wav_dir = synced_dir
        for seg in all_synced_segments:
            audio_file = seg.get("audio_file")
            if not audio_file:
                continue

            audio_path = Path(audio_file)
            # 절대 경로가 아니면 synced_dir 기준으로 해석
            if not audio_path.is_absolute():
                audio_path = synced_dir / audio_path.name

            # 파일이 없으면 S3에서 다운로드 시도
            if not audio_path.exists():
                # S3 경로 추정
                audio_s3_key = f"{project_prefix}/interim/{job_id}/text/vid/tts/synced/{audio_path.name}"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                download_from_s3(output_bucket, audio_s3_key, audio_path)

        # 7. Mux 수행
        logging.info(f"Job {job_id}: Running mux_audio_video...")
        mux_results = mux_audio_video(job_id, source_video_path)
        output_video_path = Path(mux_results["output_video"])
        final_audio_path = Path(mux_results["output_audio"])

        # 8. 최종 결과물 S3에 업로드
        output_prefix = resolve_output_prefix(
            project_id, job_id, job_details.get("output_prefix")
        )
        result_key = f"{output_prefix}/dubbed_video.mp4"
        if not upload_to_s3(output_bucket, result_key, output_video_path):
            raise Exception("Failed to upload final video to S3")

        # 9. 메타데이터 생성 및 업로드
        metadata_key = (
            job_details.get("metadata_key") or f"{output_prefix}/metadata/{job_id}.json"
        )

        # segments를 S3 경로로 변환
        metadata_segments = _segments_with_remote_audio_paths(
            all_synced_segments,
            project_prefix,
            job_id,
            paths,
        )

        # segment_index 기준으로 정렬 (번역과 매칭을 위해)
        def get_segment_index(seg):
            """세그먼트의 인덱스를 추출"""
            if "segment_index" in seg:
                return seg.get("segment_index", 0)
            elif "seg_idx" in seg:
                return int(seg.get("seg_idx", 0))
            elif "segment_id" in seg:
                seg_id = seg.get("segment_id", "")
                # "segment_XXXX" 형식에서 인덱스 추출 시도
                try:
                    if isinstance(seg_id, str) and seg_id.startswith("segment_"):
                        return int(seg_id.replace("segment_", ""))
                except (ValueError, TypeError):
                    pass
            return 0

        metadata_segments.sort(key=get_segment_index)

        # 번역 결과 병합 (모든 청크의 번역 결과)
        all_translations = []
        for chunk_idx in range(total_chunks):
            chunk_trans_key = (
                f"{project_prefix}/interim/{job_id}/chunks/chunk_{chunk_idx}/"
                f"translated.json"
            )
            chunk_trans_path = paths.interim_dir / f"chunk_{chunk_idx}_translated.json"

            if download_from_s3(output_bucket, chunk_trans_key, chunk_trans_path):
                with open(chunk_trans_path, "r", encoding="utf-8") as f:
                    chunk_trans = json.load(f)
                    all_translations.extend(chunk_trans)

        # translations를 seg_idx로 딕셔너리로 매핑
        translations_map = {}
        for trans in all_translations:
            seg_idx = trans.get("seg_idx", 0)
            translations_map[seg_idx] = trans

        # segments의 순서에 맞춰 translations 재정렬
        # segments의 segment_index (또는 seg_idx)를 기준으로 translations 매핑
        sorted_translations = []
        for seg in metadata_segments:
            seg_idx = get_segment_index(seg)
            if seg_idx in translations_map:
                sorted_translations.append(translations_map[seg_idx])
            else:
                # 매칭되는 번역이 없으면 빈 번역 추가 (segments와 translations 순서 일치 보장)
                logging.warning(
                    f"Job {job_id}: No translation found for segment_index {seg_idx}"
                )
                sorted_translations.append({"seg_idx": seg_idx, "translation": ""})

        all_translations = sorted_translations

        metadata_payload = {
            "job_id": job_id,
            "project_id": project_id,
            "target_lang": target_lang,
            "source_lang": source_lang,
            "input_bucket": input_bucket,
            "input_key": input_key,
            "result_bucket": output_bucket,
            "result_key": result_key,
            "metadata_key": metadata_key,
            "segments": metadata_segments,
            "segment_count": len(metadata_segments),
            "translations": all_translations,
        }

        if final_audio_path:
            metadata_payload["audio_artifact"] = str(final_audio_path)

        if not upload_metadata_to_s3(output_bucket, metadata_key, metadata_payload):
            raise Exception("Failed to upload metadata to S3")

        # 9.5. speaker_refs 읽기 (S3에서)
        speaker_refs_metadata = None
        refs_json_key = f"{project_prefix}/interim/{job_id}/tts/speaker_refs.json"
        refs_json_path = paths.interim_dir / "speaker_refs.json"
        if download_from_s3(output_bucket, refs_json_key, refs_json_path):
            try:
                with open(refs_json_path, "r", encoding="utf-8") as f:
                    speaker_refs_mapping = json.load(f)

                # _build_speaker_refs_metadata와 동일한 형식으로 변환
                speaker_refs_metadata = {}
                for speaker, ref_data in speaker_refs_mapping.items():
                    if isinstance(ref_data, dict):
                        audio_path = ref_data.get("audio", "")
                        prompt_text = ref_data.get("text", "")

                        # S3 경로로 변환
                        if audio_path:
                            if not Path(audio_path).is_absolute():
                                # 상대 경로인 경우 S3 키 생성
                                ref_s3_key = (
                                    f"{project_prefix}/interim/{job_id}/text/vid/tts/self_refs/"
                                    f"{Path(audio_path).name}"
                                )
                            else:
                                # 절대 경로인 경우 relative_path 계산 시도
                                try:
                                    relative_path = Path(audio_path).relative_to(
                                        paths.interim_dir
                                    )
                                    ref_s3_key = f"{project_prefix}/interim/{job_id}/{relative_path}"
                                except ValueError:
                                    ref_s3_key = (
                                        f"{project_prefix}/interim/{job_id}/text/vid/tts/self_refs/"
                                        f"{Path(audio_path).name}"
                                    )

                            speaker_refs_metadata[speaker] = {
                                "ref_wav_key": f"s3://{output_bucket}/{ref_s3_key}",
                                "prompt_text": prompt_text,
                            }
            except Exception as e:
                logging.warning(
                    f"Job {job_id}: Failed to load speaker_refs from S3: {e}",
                    exc_info=True,
                )

        # 10. 최종 콜백
        final_metadata = {
            "job_id": job_id,
            "project_id": project_id,
            "result_bucket": output_bucket,
            "result_key": result_key,
            "metadata_key": metadata_key,
            "segment_count": len(metadata_segments),
            "target_lang": target_lang,
            "translations": all_translations,
        }
        if source_lang:
            final_metadata["source_lang"] = source_lang
        if speaker_refs_metadata:
            final_metadata["speaker_refs"] = speaker_refs_metadata

        if callback_url:
            send_callback(
                callback_url,
                "done",
                "Pipeline completed successfully.",
                stage="done",
                metadata=final_metadata,
            )

        logging.info(f"Job {job_id}: Mux completed successfully")

        # 시간 측정 및 저장 (끝나기 직전)
        end_time = time.time()
        total_time = end_time - start_time

        # metadata_segments와 total_chunks는 이미 정의되어 있음
        timing_data = {
            "job_id": job_id,
            "project_id": project_id,
            "total_time_seconds": round(total_time, 3),
            "total_time_formatted": f"{int(total_time // 60)}m {int(total_time % 60)}s",
            "start_time": start_time,
            "end_time": end_time,
            "total_segments": len(metadata_segments),
            "total_chunks": total_chunks,
        }

        # S3에 저장
        if project_id:
            timing_key = f"projects/{project_id}/mux_task_time.json"
        else:
            timing_key = f"jobs/{job_id}/mux_task_time.json"

        upload_metadata_to_s3(output_bucket, timing_key, timing_data)
        logging.info(
            f"Job {job_id}: Saved timing data to s3://{output_bucket}/{timing_key}"
        )

    except Exception as e:
        logging.error(f"Mux failed for job {job_id}: {e}", exc_info=True)
        if callback_url:
            send_callback(
                callback_url,
                "failed",
                str(e),
                stage="mux_failed",
                metadata={"job_id": job_id, "project_id": project_id},
            )
        raise
