"""TTS Segments Pipeline - 세그먼트 TTS 처리"""

import logging
from pathlib import Path

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worker import (
    AWS_S3_BUCKET,
    send_callback,
    download_from_s3,
    upload_to_s3,
    _resolve_s3_location,
    _sync_segment_to_range,
)

from services.tts import (
    _transcribe_prompt_text,
    _trim_tts_artifacts,
    _synthesize_with_cosyvoice2,
)
from configs import ensure_job_dirs
from pydub import AudioSegment


def handle_tts_segments(job_details: dict) -> None:
    """segment_tts / tts 작업을 처리합니다."""
    callback_url = job_details.get("callback_url")
    segments_req = job_details.get("segments") or []

    if not callback_url:
        raise ValueError("segment_tts requires callback_url.")
    if not segments_req:
        raise ValueError("segment_tts requires at least one segment entry.")

    project_id = job_details.get("project_id")
    target_lang = job_details.get("target_lang", "ko")
    mod_raw = (job_details.get("mod") or "dynamic").strip().lower()
    mod = mod_raw if mod_raw in {"fixed", "dynamic"} else "dynamic"
    output_bucket = job_details.get("output_bucket") or AWS_S3_BUCKET

    # original_job_id가 있으면 사용 (full_pipeline job_id), 없으면 현재 job_id 사용
    original_job_id = job_details.get("original_job_id") or job_details.get("job_id")
    if not original_job_id:
        raise ValueError("segment_tts requires either original_job_id or job_id.")

    # 파일명에 사용할 job_id (현재 job_id)
    job_id = job_details.get("job_id")
    if not job_id:
        raise ValueError("segment_tts requires job_id.")

    project_prefix = f"projects/{project_id}" if project_id else "jobs"
    remote_interim_prefix = f"{project_prefix}/interim/{original_job_id}"

    send_callback(
        callback_url,
        "in_progress",
        f"Starting segment TTS for job {original_job_id}",
        stage="segment_tts_started",
        metadata={"job_id": original_job_id, "project_id": project_id, "mod": mod},
    )

    try:
        paths = ensure_job_dirs(original_job_id)
        resynth_dir = paths.vid_tts_dir / "resynth"
        resynth_dir.mkdir(parents=True, exist_ok=True)

        # --- 보이스 샘플 준비 ---
        speaker_spec = job_details.get("speaker_voices") or {}
        voice_key = speaker_spec.get("key")
        if not voice_key:
            raise ValueError("speaker_voices.key is required for segment_tts.")
        speaker_bucket = (
            speaker_spec.get("bucket")
            or job_details.get("input_bucket")
            or output_bucket
        )

        speaker_asset_dir = paths.interim_dir / "speaker_assets"
        speaker_asset_dir.mkdir(parents=True, exist_ok=True)

        voice_bucket, resolved_voice_key = _resolve_s3_location(
            voice_key, speaker_bucket
        )
        sample_path = speaker_asset_dir / Path(resolved_voice_key).name
        if not download_from_s3(voice_bucket, resolved_voice_key, sample_path):
            raise RuntimeError(
                f"Failed to download speaker sample from {resolved_voice_key}"
            )

        # CosyVoice는 30초를 초과하는 오디오에서 speech token을 추출할 수 없으므로 30초로 자름
        try:
            audio = AudioSegment.from_file(str(sample_path))
            max_duration_ms = 30 * 1000  # 30초 = 30000ms

            if len(audio) > max_duration_ms:
                logging.info(
                    f"Voice sample is {len(audio)/1000:.2f}s, trimming to 30s for CosyVoice compatibility"
                )
                audio = audio[:max_duration_ms]
                audio.export(str(sample_path), format="wav")
                logging.info("Voice sample trimmed to 30 seconds")
            else:
                logging.debug(
                    f"Voice sample is {len(audio)/1000:.2f}s, no trimming needed"
                )
        except Exception as e:
            logging.warning(
                f"Failed to check/trim voice sample duration: {e}, continuing with original file"
            )

        # --- 텍스트 프롬프트 준비 ---
        prompt_text = (speaker_spec.get("text_prompt_value") or "").strip()
        text_prompt_key = speaker_spec.get("text_prompt")
        if not prompt_text and text_prompt_key:
            prompt_bucket = (
                speaker_spec.get("text_prompt_bucket")
                or speaker_spec.get("bucket")
                or job_details.get("input_bucket")
                or output_bucket
            )
            prompt_bucket, prompt_key = _resolve_s3_location(
                text_prompt_key, prompt_bucket
            )
            prompt_path = speaker_asset_dir / Path(prompt_key).name
            if not download_from_s3(prompt_bucket, prompt_key, prompt_path):
                raise RuntimeError(
                    f"Failed to download speaker text prompt from {prompt_key}"
                )
            prompt_text = prompt_path.read_text(encoding="utf-8").strip()

        if not prompt_text:
            fallback_prompt = (job_details.get("prompt_text") or "").strip()
            prompt_text = fallback_prompt or _transcribe_prompt_text(sample_path)

        if not prompt_text:
            raise ValueError("Unable to resolve prompt text for TTS segments.")

        # --- 세그먼트별 TTS ---
        results: list[dict] = []

        def _to_seconds(value) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        for idx, seg_req in enumerate(segments_req):
            seg_id = seg_req.get("segment_id")
            text = (seg_req.get("text") or "").strip()
            if not text:
                raise ValueError("Segment is missing 'text'.")

            s_val = seg_req.get("s", seg_req.get("start"))
            e_val = seg_req.get("e", seg_req.get("end"))

            s_sec = _to_seconds(s_val) or 0.0
            e_sec = _to_seconds(e_val)

            s_ms = max(0, int(s_sec * 1000))
            e_ms = int(e_sec * 1000) if e_sec is not None else None

            # 각 세그먼트별로 고유한 파일명 사용
            local_tts = resynth_dir / f"{job_id}_{idx}.wav"
            synced_tts = resynth_dir / f"{job_id}_{idx}_synced.wav"

            # 1) CosyVoice2로 합성
            _synthesize_with_cosyvoice2(
                text=text,
                prompt_text=prompt_text,
                sample_path=sample_path,
                output_path=local_tts,
            )

            # 2) full_pipeline과 동일하게 TTS 아티팩트 / 침묵 트리밍
            _trim_tts_artifacts(local_tts)

            # 3) fixed 모드면 원래 [s, e] 길이에 맞춰 sync
            synced_path = local_tts
            if mod == "fixed":
                if e_ms is None or e_ms <= s_ms:
                    raise ValueError("Segment missing valid s/e for fixed mode.")
                target_duration_ms = max(1, e_ms - s_ms)
                synced_path = _sync_segment_to_range(
                    local_tts,
                    target_duration_ms,
                    synced_tts,
                )

            # 4) S3 업로드
            try:
                relative = synced_path.relative_to(paths.interim_dir)
            except ValueError:
                raise RuntimeError(
                    f"TTS artifact {synced_path} is outside interim dir"
                ) from None

            s3_key = f"{remote_interim_prefix}/{relative.as_posix()}"
            if not upload_to_s3(output_bucket, s3_key, synced_path):
                raise RuntimeError("Failed to upload segment to S3.")

            results.append(
                {
                    "text": text,
                    "s": s_sec,
                    "e": e_sec,
                    "audio_key": s3_key,
                    "bucket": output_bucket,
                    "mod": mod,
                    **({"segment_id": str(seg_id)} if seg_id else {}),
                }
            )

        # segment_id를 metadata에 포함 (백엔드에서 콜백 처리 시 사용)
        callback_metadata = {
            "job_id": original_job_id,
            "project_id": project_id,
            "target_lang": target_lang,
            "mod": mod,
            "segments": results,
        }
        # job_details에서 segment_id가 있으면 metadata에 포함
        segment_id = job_details.get("segment_id")
        if segment_id:
            callback_metadata["segment_id"] = segment_id

        send_callback(
            callback_url,
            "done",
            "Segment TTS completed.",
            stage="segment_tts_completed",
            metadata=callback_metadata,
        )
    except Exception as exc:
        logging.error(
            "segment_tts failed for job %s: %s", original_job_id, exc, exc_info=True
        )
        send_callback(
            callback_url,
            "failed",
            f"Segment TTS failed: {exc}",
            stage="segment_tts_failed",
            metadata={"job_id": original_job_id, "project_id": project_id, "mod": mod},
        )
        raise
