# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
import uuid
import os
import logging
from pathlib import Path
from typing import Any

import tempfile  # ⬅️ 추가
import subprocess  # ⬅️ 추가
from io import BytesIO  # ⬅️ 추가
import shutil

# 파이프라인 각 단계를 담당하는 함수 불러오기
from services.stt import run_asr
from services.demucs_split import split_vocals
from services.translate import translate_transcript
from services.tts import (
    generate_tts,
    _strip_background_from_sample,
    _transcribe_prompt_text,
    _synthesize_with_cosyvoice2,
)
from services.mux import mux_audio_video
from services.sync import sync_segments
from services.lang import normalize_lang_code
from services.speaker_embeddings import save_audio_embedding, load_embedding_index
from services.voice_recommendation import (
    load_voice_library,
    recommend_voice_replacements,
    update_voice_library_entry,
)
from configs import ensure_data_dirs, ensure_job_dirs
from pydub import AudioSegment

for name in [
    "numba",
    "numba.core",
    "numba.core.ssa",
    "numba.core.byteflow",
    "numba.core.typeinfer",
]:
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)  # or logging.ERROR
    logger.propagate = False  # 부모(root)로 안 올리게
    logger.handlers.clear()  # 혹시 자기 handler 갖고 있으면 날려버리기


# 문서화를 위한 요청/응답 모델 정의
class ASRResponse(BaseModel):
    job_id: str
    segments: list


class TranslateRequest(BaseModel):
    job_id: str
    target_lang: str
    src_lang: str | None = None


app = FastAPI(
    docs_url="/",
    title="Video Dubbing API",
    description="엔드 투 엔드 비디오 더빙 파이프라인 API",
)

# 기본 작업 폴더가 없으면 생성
ensure_data_dirs()

VOICE_SAMPLES_ROOT = Path(os.getenv("VOICE_SAMPLES_ROOT", "/data/voice-samples"))
VOICE_SAMPLES_SAMPLES_DIR = VOICE_SAMPLES_ROOT / "samples"
VOICE_SAMPLES_TTS_DIR = VOICE_SAMPLES_ROOT / "tts"
VOICE_SAMPLES_EMBED_DIR = VOICE_SAMPLES_ROOT / "embedding"
for directory in (
    VOICE_SAMPLES_ROOT,
    VOICE_SAMPLES_SAMPLES_DIR,
    VOICE_SAMPLES_TTS_DIR,
    VOICE_SAMPLES_EMBED_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


def _normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _strip_voice_samples_prefix(sample_key: str) -> str:
    key = sample_key
    if key.startswith("s3://"):
        remainder = key.split("://", 1)[1]
        if "/" in remainder:
            key = remainder.split("/", 1)[1]
        else:
            key = ""
    marker = "voice-samples/"
    if marker in key:
        key = key.split(marker, 1)[1]
    return key.lstrip("/")


def _resolve_local_voice_sample(sample_key: str | None) -> Path | None:
    if not sample_key:
        return None
    relative = _strip_voice_samples_prefix(sample_key)
    if not relative:
        return None
    return (VOICE_SAMPLES_ROOT / relative).resolve()


def _prepare_voice_replacements_local(paths, target_lang: str):
    diagnostics: dict[str, Any] = {
        "enabled": False,
        "target_lang": target_lang,
    }
    overrides: dict[str, dict] = {}
    index_path = paths.vid_tts_dir / "speaker_embeddings" / "speaker_embeddings.json"
    embeddings = load_embedding_index(index_path)
    if not embeddings:
        diagnostics["reason"] = "missing_embeddings"
        return overrides, diagnostics
    library = load_voice_library(target_lang, VOICE_SAMPLES_EMBED_DIR)
    if not library:
        diagnostics["reason"] = "library_unavailable"
        return overrides, diagnostics
    replacements = recommend_voice_replacements(
        embeddings,
        library,
        target_lang=target_lang,
    )
    if not replacements:
        diagnostics["reason"] = "no_matches"
        return overrides, diagnostics
    matches_summary: dict[str, dict] = {}
    for speaker, plan in replacements.items():
        sample_path = _resolve_local_voice_sample(plan.entry.sample_key)
        if not sample_path or not sample_path.is_file():
            continue
        overrides[speaker] = {
            "audio_path": str(sample_path),
            "prompt_text": plan.entry.prompt_text,
            "voice_id": plan.entry.voice_id,
            "similarity": plan.similarity,
            "sample_key": plan.entry.sample_key,
            "sample_bucket": plan.entry.sample_bucket,
        }
        matches_summary[speaker] = plan.summary()
    if not overrides:
        diagnostics["reason"] = "materialization_failed"
        return overrides, diagnostics
    diagnostics["enabled"] = True
    diagnostics["reason"] = "ok"
    diagnostics["matches"] = matches_summary
    diagnostics["prepared_speakers"] = sorted(overrides.keys())
    return overrides, diagnostics


@app.post("/asr", response_model=ASRResponse)
async def asr_endpoint(
    job_id: str = Form(None),
    file: UploadFile = File(None),
    src_lang: str | None = Form(None),
    speaker_count: int | None = Form(None, ge=1),
):
    """
    새 영상을 업로드하거나 기존 job_id를 지정해 WhisperX로 음성을 추출합니다.
    - 선택적으로 `src_lang`(예: 'ko', 'en')을 지정하면 해당 언어로 고정해 인식합니다.
      지정하지 않거나 'auto'이면 WhisperX가 자동으로 언어를 추론합니다.
    - `speaker_count`에 1 이상의 정수를 지정하면 pyannote diarization이 해당 화자 수를 기준으로 동작합니다.
    job_id와 화자 정보가 포함된 전사 구간 목록을 반환합니다.
    """
    if file:
        job_id = job_id or str(uuid.uuid4())
        paths = ensure_job_dirs(job_id)
        input_path = paths.input_dir / "source.mp4"
        with open(input_path, "wb") as f:
            f.write(await file.read())
    else:
        if job_id is None:
            return JSONResponse(status_code=400, content={"error": "No media provided"})
        paths = ensure_job_dirs(job_id)
        input_path = paths.input_dir / "source.mp4"
        if not input_path.is_file():
            return JSONResponse(
                status_code=404,
                content={"error": f"Input for job {job_id} not found"},
            )

    try:
        # src_lang가 제공되면 WhisperX 자동 언어 추론을 비활성화하고 해당 언어로 고정
        # 예: 'ko', 'en', 'ja' 등 ISO 언어 코드. 'auto' 또는 빈 값이면 자동 추론 유지
        segments = run_asr(
            job_id,
            source_lang=src_lang,
            speaker_count=speaker_count,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"job_id": job_id, "segments": segments}


@app.post("/translate")
async def translate_endpoint(request: TranslateRequest):
    """
    지정된 job_id의 전사 텍스트를 target_lang으로 번역합니다.
    """
    job_id = request.job_id
    target_lang = request.target_lang
    src_lang = request.src_lang
    try:
        segments = translate_transcript(job_id, target_lang, src_lang=src_lang)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {
        "job_id": job_id,
        "target_lang": target_lang,
        "src_lang": src_lang,
        "translated_segments": segments,
    }


@app.post("/tts")
async def tts_endpoint(
    job_id: str = Form(...),
    target_lang: str = Form(...),
    voice_sample: UploadFile | str | None = File(None),
    prompt_text: str | None = Form(None),
):
    """
    지정된 job_id에 대해 각 구간의 번역된 음성을 합성합니다.
    """
    paths = ensure_job_dirs(job_id)
    user_voice_sample_path: Path | None = None
    # multipart 필드가 빈 문자열로 들어오면 str("")이 되므로 None 취급
    if isinstance(voice_sample, str):
        voice_upload: UploadFile | None = None
    else:
        voice_upload = voice_sample

    # voice_sample이 실제 파일인지 확인 (빈 문자열이 아닌지)
    if voice_upload and voice_upload.filename:
        suffix = Path(voice_upload.filename).suffix.lower()
        if suffix != ".wav":
            return JSONResponse(
                status_code=400,
                content={"error": "voice_sample must be a .wav file."},
            )
        custom_ref_dir = paths.interim_dir / "tts_custom_refs"
        custom_ref_dir.mkdir(parents=True, exist_ok=True)
        user_voice_sample_path = custom_ref_dir / f"user_voice_sample{suffix}"
        data = await voice_upload.read()
        with open(user_voice_sample_path, "wb") as f:
            f.write(data)
    else:
        user_voice_sample_path = None

    prompt_text_value = prompt_text.strip() if prompt_text else None
    # 번역이 없다면 자동으로 /translate 단계를 수행해 생성
    translated_path = paths.trg_sentence_dir / "translated.json"
    if not translated_path.is_file():
        try:
            translate_transcript(job_id, target_lang)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Translation failed before TTS: {str(e)}"},
            )
    try:
        segments = generate_tts(
            job_id,
            target_lang,
            voice_sample_path=user_voice_sample_path,
            prompt_text_override=prompt_text_value,
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"job_id": job_id, "audio_segments": segments}


from fastapi import UploadFile, File, Form
from pathlib import Path
import uuid
import logging


@app.post("/pipeline")
async def pipeline_endpoint(
    file: UploadFile = File(...),
    # ⬇️ 여기: str도 허용하도록 수정
    voice_sample: UploadFile | str | None = File(None),
    job_id: str | None = Form(None),
    target_lang: str = Form(...),
    src_lang: str | None = Form(None),
    speaker_count: int | None = Form(None, ge=1),
    prompt_text: str | None = Form(None),
    replace_voice_samples: bool | str = Form(False),
):
    """
    단일 요청으로 전체 파이프라인(ASR → 번역 → TTS → Sync → Mux)을 실행합니다.
    """
    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "Input video file is required."},
        )

    job_id = job_id or str(uuid.uuid4())
    paths = ensure_job_dirs(job_id)

    video_name = Path(file.filename).name or "source.mp4"
    if not Path(video_name).suffix:
        video_name = f"{video_name}.mp4"
    source_video_path = paths.input_dir / video_name
    source_video_path.parent.mkdir(parents=True, exist_ok=True)
    media_bytes = await file.read()
    with open(source_video_path, "wb") as f:
        f.write(media_bytes)

    # === 여기부터 voice_sample 정규화 ===
    user_voice_sample_path: Path | None = None

    # 1) curl -F 'voice_sample=' 같이 들어오면: str("") 로 들어옴 → 그냥 None 취급
    if isinstance(voice_sample, str):
        voice_upload: UploadFile | None = None
    else:
        voice_upload = voice_sample

    # 2) 진짜 업로드된 파일인 경우에만 저장
    if voice_upload and voice_upload.filename:
        suffix = Path(voice_upload.filename).suffix.lower()
        if suffix != ".wav":
            return JSONResponse(
                status_code=400,
                content={"error": "voice_sample must be a .wav file."},
            )
        custom_ref_dir = paths.interim_dir / "tts_custom_refs"
        custom_ref_dir.mkdir(parents=True, exist_ok=True)
        user_voice_sample_path = custom_ref_dir / f"user_voice_sample{suffix}"
        sample_bytes = await voice_upload.read()
        with open(user_voice_sample_path, "wb") as f:
            f.write(sample_bytes)
    else:
        user_voice_sample_path = None

    prompt_text_value = prompt_text.strip() if prompt_text else None
    replace_voice_flag = _normalize_bool(replace_voice_samples)

    stage = "asr"
    translations: list[dict] = []
    segments_payload: list[dict] = []
    sync_applied = False
    voice_replacement_meta = {
        "requested": replace_voice_flag,
        "enabled": False,
        "target_lang": target_lang,
    }
    speaker_voice_overrides: dict[str, dict] = {}
    try:
        run_asr(
            job_id,
            source_video_path,
            source_lang=src_lang,
            speaker_count=speaker_count,
        )
        stage = "translate"
        translations = translate_transcript(job_id, target_lang, src_lang=src_lang)
        if replace_voice_flag:
            overrides, diagnostics = _prepare_voice_replacements_local(
                paths, target_lang
            )
            speaker_voice_overrides = overrides
            voice_replacement_meta.update(diagnostics)
        else:
            voice_replacement_meta.setdefault("reason", "not_requested")
        stage = "tts"
        segments_payload = generate_tts(
            job_id,
            target_lang,
            voice_sample_path=user_voice_sample_path,
            prompt_text_override=prompt_text_value,
            speaker_voice_overrides=(
                speaker_voice_overrides if speaker_voice_overrides else None
            ),
        )
        stage = "sync"
        try:
            synced_segments = sync_segments(job_id)
        except FileNotFoundError:
            synced_segments = []
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("Sync step failed for job %s: %s", job_id, exc)
            synced_segments = []
        else:
            if synced_segments:
                segments_payload = synced_segments
                sync_applied = True
        stage = "mux"
        mux_results = mux_audio_video(job_id, source_video_path)
    except Exception as exc:
        logging.exception("Pipeline failed at stage %s for job %s", stage, job_id)
        return JSONResponse(
            status_code=500,
            content={"error": f"{stage} failed: {exc}"},
        )

    return {
        "job_id": job_id,
        "source_lang": src_lang,
        "target_lang": target_lang,
        "translations": translations,
        "segments": segments_payload,
        "sync_applied": sync_applied,
        "output_video": mux_results["output_video"],
        "output_audio": mux_results["output_audio"],
        "voice_replacement": voice_replacement_meta,
    }


@app.post("/voice_samples/test")
async def voice_sample_test_endpoint(
    file: UploadFile = File(...),
    text: str = Form(...),
    sample_lang: str | None = Form(None),
    voice_sample_id: str | None = Form(None),
):
    """
    Upload a raw voice sample, extract a clean prompt, register it in the local
    voice-samples directory structure, and synthesize a short TTS preview.
    """
    if not file.filename:
        return JSONResponse(
            status_code=400, content={"error": "Voice sample file is required."}
        )
    job_id = str(uuid.uuid4())
    paths = ensure_job_dirs(job_id)
    local_voice_sample = paths.input_dir / "voice_sample.wav"
    local_voice_sample.parent.mkdir(parents=True, exist_ok=True)
    sample_bytes = await file.read()
    with open(local_voice_sample, "wb") as f:
        f.write(sample_bytes)

    # Trim to 30s max for stability
    try:
        audio = AudioSegment.from_file(str(local_voice_sample))
        max_duration_ms = 30 * 1000
        if len(audio) > max_duration_ms:
            audio[:max_duration_ms].export(str(local_voice_sample), format="wav")
    except Exception as exc:  # pragma: no cover - best-effort
        logging.warning("Failed to trim voice sample: %s", exc)

    audio_path_for_demucs = paths.vid_speaks_dir / "audio.wav"
    audio_path_for_demucs.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(local_voice_sample, audio_path_for_demucs)

    demucs_result = split_vocals(job_id)
    vocals_path = Path(demucs_result["vocals"])

    lang_code = normalize_lang_code(sample_lang) or "misc"
    voice_id = voice_sample_id or f"voice_{uuid.uuid4().hex[:10]}"

    sample_dir = VOICE_SAMPLES_SAMPLES_DIR / lang_code
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_output_path = sample_dir / f"{voice_id}.wav"
    shutil.copy(vocals_path, sample_output_path)

    prompt_text = _transcribe_prompt_text(sample_output_path) or ""

    embedding_tmp = paths.outputs_dir / "voice_sample_embedding.json"
    embedding_payload = save_audio_embedding(
        sample_output_path,
        embedding_tmp,
        label=voice_id,
        base_dir=VOICE_SAMPLES_ROOT,
        meta={"source": "voice_sample_test", "job_id": job_id},
    )
    library_entry = {
        "voice_id": voice_id,
        "sample_key": f"voice-samples/samples/{lang_code}/{voice_id}.wav",
        "embedding": embedding_payload.get("embedding", []),
        "prompt_text": prompt_text,
    }
    update_voice_library_entry(
        lang_code, library_entry, base_dir=VOICE_SAMPLES_EMBED_DIR
    )

    tts_output_path = VOICE_SAMPLES_TTS_DIR / f"{voice_id}.wav"
    tts_output_path.parent.mkdir(parents=True, exist_ok=True)
    _synthesize_with_cosyvoice2(
        text=text,
        prompt_text=prompt_text or text,
        sample_path=sample_output_path,
        output_path=tts_output_path,
    )

    return {
        "job_id": job_id,
        "voice_id": voice_id,
        "sample_lang": lang_code,
        "sample_path": str(sample_output_path),
        "tts_preview_path": str(tts_output_path),
        "prompt_text": prompt_text,
        "library_entry": library_entry,
    }


@app.post("/tts/test")
async def tts_test_endpoint(
    text: str = Form(...),
    voice_sample: UploadFile = File(...),
    src_lang: str | None = Form(None),
):
    """Upload a short reference clip and synthesize TTS immediately."""
    text_value = (text or "").strip()
    if not text_value:
        return JSONResponse(
            status_code=400,
            content={"error": "text is required"},
        )
    if not voice_sample.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "voice_sample file is required"},
        )

    job_id = str(uuid.uuid4())
    paths = ensure_job_dirs(job_id)
    work_dir = paths.interim_dir / "tts_test"
    work_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(voice_sample.filename).suffix or ".wav"
    raw_sample_path = work_dir / f"uploaded_sample{suffix}"
    sample_bytes = await voice_sample.read()
    with open(raw_sample_path, "wb") as f:
        f.write(sample_bytes)

    normalized_sample_path = work_dir / "tts_test_input.wav"
    try:
        audio = AudioSegment.from_file(str(raw_sample_path))
        max_duration_ms = 30 * 1000
        if len(audio) > max_duration_ms:
            audio = audio[:max_duration_ms]
        audio.set_frame_rate(16000).set_channels(1).export(
            str(normalized_sample_path),
            format="wav",
        )
    except Exception as exc:
        logging.warning("Failed to normalize uploaded voice sample: %s", exc)
        shutil.copy(raw_sample_path, normalized_sample_path)

    cleaned_sample_path = _strip_background_from_sample(
        normalized_sample_path, work_dir
    )

    final_sample_path = work_dir / "tts_test_voice.wav"
    try:
        cleaned_audio = AudioSegment.from_file(str(cleaned_sample_path))
        cleaned_audio.set_frame_rate(16000).set_channels(1).export(
            str(final_sample_path),
            format="wav",
        )
    except Exception as exc:
        logging.warning("Failed to convert cleaned sample: %s", exc)
        shutil.copy(cleaned_sample_path, final_sample_path)

    lang_hint = (src_lang or "").strip()
    prompt_value = (
        _transcribe_prompt_text(final_sample_path, language=lang_hint or None)
        or text_value
    )

    output_path = paths.outputs_dir / "tts_test.wav"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        _synthesize_with_cosyvoice2(
            text_value, prompt_value, final_sample_path, output_path
        )
    except Exception as exc:
        logging.exception("TTS test synthesis failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"TTS synthesis failed: {exc}"},
        )

    download_name = f"tts_test_{job_id}.wav"
    return FileResponse(
        output_path,
        media_type="audio/wav",
        filename=download_name,
        headers={"X-Job-Id": job_id},
    )


@app.post("/sync")
async def sync_endpoint(job_id: str = Form(...)):
    """
    TTS로 생성된 각 구간 오디오를 원본 화자 길이에 맞춰 동기화합니다.
    """
    try:
        segments = sync_segments(job_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    return {"job_id": job_id, "synced_segments": segments}


@app.post("/mux")
async def mux_endpoint(job_id: str):
    """
    합성된 음성과 배경음을 섞어 원본 영상과 결합해 더빙 영상을 생성합니다.
    최종 mp4 파일을 반환합니다.
    """
    try:
        paths = mux_audio_video(job_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    output_video = paths["output_video"]
    if not os.path.isfile(output_video):
        return JSONResponse(
            status_code=500, content={"error": "Muxing failed, output video not found"}
        )
    # 생성된 비디오 파일을 바로 다운로드할 수 있도록 응답으로 반환
    return FileResponse(
        output_video, media_type="video/mp4", filename=f"dubbed_{job_id}.mp4"
    )


@app.post("/demucs_execute")
async def demucs_execute_endpoint(
    file: UploadFile = File(...),
    stem: str = Form("vocals"),  # "vocals", "accompaniment" 등 demucs stem 이름
):
    """
    파일을 업로드하면 demucs로 분리해서,
    선택한 stem(기본: vocals) 하나를 바로 다운로드로 반환합니다.

    - job_id / 내부 데이터 디렉터리 전혀 사용 안 함
    - 임시 디렉터리에만 저장 후, 메모리로 읽어서 StreamingResponse로 내려보냄
    """
    if not file.filename:
        return JSONResponse(
            status_code=400,
            content={"error": "Input audio/video file is required."},
        )

    stem = stem.lower()

    # demucs 기본 stem 후보들 (환경에 따라 조정 가능)
    valid_stems = {"vocals", "accompaniment", "bass", "drums", "other"}
    if stem not in valid_stems:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"stem must be one of {sorted(valid_stems)}",
            },
        )

    # 1) 임시 디렉터리 안에서만 작업
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # 업로드 파일 임시 저장
            input_ext = Path(file.filename).suffix or ".wav"
            input_path = tmpdir_path / f"input{input_ext}"
            data = await file.read()
            with open(input_path, "wb") as f:
                f.write(data)

            # demucs 출력 디렉터리 지정
            output_dir = tmpdir_path / "separated"

            # demucs CLI 실행 (이미 컨테이너에 설치되어 있다고 가정)
            # 필요하면 -n 모델명 등 옵션 추가해서 써도 됨.
            cmd = [
                "demucs",
                "-o",
                str(output_dir),
                str(input_path),
            ]

            try:
                subprocess.run(cmd, check=True, cwd=tmpdir)
            except subprocess.CalledProcessError as e:
                logging.exception("demucs failed")
                return JSONResponse(
                    status_code=500, content={"error": f"demucs failed: {e}"}
                )

            # demucs 출력 구조:
            # output_dir / MODEL_NAME / BASENAME / "<stem>.wav"
            candidates = list(output_dir.rglob(f"{stem}.wav"))
            if not candidates:
                return JSONResponse(
                    status_code=500,
                    content={
                        "error": f"Could not find separated stem '{stem}.wav' in demucs output."
                    },
                )

            stem_path = candidates[0]

            # 파일 내용을 메모리로 읽어온 다음, 임시 디렉터리 삭제
            with open(stem_path, "rb") as f:
                audio_bytes = f.read()

        # 2) 임시 디렉터리 밖(이미 삭제된 후)에서 메모리 데이터를 StreamingResponse로 반환
        download_name = f"{stem}_{Path(file.filename).stem}.wav"
        return StreamingResponse(
            BytesIO(audio_bytes),
            media_type="audio/wav",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )

    except Exception as exc:
        logging.exception("demucs_execute failed")
        return JSONResponse(
            status_code=500,
            content={"error": f"demucs_execute failed: {exc}"},
        )
