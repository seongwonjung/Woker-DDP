# stt.py
import os
import logging
import shutil
import subprocess
from pathlib import Path
import torch
import json

from pydub import AudioSegment

# Transformers>=4.41 expects torch.utils._pytree.register_pytree_node.
_pytree = getattr(getattr(torch, "utils", None), "_pytree", None)
if _pytree and not hasattr(_pytree, "register_pytree_node"):
    register_impl = getattr(_pytree, "_register_pytree_node", None)
    if register_impl:

        def register_pytree_node(
            node_type,
            flatten_fn,
            unflatten_fn,
            *,
            serialized_type_name=None,
            serialized_fields=None,
        ):
            """Transformers passes extra kwargs that the old torch implementation ignores."""
            return register_impl(node_type, flatten_fn, unflatten_fn)

        _pytree.register_pytree_node = register_pytree_node

import whisperx

try:
    from whisperx.diarize import DiarizationPipeline
except ImportError:  # WhisperX<3.7 fallback
    from whisperx import DiarizationPipeline
try:
    from app.configs import WHISPERX_CACHE_DIR, ensure_job_dirs
except ModuleNotFoundError as exc:  # allow running when /app is the root
    if exc.name != "app":
        raise
    from configs import WHISPERX_CACHE_DIR, ensure_job_dirs
from services.lang import normalize_lang_code
from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    build_compact_transcript,
    save_compact_transcript,
    segment_preview,
    segment_views,
)
from services.self_reference import (
    prepare_self_reference_samples,
    serialize_reference_mapping,
)
from services.speaker_embeddings import build_reference_embeddings
from services.demucs_split import split_vocals

logger = logging.getLogger(__name__)


def _whisperx_download_root(subdir: str) -> str:
    base = Path(WHISPERX_CACHE_DIR)
    path = base / subdir
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def run_asr(
    job_id: str,
    source_video_path: Path | str | None = None,
    source_lang: str | None = None,
    speaker_count: int | None = None,
):
    """입력 영상을 WhisperX로 전사하고 화자 분리를 수행합니다.

    Args:
        job_id: 작업 식별자.
        source_video_path: 명시된 경우 해당 경로에서 오디오를 추출합니다.
        source_lang: 백엔드에서 지정한 원본 언어 코드(예: 'ko', 'en').
            지정되면 WhisperX가 언어를 자동으로 추론하지 않고 해당 언어를 사용합니다.
        speaker_count: pyannote diarization이 예상 화자 수를 고정할 수 있도록 전달합니다.
            1 이상의 정수를 지정하면 num_speakers hint로 전달되며, 생략하면 자동 추정합니다.
    """
    lang_override = normalize_lang_code(source_lang)
    normalized_speaker_count = None
    if speaker_count is not None:
        if speaker_count < 1:
            raise ValueError("speaker_count must be >= 1")
        normalized_speaker_count = int(speaker_count)

    paths = ensure_job_dirs(job_id)
    logger.info("ASR lang preference: %s", lang_override or "auto")
    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # 파일 경로 구성
    if source_video_path:
        input_video = Path(source_video_path)
    else:
        input_video = paths.input_dir / "source.mp4"

    if not input_video.is_file():
        raise FileNotFoundError(
            f"Input video not found for job {job_id} at {input_video}"
        )
    raw_audio_path = paths.vid_speaks_dir / "audio.wav"

    # 1. 영상에서 오디오 추출 (Whisper 권장 형식: 모노 16kHz)
    extract_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(raw_audio_path),
    ]
    subprocess.run(extract_cmd, check=True)

    # 1-1. Demucs로 보컬/배경을 분리해 보컬만 ASR에 사용
    demucs_result = split_vocals(job_id)
    vocals_audio_path = Path(demucs_result["vocals"])

    # 2. WhisperX 모델을 불러와 전사 수행
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CPU에서 float16은 지원되지 않아 CTranslate2가 하위 레벨에서 예외를 내며
    # basic_string::_S_construct null not valid 같은 오류를 유발할 수 있습니다.
    default_compute_type = "float16" if device == "cuda" else "int8"
    compute_type = os.getenv("WHISPERX_COMPUTE_TYPE", default_compute_type)
    logger.info(
        "Loading WhisperX ASR model (device=%s, compute_type=%s)",
        device,
        compute_type,
    )
    try:
        model = whisperx.load_model(
            "large-v3-turbo",
            device=device,
            compute_type=compute_type,
            download_root=_whisperx_download_root("asr"),
        )
    except Exception as load_exc:
        # compute_type 호환성 문제 등에 대비한 안전 폴백
        # GPU에서도 float32는 보편적으로 지원되므로 안전한 선택입니다.
        fallback_compute = "float32"
        logger.warning(
            "ASR model load failed (compute_type=%s): %s. Retrying with %s",
            compute_type,
            load_exc,
            fallback_compute,
        )
        model = whisperx.load_model(
            "large-v3-turbo",
            device=device,
            compute_type=fallback_compute,
            download_root=_whisperx_download_root("asr"),
        )

    # 3. 단어 정렬 전 단계: 오디오 전사 후 구간 정보 확보
    audio = whisperx.load_audio(str(vocals_audio_path))
    logger.info("Running ASR transcription via WhisperX")
    transcribe_kwargs: dict = {}
    if lang_override:
        transcribe_kwargs["language"] = lang_override
    try:
        result = model.transcribe(audio, **transcribe_kwargs)
    except Exception as transcribe_exc:
        # 언어 코드로 실패하면 자동 감지로 재시도
        logger.warning(
            "Transcribe failed with explicit language=%s, retrying with auto: %s",
            lang_override,
            transcribe_exc,
        )
        result = model.transcribe(audio)
        lang_override = None  # fall back to auto thereafter
    if lang_override:
        result["language"] = lang_override
    segments = result["segments"]  # 텍스트와 대략적인 타임스탬프 포함

    # align 성공/실패와 관계없이 사용할 기본 result_aligned (coarse segments)
    result_aligned = {
        "segments": segments,
        "language": result.get("language"),
    }

    # 4. 정밀한 타이밍을 위한 정렬 모델 로드
    language_code = normalize_lang_code(result.get("language")) or lang_override
    logger.info("Loading alignment model for language=%s", language_code)
    align_kwargs = {
        "language_code": language_code,
        "device": device,
    }
    align_root = _whisperx_download_root("align")
    try:
        try:
            align_model, metadata = whisperx.load_align_model(
                download_root=align_root, **align_kwargs
            )
        except TypeError as exc:
            if "unexpected keyword argument 'download_root'" in str(exc):
                align_model, metadata = whisperx.load_align_model(**align_kwargs)
            else:
                raise
        aligned = whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            device=device,
            return_char_alignments=False,
        )
        result_aligned = aligned
        segments = result_aligned["segments"]  # 단어 단위 타임스탬프가 포함된 구간
    except Exception as align_exc:
        logger.warning(
            "Alignment skipped due to error (language=%s): %s",
            language_code,
            align_exc,
        )
        # result_aligned는 coarse segments 그대로 유지

    # 5. pyannote 기반 화자 분리 (모델 접근을 위해 HF 토큰 필요)
    if hf_token:
        logger.info("Initializing WhisperX diarization pipeline via pyannote")
        diarization_model = (
            os.getenv("PYANNOTE_DIARIZATION_MODEL")
            or os.getenv("WHISPERX_DIARIZATION_MODEL")
            or "pyannote/speaker-diarization-3.1"
        )
        diarization_pipeline = DiarizationPipeline(
            diarization_model,
            use_auth_token=hf_token,
            device=device,
        )
        diarization_kwargs = {}
        if normalized_speaker_count:
            diarization_kwargs["num_speakers"] = normalized_speaker_count
            logger.info(
                "Running diarization with speaker_count=%d", normalized_speaker_count
            )
        try:
            diarization_segments = diarization_pipeline(
                str(vocals_audio_path), **diarization_kwargs
            )
        except TypeError as exc:
            if normalized_speaker_count and "num_speakers" in str(exc):
                logger.warning(
                    "Diarization pipeline rejected num_speakers hint (%s); retrying without it",
                    exc,
                )
                diarization_segments = diarization_pipeline(str(vocals_audio_path))
            else:
                raise
        # 각 구간에 화자 레이블 부여
        result_segments = whisperx.assign_word_speakers(
            diarization_segments, result_aligned
        )
        segments = result_segments["segments"]
        # 각 구간/단어에 speaker 키가 포함됨

    # 6. 문장/단어 메타데이터 정리 및 저장
    # 7. 새 compact 스키마로 저장
    bundle = build_compact_transcript(segments, language=result.get("language"))
    transcript_archive = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
    save_compact_transcript(bundle, transcript_archive)
    shutil.copyfile(
        transcript_archive,
        paths.outputs_text_dir / f"src_{COMPACT_ARCHIVE_NAME}",
    )

    # 7-1. 화자별 self-reference 샘플을 미리 생성해 둠 (TTS에서 재사용)
    try:
        base_segments = segment_views(bundle)
        vocals_audio = AudioSegment.from_wav(str(vocals_audio_path))

        tts_dir = paths.vid_tts_dir
        tts_dir.mkdir(parents=True, exist_ok=True)

        speaker_ref_dir = tts_dir / "self_refs"
        speaker_refs = prepare_self_reference_samples(
            vocals_audio, base_segments, speaker_ref_dir
        )
        # speaker -> self_ref wav & metadata 매핑 저장
        mapping = serialize_reference_mapping(speaker_refs, tts_dir)
        with open(tts_dir / "speaker_refs.json", "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        # self-reference와 동일한 위치에 스피커 임베딩을 생성한다.
        # (prod에서는 s3://<bucket>/voice-samples/ 로 동기화 예정)
        embeddings_dir = tts_dir / "speaker_embeddings"
        build_reference_embeddings(
            speaker_refs,
            embeddings_dir,
            base_dir=paths.interim_dir,
        )
        logger.info(
            "Saved self-reference mapping for %d speakers at %s",
            len(mapping),
            tts_dir / "speaker_refs.json",
        )
    except Exception as exc:
        logger.warning(
            "Failed to precompute self-reference samples for job %s: %s",
            job_id,
            exc,
        )

    # 레거시 산출물 정리 (존재할 경우)
    legacy_transcript = paths.src_sentence_dir / "transcript.json"
    if legacy_transcript.exists():
        legacy_transcript.unlink()
    aligned_path = paths.src_words_dir / "aligned_segments.json"
    if aligned_path.exists():
        aligned_path.unlink()
    for existing in paths.src_words_dir.glob("segment_*_words.json"):
        existing.unlink()

    return segment_preview(bundle)
