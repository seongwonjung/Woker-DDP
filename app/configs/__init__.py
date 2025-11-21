from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .env import (
    AWS_REGION,
    AWS_S3_BUCKET,
    CALLBACK_LOCALHOST_HOST,
    CHUNK_SIZE,
    DEFAULT_SOURCE_LANG,
    DEFAULT_TARGET_LANG,
    JOB_QUEUE_URL,
    LOG_LEVEL,
    POLL_WAIT,
    PROFILE,
    VISIBILITY_TIMEOUT,
)
from .utils import (
    JobProcessingError,
    normalize_callback_url,
    post_status,
)


def _env_path(key: str, default: str | Path) -> Path:
    return Path(os.getenv(key, str(default)))


DATA_DIR = _env_path("DATA_DIR", "/data")
INPUTS_DIR = _env_path("INPUTS_DIR", DATA_DIR / "inputs")
INTERIM_DIR = _env_path("INTERIM_DIR", DATA_DIR / "interim")
OUTPUTS_DIR = _env_path("OUTPUTS_DIR", DATA_DIR / "outputs")
MODELS_DIR = _env_path("MODELS_DIR", "/models")
WHISPERX_CACHE_DIR = _env_path("WHISPERX_CACHE_DIR", MODELS_DIR / ".cache" / "whisperx")


@dataclass(frozen=True)
class JobPaths:
    job_id: str
    input_dir: Path
    interim_dir: Path
    outputs_dir: Path
    src_sentence_dir: Path
    src_words_dir: Path
    trg_sentence_dir: Path
    trg_words_dir: Path
    vid_speaks_dir: Path
    vid_bgm_dir: Path
    vid_tts_dir: Path
    outputs_text_dir: Path
    outputs_vid_dir: Path


def ensure_data_dirs() -> None:
    for path in (INPUTS_DIR, INTERIM_DIR, OUTPUTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_job_paths(job_id: str) -> JobPaths:
    input_dir = INPUTS_DIR / job_id
    interim_dir = INTERIM_DIR / job_id
    outputs_dir = OUTPUTS_DIR / job_id

    text_root = interim_dir / "text"
    src_sentence_dir = text_root / "src" / "sentence"
    src_words_dir = text_root / "src" / "words"
    trg_sentence_dir = text_root / "trg" / "sentence"
    trg_words_dir = text_root / "trg" / "words"
    vid_root = text_root / "vid"
    vid_speaks_dir = vid_root / "speaks"
    vid_bgm_dir = vid_root / "bgm"
    vid_tts_dir = vid_root / "tts"

    outputs_text_dir = outputs_dir / "text"
    outputs_vid_dir = outputs_dir / "vid"

    return JobPaths(
        job_id=job_id,
        input_dir=input_dir,
        interim_dir=interim_dir,
        outputs_dir=outputs_dir,
        src_sentence_dir=src_sentence_dir,
        src_words_dir=src_words_dir,
        trg_sentence_dir=trg_sentence_dir,
        trg_words_dir=trg_words_dir,
        vid_speaks_dir=vid_speaks_dir,
        vid_bgm_dir=vid_bgm_dir,
        vid_tts_dir=vid_tts_dir,
        outputs_text_dir=outputs_text_dir,
        outputs_vid_dir=outputs_vid_dir,
    )


def ensure_job_dirs(job_id: str) -> JobPaths:
    paths = get_job_paths(job_id)
    for directory in (
        paths.input_dir,
        paths.interim_dir,
        paths.src_sentence_dir,
        paths.src_words_dir,
        paths.trg_sentence_dir,
        paths.trg_words_dir,
        paths.vid_speaks_dir,
        paths.vid_bgm_dir,
        paths.vid_tts_dir,
        paths.outputs_dir,
        paths.outputs_text_dir,
        paths.outputs_vid_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


__all__ = [
    "AWS_REGION",
    "AWS_S3_BUCKET",
    "CALLBACK_LOCALHOST_HOST",
    "CHUNK_SIZE",
    "DATA_DIR",
    "DEFAULT_SOURCE_LANG",
    "DEFAULT_TARGET_LANG",
    "INPUTS_DIR",
    "INTERIM_DIR",
    "JOB_QUEUE_URL",
    "JobPaths",
    "JobProcessingError",
    "LOG_LEVEL",
    "MODELS_DIR",
    "OUTPUTS_DIR",
    "POLL_WAIT",
    "PROFILE",
    "VISIBILITY_TIMEOUT",
    "WHISPERX_CACHE_DIR",
    "ensure_data_dirs",
    "ensure_job_dirs",
    "get_job_paths",
    "normalize_callback_url",
    "post_status",
]
