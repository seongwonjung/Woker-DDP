# ---------- Base ----------
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE} AS base-utils

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg sox libsox-dev libsndfile1 rubberband-cli \
    build-essential nvidia-cuda-toolkit \
 && ln -s /usr/lib/nvidia-cuda-toolkit /usr/local/cuda || true \
 && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /workspace

# ---------- Common Python ----------
FROM base-utils AS python-common
COPY requirements /tmp/requirements/
ENV PIP_CONSTRAINT=/tmp/requirements/pins.txt

RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r /tmp/requirements/base.txt \
 && python -m pip install --no-cache-dir -r /tmp/requirements/python.txt

RUN python -m pip install --no-cache-dir "nvidia-cudnn-cu12>=9.1,<9.2"
ENV LD_LIBRARY_PATH=/opt/conda/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# ---------- WhisperX ----------
FROM python-common AS stt-whisperx
RUN python -m pip install --no-cache-dir -r /tmp/requirements/whisperx.txt

# ---------- CosyVoice ----------
FROM stt-whisperx AS tts-cosyvoice

# CosyVoice 주변 의존성
RUN python -m pip install --no-cache-dir -r /tmp/requirements/cosyvoice.txt

ARG CLONE_COSYVOICE=true
ENV COSYVOICE_DIR=/opt/CosyVoice

RUN if [ "$CLONE_COSYVOICE" = "true" ]; then \
      git clone https://github.com/FunAudioLLM/CosyVoice.git ${COSYVOICE_DIR} && \
      cd ${COSYVOICE_DIR} && git submodule update --init --recursive && \
      # torch/torchaudio는 상위에서 관리, openai-whisper는 나중에 별도로(노-디펜던시) 설치
      sed -i '/^torch/d' requirements.txt && \
      sed -i '/^torchaudio/d' requirements.txt && \
      sed -i '/^openai-whisper/d' requirements.txt && \
      python -m pip install --no-cache-dir -r requirements.txt; \
    fi

# wetext를 명시적으로 설치 (ttsfrd 대신 사용)
RUN python -m pip install --no-cache-dir wetext

# ★ 핵심: torch 2.8의 triton(=3.4.0) 고정과 충돌 피하기 위해
# openai-whisper를 --no-deps로 설치하고 필요한 최소 deps만 직접 핀으로 설치
# (CosyVoice는 import만 필요. 실제 STT는 WhisperX/ faster-whisper 사용)
RUN python -m pip install --no-cache-dir tiktoken==0.11.0 more-itertools==10.7.0 \
 && python -m pip install --no-cache-dir --no-deps openai-whisper==20231117

# CosyVoice/Matcha-TTS 검색 경로
ENV PYTHONPATH=${COSYVOICE_DIR}:${COSYVOICE_DIR}/third_party/Matcha-TTS:$PYTHONPATH

# ---------- Extra ----------
FROM tts-cosyvoice AS extra
RUN if [ -s /tmp/requirements/extra.txt ]; then \
      python -m pip install --no-cache-dir -r /tmp/requirements/extra.txt; \
    fi

# ---------- App ----------
FROM extra AS app-base
WORKDIR /app
COPY . /app

ENV APP_MODULE=main:app \
    APP_PORT=8000
EXPOSE 8000

# ---------- Runtime ----------
FROM app-base AS runtime
ENV UVICORN_RELOAD=
CMD ["sh", "-c", "uvicorn ${APP_MODULE:-main:app} --host 0.0.0.0 --port ${APP_PORT:-8000} ${UVICORN_RELOAD:-}"]

# ---------- Dev ----------
FROM app-base AS dev
ENV UVICORN_RELOAD=--reload
CMD ["sh", "-c", "uvicorn ${APP_MODULE:-main:app} --host 0.0.0.0 --port ${APP_PORT:-8000} ${UVICORN_RELOAD:-}"]

FROM runtime
