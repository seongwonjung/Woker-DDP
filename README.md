# Docker 빌드 & 실행 가이드

## 개요

- `app/Dockerfile`은 CUDA 기반 이미지를 다중 단계로 쪼개서, 자주 바뀌는 영역만 다시 빌드할 수 있도록 했습니다.
- 각 단계는 자신이 필요한 요구사항 파일만 COPY 하므로, 해당 파일이 바뀌지 않으면 아랫단 캐시가 그대로 유지됩니다.
- `docker-compose.yml`은 기본적으로 `runtime` 타깃을 빌드하고 `./app`을 바인드 마운트하여 Python 코드를 수정하면 곧바로 reload 되도록 구성했습니다.

## 요구사항 파일

| 파일                             | 용도                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------- |
| `app/requirements/base.txt`      | numpy, ffmpeg 등 CUDA 위 공통 유틸                                                    |
| `app/requirements/python.txt`    | FastAPI, uvicorn, pydantic 등 서버 공통 의존성                                        |
| `app/requirements/whisperx.txt`  | WhisperX, faster-whisper, Demucs, Torch 계열 STT 전용 패키지                          |
| `app/requirements/cosyvoice.txt` | CosyVoice(ModelScope) 관련 패키지                                                     |
| `app/requirements/extra.txt`     | gTTS, googletrans 등 선택 기능                                                        |
| `app/requirements/pins.txt`      | 모든 pip install에 강제하는 버전 핀(예: `numpy>=1.26.4,<2`, `faster-whisper>=1.1,<2`) |
| `app/requirements.txt`           | 위 파일들을 `-r`로 모은 레거시 진입점 (여전히 호환성 유지용)                          |

## Docker 빌드 단계

1. `base-utils` : CUDA 13.0.2 + Ubuntu 22.04 + 시스템 패키지
2. `python-common` : `base.txt`, `python.txt` 설치
3. `stt-whisperx` : `whisperx.txt` 설치
4. `tts-cosyvoice` : `cosyvoice.txt` 설치 + CosyVoice 리포지토리(옵션)
5. `extra` : `extra.txt` 설치
6. `app-base` : 애플리케이션 코드 COPY
7. `runtime` / `dev` : uvicorn 실행 방식만 다름 (`dev`는 `--reload`)

필요 시 특정 단계까지만 빠르게 빌드:

```bash
docker build -f app/Dockerfile --target stt-whisperx ./app
```

## 데이터 & 모델 디렉터리

- 루트 `data/` 폴더가 컨테이너 `/data`로 마운트되며, 그 안에 `inputs/`, `interim/`, `outputs/`가 생성됩니다. 호스트에서도 `mkdir -p data/{inputs,interim,outputs}`로 미리 준비해 두면 정리가 쉽습니다.
- `DATA_DIR`, `INPUTS_DIR`, `INTERIM_DIR`, `OUTPUTS_DIR` 환경변수로 각 경로를 바꿀 수 있으며, 기본값은 `/data/...`입니다.
- 모델·캐시는 `./models` → `/models`에 저장됩니다. `MODELS_DIR`(기본 `/models`)과 `WHISPERX_CACHE_DIR=/models/.cache/whisperx`, `XDG_CACHE_HOME=/models/.cache` 설정 덕분에 Hugging Face Hub에서 내려받은 WhisperX/pyannote 자산이 컨테이너 재시작 시에도 유지됩니다.

### job_id 단위 디렉터리 구조

```
data/
├── inputs/<job_id>/source.mp4
├── outputs/<job_id>/
│   ├── text/ (src_transcript.comp.json, trg_translated.json)
│   └── vid/ (dubbed_audio.wav, dubbed_video.mp4)
└── interim/<job_id>/
    └── text/
        ├── src/
        │   └── sentence/transcript.comp.json
        ├── trg/
        │   └── sentence/translated.json
        └── vid/
            ├── speaks/ (audio.wav, vocals.wav, ...)
            ├── bgm/background.wav
            └── tts/*.wav
```

각 단계는 위 경로를 사용하며 `/mux` 결과물은 `outputs/<job_id>/vid`에 저장됩니다.

### Compact transcript schema

- STT 단계는 세그먼트/단어 메타데이터를 `transcript.comp.json` 하나에 저장합니다. JSON 구조는 다음과 같습니다.
  ```json
  {
    "v": 1,
    "speakers": ["SPEAKER_00", "SPEAKER_01"],
    "segments": [
      {
        "s": 1955,
        "e": 18698,
        "sp": 0,
        "txt": "...",
        "gap": [4519, 4519],
        "w_off": [0, 23]
      }
    ],
    "vocab": ["메사추세추주의", "케이프", "..."],
    "words": [
      [0, 0, 742, 0, 191],
      [0, 1163, 1430, 1, 255]
    ]
  }
  ```
- 시간 단위는 ms이며, `segments[i].w_off = [start,count]`가 평면 `words` 배열의 위치를 가리킵니다.
- 각 단어 항목은 `[segIdx, segRelativeStart, segRelativeEnd, vocabIdx, scoreQ(0..255)]`입니다.
- 문자열 ID는 저장하지 않고, 필요 시 `segment_{idx:04d}` 형태로 on-the-fly 생성합니다.
- 번역 단계 결과(`trg/sentence/translated.json`, `outputs/.../trg_translated.json`)는 `{"seg_idx": int, "translation": str}` 목록만 담고, 나머지 메타데이터는 source compact 파일을 참조합니다.

#### Metadata storage rationale

- **파일 수 최소화**: 세그먼트/단어 JSON을 한데 묶어 S3와 파일시스템 오버헤드를 줄입니다. 키 폭이 늘어나더라도 gzip/zstd 압축으로 충분히 상쇄됩니다.
- **인덱스 우선 구조**: 화자/어휘/단어는 숫자 인덱스로 참조하여 문자열을 재사용합니다. 필요할 때만 `segment_{idx}` 같은 사람이 읽을 수 있는 ID를 생성하세요.
- **상대 시간 표현**: 단어 수준 타임스탬프는 세그먼트 시작으로부터의 오프셋만 저장하여 절대시간 중복을 제거합니다. 세그먼트에는 시작·끝(ms)만 남기고 duration은 파생 값입니다.
- **정규화된 점수**: WhisperX `score`(0~1 float)를 0~255 양자화 값으로 저장합니다. 이 값은 신뢰도 heatmap 등에 직접 사용할 수 있고 필요한 경우 다시 0~1 범위로 복원하면 됩니다.
- **단일 소스**: `transcript.comp.json`가 모든 언어/자막/단어 정렬의 기준입니다. 파생 산출물(`translated.json`, `segments.json` 등)은 compact 파일의 `seg_idx`나 `segment_id`만 참조하면 됩니다.

## 모델 다운로드

- Dockerfile 기본값은 `DOWNLOAD_COSYVOICE_MODELS=false`라서 자동 다운로드를 수행하지 않습니다.
- 직접 받은 모델을 `./models` 아래에 두면 Compose가 `/models`로 마운트합니다. WhisperX 캐시도 같은 경로 아래 `.cache/whisperx`에 저장됩니다.
- 만약 Docker 빌드 중에 받으려면:

```bash
docker compose build --build-arg DOWNLOAD_COSYVOICE_MODELS=true worker
```

## 재빌드 & 실행 흐름

1. **최초 세팅**
   ```bash
   docker compose build --no-cache worker
   docker compose up -d
   ```
2. **Python 코드 변경**
   - `./app`가 바인드되어 있고 `UVICORN_RELOAD` 기본값이 `--reload`이므로 `docker compose up`만 유지하면 자동 반영됩니다.
   - 강제로 재기동하고 싶으면 `docker compose restart worker`.
3. **요구사항 변경**
   - 수정한 파일에 따라 영향을 받는 단계만 빌드되지만, 명령은 동일합니다.
   ```bash
   docker compose build worker
   docker compose up -d
   ```
   - 예: STT 패키지만 수정했다면 `whisperx.txt`만 바뀌므로 `stt-whisperx` 이상 단계만 다시 빌드됩니다.
4. **특정 단계만 검증**
   ```bash
   docker build -f app/Dockerfile --target tts-cosyvoice ./app
   ```
   - 결과가 만족스러우면 `docker compose up --build worker`로 최종 이미지를 올립니다.

## 트러블슈팅 팁

- CosyVoice 리포지토리를 완전히 생략하려면 build arg `CLONE_COSYVOICE=false`를 전달하세요.
- 운영 배포에서 핫 리로드를 끄고 싶다면 `.env` 또는 Compose override에서 `UVICORN_RELOAD=`(빈 값)으로 설정하고, `./app:/app` 바인드 마운트도 제거하는 편이 좋습니다.
- NumPy 2.x로 강제 업그레이드를 막기 위해 모든 pip 설치에 `app/requirements/pins.txt`를 constraint로 적용하고 있습니다. 버전을 바꾸려면 pins 파일을 수정하고 `docker compose build worker`를 다시 실행하세요.
- WhisperX 3.7.x 는 `faster-whisper>=1.1,<2`와 `huggingface_hub>=0.24`를 요구하므로, `app/requirements/whisperx.txt`와 `requirements/pins.txt`에서 해당 범위를 유지하세요. 버전을 바꿨다면 `docker compose build worker`(필요 시 `--no-cache`)로 다시 이미지를 빌드해야 전체 레이어가 새 의존성을 사용합니다.
- 설치된 버전을 확인하려면 컨테이너에서 `python - <<'PY'\nimport importlib.metadata as md\nprint(md.version('faster-whisper'))\nPY` 명령을 실행하세요. 범위를 벗어나면 빌드 중 pip가 실패하도록 constraint가 적용돼 있습니다.
- WhisperX/pyannote 모델을 오프라인으로 유지하고 싶다면 `.env`에 `WHISPERX_CACHE_DIR`을 원하는 경로로 지정하고, 해당 폴더를 호스트에서 미리 채워 넣으면 런타임 다운로드를 피할 수 있습니다. 기본값은 `/models/.cache/whisperx`이며, `services/stt.py`가 각 단계마다 로깅을 남겨 어디에서 실패했는지 추적할 수 있습니다.

## HuggingFace 토큰 사용

1. 루트 `.env` 파일에 `HUGGINGFACE_TOKEN=hf_xxx` 형태로 저장합니다. (커밋 금지)
2. `docker-compose.yml`이 `.env`를 `env_file`로 읽어 컨테이너 환경변수에 주입합니다.
3. FastAPI `/asr` 엔드포인트는 더 이상 `hf_token` 입력을 받지 않으며, `run_asr`가 `HF_TOKEN`, `HUGGINGFACE_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `HUGGINGFACEHUB_API_TOKEN` 중 하나를 자동으로 찾아 pyannote diarization에 전달합니다.
4. `/asr` 및 `/pipeline` 요청 본문에 `speaker_count`(정수, 1 이상)를 넘기면 pyannote diarization이 해당 화자 수를 기준으로 동작합니다. 지정하지 않으면 기존처럼 화자 수를 자동으로 추정합니다.

## Vertex AI 설정(번역)

- 번역 단계(`services.translate.translate_transcript`)는 기본적으로 Vertex AI Gemini(Flash‑Lite) 모델을 사용합니다. 서비스 계정 JSON 기반 인증을 지원하며, 최소 10개 세그먼트를 묶어서 배치 번역합니다.

### 사전 준비

- GCP 프로젝트에서 Vertex AI API를 사용 설정합니다.
- 서비스 계정을 만들고 적절한 권한(예: `roles/aiplatform.user`)을 부여합니다.
- 서비스 계정 키(JSON)를 다운로드합니다.

### 환경변수

- `VERTEX_SERVICE_ACCOUNT_JSON` – 서비스 계정 JSON 파일 경로(컨테이너에서 접근 가능해야 함). 예: `/run/secrets/vertex_sa.json`
- `VERTEX_PROJECT_ID` – GCP 프로젝트 ID(JSON에 포함되어 있으면 생략 가능)
- `VERTEX_LOCATION` – 기본 `us-central1`
- `VERTEX_GEMINI_MODEL` – 기본 `gemini-1.5-flash-8b` (경량/고속 번역용)
- `MT_MIN_BATCH_SIZE` – 배치 최소 크기(기본 `10`)

`.env` 예시:

```
VERTEX_SERVICE_ACCOUNT_JSON=/run/secrets/vertex_sa.json
VERTEX_PROJECT_ID=your-gcp-project
VERTEX_LOCATION=us-central1
VERTEX_GEMINI_MODEL=gemini-1.5-flash-8b
MT_MIN_BATCH_SIZE=10
```

Docker 실행 시, 서비스 계정 JSON 파일을 컨테이너에 마운트하거나 Docker secrets를 사용해 `/run/secrets/vertex_sa.json` 경로로 제공하세요.

### 의존성

- `google-cloud-aiplatform` 패키지를 추가했습니다. Docker 이미지를 다시 빌드해야 합니다.

```
docker compose build worker
docker compose up -d
```

### 동작 개요

- STT 결과의 세그먼트를 최소 10개 단위로 청크하여 각 청크를 한 번의 Gemini 호출로 번역합니다.
- 프롬프트는 입력을 JSON 배열로 전달하고, 출력도 `[{"seg_idx", "translation"}]` 형태의 JSON 배열만 반환하도록 유도합니다.
- 결과 파일은 다음 위치에 저장됩니다.
  - `interim/<job_id>/text/trg/sentence/translated.json`
  - `outputs/<job_id>/text/trg_translated.json`
