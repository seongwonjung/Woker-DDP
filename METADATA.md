# 메타데이터 저장 가이드

이 문서는 **STT → 번역 → TTS** 각 단계가 어떤 메타데이터를 어디에 남기는지 정리한 것입니다. 경로는 모두 `data/`(컨테이너 내부 `/data`)를 기준으로 하며 `job_id` 별로 나뉩니다.

## 디렉터리 규칙

- `inputs/<job_id>/source.mp4` – `/asr` 업로드를 통해 유입된 원본 영상.
- `interim/<job_id>` – 모든 중간 산출물이 쌓이는 워크트리.
  - `text/src/sentence` – WhisperX compact transcript (`transcript.comp.json`).
  - `text/trg/sentence` – 번역 결과(`translated.json`).
  - `text/vid/tts` – TTS 원본 wav 및 구간별 메타데이터.
- `outputs/<job_id>` – 사용자에게 제공하는 결과 복사본(`text/`, `vid/`).

## 1단계 – WhisperX STT 메타데이터

생성 위치: `services.stt.run_asr`.

### 산출물

- `interim/<job_id>/text/src/sentence/transcript.comp.json`
- `outputs/<job_id>/text/src_transcript.comp.json` (다운로드용 복사본)

### Compact Transcript 스키마

최상위 키:

| key        | type            | 설명                                                                  |
| ---------- | --------------- | --------------------------------------------------------------------- |
| `v`        | int             | 스키마 버전 (`1`).                                                    |
| `unit`     | str             | 시간 단위(현재 `"ms"`).                                               |
| `lang`     | str 또는 null   | WhisperX가 반환한 ISO 언어 코드.                                      |
| `speakers` | list[str]       | 정규화된 화자 라벨 목록(`SPEAKER_00` 등).                             |
| `segments` | list[object]    | 구간 타임라인(`words` 인덱스를 참조).                                 |
| `vocab`    | list[str]       | 단어 항목이 참조하는 유니크 토큰.                                     |
| `words`    | list[list[int]] | `[seg_idx, offset_start_ms, offset_end_ms, vocab_idx, score_q]` 형식. |

각 세그먼트(`segments[i]`) 필드:

| field    | 설명                                                     |
| -------- | -------------------------------------------------------- |
| `s`, `e` | 세그먼트 시작/끝(ms).                                    |
| `sp`     | `speakers` 인덱스.                                       |
| `txt`    | WhisperX가 출력한 텍스트(화자 분리 후).                  |
| `gap`    | `[gap_after_ms, gap_after_vad_ms]`; 싱크/뮤직킹 참고 값. |
| `w_off`  | 전역 `words` 리스트에서 `[시작 인덱스, 개수]`.           |
| `o`      | 원본 Whisper 세그먼트 ID(추적용).                        |
| `ov`     | 이전 구간과 겹치면 `true`.                               |

`words` 항목은 첫 열의 세그먼트를 가리키며, 시간은 부모 세그먼트 기준 상대 ms, `score_q`는 0~255로 양자화된 confidence입니다. `/asr` 응답은 `segment_preview()` → `SegmentView.to_public_dict()` 경유로 만들어지며, 여기서도 `segment_{idx:04d}` ID가 유지됩니다.

## 2단계 – 번역 메타데이터

생성 위치: `services.translate.translate_transcript`.

### 산출물

- `interim/<job_id>/text/trg/sentence/translated.json`
- `outputs/<job_id>/text/trg_translated.json`

### 스키마

UTF-8 JSON 배열이며 각 요소는 다음을 포함해야 합니다.

| field         | 설명                                                               |
| ------------- | ------------------------------------------------------------------ |
| `seg_idx`     | compact transcript의 `segments[seg_idx]`를 가리키는 0-base 인덱스. |
| `translation` | 해당 구간의 번역 텍스트.                                           |

### 선택 입력(Overrides)

`generate_tts`는 동일한 JSON을 다시 읽으며, 특정 키가 있으면 아래와 같이 사용합니다.

| optional key                         | TTS 단계에서의 효과                                             |
| ------------------------------------ | --------------------------------------------------------------- |
| `speaker`                            | 구간을 다른 화자 라벨로 바꿀 때 사용.                           |
| `voice_sample_path` / `voice_sample` | 구간별 커스텀 레퍼런스 wav 경로(상대 경로는 job 디렉터리 기준). |
| `prompt_text` / `prompt`             | CosyVoice에 전달할 직접 입력 프롬프트.                          |
| `reference_text`                     | `prompt_text`가 없을 때 대체로 사용.                            |

위 키 외의 정보도 JSON 안에 남길 수 있으며, 현재는 `generate_tts`가 표에 있는 필드만 소비합니다.

## 3단계 – TTS 메타데이터

생성 위치: `services.tts.generate_tts`.

### 보이스 레퍼런스 결정 로직

각 세그먼트마다 다음 정보를 병합해 CosyVoice 호출 페이로드를 만듭니다.

1. compact transcript 기반 원본 메타데이터(`speaker`, `start`, `end`, `duration`).
2. `translated.json` overrides(번역, 화자, 샘플 경로, 프롬프트 등).
3. `/tts/dub` 요청 시 넘겨지는 배우 매핑(`SPEAKER_XX` → `actor_id`). 각 배우의 샘플/프롬프트는 `interim/<job_id>/tts_custom_refs/actors/<actor_id>/prompt.txt` 아래에 저장됩니다.
4. 명시 샘플이 없을 때 자동으로 추출한 self-reference(`text/vid/tts/<speaker>_self_ref.wav`).

선택 순서는 “배우 샘플 → per-segment override → `tts_custom_refs/<speaker>` → self-reference” 입니다.

### 산출물

- `interim/<job_id>/text/vid/tts/*.wav` – 세그먼트별 CosyVoice 원본 오디오.
- `interim/<job_id>/text/vid/tts/segments.json` – 각 wav 생성 과정을 기술한 메타데이터.

### `segments.json` 필드

각 요소는 다음 키를 가집니다.

| field             | 설명                                       |
| ----------------- | ------------------------------------------ |
| `segment_id`      | STT와 동일한 안정 ID(`segment_{idx:04d}`). |
| `seg_idx`         | compact transcript 인덱스.                 |
| `speaker`         | 합성 시 실제로 사용된 화자 라벨.           |
| `start`, `end`    | 원본 구간 시작/끝(초 단위).                |
| `target_duration` | 목표 길이(초), 보통 원본 세그먼트 길이.    |
| `audio_file`      | 동기화 전 TTS wav 절대 경로.               |
| `voice_sample`    | CosyVoice에 투입된 레퍼런스 오디오 경로.   |
| `prompt_text`     | TTS 백엔드에 전달된 프롬프트 문자열.       |
| `tts_backend`     | 합성기 식별자(현재 `"cosyvoice2"`).        |

### 참고 – Sync 단계 메타데이터

`/sync`는 `segments.json`을 입력으로 받아 각 wav를 타겟 길이에 맞게 타임스트레치한 다음 `interim/<job_id>/text/vid/tts/synced/segments_synced.json`을 만듭니다. 이 파일은 길이 비율, 패딩/무음 보정, 사용한 백엔드(`stretch_backend`) 등을 추가하며, 상위 단계 메타데이터에 전적으로 의존합니다.

---

위 표를 따르면 사용자에게 노출되는 어떤 산출물도 compact STT 아카이브까지 추적할 수 있고, 번역/음성 합성 단계에서 필요한 매개변수나 override 포인트도 정확히 파악할 수 있습니다.
