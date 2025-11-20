"""Vertex AI Gemini 기반 기계번역 서비스 (단순/안정 버전).

변경 요약
- googletrans 제거, Vertex AI Gemini로 대체.
- 최소 10개 세그먼트 단위 배치 호출.
- '입력 N개 → 출력 N개'를 seg_idx로 매핑해서 복원.
- 길이 비율/char_count/2차 보정 호출 제거 → 프롬프트 간단 + MAX_TOKENS 회피.
- 비동기 병렬 처리 추가 (MT_MAX_CONCURRENT 환경 변수로 제어)
- ENV: VERTEX_PROJECT_ID, VERTEX_LOCATION, VERTEX_GEMINI_MODEL,
       GOOGLE_APPLICATION_CREDENTIALS 또는 VERTEX_SERVICE_ACCOUNT_JSON/SA_PATH,
       MT_MIN_BATCH_SIZE(기본 10), MT_BACKEND, MT_STRICT, MT_MAX_CONCURRENT(기본 5)
"""

from __future__ import annotations

import json
import os
import shutil
import logging
import asyncio
from typing import Iterable, List, Dict, Any

from configs import get_job_paths
from services.transcript_store import (
    COMPACT_ARCHIVE_NAME,
    load_compact_transcript,
    segment_views,
)

# 선택 의존성: 라이브러리가 없어도 모듈 import 자체는 되도록 지연 임포트
_VERTEX_AVAILABLE = True
try:
    import vertexai  # type: ignore
    from vertexai.generative_models import (  # type: ignore
        GenerativeModel,
        GenerationConfig,
    )
    from google.oauth2 import service_account  # type: ignore
except Exception:  # pragma: no cover
    _VERTEX_AVAILABLE = False


def _fallback_translate_batch(
    items: List[Dict[str, Any]], target_lang: str, src_lang: str | None = None
) -> List[Dict[str, Any]]:
    """최소한의 폴백 번역기.

    - 1순위: googletrans 사용(설치되어 있고 네트워크 가능할 때)
    - 실패 시: 입력 텍스트를 그대로 반환(아이덴티티)
    """
    try:
        from googletrans import Translator  # type: ignore

        tr = Translator()
        src = src_lang if src_lang else "auto"
        texts = [str(o["text"]) for o in items]
        res = tr.translate(texts, dest=target_lang, src=src)
        outputs: List[Dict[str, Any]] = []
        for i, o in enumerate(items):
            translated = res[i].text if i < len(res) else str(o["text"])  # type: ignore
            outputs.append({"seg_idx": int(o["seg_idx"]), "translation": translated})
        return outputs
    except Exception:
        # 네트워크/설치 문제 시 아이덴티티 폴백
        return [
            {"seg_idx": int(o["seg_idx"]), "translation": str(o["text"])} for o in items
        ]


def _env_str(key: str, default: str | None = None) -> str | None:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


def _env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}


def _chunked(seq: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """이터러블을 최대 `size` 길이의 리스트들로 잘라 순차 반환."""
    batch: List[Any] = []
    for item in seq:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


class GeminiTranslator:
    """Vertex AI Gemini 기반 번역기 (단순/안정 버전).

    - ENV로 프로젝트/리전/모델/인증을 읽어 초기화.
    - JSON 배열만 요청하고, seg_idx로 N개→N개 매핑 복원.
    """

    def __init__(self) -> None:
        if not _VERTEX_AVAILABLE:
            raise RuntimeError(
                "Vertex AI libraries not installed. Add google-cloud-aiplatform to requirements."
            )

        # 모델/리전
        self.model_name = _env_str("VERTEX_GEMINI_MODEL", "gemini-2.5-pro")
        self.location = _env_str("VERTEX_LOCATION", "us-central1")
        self.project_id = _env_str("VERTEX_PROJECT_ID")

        # 서비스 계정 JSON (여러 키명 지원)
        sa_path = (
            _env_str("VERTEX_SERVICE_ACCOUNT_JSON")
            or _env_str("VERTEX_SA_PATH")
            or _env_str("GOOGLE_APPLICATION_CREDENTIALS")
        )

        creds = None
        if sa_path and os.path.isfile(sa_path):
            # 프로젝트 ID가 없으면 JSON에서 복구 시도
            if not self.project_id:
                try:
                    with open(sa_path, "r", encoding="utf-8") as f:
                        j = json.load(f)
                        self.project_id = j.get("project_id")
                except Exception:
                    pass
            try:
                creds = service_account.Credentials.from_service_account_file(
                    sa_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load service account JSON at {sa_path}: {e}"
                )

        if not self.project_id:
            raise RuntimeError(
                "VERTEX_PROJECT_ID is required (or set in service account JSON)."
            )

        vertexai.init(
            project=self.project_id, location=self.location, credentials=creds
        )
        self._model = GenerativeModel(self.model_name)

    def translate_batch(
        self,
        items: List[Dict[str, Any]],
        target_lang: str,
        src_lang: str | None = None,
    ) -> List[Dict[str, Any]]:
        """배치 번역 수행 (심플 버전).

        items: [{"seg_idx": int, "text": str}, ...]
        반환: [{"seg_idx": int, "translation": str}, ...] (seg_idx 기준으로 N개 복원)
        """
        n = len(items)
        if n == 0:
            return []

        src_texts = [str(o["text"]) for o in items]
        seg_idxs = [int(o["seg_idx"]) for o in items]

        sys = (
            "You are a dubbing subtitle translator.\n"
            "- Translate with dubbing in mind.\n"
            "- Idioms can be paraphrased.\n"
            "- translate short texts as briefly as possible.\n"
            "- For numbers, translate only with the pronunciation that fits the context.\n"
            "- Proper nouns like AI should be translated phonetically.\n"
            "- Characters other than the target language must not be entered.\n"
            "- Translate each item from source language to the target language.\n"
            "- Do NOT merge or split items.\n"
            "- Do NOT add explanations, numbering, or any extra text.\n"
            "- Return ONLY one JSON array of length N.\n"
            "- Each element must be an object with exactly two fields: "
            "seg_idx (int) and translation (string).\n"
        )
        src_lang_txt = (
            f"Source language: {src_lang}"
            if src_lang
            else "Source language: auto-detect"
        )
        user = (
            f"N={n}\n"
            f"{src_lang_txt}\nTarget language: {target_lang}\n\n"
            "Inputs (keep order and seg_idx):\n"
            + "\n".join(
                f"[{i}] seg_idx={seg_idxs[i]} text={src_texts[i]}" for i in range(n)
            )
            + "\n\nReturn JSON ONLY like:\n"
            '[{"seg_idx": 0, "translation": "..."}, ...]'
        )

        gen_cfg = GenerationConfig(
            temperature=0.1,
            max_output_tokens=8192,  # 프롬프트가 짧아졌으니 2k면 충분
        )

        resp = self._model.generate_content(
            contents=[sys, user],
            generation_config=gen_cfg,
        )
        logging.info(f"Gemini raw response: {resp}")
        text = self._extract_text(resp)
        data = self._parse_json_array(text)

        # seg_idx → 번역 매핑
        mapping: Dict[int, str] = {}
        for obj in data or []:
            try:
                si = int(obj.get("seg_idx"))
                tr = obj.get("translation")
                if isinstance(tr, str):
                    mapping[si] = tr
            except Exception:
                continue

        # 최종 N개 복원 (누락은 원문 폴백)
        out: List[Dict[str, Any]] = []
        for idx, src in zip(seg_idxs, src_texts):
            out.append({"seg_idx": idx, "translation": mapping.get(idx, src)})
        return out

    @staticmethod
    def _extract_text(resp: Any) -> str:
        # SDK 버전에 따라 텍스트 접근 경로가 다를 수 있음
        try:
            cands = getattr(resp, "candidates", None)
            if cands:
                content = cands[0].content
                parts = getattr(content, "parts", None)
                # parts 안에 text가 있으면 사용
                for p in parts or []:
                    t = getattr(p, "text", None)
                    if isinstance(t, str) and t.strip():
                        return t
        except Exception:
            pass
        # 그 다음 resp.text
        try:
            if hasattr(resp, "text") and isinstance(resp.text, str):
                return resp.text
        except Exception:
            pass
        # 최후 수단
        return str(resp)

    @staticmethod
    def _parse_json_array(text: str) -> List[Dict[str, Any]]:
        # 1) 순수 JSON 시도
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj  # type: ignore[return-value]
        except Exception:
            pass
        # 2) 주변 텍스트에서 JSON 배열 부분 추출 시도
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                obj = json.loads(snippet)
                if isinstance(obj, list):
                    return obj  # type: ignore[return-value]
            except Exception:
                pass
        # 3) 실패 시 빈 배열
        return []


def _merge_batches(
    original_items: List[Dict[str, Any]], batch_outputs: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """배치 결과를 원본 순서로 병합.

    누락된 번역은 원문 텍스트로 대체.
    """
    merged: Dict[int, str] = {}
    for out in batch_outputs:
        for obj in out:
            if isinstance(obj.get("seg_idx"), int) and isinstance(
                obj.get("translation"), str
            ):
                merged[obj["seg_idx"]] = obj["translation"]

    result: List[Dict[str, Any]] = []
    for item in original_items:
        idx = int(item["seg_idx"])  # type: ignore[arg-type]
        txt = str(item["text"])  # type: ignore[arg-type]
        result.append({"seg_idx": idx, "translation": merged.get(idx, txt)})
    return result


def translate_transcript(job_id: str, target_lang: str, src_lang: str | None = None):
    """전사된 구간 텍스트를 지정 언어로 번역.

    - 기본적으로 Vertex AI Gemini 사용(ENV로 모델/리전 지정)
    - 세그먼트를 최소 10개 단위로 묶어 배치 호출
    - 비동기 병렬 처리로 여러 배치를 동시에 처리 (MT_MAX_CONCURRENT로 제어)
    - 결과를 translated.json으로 저장하고 outputs에도 복사
    """
    paths = get_job_paths(job_id)
    transcript_path = paths.src_sentence_dir / COMPACT_ARCHIVE_NAME
    if not transcript_path.is_file():
        raise FileNotFoundError("Transcript not found. Run ASR stage first.")

    bundle = load_compact_transcript(transcript_path)
    seg_views = segment_views(bundle)

    # MT 입력 준비
    items = [{"seg_idx": s.idx, "text": s.text} for s in seg_views]

    min_batch = int(_env_str("MT_MIN_BATCH_SIZE", "10") or "10")
    if min_batch < 1:
        min_batch = 10

    # 최대 동시 실행 배치 수 (환경 변수로 제어 가능, 기본값: 5)
    max_concurrent = int(_env_str("MT_MAX_CONCURRENT", "5") or "5")
    if max_concurrent < 1:
        max_concurrent = 5

    # 백엔드 선택: 기본 vertex, 환경변수로 강제 가능
    backend = (os.getenv("MT_BACKEND") or "vertex").strip().lower()
    strict = _env_bool("MT_STRICT", True)
    translator: Any | None = None
    use_vertex = backend in {"vertex", "gemini", "gemini-vertex"}
    if use_vertex:
        try:
            translator = GeminiTranslator()
        except Exception as exc:
            if strict:
                # 무조건 Vertex만 허용: 즉시 실패
                raise RuntimeError(
                    f"Vertex translator initialization failed under MT_STRICT: {exc}"
                )
            # 라이브러리/인증 누락 시 폴백 허용
            use_vertex = False

    if not use_vertex or translator is None:
        # 배치 단위로 폴백 번역 수행 후 병합 로직 재사용 (비동기 병렬 처리)
        batches = list(_chunked(items, size=min_batch))

        async def fallback_translate_batch_async(
            batch: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            """비동기로 폴백 배치 번역 수행"""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, _fallback_translate_batch, batch, target_lang, src_lang
            )

        async def translate_all_fallback_batches() -> List[List[Dict[str, Any]]]:
            """모든 폴백 배치를 병렬로 번역 (동시 실행 수 제한)"""
            semaphore = asyncio.Semaphore(max_concurrent)

            async def translate_with_semaphore(
                batch: List[Dict[str, Any]],
            ) -> List[Dict[str, Any]]:
                async with semaphore:
                    return await fallback_translate_batch_async(batch)

            tasks = [translate_with_semaphore(batch) for batch in batches]
            return await asyncio.gather(*tasks)

        # 비동기 실행 (이미 실행 중인 이벤트 루프 처리)
        logging.info(
            f"Translating {len(batches)} batches (fallback) with max {max_concurrent} concurrent requests"
        )

        try:
            # 이미 실행 중인 이벤트 루프가 있는지 확인
            running_loop = asyncio.get_running_loop()
            # 실행 중인 루프가 있으면 nest_asyncio 사용
            try:
                import nest_asyncio

                nest_asyncio.apply()
                batch_outputs = running_loop.run_until_complete(
                    translate_all_fallback_batches()
                )
            except ImportError:
                # nest_asyncio가 없으면 동기적으로 순차 실행 (폴백)
                logging.warning(
                    "nest_asyncio not available and event loop is running. "
                    "Falling back to sequential translation. "
                    "Install nest_asyncio for parallel processing: pip install nest_asyncio"
                )
                batch_outputs = []
                for batch in batches:
                    out = _fallback_translate_batch(
                        batch, target_lang, src_lang=src_lang
                    )
                    batch_outputs.append(out)
        except RuntimeError:
            # 실행 중인 루프가 없으면 새로 생성
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            batch_outputs = loop.run_until_complete(translate_all_fallback_batches())

        translated_segments = _merge_batches(items, batch_outputs)

        # 결과 저장(폴백도 동일한 경로 유지)
        trans_out_path = paths.trg_sentence_dir / "translated.json"
        trans_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(trans_out_path, "w", encoding="utf-8") as f:
            json.dump(translated_segments, f, ensure_ascii=False, indent=2)
        paths.outputs_text_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(trans_out_path, paths.outputs_text_dir / "trg_translated.json")
        return translated_segments

    # 여기로 왔으면 Vertex 사용 (translator 준비됨)

    # 배치를 리스트로 수집
    batches = list(_chunked(items, size=min_batch))

    # 비동기 병렬 처리
    async def translate_batch_async(
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """비동기로 배치 번역 수행 (스레드 풀 사용)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, translator.translate_batch, batch, target_lang, src_lang
        )

    async def translate_all_batches() -> List[List[Dict[str, Any]]]:
        """모든 배치를 병렬로 번역 (동시 실행 수 제한)"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def translate_with_semaphore(
            batch: List[Dict[str, Any]],
        ) -> List[Dict[str, Any]]:
            async with semaphore:
                return await translate_batch_async(batch)

        tasks = [translate_with_semaphore(batch) for batch in batches]
        return await asyncio.gather(*tasks)

    # 비동기 실행 (이미 실행 중인 이벤트 루프 처리)
    logging.info(
        f"Translating {len(batches)} batches with max {max_concurrent} concurrent requests"
    )

    try:
        # 이미 실행 중인 이벤트 루프가 있는지 확인
        running_loop = asyncio.get_running_loop()
        # 실행 중인 루프가 있으면 nest_asyncio 사용
        try:
            import nest_asyncio

            nest_asyncio.apply()
            batch_outputs = running_loop.run_until_complete(translate_all_batches())
        except ImportError:
            # nest_asyncio가 없으면 동기적으로 순차 실행 (폴백)
            logging.warning(
                "nest_asyncio not available and event loop is running. "
                "Falling back to sequential translation. "
                "Install nest_asyncio for parallel processing: pip install nest_asyncio"
            )
            batch_outputs = []
            for batch in batches:
                out = translator.translate_batch(batch, target_lang, src_lang=src_lang)
                batch_outputs.append(out)
    except RuntimeError:
        # 실행 중인 루프가 없으면 새로 생성
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        batch_outputs = loop.run_until_complete(translate_all_batches())

    translated_segments = _merge_batches(items, batch_outputs)

    # 결과 저장
    trans_out_path = paths.trg_sentence_dir / "translated.json"
    trans_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trans_out_path, "w", encoding="utf-8") as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    paths.outputs_text_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(trans_out_path, paths.outputs_text_dir / "trg_translated.json")
    return translated_segments
