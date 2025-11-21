from __future__ import annotations

import os
from typing import Any, Dict, Optional
from urllib.parse import urlparse, urlunparse

import requests

from .env import CALLBACK_LOCALHOST_HOST


class JobProcessingError(Exception):
    """Raised when a job fails irrecoverably during processing."""


def normalize_callback_url(callback_url: str) -> str:
    parsed = urlparse(callback_url)
    if parsed.hostname in {"localhost", "127.0.0.1"} and CALLBACK_LOCALHOST_HOST:
        host = CALLBACK_LOCALHOST_HOST
        netloc = host
        if parsed.port:
            netloc = f"{host}:{parsed.port}"
        parsed = parsed._replace(netloc=netloc)
    return urlunparse(parsed)


def post_status(
    http: requests.Session,
    callback_url: str,
    status: str,
    *,
    result_key: Optional[str] = None,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    stage_id: Optional[str] = None,
    stage_status: Optional[str] = None,
    project_id: Optional[str] = None,
) -> None:
    stage_id = stage_id or "pipeline"
    stage_status = stage_status or ("done" if status == "done" else "processing")

    payload: Dict[str, Any] = {
        "status": status,
        "stage_id": stage_id,
        "stage_status": stage_status,
    }
    if project_id is not None:
        payload["project_id"] = project_id
    if result_key is not None:
        payload["result_key"] = result_key
    if error is not None:
        payload["error"] = error
    if metadata is not None:
        payload["metadata"] = metadata

    target_url = normalize_callback_url(callback_url)

    try:
        resp = http.post(target_url, json=payload, timeout=30)
    except requests.RequestException as exc:
        raise JobProcessingError(f"Callback request failed: {exc}") from exc

    if not resp.ok:
        raise JobProcessingError(
            f"Callback responded with {resp.status_code}: {resp.text[:200]}"
        )


def ensure_workdir(job_id: str) -> str:
    workdir = os.path.join("/app/data", job_id)
    os.makedirs(workdir, exist_ok=True)
    return workdir


def send_callback(
    callback_url: str,
    status: str,
    message: str,
    stage: str | None = None,
    metadata: dict | None = None,
):
    """백엔드로 진행 상황 콜백을 보냅니다."""
    import logging
    import requests

    try:
        payload = {"status": status, "message": message}

        # 메타데이터 구성
        callback_metadata = metadata or {}
        if stage:
            callback_metadata["stage"] = stage

        if callback_metadata:
            payload["metadata"] = callback_metadata

        response = requests.post(callback_url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info(
            f"Sent callback to {callback_url} with status: {status}, stage: {stage or 'N/A'}"
        )
    except requests.RequestException as e:
        logging.error(f"Failed to send callback to {callback_url}: {e}")


def resolve_output_prefix(
    project_id: str | None, job_id: str, override: str | None
) -> str:
    """결과물을 저장할 기본 경로를 계산합니다."""
    if override:
        return override.rstrip("/")
    if project_id:
        return f"projects/{project_id}/outputs/{job_id}"
    return f"jobs/{job_id}/outputs"


def parse_bool(value) -> bool:
    """값을 boolean으로 변환합니다."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_positive_int(value, field_name: str) -> int | None:
    """Optional int parser that tolerates strings and invalid inputs."""
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        import logging

        logging.warning(
            "Ignoring %s=%r because it is not an integer", field_name, value
        )
        return None
    if parsed < 1:
        import logging

        logging.warning("Ignoring %s=%r because it must be >= 1", field_name, value)
        return None
    return parsed
