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