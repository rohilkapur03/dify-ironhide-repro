"""Ironhide adapter for the deployed Dify File Translation application."""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_USER = "ironhide-ci"
_TIMEOUT_SECONDS = 120
_USER_AGENT = "ironhide-dify-adapter/1.0"


class DifyAdapterError(RuntimeError):
    """Raised when the Dify application cannot complete an Ironhide episode."""


def _settings() -> tuple[str, str]:
    api_url = os.environ.get("DIFY_API_URL", "https://api.dify.ai/v1").rstrip("/")
    api_key = os.environ.get("DIFY_API_KEY", "")
    if not api_key:
        raise DifyAdapterError("DIFY_API_KEY is required")
    return api_url, api_key


def _request(request: Request) -> dict[str, Any]:
    try:
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise DifyAdapterError(f"Dify returned HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise DifyAdapterError(f"Dify request failed: {exc}") from exc


def _upload_episode(api_url: str, api_key: str, prompt: dict[str, Any]) -> str:
    boundary = f"ironhide-{uuid.uuid4().hex}"
    filename = "ironhide-episode.txt"
    content = json.dumps(prompt, ensure_ascii=False, indent=2).encode("utf-8")
    mime_type = mimetypes.guess_type(filename)[0] or "text/plain"
    body = b"".join(
        [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"user\"\r\n\r\n{_USER}\r\n".encode(),
            (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"file\"; filename=\"{filename}\"\r\n"
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode(),
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    result = _request(
        Request(
            f"{api_url}/files/upload",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )
    )
    file_id = result.get("id")
    if not isinstance(file_id, str) or not file_id:
        raise DifyAdapterError("Dify file upload did not return an id")
    return file_id


def _run_chat(api_url: str, api_key: str, file_id: str) -> str:
    payload = {
        "inputs": {
            "text": {
                "type": "document",
                "transfer_method": "local_file",
                "upload_file_id": file_id,
            },
            "target_language": "English",
        },
        "query": "Process the attached document according to the published application workflow.",
        "response_mode": "blocking",
        "conversation_id": "",
        "user": _USER,
        "files": [],
    }
    result = _request(
        Request(
            f"{api_url}/chat-messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": _USER_AGENT,
            },
            method="POST",
        )
    )
    answer = result.get("answer")
    if not isinstance(answer, str):
        raise DifyAdapterError("Dify chat response did not contain an answer")
    return answer


def _turn_number(prompt: dict[str, Any]) -> int:
    episode = prompt.get("episode")
    if isinstance(episode, dict):
        script = episode.get("script")
        if isinstance(script, list) and script:
            last = script[-1]
            if isinstance(last, dict) and isinstance(last.get("turn"), int):
                return last["turn"]
            return len(script)
    return 1


def agent(prompt: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    """Drive the configured Dify app for one Ironhide contract-v2 round."""
    del tools  # This deployed app exposes no tool-call surface through its API.
    api_url, api_key = _settings()
    file_id = _upload_episode(api_url, api_key, prompt)
    answer = _run_chat(api_url, api_key, file_id)

    if prompt.get("episode_type") == "behavioral":
        return {"detections": []}

    return {
        "actions": [
            {
                "turn": _turn_number(prompt),
                "action": "comply",
                "response": answer,
                "parsed": False,
            }
        ],
        "tool_calls": [],
    }
