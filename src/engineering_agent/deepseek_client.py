"""Small standard-library client for DeepSeek JSON decisions."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .decision_loop import ModelResult


@dataclass(slots=True)
class DeepSeekChatClient:
    api_key: str
    model: str
    base_url: str = "https://api.deepseek.com"
    timeout_seconds: float = 180.0
    max_tokens: int = 8192

    @classmethod
    def from_environment(cls) -> "DeepSeekChatClient":
        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        model = os.environ.get("DEEPSEEK_MODEL", "").strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured")
        if not model:
            raise RuntimeError(
                "DEEPSEEK_MODEL is not configured; the V5 run will not "
                "silently choose a model version"
            )
        return cls(
            api_key=api_key,
            model=model,
            base_url=os.environ.get(
                "DEEPSEEK_BASE_URL",
                "https://api.deepseek.com",
            ).rstrip("/"),
            timeout_seconds=float(
                os.environ.get("DEEPSEEK_TIMEOUT_SECONDS", "180")
            ),
            max_tokens=int(
                os.environ.get("DEEPSEEK_MAX_TOKENS", "8192")
            ),
        )

    def complete_json(
        self,
        messages: Sequence[Mapping[str, str]],
    ) -> ModelResult:
        body = {
            "model": self.model,
            "messages": list(messages),
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                raw_bytes = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DeepSeek HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"DeepSeek connection failed: {exc.reason}"
            ) from exc

        raw: dict[str, Any] = json.loads(raw_bytes.decode("utf-8"))
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError("DeepSeek response contains no choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("DeepSeek returned empty JSON content")
        payload = json.loads(content)
        if not isinstance(payload, Mapping):
            raise RuntimeError("DeepSeek JSON content is not an object")
        return ModelResult(
            payload=dict(payload),
            raw_response=raw,
            provider="deepseek",
            model=str(raw.get("model") or self.model),
        )
