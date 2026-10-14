"""OpenRouter fallback used when Gemini is unavailable.

Uses the OpenAI-compatible OpenRouter chat-completions endpoint and validates
that the provider response is a JSON object. Recoverable JSON formatting errors
are repaired with json-repair; schema validation remains the caller's job.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from app.config import get_settings

try:
    from json_repair import repair_json
except ImportError:
    repair_json = None


class FallbackProviderError(RuntimeError):
    """The backup provider could not return usable JSON."""


def _extract_json_candidate(value: str) -> str:
    """Remove common Markdown fences and isolate the JSON object."""
    text = value.strip()
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text).strip()

    start = text.find("{")

    if start < 0:
        raise FallbackProviderError("OpenRouter did not return a JSON object.")

    end = text.rfind("}")

    if end >= start:
        return text[start : end + 1]

    # Allow the repair library to try recovering a missing closing brace.
    return text[start:]


def _json_object(value: str) -> str:
    """Return a normalized JSON object string or raise a useful error."""
    if not isinstance(value, str) or not value.strip():
        raise FallbackProviderError("OpenRouter returned an empty response.")

    candidate = _extract_json_candidate(value)

    try:
        parsed: Any = json.loads(candidate)

    except json.JSONDecodeError as original_error:
        if repair_json is None:
            raise FallbackProviderError(
                "OpenRouter returned malformed JSON. Install the "
                "json-repair package to attempt recovery."
            ) from original_error

        try:
            parsed = repair_json(
                candidate,
                return_objects=True,
            )
        except Exception as repair_error:
            raise FallbackProviderError(
                "OpenRouter returned malformed JSON and automatic " "repair failed."
            ) from repair_error

    if not isinstance(parsed, dict):
        raise FallbackProviderError("OpenRouter response must be a JSON object.")

    # Return strict, normalized JSON for downstream Pydantic validation.
    return json.dumps(
        parsed,
        ensure_ascii=False,
        separators=(",", ":"),
    )


class OpenRouterFallback:
    """Generate structured JSON through OpenRouter."""

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.ai_fallback_enabled:
            raise FallbackProviderError("OpenRouter fallback is disabled.")

        if not settings.openrouter_api_key:
            raise FallbackProviderError(
                "OpenRouter fallback is not configured: "
                "OPENROUTER_API_KEY is missing."
            )

        self.api_key = settings.openrouter_api_key
        self.base_url = settings.openrouter_base_url.rstrip("/")
        self.model = settings.openrouter_model

    def generate_json(self, prompt: str) -> str:
        """Call OpenRouter and return a syntactically valid JSON object."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("OpenRouter prompt cannot be empty.")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one complete, valid JSON object. "
                        "Do not use Markdown fences or explanatory text. "
                        "Do not omit required fields. Do not truncate the "
                        "response. Use empty arrays or null only when "
                        "appropriate to the requested schema."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_object",
            },
        }

        try:
            with httpx.Client(
                timeout=httpx.Timeout(
                    25.0,
                    connect=8.0,
                ),
                follow_redirects=True,
            ) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

                response.raise_for_status()
                body = response.json()

        except httpx.TimeoutException as exc:
            raise FallbackProviderError("OpenRouter request timed out.") from exc

        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            detail = exc.response.text[:500]

            raise FallbackProviderError(
                f"OpenRouter returned HTTP {status}: {detail}"
            ) from exc

        except httpx.HTTPError as exc:
            raise FallbackProviderError(
                f"OpenRouter network request failed: {exc}"
            ) from exc

        except ValueError as exc:
            raise FallbackProviderError(
                "OpenRouter returned a response that was not valid JSON."
            ) from exc

        try:
            choices = body["choices"]

            if not isinstance(choices, list) or not choices:
                raise KeyError("choices")

            choice = choices[0]
            message = choice["message"]
            content = message.get("content")
            finish_reason = choice.get("finish_reason")

        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise FallbackProviderError(
                "OpenRouter response did not contain the expected "
                "choices[0].message.content structure."
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise FallbackProviderError(
                "OpenRouter returned an empty message. "
                f"Finish reason: {finish_reason or 'unknown'}."
            )

        if finish_reason == "length":
            raise FallbackProviderError(
                "OpenRouter truncated the response because the output "
                "token limit was reached. Reduce the prompt size or use "
                "a model with a larger output limit."
            )

        return _json_object(content)
