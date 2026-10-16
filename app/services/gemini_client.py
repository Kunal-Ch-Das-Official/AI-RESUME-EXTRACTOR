"""One shared Gemini client (connection pooling) instead of one per request."""

from __future__ import annotations

from functools import lru_cache

from google import genai
from google.genai import types

from app.config import get_settings


@lru_cache
def get_gemini_client() -> genai.Client:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    # Real deadlines are enforced by asyncio.wait_for in the LLM router; this
    # is only a generous transport-level safety net (milliseconds).
    return genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(timeout=90_000),
    )


def reset_gemini_client() -> None:
    get_gemini_client.cache_clear()
