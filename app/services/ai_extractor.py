"""AI resume extraction: one compact call, validated and converted in code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.services.ai_schema import parse_wire, wire_to_payload
from app.services.llm import LLMUnavailable, get_router, make_cache_key
from app.services.prompts import SYSTEM, build_extraction_prompt, prepare_resume_text

CACHE_TTL_SECONDS = 900  # re-uploading the same resume must not cost tokens again


@dataclass
class AIExtraction:
    payload: dict[str, Any]  # ResumeInfo-shaped dict
    provider: str
    model: str
    cached: bool = False


class AIExtractor:
    async def extract(self, raw_text: str) -> AIExtraction:
        settings = get_settings()
        if not settings.any_ai_configured:
            raise LLMUnavailable("Configure GEMINI_API_KEY or OPENROUTER_API_KEY.")

        text = prepare_resume_text(raw_text, settings.ai_max_input_chars)
        result = await get_router().generate_json(
            build_extraction_prompt(text),
            lambda data: wire_to_payload(parse_wire(data), text),
            task="resume_extraction",
            system=SYSTEM,
            max_tokens=settings.ai_extract_max_output_tokens,
            temperature=0.0,
            deadline_seconds=settings.ai_extract_deadline_seconds,
            timeout_scale=4.0,  # extraction legitimately needs longer than short tasks
            cache_key=make_cache_key("extract", text),
            cache_ttl=CACHE_TTL_SECONDS,
        )
        return AIExtraction(result.value, result.provider, result.model, result.cached)
