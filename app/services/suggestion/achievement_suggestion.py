"""Achievement bullet suggestions for one award/achievement entry (token-lean)."""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.services.llm import LLMUnavailable, get_router, make_cache_key

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30
OPTION_TITLES = [
    "Impact & Recognition",
    "Technical Scope & Execution",
    "Leadership & Distinction",
]
FOCUS = [
    "measurable impact, milestone significance, and core recognition",
    "technical or domain execution, competition rigor, and problem-solving",
    "peer distinction, leadership, and standout contributions",
]

SYSTEM = "You write resume bullets for career awards and achievements. Reply with one JSON object only."


def _clean_text(
    value: Any, field: str, max_length: int, required: bool = True
) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"{field} is required.")
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text.")
    value = " ".join(value.split())
    if required and not value:
        raise ValueError(f"{field} is required.")
    if len(value) > max_length:
        raise ValueError(f"{field} must be {max_length} characters or fewer.")
    return value or None


def _build_prompt(data: dict[str, Any]) -> str:
    focus = "\n".join(f"{i}. {f}" for i, f in enumerate(FOCUS, 1))
    title = data["title"]
    initial_desc = data.get("description", "")

    return (
        "Write concise resume accomplishment bullets for this achievement based on its draft description. "
        'Return {"options":[[b,b],[...],[...]]}: '
        "3 options, 2 to 3 bullets each (10-24 words), focused on:\n"
        f"{focus}\n"
        "Style: ATS-friendly, start with a strong action verb (e.g. Awarded, Recognized, Built, Selected), "
        "past tense, no first person pronouns, no markdown formatting. "
        "Elevate the user's initial draft to sound distinguished, clear, and professional. "
        "Do NOT invent fabricated rankings, percentages, prize money, or metrics not present in the draft.\n"
        f"Achievement Title: {title}\n"
        f"Draft Description: {initial_desc}"
    )


def _parse(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("options")
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("expected three options")
    options = []
    for index, item in enumerate(raw[:3]):
        bullets_raw = item.get("bullets") if isinstance(item, dict) else item
        if not isinstance(bullets_raw, list):
            raise ValueError("option is not a list of bullets")
        bullets = list(
            dict.fromkeys(
                " ".join(b.split())[:260].lstrip("-•* ")
                for b in bullets_raw
                if isinstance(b, str) and b.strip()
            )
        )[:4]
        if len(bullets) < 1:
            raise ValueError("each option needs at least one bullet")
        options.append(
            {
                "title": (
                    OPTION_TITLES[index]
                    if index < len(OPTION_TITLES)
                    else f"Suggestion {index + 1}"
                ),
                "bullets": bullets,
            }
        )
    return {"options": options}


def clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": _clean_text(payload.get("title"), "title", 160),
        "description": _clean_text(payload.get("description"), "description", 3000),
    }


async def generate_achievement_suggestions(
    payload: dict[str, Any], *, regenerate: bool = False
) -> dict[str, Any]:
    data = clean_payload(payload)
    settings = get_settings()
    if not settings.any_ai_configured:
        raise LLMUnavailable(
            "No AI provider is configured for achievement suggestions."
        )
    result = await get_router().generate_json(
        _build_prompt(data),
        _parse,
        task="achievement_suggestion",
        system=SYSTEM,
        max_tokens=650,
        temperature=0.5,
        deadline_seconds=settings.ai_short_deadline_seconds,
        cache_key=None if regenerate else make_cache_key("ach_sug", data),
        cache_ttl=0 if regenerate else CACHE_TTL_SECONDS,
    )
    return result.value
