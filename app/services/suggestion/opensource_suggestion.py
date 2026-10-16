"""Open source contribution bullet suggestions for one contribution entry (token-lean)."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import get_settings
from app.services.llm import LLMUnavailable, get_router, make_cache_key

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30
OPTION_TITLES = [
    "Core Code & Feature Implementation",
    "Bug Fixes, Performance & Testing",
    "Community Collaboration & Maintenance",
]
FOCUS = [
    "core code contributions, architectural features, and pull request implementations",
    "bug resolutions, testing coverage, performance optimizations, and refactoring",
    "code reviews, issue triage, documentation improvements, and open-source collaboration",
]
PRESENT_RE = re.compile(
    r"\b(present|current|currently|ongoing|till date|to date|now)\b", re.I
)

SYSTEM = "You write resume bullets for open-source software contributions. Reply with one JSON object only."


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
    duration_str = data.get("duration") or ""
    present = bool(PRESENT_RE.search(duration_str))
    tense = "present" if present else "past"
    focus = "\n".join(f"{i}. {f}" for i, f in enumerate(FOCUS, 1))
    initial_desc = data.get("description", "")
    project_name = data["project_name"]

    return (
        "Write concise resume accomplishment bullets for this open-source contribution based on its draft description. "
        'Return {"options":[[b,b,b],[...],[...]]}: '
        "3 options, 2 to 3 bullets each (10-24 words), focused on:\n"
        f"{focus}\n"
        f"Style: ATS-friendly, start with an action verb, {tense} tense, no first person pronouns, "
        "no markdown formatting. Elevate the user's initial draft to sound technically precise, impactful, and professional. "
        "Do NOT invent fabricated star counts, download numbers, PR numbers, or tech stacks not mentioned.\n"
        f"Project Name: {project_name}\n"
        f"Duration: {duration_str or 'n/a'}\n"
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
        "project_name": _clean_text(payload.get("project_name"), "project_name", 150),
        "duration": _clean_text(payload.get("duration"), "duration", 80, required=True),
        "description": _clean_text(
            payload.get("description"), "description", 3000, required=True
        ),
    }


async def generate_opensource_suggestions(
    payload: dict[str, Any], *, regenerate: bool = False
) -> dict[str, Any]:
    data = clean_payload(payload)
    settings = get_settings()
    if not settings.any_ai_configured:
        raise LLMUnavailable(
            "No AI provider is configured for open source suggestions."
        )
    result = await get_router().generate_json(
        _build_prompt(data),
        _parse,
        task="opensource_suggestion",
        system=SYSTEM,
        max_tokens=650,
        temperature=0.5,
        deadline_seconds=settings.ai_short_deadline_seconds,
        cache_key=None if regenerate else make_cache_key("os_sug", data),
        cache_ttl=0 if regenerate else CACHE_TTL_SECONDS,
    )
    return result.value
