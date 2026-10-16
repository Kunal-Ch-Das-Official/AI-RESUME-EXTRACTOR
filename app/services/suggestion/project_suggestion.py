"""Project description bullet suggestions for one project entry (token-lean)."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import get_settings
from app.services.llm import LLMUnavailable, get_router, make_cache_key

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30
OPTION_TITLES = [
    "Architecture & Core Implementation",
    "Performance, Features & Scalability",
    "Engineering Quality & Deliverables",
]
FOCUS = [
    "core architecture, feature implementation, and problem solved",
    "performance optimizations, data flows, and technical impact (no fabricated statistics)",
    "best practices, modular design, clean engineering, and integration",
]
PRESENT_RE = re.compile(
    r"\b(present|current|currently|ongoing|till date|to date|now)\b", re.I
)

SYSTEM = "You write resume bullets for engineering projects. Reply with one JSON object only."


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


def _clean_tech_stack(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            tech = " ".join(item.split())[:50]
            if tech and tech.lower() not in [t.lower() for t in cleaned]:
                cleaned.append(tech)
    return cleaned[:15]


def _build_prompt(data: dict[str, Any]) -> str:
    present = (
        data.get("is_working") is True
        or not data.get("end_date")
        or bool(PRESENT_RE.search(str(data.get("end_date", ""))))
    )
    tense = "present" if present else "past"
    focus = "\n".join(f"{i}. {f}" for i, f in enumerate(FOCUS, 1))
    tech_stack_str = ", ".join(data.get("tech_stack", [])) or "Modern technologies"
    initial_desc = data.get("description", "")

    return (
        "Write resume bullets for this project based on its draft description. "
        'Return {"options":[[b,b,b],[...],[...]]}: '
        "3 options, 3 to 4 bullets each (10-24 words), focused on:\n"
        f"{focus}\n"
        f"Style: ATS-friendly, start with a strong action verb, {tense} tense, no first person pronouns, "
        "no markdown formatting. Polish the user's initial draft to make it professional and impactful. "
        "Do NOT invent fabricated numbers, metrics, or technologies not mentioned.\n"
        f"Project Name: {data['name']}\n"
        f"Tech Stack: {tech_stack_str}\n"
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
        )[:5]
        if len(bullets) < 2:
            raise ValueError("each option needs at least two bullets")
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
        "name": _clean_text(payload.get("name"), "name", 150),
        "description": _clean_text(payload.get("description"), "description", 3000),
        "tech_stack": _clean_tech_stack(payload.get("tech_stack")),
        "is_working": bool(payload.get("is_working", False)),
        "start_date": _clean_text(
            payload.get("start_date"), "start_date", 40, required=False
        ),
        "end_date": _clean_text(
            payload.get("end_date"), "end_date", 40, required=False
        ),
    }


async def generate_project_suggestions(
    payload: dict[str, Any], *, regenerate: bool = False
) -> dict[str, Any]:
    data = clean_payload(payload)
    settings = get_settings()
    if not settings.any_ai_configured:
        raise LLMUnavailable("No AI provider is configured for project suggestions.")
    result = await get_router().generate_json(
        _build_prompt(data),
        _parse,
        task="project_suggestion",
        system=SYSTEM,
        max_tokens=750,
        temperature=0.5,
        deadline_seconds=settings.ai_short_deadline_seconds,
        cache_key=None if regenerate else make_cache_key("proj_sug", data),
        cache_ttl=0 if regenerate else CACHE_TTL_SECONDS,
    )
    return result.value
