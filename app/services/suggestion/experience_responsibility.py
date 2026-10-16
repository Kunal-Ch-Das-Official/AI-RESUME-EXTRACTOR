"""Responsibility bullet suggestions for one job entry (token-lean)."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import get_settings
from app.services.llm import LLMUnavailable, get_router, make_cache_key

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30
OPTION_TITLES = ["Core responsibilities", "Delivery & quality", "Collaboration & communication"]
FOCUS = [
    "day-to-day core duties",
    "delivery, quality and impact (no numbers)",
    "teamwork, stakeholders and communication",
]
PRESENT_RE = re.compile(r"\b(present|current|currently|ongoing|till date|to date|now)\b", re.I)

SYSTEM = "You write resume bullets. Reply with one JSON object only."


def _clean_text(value: Any, field: str, max_length: int, required: bool = True) -> str | None:
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
    present = not data.get("end_date") or bool(PRESENT_RE.search(data["end_date"]))
    tense = "present" if present else "past"
    focus = "\n".join(f"{i}. {f}" for i, f in enumerate(FOCUS, 1))
    end = data.get("end_date") or "Present"
    return (
        'Write resume bullets for this job. Return {"options":[[b,b,b,b],[...],[...]]}: '
        "3 options, 4 bullets each (8-20 words), focused on:\n"
        f"{focus}\n"
        f"Style: ATS-friendly, start with an action verb, {tense} tense, no first person, "
        "no markdown. Stay generic to what the title/company/type imply; never invent metrics, "
        "tools, technologies, team sizes, clients or outcomes.\n"
        f"Job: {data['job_title']} | {data['company']} | {data.get('location') or 'n/a'} | "
        f"{data['job_conditions']}, {data['job_type']} | {data['start_date']} - {end}"
    )


def _parse(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("options")
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError("expected three options")
    options = []
    for index, item in enumerate(raw[:3]):
        bullets_raw = item.get("responsibilities") if isinstance(item, dict) else item
        if not isinstance(bullets_raw, list):
            raise ValueError("option is not a list of bullets")
        bullets = list(
            dict.fromkeys(
                " ".join(b.split())[:240].lstrip("-•* ")
                for b in bullets_raw
                if isinstance(b, str) and b.strip()
            )
        )[:5]
        if len(bullets) < 3:
            raise ValueError("each option needs at least three bullets")
        options.append({"title": OPTION_TITLES[index], "responsibilities": bullets})
    return {"options": options}


def clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "job_title": _clean_text(payload.get("job_title"), "job_title", 120),
        "company": _clean_text(payload.get("company"), "company", 160),
        "location": _clean_text(payload.get("location"), "location", 160, required=False),
        "job_conditions": _clean_text(payload.get("job_conditions"), "job_conditions", 40),
        "job_type": _clean_text(payload.get("job_type"), "job_type", 50),
        "start_date": _clean_text(payload.get("start_date"), "start_date", 40),
        "end_date": _clean_text(payload.get("end_date"), "end_date", 40, required=False),
    }


async def generate_experience_responsibilities(
    payload: dict[str, Any], *, regenerate: bool = False
) -> dict[str, Any]:
    data = clean_payload(payload)
    settings = get_settings()
    if not settings.any_ai_configured:
        raise LLMUnavailable("No AI provider is configured for experience responsibilities.")
    result = await get_router().generate_json(
        _build_prompt(data),
        _parse,
        task="experience_responsibility",
        system=SYSTEM,
        max_tokens=700,
        temperature=0.5,
        deadline_seconds=settings.ai_short_deadline_seconds,
        cache_key=None if regenerate else make_cache_key("er", data),
        cache_ttl=0 if regenerate else CACHE_TTL_SECONDS,
    )
    return result.value
