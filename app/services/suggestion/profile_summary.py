"""Profile-summary suggestions.

Token strategy: the model only writes the four summary texts. Titles/styles are
fixed server-side (they were ~20% of the old output), thinking is disabled, the
prompt is ~120 tokens, and identical requests inside 30s are served from cache.
"""

from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings
from app.services.llm import LLMUnavailable, get_router, make_cache_key

logger = logging.getLogger(__name__)

MAX_DRAFT_CHARS = 5_000
MIN_DRAFT_CHARS = 10
MAX_ROLE_CHARS = 120
CACHE_TTL_SECONDS = 30  # absorbs double clicks; "regenerate" bypasses it

# (title, style, writing instruction). Order matters: it is the output order.
STYLES: list[tuple[str, str, str]] = [
    ("Concise", "Direct and compact", "30-45 words, one tight paragraph"),
    ("Impact-focused", "Leads with strengths", "50-70 words, open with strongest value"),
    ("Detailed", "Fuller professional context", "70-90 words, richer context"),
    ("Keyword-optimized", "ATS keyword-friendly", "45-70 words, reuse role/skill terms from the draft and target role"),
]
SUMMARY_COUNT = len(STYLES)

SYSTEM = "You are an expert resume editor. Reply with one JSON object only."


def _build_prompt(draft: str, target_role: str) -> str:
    styles = "\n".join(f"{i}. {title}: {rule}" for i, (title, _, rule) in enumerate(STYLES, 1))
    return (
        f'Rewrite the draft profile summary for a resume targeting "{target_role}".\n'
        f'Return {{"summaries":["...","...","...","..."]}} with {SUMMARY_COUNT} versions in this order:\n'
        f"{styles}\n"
        "Rules: use only facts stated in the draft; never invent employers, years, degrees, "
        "skills, metrics, tools, certifications or achievements; the target role only guides "
        "emphasis. Resume style (no \"I\"), plain text, same language as the draft, "
        "better grammar and ATS readability.\n"
        f'Draft: """{draft}"""'
    )


def _parse(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("summaries")
    if raw is None:  # tolerate {"options":[{"summary":...}]} from weaker models
        raw = payload.get("options")
    if not isinstance(raw, list):
        raise ValueError("missing summaries list")
    texts: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            item = item.get("summary") or item.get("text")
        if isinstance(item, str) and item.strip():
            texts.append(" ".join(item.split()))
    if len(texts) < SUMMARY_COUNT:
        raise ValueError("fewer than four valid summaries")
    return {
        "options": [
            {"title": title, "style": style, "summary": text}
            for (title, style, _), text in zip(STYLES, texts[:SUMMARY_COUNT])
        ]
    }


def validate_inputs(draft: str, target_role: str) -> tuple[str, str]:
    draft = (draft or "").strip()
    target_role = " ".join((target_role or "").split())
    if not draft:
        raise ValueError("An initial profile draft is required.")
    if len(draft) < MIN_DRAFT_CHARS:
        raise ValueError(f"Please write at least {MIN_DRAFT_CHARS} characters in your profile draft.")
    if len(draft) > MAX_DRAFT_CHARS:
        raise ValueError(f"The profile draft must be {MAX_DRAFT_CHARS} characters or fewer.")
    if not target_role:
        raise ValueError("A target role is required.")
    if len(target_role) > MAX_ROLE_CHARS:
        raise ValueError(f"The target role must be {MAX_ROLE_CHARS} characters or fewer.")
    return draft, target_role


async def generate_profile_summaries(
    draft: str, target_role: str, *, regenerate: bool = False
) -> dict[str, Any]:
    """Return four summary alternatives. Raises ValueError (bad input) or LLMUnavailable."""
    draft, target_role = validate_inputs(draft, target_role)
    settings = get_settings()
    if not settings.any_ai_configured:
        raise LLMUnavailable("Configure GEMINI_API_KEY or enable the OpenRouter fallback.")

    result = await get_router().generate_json(
        _build_prompt(draft, target_role),
        _parse,
        task="profile_summary",
        system=SYSTEM,
        max_tokens=900,
        temperature=0.5,
        deadline_seconds=settings.ai_short_deadline_seconds,
        cache_key=None if regenerate else make_cache_key("ps", draft, target_role.lower()),
        cache_ttl=0 if regenerate else CACHE_TTL_SECONDS,
    )
    return result.value
