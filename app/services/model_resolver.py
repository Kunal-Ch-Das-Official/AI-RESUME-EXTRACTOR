"""Keeps Gemini model names valid even when Google retires or renames models.

Google shuts models down on a schedule (e.g. gemini-2.5-flash was retired on
2026-06-17), after which every call returns 404. Instead of hard failing, the
resolver asks the API which models this key can use, and substitutes the best
available stable Flash model when a configured one is gone. A loud warning is
logged so the .env can be updated.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from app.services.gemini_client import get_gemini_client

logger = logging.getLogger(__name__)

_TTL = 3600.0
_NAME = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash(-lite)?$")  # stable names, no previews

_available: set[str] | None = None
_fetched_at = 0.0
_lock: asyncio.Lock | None = None
_warned: set[str] = set()


async def _fetch() -> set[str]:
    names: set[str] = set()
    pager = await get_gemini_client().aio.models.list()
    async for model in pager:
        actions = model.supported_actions or []
        if actions and "generateContent" not in actions:
            continue
        names.add((model.name or "").removeprefix("models/"))
    return names


async def available_models(force: bool = False) -> set[str] | None:
    """Models this key can call, cached for an hour. None if the lookup fails."""
    global _available, _fetched_at, _lock
    if _lock is None:
        _lock = asyncio.Lock()
    async with _lock:
        if not force and _available is not None and time.monotonic() - _fetched_at < _TTL:
            return _available
        try:
            _available = await asyncio.wait_for(_fetch(), timeout=8)
            _fetched_at = time.monotonic()
        except Exception as exc:  # noqa: BLE001 - listing is best effort
            logger.warning("Could not list Gemini models (%s); using configured names", type(exc).__name__)
            return _available
        return _available


def _rank(name: str) -> tuple[float, int]:
    m = _NAME.match(name)
    return (float(m.group(1)), 0 if m.group(2) else 1) if m else (-1.0, 0)


def best_models(available: set[str]) -> tuple[str | None, str | None]:
    """(best full flash model, best flash-lite model) among stable names."""
    flash = sorted((n for n in available if (m := _NAME.match(n)) and not m.group(2)), key=_rank)
    lite = sorted((n for n in available if (m := _NAME.match(n)) and m.group(2)), key=_rank)
    return (flash[-1] if flash else None, lite[-1] if lite else None)


async def resolve(configured: str, *, lite: bool = False, force: bool = False) -> str:
    """Return ``configured`` if it exists, otherwise the best available substitute."""
    available = await available_models(force=force)
    if not available or configured in available:
        return configured
    flash, flash_lite = best_models(available)
    substitute = (flash_lite or flash) if lite else (flash or flash_lite)
    if not substitute:
        return configured
    if configured not in _warned:
        _warned.add(configured)
        logger.error(
            "Gemini model %r is not available for this API key. Using %r instead. "
            "Set GEMINI_MODEL / GEMINI_FALLBACK_MODEL in .env. Available flash models: %s",
            configured, substitute, ", ".join(sorted(n for n in available if _NAME.match(n))),
        )
    return substitute
