"""Provider-agnostic, deadline-bounded JSON generation.

Every AI feature (profile summaries, responsibilities, resume extraction) goes
through ``LLMRouter.generate_json``. It gives them, in one place:

* a hard wall-clock deadline across ALL providers (asyncio cancellation, not
  just socket read timeouts, so a slow provider can never hold a request open);
* ordered failover: Gemini -> Gemini fallback model -> OpenRouter;
* a per-provider circuit breaker so a provider that is down is skipped
  instead of costing every user a timeout;
* validation inside the loop: unusable output falls through to the next
  provider instead of failing the request;
* a small TTL cache + single-flight so double clicks / retries never pay twice;
* token-usage logging so cost regressions are visible.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Generic, TypeVar

import httpx
from google.genai import errors as genai_errors
from google.genai import types

from app.config import Settings, get_settings
from app.services.gemini_client import get_gemini_client
from app.services import model_resolver

logger = logging.getLogger(__name__)

T = TypeVar("T")

try:  # optional dependency, listed in requirements.txt
    _repair_json = import_module("json_repair").repair_json
except Exception:  # pragma: no cover
    _repair_json = None


# --------------------------------------------------------------------- errors
class LLMError(RuntimeError):
    """Base class for AI-layer failures."""


class LLMUnavailable(LLMError):
    """No provider produced a usable answer inside the time budget."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


class LLMMisconfigured(LLMUnavailable):
    """Every provider that was tried failed because of configuration
    (bad API key or unknown model name), so retrying will not help."""


class ProviderError(LLMError):
    """A single provider attempt failed.

    ``fatal``  -> misconfiguration (bad key / model); don't retry for a while.
    ``counts`` -> counts toward the circuit breaker (outage, overload, timeout).
    """

    def __init__(self, message: str, *, fatal: bool = False, counts: bool = True):
        super().__init__(message)
        self.fatal = fatal
        self.counts = counts


class InvalidOutput(LLMError):
    """The provider answered, but not with usable JSON for this task."""


# ------------------------------------------------------------------ JSON utils
def parse_json_object(raw: str) -> dict[str, Any]:
    """Parse a JSON object from model output (fences / chatter / minor damage)."""
    text = (raw or "").strip().lstrip("\ufeff")
    if not text:
        raise InvalidOutput("empty response")
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text).strip()

    start = text.find("{")
    if start < 0:
        raise InvalidOutput("no JSON object in response")
    end = text.rfind("}")
    candidate = text[start : end + 1] if end > start else text[start:]

    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        if _repair_json is None:
            raise InvalidOutput("malformed JSON") from None
        try:
            value = _repair_json(candidate, return_objects=True)
        except Exception as exc:  # noqa: BLE001
            raise InvalidOutput("malformed JSON (repair failed)") from exc
    if not isinstance(value, dict):
        raise InvalidOutput("JSON root is not an object")
    return value


# ------------------------------------------------------------------- providers
@dataclass
class Completion:
    text: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None


class Provider:
    """Interface for one model endpoint."""

    name: str = "provider"
    model: str = ""
    timeout: float = 10.0

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None,
        max_tokens: int,
        temperature: float,
    ) -> Completion:  # pragma: no cover - interface
        raise NotImplementedError


def _generation(model: str) -> float:
    m = re.search(r"gemini-(\d+(?:\.\d+)?)", model.lower())
    return float(m.group(1)) if m else 0.0


def _thinking_config(settings: Settings, model: str) -> types.ThinkingConfig | None:
    """Keep hidden reasoning (billed as output tokens) to a minimum.

    2.x models take a token budget; 3.x models take a level instead.
    GEMINI_THINKING_BUDGET: 0 = as little as the model allows, -1 = model default.
    """
    budget = settings.gemini_thinking_budget
    if budget < 0:
        return None
    name, generation = model.lower(), _generation(model)
    if generation >= 3:
        level = types.ThinkingLevel.LOW if "pro" in name else types.ThinkingLevel.MINIMAL
        if budget > 0:
            level = types.ThinkingLevel.MEDIUM if budget > 2048 else types.ThinkingLevel.LOW
        return types.ThinkingConfig(thinking_level=level)
    if "2.5" in name:
        if "pro" in name:
            budget = max(budget, 128)  # 2.5 Pro cannot fully disable thinking
        return types.ThinkingConfig(thinking_budget=budget)
    return None


class GeminiProvider(Provider):
    def __init__(
        self, settings: Settings, model: str, timeout: float, label: str, *, lite: bool = False
    ):
        self.settings = settings
        self.model = model
        self.timeout = timeout
        self.name = label
        self.lite = lite

    async def _call(self, model, prompt, system, max_tokens, temperature):
        kwargs: dict[str, Any] = {}
        # Google recommends the default temperature for Gemini 3+ models;
        # lowering it can cause looping/degraded output.
        if _generation(model) < 3:
            kwargs["temperature"] = temperature
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            system_instruction=system,
            thinking_config=_thinking_config(self.settings, model),
            **kwargs,
        )
        return await get_gemini_client().aio.models.generate_content(
            model=model, contents=prompt, config=config
        )

    async def complete(self, prompt, *, system, max_tokens, temperature) -> Completion:
        # Swap in a working model name if the configured one was retired.
        self.model = await model_resolver.resolve(self.model, lite=self.lite)
        try:
            try:
                response = await self._call(self.model, prompt, system, max_tokens, temperature)
            except genai_errors.APIError as exc:
                if getattr(exc, "code", None) != 404:
                    raise
                # Model may have been retired since the list was cached: re-check once.
                model = await model_resolver.resolve(self.model, lite=self.lite, force=True)
                if model == self.model:
                    raise
                self.model = model
                response = await self._call(model, prompt, system, max_tokens, temperature)
        except genai_errors.APIError as exc:
            raise _from_genai_error(exc) from exc

        candidates = response.candidates or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        if finish is not None and getattr(finish, "name", str(finish)) == "MAX_TOKENS":
            raise ProviderError("Gemini output was truncated (MAX_TOKENS)", counts=False)
        try:
            text = response.text or ""
        except Exception:  # noqa: BLE001 - blocked / empty candidates
            text = ""
        if not text.strip():
            raise ProviderError(f"Gemini returned no text (finish={finish})", counts=False)
        usage = response.usage_metadata
        return Completion(
            text=text,
            prompt_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            thinking_tokens=getattr(usage, "thoughts_token_count", None),
        )


def _from_genai_error(exc: genai_errors.APIError) -> ProviderError:
    # NOTE: google-genai exposes the HTTP status as ``.code`` (there is no
    # ``.status_code``), which is why status-based retry logic must use it.
    code = getattr(exc, "code", None)
    label = f"Gemini HTTP {code} {getattr(exc, 'status', '')}".strip()
    if code in (401, 403, 404):
        return ProviderError(
            f"{label}: check GEMINI_API_KEY and GEMINI_MODEL (model may be retired)",
            fatal=True
        )
    if code == 400:
        return ProviderError(f"{label}: {str(exc.message)[:160]}", counts=False)
    return ProviderError(label)  # 429 / 5xx -> transient, counts toward breaker


class OpenRouterProvider(Provider):
    name = "openrouter"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.openrouter_model
        self.timeout = settings.openrouter_timeout_seconds
        self._url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"

    async def complete(self, prompt, *, system, max_tokens, temperature) -> Completion:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            # Reasoning tokens are billed/slow and unnecessary here.
            "reasoning": {"enabled": False},
        }
        try:
            response = await _http_client().post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter network error: {type(exc).__name__}") from exc

        if response.status_code in (401, 403):
            raise ProviderError("OpenRouter auth failed: check OPENROUTER_API_KEY", fatal=True)
        if response.status_code >= 400:
            snippet = " ".join(response.text.split())[:160]
            raise ProviderError(
                f"OpenRouter HTTP {response.status_code}: {snippet}",
                counts=response.status_code in (408, 429) or response.status_code >= 500,
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError("OpenRouter returned non-JSON HTTP body") from exc
        if isinstance(body.get("error"), dict):
            raise ProviderError(
                f"OpenRouter error: {str(body['error'].get('message', 'unknown'))[:160]}"
            )
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError("OpenRouter returned no choices")
        choice = choices[0]
        text = (choice.get("message") or {}).get("content")
        if choice.get("finish_reason") == "length":
            raise ProviderError("OpenRouter output was truncated", counts=False)
        if not isinstance(text, str) or not text.strip():
            raise ProviderError("OpenRouter returned empty content", counts=False)
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            prompt_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )


_client: httpx.AsyncClient | None = None


def _http_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=5.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            follow_redirects=True,
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None


# -------------------------------------------------------------- breaker / cache
class CircuitBreaker:
    def __init__(self, threshold: int, cooldown: float) -> None:
        self.threshold = max(1, threshold)
        self.cooldown = cooldown
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}

    def is_open(self, name: str) -> bool:
        until = self._open_until.get(name)
        if until is None:
            return False
        if time.monotonic() >= until:  # half-open: allow one trial
            self._open_until.pop(name, None)
            self._failures[name] = self.threshold - 1
            return False
        return True

    def success(self, name: str) -> None:
        self._failures.pop(name, None)
        self._open_until.pop(name, None)

    def failure(self, name: str, *, fatal: bool = False) -> None:
        count = self._failures.get(name, 0) + 1
        self._failures[name] = count
        if fatal:
            self._open_until[name] = time.monotonic() + max(self.cooldown * 6, 120.0)
        elif count >= self.threshold:
            self._open_until[name] = time.monotonic() + self.cooldown
            logger.warning("llm breaker opened provider=%s cooldown=%ss", name, self.cooldown)


class TTLCache:
    def __init__(self, max_items: int = 256) -> None:
        self.max_items = max_items
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Any | None:
        item = self._data.get(key)
        if item is None:
            return None
        expires, value = item
        if time.monotonic() >= expires:
            self._data.pop(key, None)
            return None
        self._data.move_to_end(key)
        return copy.deepcopy(value)

    def set(self, key: str, value: Any, ttl: float) -> None:
        self._data[key] = (time.monotonic() + ttl, copy.deepcopy(value))
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


def make_cache_key(task: str, *parts: Any) -> str:
    digest = hashlib.sha256(
        json.dumps([task, *parts], ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    return f"{task}:{digest[:32]}"


# ---------------------------------------------------------------------- router
@dataclass
class LLMResult(Generic[T]):
    value: T
    provider: str
    model: str
    cached: bool = False
    attempts: list[str] = field(default_factory=list)


class LLMRouter:
    def __init__(self, providers: list[Provider], settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.providers = providers
        self.breaker = CircuitBreaker(
            self.settings.ai_breaker_threshold, self.settings.ai_breaker_cooldown_seconds
        )
        self.cache = TTLCache()
        self._inflight: dict[str, asyncio.Task] = {}
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def configured(self) -> bool:
        return bool(self.providers)

    def _sem(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(max(1, self.settings.ai_max_concurrency))
        return self._semaphore

    async def generate_json(
        self,
        prompt: str,
        parse: Callable[[dict[str, Any]], T],
        *,
        task: str,
        system: str | None = None,
        max_tokens: int = 800,
        temperature: float = 0.3,
        deadline_seconds: float | None = None,
        timeout_scale: float = 1.0,
        cache_key: str | None = None,
        cache_ttl: float = 0.0,
    ) -> LLMResult[T]:
        """Return ``parse(json)`` from the first provider that yields valid output.

        ``parse`` must raise (ValueError / pydantic.ValidationError) on unusable
        data; that sends the request to the next provider.
        Raises ``LLMUnavailable`` when the budget or all providers are exhausted.
        """
        if not self.providers:
            raise LLMUnavailable("No AI provider is configured.")
        deadline_seconds = deadline_seconds or self.settings.ai_short_deadline_seconds

        if cache_key and cache_ttl > 0:
            hit = self.cache.get(cache_key)
            if hit is not None:
                logger.info("llm task=%s cache=hit", task)
                return LLMResult(hit["value"], hit["provider"], hit["model"], cached=True)

        if cache_key:  # single-flight: identical concurrent requests share one call
            running = self._inflight.get(cache_key)
            if running is not None:
                return await asyncio.shield(running)

        coro = self._run(
            prompt, parse, task, system, max_tokens, temperature,
            deadline_seconds, timeout_scale,
        )
        if not cache_key:
            return await coro

        loop_task = asyncio.ensure_future(coro)
        self._inflight[cache_key] = loop_task
        try:
            result = await asyncio.shield(loop_task)
        finally:
            if loop_task.done():
                self._inflight.pop(cache_key, None)
        if cache_ttl > 0:
            self.cache.set(
                cache_key,
                {"value": result.value, "provider": result.provider, "model": result.model},
                cache_ttl,
            )
        return result

    async def _run(
        self, prompt, parse, task, system, max_tokens, temperature,
        deadline_seconds, timeout_scale,
    ) -> LLMResult:
        started = time.monotonic()
        deadline = started + deadline_seconds
        failures: list[str] = []
        fatal_failures = 0
        tried = 0

        for provider in self.providers:
            remaining = deadline - time.monotonic()
            if remaining < 1.5:
                failures.append("time budget exhausted")
                break
            if self.breaker.is_open(provider.name):
                failures.append(f"{provider.name}: skipped (recent failures)")
                continue

            attempt_timeout = min(provider.timeout * timeout_scale, remaining)
            tried += 1
            t0 = time.monotonic()
            try:
                async with self._sem():
                    completion = await asyncio.wait_for(
                        provider.complete(
                            prompt,
                            system=system,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        ),
                        timeout=attempt_timeout,
                    )
                value = parse(parse_json_object(completion.text))
            except asyncio.TimeoutError:
                self.breaker.failure(provider.name)
                failures.append(f"{provider.name}: timed out after {attempt_timeout:.0f}s")
                logger.warning(
                    "llm task=%s provider=%s timeout=%.0fs", task, provider.name, attempt_timeout
                )
                continue
            except ProviderError as exc:
                if exc.counts or exc.fatal:
                    self.breaker.failure(provider.name, fatal=exc.fatal)
                fatal_failures += exc.fatal
                failures.append(f"{provider.name}: {exc}")
                logger.warning("llm task=%s provider=%s failed: %s", task, provider.name, exc)
                continue
            except (InvalidOutput, ValueError) as exc:  # pydantic errors subclass ValueError
                failures.append(f"{provider.name}: invalid output ({type(exc).__name__})")
                logger.warning(
                    "llm task=%s provider=%s invalid_output=%s", task, provider.name,
                    str(exc)[:160],
                )
                continue
            except Exception as exc:  # noqa: BLE001 - never let one provider kill the chain
                self.breaker.failure(provider.name)
                failures.append(f"{provider.name}: {type(exc).__name__}")
                logger.exception("llm task=%s provider=%s unexpected error", task, provider.name)
                continue

            self.breaker.success(provider.name)
            logger.info(
                "llm task=%s provider=%s model=%s ms=%d in=%s out=%s think=%s",
                task, provider.name, provider.model,
                (time.monotonic() - t0) * 1000,
                completion.prompt_tokens, completion.output_tokens, completion.thinking_tokens,
            )
            return LLMResult(value, provider.name, provider.model, attempts=failures)

        logger.error("llm task=%s failed after %.1fs: %s", task, time.monotonic() - started, failures)
        if tried and fatal_failures == tried:
            raise LLMMisconfigured(
                "AI providers are misconfigured (invalid API key or unknown model).", failures
            )
        raise LLMUnavailable("All AI providers failed or timed out.", failures)


# --------------------------------------------------------------------- factory
_router: LLMRouter | None = None


def build_providers(settings: Settings) -> list[Provider]:
    providers: list[Provider] = []
    if settings.gemini_api_key:
        providers.append(
            GeminiProvider(settings, settings.gemini_model, settings.gemini_timeout_seconds, "gemini")
        )
        fallback = (settings.gemini_fallback_model or "").strip()
        if fallback and fallback != settings.gemini_model:
            providers.append(
                GeminiProvider(
                    settings, fallback, settings.gemini_fallback_timeout_seconds,
                    "gemini-fallback", lite="lite" in fallback.lower(),
                )
            )
    if settings.openrouter_enabled:
        providers.append(OpenRouterProvider(settings))
    return providers


def get_router() -> LLMRouter:
    global _router
    if _router is None:
        settings = get_settings()
        _router = LLMRouter(build_providers(settings), settings)
    return _router


def reset_router() -> None:
    global _router
    _router = None
