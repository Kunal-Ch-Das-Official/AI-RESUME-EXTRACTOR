import asyncio
import time

import pytest

from app.config import Settings
from app.services.llm import (
    Completion, InvalidOutput, LLMRouter, LLMUnavailable, Provider, ProviderError,
    make_cache_key, parse_json_object,
)


class Fake(Provider):
    def __init__(self, name, behaviour, timeout=2.0):
        self.name, self.model, self.timeout = name, name, timeout
        self.behaviour, self.calls = behaviour, 0

    async def complete(self, prompt, *, system, max_tokens, temperature):
        self.calls += 1
        return await self.behaviour()


def ok(text='{"a": 1}'):
    async def run():
        return Completion(text=text)
    return run


def boom(message="HTTP 503", **kw):
    async def run():
        raise ProviderError(message, **kw)
    return run


def hang():
    async def run():
        await asyncio.sleep(30)
    return run


def router(*providers):
    return LLMRouter(list(providers), Settings(AI_BREAKER_THRESHOLD=2, AI_BREAKER_COOLDOWN_SECONDS=60))


def parse(d):
    if "a" not in d:
        raise ValueError("no a")
    return d["a"]


def run(coro):
    return asyncio.run(coro)


def test_fails_over_after_503():
    g, o = Fake("gemini", boom()), Fake("openrouter", ok())
    result = run(router(g, o).generate_json("p", parse, task="t"))
    assert result.value == 1 and result.provider == "openrouter"


def test_hanging_provider_is_cancelled_within_budget():
    g, o = Fake("gemini", hang(), timeout=1.0), Fake("openrouter", ok())
    started = time.monotonic()
    result = run(router(g, o).generate_json("p", parse, task="t", deadline_seconds=5))
    assert result.provider == "openrouter"
    assert time.monotonic() - started < 3  # not 30s


def test_total_deadline_is_hard():
    g, o = Fake("g", hang(), timeout=10), Fake("o", hang(), timeout=10)
    started = time.monotonic()
    with pytest.raises(LLMUnavailable):
        run(router(g, o).generate_json("p", parse, task="t", deadline_seconds=3))
    assert time.monotonic() - started < 4.5


def test_invalid_output_falls_through():
    g, o = Fake("g", ok('{"nope": 1}')), Fake("o", ok())
    assert run(router(g, o).generate_json("p", parse, task="t")).provider == "o"


def test_breaker_skips_failing_provider():
    g, o = Fake("g", boom()), Fake("o", ok())
    r = router(g, o)
    for _ in range(4):
        run(r.generate_json("p", parse, task="t"))
    assert g.calls == 2  # opened after threshold=2, then skipped


def test_cache_and_single_flight():
    g = Fake("g", ok())
    r = router(g)

    async def go():
        key = make_cache_key("t", "same")
        results = await asyncio.gather(
            *[r.generate_json("p", parse, task="t", cache_key=key, cache_ttl=30) for _ in range(5)]
        )
        again = await r.generate_json("p", parse, task="t", cache_key=key, cache_ttl=30)
        return results, again

    results, again = run(go())
    assert g.calls == 1 and again.cached and all(x.value == 1 for x in results)


def test_all_fail_raises_with_reasons():
    with pytest.raises(LLMUnavailable) as info:
        run(router(Fake("g", boom("Gemini HTTP 503"))).generate_json("p", parse, task="t"))
    assert "Gemini HTTP 503" in info.value.errors[0]


def test_parse_json_object_handles_fences_and_damage():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Here you go: {"a": 1} thanks') == {"a": 1}
    assert parse_json_object('{"a": [1, 2,') == {"a": [1, 2]}
    with pytest.raises(InvalidOutput):
        parse_json_object("no json here")
