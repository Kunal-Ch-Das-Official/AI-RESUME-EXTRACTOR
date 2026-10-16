import asyncio

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.config import Settings
from app.main import app
from app.services import llm, model_resolver
from app.services.llm import (
    GeminiProvider, LLMMisconfigured, LLMRouter, Provider, ProviderError, _thinking_config,
)

AVAILABLE = {"gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3-flash-preview"}


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    model_resolver._warned.clear()

    async def fake_available(force=False):
        return AVAILABLE

    monkeypatch.setattr(model_resolver, "available_models", fake_available)


def test_best_models_ignores_previews_and_picks_newest():
    assert model_resolver.best_models(AVAILABLE) == ("gemini-3.5-flash", "gemini-3.5-flash-lite")


def test_retired_model_is_replaced_and_existing_kept():
    run = asyncio.run
    assert run(model_resolver.resolve("gemini-2.5-flash")) == "gemini-3.5-flash"
    assert run(model_resolver.resolve("gemini-2.5-flash-lite", lite=True)) == "gemini-3.5-flash-lite"
    assert run(model_resolver.resolve("gemini-3.1-flash-lite", lite=True)) == "gemini-3.1-flash-lite"


def test_thinking_is_minimal_for_gemini3_and_budgeted_for_25():
    s = Settings()
    assert _thinking_config(s, "gemini-3.5-flash").thinking_level.name == "MINIMAL"
    assert _thinking_config(s, "gemini-3.5-pro").thinking_level.name == "LOW"
    assert _thinking_config(s, "gemini-2.5-flash").thinking_budget == 0
    assert _thinking_config(Settings(GEMINI_THINKING_BUDGET=-1), "gemini-3.5-flash") is None


def test_404_is_fatal_and_mentions_model():
    class Models:
        async def generate_content(self, **kw):
            raise genai_errors.ClientError(404, {"error": {"status": "NOT_FOUND", "message": "x"}})

    client = type("C", (), {"aio": type("A", (), {"models": Models()})()})()
    import app.services.llm as m
    m.get_gemini_client = lambda: client
    provider = GeminiProvider(Settings(GEMINI_API_KEY="x"), "gemini-3.5-flash", 5, "gemini")
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.complete("p", system=None, max_tokens=10, temperature=0))
    assert info.value.fatal and "GEMINI_MODEL" in str(info.value)


class AlwaysFatal(Provider):
    def __init__(self, name):
        self.name = self.model = name
        self.timeout = 2

    async def complete(self, *a, **k):
        raise ProviderError("HTTP 404", fatal=True)


def test_all_fatal_raises_misconfigured_and_route_explains(monkeypatch):
    import app.api.routes as routes
    import app.services.suggestion.profile_summary as ps

    s = Settings(GEMINI_API_KEY="x")
    monkeypatch.setattr(llm, "_router", LLMRouter([AlwaysFatal("g"), AlwaysFatal("o")], s))
    monkeypatch.setattr(ps, "get_settings", lambda: s)
    monkeypatch.setattr(routes, "get_settings", lambda: s)

    with pytest.raises(LLMMisconfigured):
        asyncio.run(llm._router.generate_json("p", lambda d: d, task="t"))

    llm._router = LLMRouter([AlwaysFatal("g"), AlwaysFatal("o")], s)
    r = TestClient(app).post(
        "/api/v1/resumes/profile-summary/generate",
        json={"draft": "Backend developer with Python experience.", "target_role": "Engineer"},
    )
    assert r.status_code == 503 and "not configured correctly" in r.json()["detail"]
