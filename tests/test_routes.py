import asyncio
import time

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app.config import Settings
from app.main import app
from app.services import llm
from app.services import suggestion  # noqa: F401
from app.services.llm import GeminiProvider, LLMRouter, ProviderError
from app.services.suggestion import profile_summary as ps

BODY = {"draft": "Backend developer with three years of Python and FastAPI.", "target_role": "Backend Engineer"}
GOOD = '{"summaries": ["a a", "b b", "c c", "d d"]}'


class Scripted(llm.Provider):
    def __init__(self, name, fn, timeout=2.0):
        self.name = self.model = name
        self.timeout, self.fn = timeout, fn

    async def complete(self, prompt, *, system, max_tokens, temperature):
        return await self.fn()


def install(monkeypatch, *providers, deadline=6):
    settings = Settings(GEMINI_API_KEY="x", AI_SHORT_DEADLINE_SECONDS=deadline)
    monkeypatch.setattr(llm, "_router", LLMRouter(list(providers), settings))
    monkeypatch.setattr(ps, "get_settings", lambda: settings)
    import app.api.routes as routes
    monkeypatch.setattr(routes, "get_settings", lambda: settings)


def test_gemini_503_then_fallback_returns_200(monkeypatch):
    async def gemini():
        raise ProviderError("Gemini HTTP 503 UNAVAILABLE")

    async def fallback():
        return llm.Completion(text=GOOD)

    install(monkeypatch, Scripted("gemini", gemini), Scripted("openrouter", fallback))
    r = TestClient(app).post("/api/v1/resumes/profile-summary/generate", json=BODY)
    assert r.status_code == 200 and len(r.json()["options"]) == 4


def test_every_provider_hanging_gives_clean_503_not_504(monkeypatch):
    async def hang():
        await asyncio.sleep(60)

    install(monkeypatch, Scripted("gemini", hang, 2), Scripted("openrouter", hang, 2), deadline=4)
    started = time.monotonic()
    r = TestClient(app).post("/api/v1/resumes/profile-summary/generate", json=BODY)
    assert r.status_code == 503 and r.headers["retry-after"] == "5"
    assert time.monotonic() - started < 6


def test_genai_server_error_maps_to_transient_provider_error(monkeypatch):
    class FakeModels:
        async def generate_content(self, **kw):
            raise genai_errors.ServerError(503, {"error": {"status": "UNAVAILABLE", "message": "overloaded"}})

    class FakeClient:
        aio = type("A", (), {"models": FakeModels()})()

    monkeypatch.setattr(llm, "get_gemini_client", lambda: FakeClient())
    provider = GeminiProvider(Settings(GEMINI_API_KEY="x"), "gemini-2.5-flash", 5, "gemini")
    with pytest.raises(ProviderError) as info:
        asyncio.run(provider.complete("p", system=None, max_tokens=10, temperature=0))
    assert "503" in str(info.value) and info.value.counts and not info.value.fatal


def test_validation_error_is_422():
    r = TestClient(app).post("/api/v1/resumes/profile-summary/generate", json={"draft": "x", "target_role": "y"})
    assert r.status_code == 422


def test_health_without_mongo():
    assert TestClient(app).get("/health").json()["status"] == "ok"
