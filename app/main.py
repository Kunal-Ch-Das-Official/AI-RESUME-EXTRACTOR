import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.db import close_client, get_client
from app.services.llm import close_http_client

_settings = get_settings()
logging.basicConfig(
    level=getattr(logging, _settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
for noisy in ("httpx", "google_genai.models", "google_genai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)  # no URLs/keys or "AFC" noise

logger = logging.getLogger(__name__)


def _mongo_enabled() -> bool:
    s = get_settings()
    return s.database_mode != "local" and bool(s.mongodb_uri)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MongoDB is optional; repository.py transparently persists locally when it
    # is unavailable or not configured.
    if _mongo_enabled():
        try:
            await get_client().admin.command("ping")
        except Exception as exc:  # noqa: BLE001
            logger.warning("MongoDB unreachable at startup (%s); continuing", type(exc).__name__)
    settings = get_settings()
    logger.info(
        "AI providers: gemini=%s openrouter=%s",
        bool(settings.gemini_api_key), settings.openrouter_enabled,
    )
    yield
    await close_http_client()
    await close_client()


app = FastAPI(
    title="AI-Resume Importer",
    description="Resume extraction and AI writing assistance (Gemini with failover).",
    version="4.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    if not _mongo_enabled():
        return {"status": "ok", "mongodb": "not_configured", "storage": "local"}
    try:
        await get_client().admin.command("ping")
        return {"status": "ok", "mongodb": "connected"}
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ok", "mongodb": "unavailable", "storage": "local",
            "detail": type(exc).__name__,
        }
