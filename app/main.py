from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.config import get_settings
from app.db import close_client, get_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MongoDB is optional for imports; repository.py transparently persists
    # locally when it is unavailable or not configured.
    settings = get_settings()
    if settings.database_mode != "local" and settings.mongodb_uri:
        try:
            client = get_client()
            await client.admin.command("ping")
        except Exception:
            pass
    yield
    await close_client()


app = FastAPI(
    title="AI-Resume Importer",
    description="Hybrid semantic resume extraction with RAG grounding and verification.",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    try:
        await get_client().admin.command("ping")
        return {"status": "ok", "mongodb": "connected"}
    except Exception as exc:
        return {"status": "ok", "mongodb": "unavailable", "storage": "local", "detail": str(exc)}
