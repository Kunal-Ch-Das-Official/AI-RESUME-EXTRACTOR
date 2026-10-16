from __future__ import annotations

from functools import lru_cache

from google.genai import types

from app.config import get_settings
from app.services.gemini_client import get_gemini_client

_BATCH = 100  # Gemini embed_content accepts up to 100 texts per request


class EmbeddingService:
    """Async, batched embeddings (one request per 100 texts, not one per text)."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required.")
        self.model = settings.gemini_embedding_model
        self.dimensions = settings.gemini_embedding_dimensions

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        client = get_gemini_client()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            batch = texts[start : start + _BATCH]
            result = await client.aio.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=self.dimensions),
            )
            embeddings = result.embeddings or []
            if len(embeddings) != len(batch) or any(not e.values for e in embeddings):
                raise RuntimeError("Gemini returned an incomplete embedding batch.")
            vectors.extend(list(e.values) for e in embeddings)
        return vectors

    async def embed(self, text: str) -> list[float]:
        return (await self.embed_many([text]))[0]


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
