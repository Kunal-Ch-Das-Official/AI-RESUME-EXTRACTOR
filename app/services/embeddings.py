from __future__ import annotations

from google import genai
from google.genai import types

from app.config import get_settings


class EmbeddingService:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required.")
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_embedding_model
        self.dimensions = settings.gemini_embedding_dimensions

    def embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
            ),
        )
        if not result.embeddings:
            raise RuntimeError("Gemini returned no embedding.")
        values = result.embeddings[0].values
        if not values:
            raise RuntimeError("Gemini returned an empty embedding.")
        return list(values)
