from __future__ import annotations

import logging
import re

from app.config import get_settings
from app.db import get_knowledge_collection
from app.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    def __init__(self) -> None:
        self.embeddings = EmbeddingService()

    async def search(self, query: str, limit: int = 8) -> list[dict]:
        """Search the knowledge collection with vector search, then lexical fallback."""
        settings = get_settings()
        collection = get_knowledge_collection()
        query = (query or "").strip()
        if not query or limit < 1:
            return []

        # Atlas vector search is preferred, but a missing/unavailable vector index
        # must not make resume import fail. PyMongo AsyncCollection.aggregate()
        # is awaited before iterating the returned async cursor.
        try:
            vector = self.embeddings.embed(query)
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": settings.mongodb_vector_index,
                        "path": "embedding",
                        "queryVector": vector,
                        "numCandidates": max(limit * 8, 40),
                        "limit": limit,
                    }
                },
                {
                    "$project": {
                        "_id": 0,
                        "text": 1,
                        "kind": 1,
                        "field": 1,
                        "source": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]
            cursor = await collection.aggregate(pipeline)
            results = [document async for document in cursor]
            if results:
                return results
        except Exception as exc:
            logger.warning("Vector knowledge search unavailable; using lexical fallback: %s", exc)

        terms = list(dict.fromkeys(re.findall(r"[a-zA-Z0-9+#.-]{3,}", query.lower())))[:12]
        if not terms:
            return []

        try:
            cursor = collection.find(
                {},
                {"_id": 0, "text": 1, "kind": 1, "field": 1, "source": 1},
            ).sort("_id", -1).limit(1000)
            scored: list[tuple[int, dict]] = []
            async for document in cursor:
                text = str(document.get("text", ""))
                normalized = text.lower()
                score = sum(1 for term in terms if term in normalized)
                if score:
                    scored.append((score, document))
            scored.sort(key=lambda item: item[0], reverse=True)
            return [document for _, document in scored[:limit]]
        except Exception as exc:
            logger.warning("Lexical knowledge search failed: %s", exc)
            return []
