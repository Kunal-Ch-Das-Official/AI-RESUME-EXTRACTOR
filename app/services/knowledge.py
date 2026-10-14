from __future__ import annotations

from datetime import datetime, timezone

from app.db import get_knowledge_collection
from app.services.embeddings import EmbeddingService


async def upsert_knowledge(
    *,
    text: str,
    kind: str,
    field: str | None = None,
    source: str = "manual",
    metadata: dict | None = None,
) -> str:
    collection = get_knowledge_collection()
    embedding = EmbeddingService().embed(text)

    document = {
        "text": text,
        "kind": kind,
        "field": field,
        "source": source,
        "metadata": metadata or {},
        "embedding": embedding,
        "createdAt": datetime.now(timezone.utc),
    }

    result = await collection.insert_one(document)
    return str(result.inserted_id)
