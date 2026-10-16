
from __future__ import annotations

from uuid import uuid4

from app.db import get_knowledge_collection
from app.services.embeddings import get_embedding_service


class KnowledgeIngestor:
    def __init__(self) -> None:
        self.embeddings = get_embedding_service()

    async def ingest(
        self,
        *,
        filename: str,
        text: str,
        chunks: list[dict] | None = None,
    ) -> dict:
        text = self._clean_text(text)

        if not text:
            raise ValueError("No readable text was extracted from the file.")

        # Use the application's existing chunker when chunks are supplied.
        # Fall back to text chunking only if the caller does not provide chunks.
        if chunks:
            knowledge_chunks = self._normalize_chunks(chunks)
        else:
            knowledge_chunks = self._chunk_text(text)

        if not knowledge_chunks:
            raise ValueError(
                "The document did not contain enough text to create knowledge chunks."
            )

        collection = get_knowledge_collection()

        prepared: list[dict] = []

        for index, chunk in enumerate(knowledge_chunks):
            if isinstance(chunk, dict):
                chunk_text = str(chunk.get("text", "")).strip()
                kind = chunk.get("kind", "resume_example")
                field = chunk.get("field") or self._detect_field(chunk_text)
                source_chunk_id = chunk.get("chunk_id") or chunk.get("id")
            else:
                chunk_text = str(chunk).strip()
                kind = "resume_example"
                field = self._detect_field(chunk_text)
                source_chunk_id = None

            if not chunk_text:
                continue

            document = {
                "knowledge_id": str(uuid4()),
                "source": filename,
                "chunk_index": index,
                "text": chunk_text,
                "kind": kind,
                "field": field,
            }
            if source_chunk_id:
                document["source_chunk_id"] = source_chunk_id
            prepared.append(document)

        # One batched embedding request instead of one network call per chunk.
        vectors = (
            await self.embeddings.embed_many([d["text"] for d in prepared])
            if prepared
            else []
        )
        documents = [{**d, "embedding": v} for d, v in zip(prepared, vectors)]

        if not documents:
            raise ValueError(
                "No valid knowledge chunks were generated from the document."
            )

        await collection.insert_many(documents)

        return {
            "success": True,
            "source": filename,
            "chunks_created": len(documents),
            "message": (
                f"Successfully added {len(documents)} knowledge chunks."
            ),
        }

    @staticmethod
    def _normalize_chunks(chunks: list[dict]) -> list[dict]:
        normalized = []

        for chunk in chunks:
            if isinstance(chunk, dict):
                text = str(chunk.get("text", "")).strip()

                if not text:
                    continue

                normalized.append(
                    {
                        **chunk,
                        "text": text,
                    }
                )
            else:
                text = str(chunk).strip()

                if text:
                    normalized.append(
                        {
                            "text": text,
                            "kind": "resume_example",
                            "field": "general",
                        }
                    )

        return normalized

    @staticmethod
    def _clean_text(text: str) -> str:
        lines = []

        for line in text.splitlines():
            line = " ".join(line.split())

            if line:
                lines.append(line)

        return "\n".join(lines).strip()

    @staticmethod
    def _chunk_text(
        text: str,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> list[str]:
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)

            if end < text_length:
                boundary = text.rfind("\n", start, end)

                if boundary > start + (chunk_size // 2):
                    end = boundary

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            start = max(end - overlap, start + 1)

        return chunks

    @staticmethod
    def _detect_field(text: str) -> str:
        normalized = text.lower()

        field_keywords = {
            "education": [
                "education",
                "qualification",
                "academic",
                "university",
                "college",
                "degree",
                "bachelor",
                "master",
                "phd",
            ],
            "experience": [
                "experience",
                "employment",
                "work history",
                "professional experience",
                "career",
            ],
            "skills": [
                "skills",
                "technical skills",
                "skill set",
                "technologies",
                "technical expertise",
            ],
            "projects": [
                "projects",
                "project experience",
                "selected projects",
            ],
            "certifications": [
                "certification",
                "certifications",
                "certificate",
                "certificates",
            ],
            "publications": [
                "publication",
                "publications",
                "research paper",
                "journal",
            ],
            "achievements": [
                "achievement",
                "achievements",
                "awards",
                "honors",
                "honours",
            ],
        }

        for field, keywords in field_keywords.items():
            if any(keyword in normalized for keyword in keywords):
                return field

        return "general"
