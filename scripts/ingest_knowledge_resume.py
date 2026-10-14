import asyncio
import sys
from pathlib import Path

from app.services.chunker import build_chunks
from app.services.file_extractor import extract_document
from app.services.knowledge import upsert_knowledge


async def main(path: str):
    file_path = Path(path)
    data = file_path.read_bytes()
    parsed = extract_document(data, file_path.name)
    chunks = build_chunks(parsed["blocks"])

    for chunk in chunks:
        # Only put approved/sanitized examples into the reusable knowledge base.
        text = (
            "Approved resume example.\n"
            f"Pages: {chunk['pages']}\n"
            f"Example content:\n{chunk['text']}"
        )
        await upsert_knowledge(
            text=text,
            kind="resume_example",
            source=file_path.name,
            metadata={"chunk_id": chunk["chunk_id"], "pages": chunk["pages"]},
        )

    print(f"Indexed {len(chunks)} chunks from {file_path.name}.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/ingest_knowledge_resume.py /path/resume.pdf")
    asyncio.run(main(sys.argv[1]))
