from __future__ import annotations



def build_chunks(blocks: list[dict], max_chars: int = 1800) -> list[dict]:
    """
    Create retrieval-friendly chunks while preserving source block IDs.

    This deliberately does not classify headings. A heading can be unusual,
    absent, or ambiguous. The model receives the original text and evidence.
    """
    chunks = []
    current = []
    current_chars = 0

    for block in blocks:
        text = str(block.get("text", "")).strip()
        if not text:
            continue

        if current and current_chars + len(text) > max_chars:
            chunks.append(_make_chunk(current, len(chunks) + 1))
            current = []
            current_chars = 0

        current.append(block)
        current_chars += len(text) + 1

    if current:
        chunks.append(_make_chunk(current, len(chunks) + 1))

    return chunks


def _make_chunk(blocks: list[dict], index: int) -> dict:
    text = "\n".join(block["text"] for block in blocks)
    pages = sorted({int(block.get("page", 1)) for block in blocks})
    block_ids = [block["id"] for block in blocks]

    return {
        "chunk_id": f"chunk-{index}",
        "text": text,
        "pages": pages,
        "block_ids": block_ids,
    }
