from app.services.chunker import build_chunks


def test_chunker_preserves_source_ids():
    blocks = [
        {"id": "a", "page": 1, "text": "Education"},
        {"id": "b", "page": 1, "text": "Bachelor of Science"},
    ]
    chunks = build_chunks(blocks, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0]["block_ids"] == ["a", "b"]
    assert "Bachelor" in chunks[0]["text"]
