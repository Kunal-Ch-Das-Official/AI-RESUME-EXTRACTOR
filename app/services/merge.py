from __future__ import annotations

from typing import Any


def _merge_extraction_data(local: Any, ai: Any) -> Any:
    """Merge AI extraction with deterministic evidence without dropping local records.

    AI values win for non-empty scalar conflicts. Mapping keys and list records
    found only by either extractor are retained, preventing a partial AI result
    from erasing information recovered from the source document.
    """
    if ai is None or ai == "" or ai == [] or ai == {}:
        return local
    if local is None or local == "" or local == [] or local == {}:
        return ai
    if isinstance(local, dict) and isinstance(ai, dict):
        merged = dict(local)
        for key, value in ai.items():
            merged[key] = _merge_extraction_data(local.get(key), value) if key in local else value
        return merged
    if isinstance(local, list) and isinstance(ai, list):
        result = list(ai)
        for item in local:
            if item not in result:
                result.append(item)
        return result
    return ai
