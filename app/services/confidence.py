from __future__ import annotations

from typing import Any

from app.models.resume import ResumeInfo


def calculate_confidence(
    *,
    resume: ResumeInfo,
    knowledge: list[dict] | None,
    warnings: list[str],
    source_text: str,
) -> dict[str, Any]:
    """Conservative, explainable heuristic.

    ``knowledge=None`` means RAG was not used for this import, which is the
    normal case, so it is not penalised (it used to cost 0.12 on every import).
    """
    data = resume.model_dump(mode="json")
    score = 0.90
    reasons: list[str] = []

    if not source_text.strip():
        score -= 0.40
        reasons.append("No source text.")

    if knowledge is not None and not knowledge:
        score -= 0.12
        reasons.append("No RAG knowledge was retrieved.")

    if warnings:
        score -= min(0.20, 0.04 * len(warnings))
        reasons.append(f"{len(warnings)} deterministic validation warning(s).")

    if (data.get("basicInfo", {}).get("fullName") or "").strip().lower() in {"", "unknown"}:
        score -= 0.10
        reasons.append("Name is empty.")

    if not (data.get("contactInfo", {}).get("primaryEmail") or "").strip():
        score -= 0.10
        reasons.append("Primary email is empty.")

    return {"score": round(max(0.0, min(1.0, score)), 4), "reasons": reasons}
