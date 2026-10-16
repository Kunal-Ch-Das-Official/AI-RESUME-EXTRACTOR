from __future__ import annotations

import copy
from typing import Any

# Placeholder values the deterministic parser emits when it finds nothing.
SENTINELS = {"", "unknown", "imported resume", "not specified", "employer not specified"}

RECORD_SECTIONS = {
    "educations": "qualifications",
    "workExperience": "companies",
    "projects": "projects",
    "certifications": "certificates",
    "publications": "publications",
    "openSource": "contributions",
}


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in SENTINELS
    return value in ([], {})


def _fill(primary: Any, fallback: Any) -> Any:
    return fallback if _blank(primary) else primary


def _fill_dict(primary: dict | None, fallback: dict | None) -> dict:
    primary, fallback = primary or {}, fallback or {}
    merged = dict(fallback)
    for key, value in primary.items():
        merged[key] = _fill(value, fallback.get(key))
    return merged


def _has_records(section: Any, inner: str) -> bool:
    return isinstance(section, dict) and bool(section.get(inner))


def _has_skills(section: Any) -> bool:
    return isinstance(section, dict) and any((section.get("skills") or {}).values())


def merge_extraction(local: dict[str, Any], ai: dict[str, Any]) -> dict[str, Any]:
    """Combine the AI result (primary) with the deterministic parse (gap filler).

    * Scalars/contact/address: AI value, else the deterministic value
      (placeholders such as "Unknown" never win).
    * Record sections (education, jobs, projects, ...): the AI section replaces
      the local one whenever it has records. Mixing the two produced duplicated
      and placeholder entries ("Employer not specified").
    * Anything only the local parser found (e.g. rawSections) is kept.
    """
    merged = copy.deepcopy(local)

    merged["basicInfo"] = _fill_dict(ai.get("basicInfo"), local.get("basicInfo"))
    merged["address"] = _fill_dict(ai.get("address"), local.get("address"))
    merged["contactInfo"] = _fill_dict(ai.get("contactInfo"), local.get("contactInfo"))

    ai_summary = ai.get("profileSummary") or {}
    if not _blank(ai_summary.get("objective")):
        merged["profileSummary"] = ai_summary

    for section, inner in RECORD_SECTIONS.items():
        if _has_records(ai.get(section), inner):
            merged[section] = ai[section]

    if _has_skills(ai.get("skills")):
        merged["skills"] = ai["skills"]

    if (ai.get("languageProficiency") or {}).get("languageKnows"):
        merged["languageProficiency"] = ai["languageProficiency"]

    if (ai.get("awardsAndAchievements") or {}).get("achievements"):
        merged["awardsAndAchievements"] = ai["awardsAndAchievements"]

    return merged
