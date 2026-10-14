from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from app.models.resume import ResumeInfo


URL_FIELDS = {
    "linkedin",
    "github",
    "portfolio",
    "projectUrl",
    "certificateUrl",
    "githubUrl",
    "referenceUrl",
    "url",
}


def normalize_resume(data: ResumeInfo | dict[str, Any]) -> ResumeInfo:
    if isinstance(data, ResumeInfo):
        return data

    cleaned = _clean_strings(data)

    if not isinstance(cleaned, dict):
        raise ValueError("Resume data must be a dictionary.")

    _normalize_urls(cleaned)

    cleaned.setdefault("templateName", "Imported Resume")
    cleaned.setdefault("basicInfo", {"fullName": "", "position": ""})
    cleaned.setdefault(
        "address",
        {
            "streetName": "",
            "city": "",
            "district": "",
            "pincode": "",
            "country": "",
        },
    )
    cleaned.setdefault(
        "contactInfo",
        {"primaryEmail": "", "primaryMobile": ""},
    )
    cleaned.setdefault(
        "profileSummary",
        {"priority": 1, "subject": "", "objective": None},
    )
    cleaned.setdefault(
        "educations",
        {
            "priority": 1,
            "sectionTitle": None,
            "qualifications": {},
        },
    )
    cleaned.setdefault(
        "skills",
        {
            "priority": 1,
            "sectionTitle": None,
            "skills": {},
        },
    )
    cleaned.setdefault(
        "languageProficiency",
        {
            "priority": 1,
            "sectionTitle": None,
            "languageKnows": [],
        },
    )

    # Intelligently sort and deduplicate dynamic skill sets
    if "skills" in cleaned and isinstance(cleaned["skills"], dict):
        skill_sets = cleaned["skills"].get("skills", {})
        if isinstance(skill_sets, dict):
            cleaned["skills"]["skills"] = {
                k: sorted(list(dict.fromkeys(v)))
                for k, v in skill_sets.items()
                if isinstance(v, list)
            }

    _repair_common_model_key_variants(cleaned)

    return ResumeInfo.model_validate(cleaned)


def _normalize_urls(value: Any, parent_key: str | None = None) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in URL_FIELDS and isinstance(item, str):
                value[key] = _normalize_url(item)
            else:
                _normalize_urls(item, key)
    elif isinstance(value, list):
        for item in value:
            _normalize_urls(item, parent_key)


def _normalize_url(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return value

    parsed = urlparse(value)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return value

    if value.startswith("//"):
        return f"https:{value}"

    if re.match(
        r"^(?:www\.)?(?:linkedin\.com|github\.com|"
        r"github\.io|[a-z0-9-]+(?:\.[a-z0-9-]+)+)(?::\d+)?(?:/|$)",
        value,
        re.IGNORECASE,
    ):
        candidate = f"https://{value}"
        parsed = urlparse(candidate)
        if parsed.scheme == "https" and parsed.netloc:
            return candidate

    return value


def _repair_common_model_key_variants(data: dict[str, Any]) -> None:
    contact = data.get("contactInfo")
    if isinstance(contact, dict):
        if not contact.get("primaryEmail"):
            contact["primaryEmail"] = _first_email(data)
        if not contact.get("primaryMobile"):
            contact["primaryMobile"] = _first_phone(data)

    educations = data.get("educations")
    if isinstance(educations, dict):
        qualifications = educations.get("qualifications")
        if isinstance(qualifications, dict):
            for item in qualifications.values():
                if isinstance(item, dict):
                    if "achivedMarks" not in item and "achievedMarks" in item:
                        item["achivedMarks"] = item.pop("achievedMarks")


def _first_email(value: Any) -> str:
    pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    for item in _find_strings(value):
        match = pattern.search(item)
        if match:
            return match.group(0)
    return ""


def _first_phone(value: Any) -> str:
    pattern = re.compile(r"(?<!\d)\+?\d[\d\s().-]{7,}\d(?!\d)")
    for item in _find_strings(value):
        match = pattern.search(item)
        if match:
            return match.group(0).strip()
    return ""


def _find_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            result.extend(_find_strings(item))
        return result
    if isinstance(value, dict):
        result: list[str] = []
        for item in value.values():
            result.extend(_find_strings(item))
        return result
    return []


def _clean_strings(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return [_clean_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_strings(item) for key, item in value.items()}
    return value