from __future__ import annotations

import re
from typing import Any

from app.models.resume import ResumeInfo


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
URL_RE = re.compile(r"^https?://", re.I)


def validate_semantics(resume: ResumeInfo) -> list[str]:
    warnings: list[str] = []
    data = resume.model_dump(mode="json")

    email = data.get("contactInfo", {}).get("primaryEmail", "")
    if email and not EMAIL_RE.match(email):
        warnings.append("Primary email has an invalid format.")

    contact = data.get("contactInfo", {})
    for key in ("linkedin", "github", "portfolio"):
        value = contact.get(key)
        if value and not URL_RE.match(value):
            warnings.append(f"{key} is not an absolute HTTP(S) URL.")

    for section_name, section in (
        ("projects", data.get("projects")),
        ("workExperience", data.get("workExperience")),
        ("educations", data.get("educations")),
    ):
        if not section:
            continue
        if section.get("priority", 1) < 1:
            warnings.append(f"{section_name} has invalid priority.")

    for language in data.get("languageProficiency", {}).get("languageKnows", []):
        score = language.get("proficiencyOutOfTen")
        if score is not None and not 0 <= float(score) <= 10:
            warnings.append(
                f"Language proficiency for {language.get('languageName')} is outside 0-10."
            )

    return warnings
