"""Compact extraction prompt (resume text is sent exactly once)."""

from __future__ import annotations

import re

from app.services.ai_schema import WIRE_SHAPE

SYSTEM = "You convert resume text to structured JSON. Reply with one JSON object only."

INSTRUCTIONS = f"""Convert the resume below into JSON of exactly this shape (use "" / [] / false when absent; keep every key):
{WIRE_SHAPE}

Rules:
- Use only what this resume states. Never invent dates, employers, titles, institutions, scores, URLs, emails or skills.
- Classify by meaning, not heading text ("Academic Journey" = education, "Where I've Made an Impact" = experience, "Core Stack" = skills). Never drop a section: map unfamiliar ones to the closest field.
- Keep source order and original wording. Every bullet is its own string. Join wrapped lines; ignore [PAGE n] markers and repeated headers/footers. Do not repeat the same text in several fields.
- Dates exactly as written ("Jan 2021", "2019"). Present/current/ongoing/till date => current:true and end:"".
- experience: jobs, internships, freelance. type = Full-time/Part-time/Internship/Contract and mode = On-site/Remote/Hybrid, only if stated. Academic or personal work => projects.
- skills: group by the resume's own labels, else one group "Skills". Only skills/tools/technologies, never jobs, degrees or sentences. Human languages go in languages (level 0-10 only if stated: native 10, fluent 9, professional 8, intermediate 6, basic 3; else 0).
- education.score: marks/CGPA/percentage as written. summary.heading = the section's own title; summary.text = its prose.
- URLs: copy exactly as shown (including those listed under links); never guess.

RESUME:
"""

_PAGE = re.compile(r"^\[\s*page\s+\d+\s*\]$", re.I)


def prepare_resume_text(raw_text: str, max_chars: int) -> str:
    """Shrink resume text without losing content: drop page markers, blank
    lines, repeated whitespace and consecutive duplicate lines; cap length."""
    lines: list[str] = []
    previous = ""
    for line in raw_text.splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", line.replace("\uf0b7", "•")).strip()
        if not line or _PAGE.match(line) or line.lower() == "[embedded links]":
            continue
        if line == previous:
            continue
        lines.append(line)
        previous = line
    text = "\n".join(lines)
    if len(text) > max_chars:
        cut = text.rfind("\n", 0, max_chars)
        text = text[: cut if cut > max_chars // 2 else max_chars]
    return text


def build_extraction_prompt(resume_text: str) -> str:
    return INSTRUCTIONS + resume_text
