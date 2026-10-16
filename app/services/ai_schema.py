"""Lean "wire" format for AI extraction and its conversion to ResumeInfo.

Why not send ResumeInfo's JSON schema to the model?
* It is thousands of tokens, resent on every call.
* ResumeInfo uses ``dict[str, Item]`` maps, which Gemini's structured-output
  schema cannot describe (``additionalProperties`` is stripped), so the model
  never saw the item shape.

The model instead fills a short, list-based shape (WIRE_SHAPE); this module
validates it leniently and builds the dict-keyed ResumeInfo payload in code.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field


# ----------------------------------------------------------- lenient coercers
def _s(v: Any) -> str:
    if v is None or isinstance(v, (dict, list)):
        return ""
    if isinstance(v, bool):
        return ""
    return " ".join(str(v).split())


def _sl(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = re.split(r"[,;\n]", v)
    if not isinstance(v, list):
        return []
    out = [_s(x) for x in v]
    return list(dict.fromkeys(x for x in out if x))


def _b(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"true", "yes", "y", "1", "present", "current"}
    return bool(v)


def _n(v: Any) -> float:
    try:
        return max(0.0, min(10.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _records(v: Any) -> list[Any]:
    if isinstance(v, dict):
        return [v]
    if not isinstance(v, list):
        return []
    return [x for x in v if isinstance(x, dict)]


def _obj(v: Any) -> Any:
    return v if isinstance(v, dict) else {}


Str = Annotated[str, BeforeValidator(_s)]
StrList = Annotated[list[str], BeforeValidator(_sl)]
Bool = Annotated[bool, BeforeValidator(_b)]
Num = Annotated[float, BeforeValidator(_n)]


class _W(BaseModel):
    model_config = ConfigDict(extra="ignore")


class WAddress(_W):
    street: Str = ""
    city: Str = ""
    district: Str = ""
    pincode: Str = ""
    country: Str = ""


class WSummary(_W):
    heading: Str = ""
    text: Str = ""


class WEducation(_W):
    degree: Str = ""
    institution: Str = ""
    start: Str = ""
    end: Str = ""
    pursuing: Bool = False
    score: Str = ""
    description: Str = ""


class WExperience(_W):
    company: Str = ""
    role: Str = ""
    type: Str = ""
    mode: Str = ""
    location: Str = ""
    start: Str = ""
    end: Str = ""
    current: Bool = False
    bullets: StrList = Field(default_factory=list)


class WSkillGroup(_W):
    group: Str = ""
    items: StrList = Field(default_factory=list)


class WLanguage(_W):
    name: Str = ""
    level: Num = 0.0


class WProject(_W):
    name: Str = ""
    description: Str = ""
    url: Str = ""
    start: Str = ""
    end: Str = ""
    ongoing: Bool = False
    tech: StrList = Field(default_factory=list)
    skills: StrList = Field(default_factory=list)


class WCertification(_W):
    name: Str = ""
    overview: Str = ""
    skills: StrList = Field(default_factory=list)
    url: Str = ""
    duration: Str = ""


class WOpenSource(_W):
    name: Str = ""
    url: Str = ""
    description: Str = ""
    duration: Str = ""


class WPublication(_W):
    title: Str = ""
    description: Str = ""
    url: Str = ""


class WAward(_W):
    title: Str = ""
    description: Str = ""
    url: Str = ""


Recs = BeforeValidator(_records)


class WireResume(_W):
    name: Str = ""
    title: Str = ""
    email: Str = ""
    phone: Str = ""
    linkedin: Str = ""
    github: Str = ""
    portfolio: Str = ""
    address: Annotated[WAddress, BeforeValidator(_obj)] = Field(default_factory=WAddress)
    summary: Annotated[WSummary, BeforeValidator(_obj)] = Field(default_factory=WSummary)
    education: Annotated[list[WEducation], Recs] = Field(default_factory=list)
    experience: Annotated[list[WExperience], Recs] = Field(default_factory=list)
    skills: Annotated[list[WSkillGroup], Recs] = Field(default_factory=list)
    languages: Annotated[list[WLanguage], Recs] = Field(default_factory=list)
    projects: Annotated[list[WProject], Recs] = Field(default_factory=list)
    certifications: Annotated[list[WCertification], Recs] = Field(default_factory=list)
    openSource: Annotated[list[WOpenSource], Recs] = Field(default_factory=list)
    publications: Annotated[list[WPublication], Recs] = Field(default_factory=list)
    awards: Annotated[list[WAward], Recs] = Field(default_factory=list)


# What the model is shown (≈330 tokens, vs ≈4k for ResumeInfo's JSON schema).
WIRE_SHAPE = """{
"name":"","title":"","email":"","phone":"","linkedin":"","github":"","portfolio":"",
"address":{"street":"","city":"","district":"","pincode":"","country":""},
"summary":{"heading":"","text":""},
"education":[{"degree":"","institution":"","start":"","end":"","pursuing":false,"score":"","description":""}],
"experience":[{"company":"","role":"","type":"","mode":"","location":"","start":"","end":"","current":false,"bullets":[""]}],
"skills":[{"group":"","items":[""]}],
"languages":[{"name":"","level":0}],
"projects":[{"name":"","description":"","url":"","start":"","end":"","ongoing":false,"tech":[""],"skills":[""]}],
"certifications":[{"name":"","overview":"","skills":[""],"url":"","duration":""}],
"openSource":[{"name":"","url":"","description":"","duration":""}],
"publications":[{"title":"","description":"","url":""}],
"awards":[{"title":"","description":"","url":""}]
}"""


def parse_wire(payload: dict[str, Any]) -> WireResume:
    if isinstance(payload.get("resume"), dict):  # some models wrap the object
        payload = payload["resume"]
    wire = WireResume.model_validate(payload)
    has_content = bool(
        wire.name or wire.email or wire.education or wire.experience or wire.skills
        or wire.projects or wire.summary.text
    )
    if not has_content:
        raise ValueError("model returned an empty resume")
    return wire


# --------------------------------------------------------- wire -> ResumeInfo
def _norm_url(value: str) -> str:
    v = value.lower().strip().rstrip("/.,;)")
    v = re.sub(r"^https?://", "", v)
    return re.sub(r"^www\.", "", v)


def _supported_url(url: str, source_norm: str) -> str | None:
    """Keep a URL only if it actually appears in the source text (anti-hallucination)."""
    if not url:
        return None
    return url if _norm_url(url) in source_norm else None


def _unique(key: str, used: dict, fallback: str) -> str:
    base = (key or fallback).strip()[:160] or fallback
    candidate, n = base, 2
    while candidate in used:
        candidate = f"{base} ({n})"
        n += 1
    return candidate


_FRACTION = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)")


def _score_fields(score: str) -> dict[str, Any]:
    if not score:
        return {}
    out: dict[str, Any] = {"percentage": score}
    m = _FRACTION.search(score)
    if m:
        out["achivedMarks"], out["totalMarks"] = float(m.group(1)), float(m.group(2))
    return out


def wire_to_payload(wire: WireResume, source_text: str) -> dict[str, Any]:
    """Build a ResumeInfo-shaped dict. Only sections that have data are emitted."""
    src = _norm_url(source_text)
    src_lower = source_text.lower()

    email = wire.email if wire.email and wire.email.lower() in src_lower else ""

    payload: dict[str, Any] = {
        "basicInfo": {"fullName": wire.name, "position": wire.title},
        "address": {
            "streetName": wire.address.street, "city": wire.address.city,
            "district": wire.address.district, "pincode": wire.address.pincode,
            "country": wire.address.country,
        },
        "contactInfo": {
            "primaryEmail": email,
            "primaryMobile": wire.phone,
            "linkedin": _supported_url(wire.linkedin, src),
            "github": _supported_url(wire.github, src),
            "portfolio": _supported_url(wire.portfolio, src),
        },
        "profileSummary": {
            "priority": 1,
            "subject": wire.summary.heading or "Professional Summary",
            "objective": wire.summary.text or None,
        },
        "templateName": "Imported Resume",
    }

    # education
    quals: dict[str, Any] = {}
    for i, e in enumerate(wire.education, 1):
        if not (e.degree or e.institution):
            continue
        item = {
            "priority": i, "institutionName": e.institution, "startedAt": e.start,
            "yearOfComplete": e.end or None, "pursuing": e.pursuing,
            "description": e.description or None, **_score_fields(e.score),
        }
        quals[_unique(e.degree, quals, f"Qualification {i}")] = item
    payload["educations"] = {"priority": 2, "sectionTitle": "Education", "qualifications": quals}

    # experience
    companies: dict[str, Any] = {}
    for i, x in enumerate(wire.experience, 1):
        if not (x.company or x.role):
            continue
        companies[_unique(x.company, companies, f"Employer {i}")] = {
            "priority": i, "jobTitle": x.role, "jobConditions": x.mode, "jobTypes": x.type,
            "jobLocation": x.location or None, "responsibility": x.bullets or None,
            "startDate": x.start, "endDate": x.end or None,
            "isPresentJob": x.current and not x.end,
        }
    if companies:
        payload["workExperience"] = {
            "priority": 1, "sectionTitle": "Professional Experience", "companies": companies
        }

    # skills
    skills: dict[str, list[str]] = {}
    for g in wire.skills:
        if g.items:
            key = _unique(g.group or "Skills", skills, "Skills")
            skills[key] = g.items
    payload["skills"] = {"priority": 3, "sectionTitle": "Skills", "skills": skills}

    payload["languageProficiency"] = {
        "priority": 4, "sectionTitle": "Languages",
        "languageKnows": [
            {"languageName": l.name, "proficiencyOutOfTen": l.level}
            for l in wire.languages if l.name
        ],
    }

    projects: dict[str, Any] = {}
    for i, p in enumerate(wire.projects, 1):
        if not (p.name or p.description):
            continue
        projects[_unique(p.name, projects, f"Project {i}")] = {
            "priority": i, "description": p.description,
            "projectUrl": _supported_url(p.url, src),
            "startDate": p.start, "endDate": p.end or None,
            "isWorking": p.ongoing and not p.end, "techStack": p.tech, "skills": p.skills,
        }
    if projects:
        payload["projects"] = {"priority": 5, "sectionTitle": "Projects", "projects": projects}

    certs: dict[str, Any] = {}
    for i, c in enumerate(wire.certifications, 1):
        if not c.name:
            continue
        certs[_unique(c.name, certs, f"Certification {i}")] = {
            "priority": i, "overview": c.overview or None, "skillLearned": c.skills,
            "certificateUrl": _supported_url(c.url, src), "duration": c.duration,
        }
    if certs:
        payload["certifications"] = {
            "priority": 6, "sectionTitle": "Certifications", "certificates": certs
        }

    pubs: dict[str, Any] = {}
    for i, p in enumerate(wire.publications, 1):
        if not (p.title or p.description):
            continue
        pubs[_unique(p.title or p.description[:120], pubs, f"Publication {i}")] = {
            "priority": i, "description": p.description or None,
            "referenceUrl": _supported_url(p.url, src),
        }
    if pubs:
        payload["publications"] = {"priority": 7, "sectionTitle": "Publications", "publications": pubs}

    oss: dict[str, Any] = {}
    for i, o in enumerate(wire.openSource, 1):
        if not (o.name or o.url):
            continue
        oss[_unique(o.name or o.url, oss, f"Contribution {i}")] = {
            "priority": i, "githubUrl": _supported_url(o.url, src),
            "description": o.description or None, "duration": o.duration or None,
        }
    if oss:
        payload["openSource"] = {"priority": 8, "sectionTitle": "Open Source", "contributions": oss}

    awards = [
        {"priority": i, "title": a.title or None, "description": a.description or None,
         "url": _supported_url(a.url, src)}
        for i, a in enumerate(wire.awards, 1) if a.title or a.description
    ]
    if awards:
        payload["awardsAndAchievements"] = {
            "priority": 9, "sectionTitle": "Awards & Achievements", "achievements": awards
        }
    return payload
