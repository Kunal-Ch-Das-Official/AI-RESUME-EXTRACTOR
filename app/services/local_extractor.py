"""Schema-aligned deterministic resume parser used when AI extraction is unavailable."""
from __future__ import annotations
import re
from collections import OrderedDict
from typing import Any
from app.models.resume import ResumeInfo

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
URL = re.compile(r"(?:https?://|www\.|(?:linkedin\.com|github\.com|[a-z0-9-]+\.[a-z]{2,}/))[^\s|,;]+", re.I)
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
PAGE_MARKER = re.compile(r"^\[\s*page\s+\d+\s*\]$", re.I)

HEADINGS: dict[str, set[str]] = {
    "summary": {
        "career objective", "objective", "career summary", "profile summary",
        "professional summary", "summary", "profile", "about me", "about",
        "personal profile", "executive summary", "professional profile",
        "career profile", "overview", "personal statement",
    },
    "education": {
        "academic career", "education", "educations", "academic qualifications",
        "educational qualifications", "qualification", "qualifications",
        "educational background", "academic background", "academic history",
        "education history", "schooling", "college", "colleges", "degrees",
        "degree", "academic details", "educational details", "training and education",
    },
    "experience": {
        "professional experience", "work experience", "employment history",
        "work history", "experience", "experiences", "career history",
        "professional background", "employment", "employment experience",
        "industry experience", "work profile", "career", "internships",
        "internship experience", "professional journey",
    },
    "projects": {
        "academic project undertaken", "academic projects", "projects",
        "project", "project experience", "selected projects", "selected work",
        "personal projects", "key projects", "portfolio", "research projects",
    },
    "certifications": {
        "certification courses", "certifications", "certificates", "certification",
        "licenses and certifications", "courses and certifications",
        "training", "courses", "professional certifications", "licenses",
    },
    "publications": {"publications", "publication", "research publications", "papers", "research papers"},
    "instruments": {"analytical instruments", "laboratory instruments", "instruments"},
    "professional_skills": {
        "professional skills", "soft skills", "personal skills", "interpersonal skills",
        "behavioral skills", "core competencies", "competencies", "strengths",
        "key strengths", "areas of strength",
    },
    "languages": {
        "languages known", "language proficiency", "languages", "language skills",
        "known languages", "linguistic skills",
    },
    "technical_skills": {
        "technical skills", "technical skill", "technical competencies",
        "technical expertise", "skills", "skill set", "key skills",
        "core skills", "areas of expertise", "expertise", "technologies",
        "technology stack", "tools and technologies", "tools", "software skills",
        "computer skills", "it skills", "knowledge and skills",
    },
}


def _clean_line(line: str) -> str:
    line = line.replace("\uf0b7", "•").replace("\u2022", "•").strip()
    line = re.sub(r"\s+", " ", line)
    if not line or PAGE_MARKER.fullmatch(line) or line.lower() in {"i", "[embedded links]"}:
        return ""
    return line


def _heading(line: str) -> str | None:
    normalized = re.sub(r"[^a-z0-9 ]", "", line.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # 1. Check exact matches in the known dictionary
    matched = next((section for section, names in HEADINGS.items() if normalized in names), None)
    if matched:
        return matched

    # 2. Dynamic heuristic classification for unknown or custom section headers
    if _is_likely_heading(normalized):
        if any(w in normalized for w in ["edu", "college", "school", "academic", "degree", "qualification"]):
            return "education"
        if any(w in normalized for w in ["exp", "work", "job", "career", "employment", "history", "role", "professional"]):
            return "experience"
        if any(w in normalized for w in ["skill", "tech", "stack", "tool", "competenc", "expertise"]):
            return "technical_skills"
        if any(w in normalized for w in ["project", "portfolio", "build", "ventures"]):
            return "projects"
        if any(w in normalized for w in ["cert", "license", "course", "training"]):
            return "certifications"
        if any(w in normalized for w in ["publi", "paper", "article"]):
            return "publications"
        if any(w in normalized for w in ["award", "honor", "achievement", "accomplish"]):
            return "awardsAndAchievements"

    return None


def _is_likely_heading(line: str) -> bool:
    """Detects custom/dynamic section titles that aren't in the static dictionary."""
    if not line or len(line) > 50 or len(line.split()) > 6:
        return False
    if EMAIL.search(line) or URL.search(line) or PHONE.search(line) or YEAR.search(line):
        return False
    if re.search(r"[.!?;,]", line):
        return False
    words = line.split()
    is_title_or_upper = all(w[0].isupper() or w.isupper() for w in words if w)
    return is_title_or_upper or line.endswith(":")


def _sections(lines: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {key: [] for key in HEADINGS}
    current: str | None = None
    for line in lines:
        found = _heading(line)
        if found:
            current = found
        elif current and line:
            result[current].append(line)
    return result


def _looks_like_unmapped_heading(line: str) -> bool:
    """Conservative heading heuristic for resume-specific/unfamiliar sections."""
    if re.match(r"^\s*[•*\-–—]\s+", line):
        return False
    value = _without_bullet(line).strip().rstrip(":")
    if not value or len(value) > 72 or len(value.split()) > 8:
        return False
    if EMAIL.search(value) or URL.search(value) or PHONE.search(value):
        return False
    if re.search(r"[.!?;,]", value) or re.search(r"\d", value):
        return False
    if len(value.split()) == 1 and len(value) < 3:
        return False
    words = value.split()
    title_case = len(words) <= 6 and all(
        word[:1].isupper() or word.isupper() or word.lower() in {"and", "of", "for", "in", "to", "&"}
        for word in words
    )
    uppercase = value.upper() == value and any(ch.isalpha() for ch in value)
    return uppercase or title_case or line.strip().endswith(":")


def _extract_raw_sections(lines: list[str]) -> dict[str, str]:
    """Preserve every recognizable section as source text, including unknown headings."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    started = False
    for line in lines:
        normalized = _heading(line)
        is_heading = bool(normalized) or (started and _looks_like_unmapped_heading(line))
        if is_heading:
            heading = re.sub(r"\s+", " ", _without_bullet(line).strip().rstrip(":"))
            if not heading:
                continue
            key = heading
            count = 2
            while key in sections:
                key = f"{heading} ({count})"
                count += 1
            sections[key] = []
            current = key
            started = True
        elif current and line.strip():
            sections[current].append(line.strip())
    return {heading: "\n".join(content).strip() for heading, content in sections.items()}


def _without_bullet(line: str) -> str:
    return re.sub(r"^[\s•*\-–—]+", "", line).strip()


def _dedupe(values: list[str]) -> list[str]:
    return list(OrderedDict.fromkeys(v.strip() for v in values if v and v.strip()))


def _join_wrapped(lines: list[str]) -> str:
    return re.sub(r"\s+", " ", " ".join(_without_bullet(line) for line in lines if _without_bullet(line))).strip()


def _name_position(lines: list[str]) -> tuple[str, str]:
    all_candidates = [_without_bullet(line) for line in lines]
    candidates = all_candidates[:30]
    name = next((line for line in candidates
        if 1 < len(line.split()) <= 4 and not any(ch.isdigit() for ch in line)
        and not EMAIL.search(line) and not PHONE.search(line) and not URL.search(line)
        and not _heading(line) and "|" not in line
        and not any(word in line.lower() for word in ("resume", "curriculum vitae", "career objective", "academic career", "technical skills", "languages known"))), "Unknown")
    position = next((line for line in all_candidates if "|" in line and len(line) <= 80
        and not EMAIL.search(line) and not PHONE.search(line) and not URL.search(line)), "")
    if not position and name in all_candidates:
        name_index = all_candidates.index(name)
        for line in all_candidates[name_index + 1:name_index + 6]:
            if (line and len(line) <= 100 and not _heading(line)
                and not EMAIL.search(line) and not PHONE.search(line) and not URL.search(line)
                and not re.search(r"@|https?://|www\.|\b(?:street|road|lane|kolkata|india|west bengal)\b", line, re.I)):
                position = line
                break
    return name, position or "Imported Resume"


def _education_records(lines: list[str]) -> list[str]:
    records: list[str] = []
    current: list[str] = []
    start = re.compile(r"^(?:M\s*\.?\s*Sc\.?|B\s*\.?\s*Sc\.?|Ph\s*\.?\s*D\.?|MBA|BTech|B\.?\s*Tech\.?|MTech|M\.?\s*Tech\.?|12th|10th|Higher Secondary|Secondary|Diploma)\b", re.I)
    for line in lines:
        item = _without_bullet(line)
        if not item:
            continue
        if start.match(item):
            if current:
                records.append(_join_wrapped(current))
            current = [item]
        elif current:
            current.append(item)
    if current:
        records.append(_join_wrapped(current))
    return records


def _parse_education(lines: list[str]) -> dict[str, dict[str, Any]]:
    qualifications: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(_education_records(lines), start=1):
        year_match = YEAR.search(record)
        year = year_match.group(0) if year_match else None
        marks_match = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*%", record)
        if re.match(r"^M\s*\.?\s*Sc", record, re.I):
            degree = "M.Sc. Chemistry" if "chemistry" in record.lower() else "M.Sc."
        elif re.match(r"^B\s*\.?\s*Sc", record, re.I):
            degree = "B.Sc. Chemistry" if "chemistry" in record.lower() else "B.Sc."
        elif re.match(r"^12th", record, re.I):
            degree = "12th"
        elif re.match(r"^10th", record, re.I):
            degree = "10th"
        else:
            degree = record.split(" from ", 1)[0].strip() or f"Qualification {index}"
        match = re.search(r"\bfrom\s+(.+?)(?=,?\s*(?:in\s*)?(?:19|20)\d{2}\b|,?\s*having\b|$)", record, re.I)
        institution = match.group(1).strip(" ,.;") if match else record[len(degree):].strip(" ,.;")
        qualifications[degree] = {
            "priority": index, "institutionName": institution, "startedAt": "",
            "yearOfComplete": year,
            "pursuing": bool(re.search(r"\b(pursuing|currently studying|in progress)\b", record, re.I)),
            "percentage": f"{marks_match.group(1)}%" if marks_match else None,
            "description": record,
        }
    return qualifications


def _bullet_items(lines: list[str]) -> list[str]:
    items: list[str] = []
    current = ""
    for raw in lines:
        line = raw.strip()
        if not line or line in {"•", "*", "-", "–"}:
            if current:
                items.append(current.strip())
                current = ""
            continue
        has_bullet = bool(re.match(r"^[•*\-–—]\s*", line))
        clean = _without_bullet(line)
        if has_bullet:
            if current:
                items.append(current.strip())
            current = clean
        elif current:
            current += " " + clean
        else:
            items.append(clean)
    if current:
        items.append(current.strip())
    return _dedupe(items)


def _split_skill_values(lines: list[str]) -> list[str]:
    values: list[str] = []
    for raw in lines:
        line = _without_bullet(raw).strip()
        if not line:
            continue
        if ":" in line and len(line.split(":", 1)[0].split()) <= 5:
            _, remainder = line.split(":", 1)
            if remainder.strip():
                line = remainder.strip()
        parts = re.split(r"[,;|•·]+|\s+\u2022\s+", line)
        for part in parts:
            item = re.sub(r"\s+", " ", part).strip(" .\t-–—")
            if item and item.lower() not in {"skills", "technical skills", "professional skills"}:
                values.append(item)
    return _dedupe(values)


def _parse_skills(sections: dict[str, list[str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    technical_lines = sections.get("technical_skills", [])
    technical_text = _join_wrapped(technical_lines)
    if technical_text:
        matches = re.findall(
            r"(Software|Standards Knowledge)\s*:\s*(.*?)(?=(?:Software|Standards Knowledge)\s*:|$)",
            technical_text, re.I
        )
        if matches:
            for label, value in matches:
                result[label.strip().title()] = _split_skill_values([value])
        else:
            values = _split_skill_values(technical_lines)
            if values:
                result["Imported skills"] = values

    instruments = _split_skill_values(sections.get("instruments", []))
    if instruments:
        result["Analytical Instruments"] = instruments
    professional = _split_skill_values(sections.get("professional_skills", []))
    if professional:
        result["Professional Skills"] = professional
    return {key: value for key, value in result.items() if value}


def _parse_experience(lines: list[str]) -> dict[str, Any] | None:
    content = [_without_bullet(line) for line in lines if _without_bullet(line)]
    if not content:
        return None
    title = company = date_text = ""
    responsibilities: list[str] = []
    for line in content:
        if not title and not YEAR.search(line) and not re.search(r"\b(ongoing|present|current)\b", line, re.I):
            title = line
            continue
        if not company and re.search(r"\b(center|centre|company|ltd|limited|inc\.?|corporation|laborator|research)\b", line, re.I):
            company = line.rstrip(".")
            continue
        if not date_text and YEAR.search(line):
            date_text = line
            continue
        if re.match(r"^(conducted|performed|maintained|managed|developed|responsible|analyzed|analysed|ensured|prepared|operated|supported|assisted|led|created|implemented)\b", line, re.I):
            responsibilities.append(line)
        elif responsibilities:
            responsibilities[-1] += " " + line
    year = YEAR.search(date_text)
    month = re.search(r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b", date_text, re.I)
    start_date = f"{month.group(0)} {year.group(0)}" if year and month else (year.group(0) if year else "")
    return {"priority": 1, "sectionTitle": "Professional Experience", "companies": {
        company or "Employer not specified": {
            "priority": 1, "jobTitle": title or "Not specified", "jobConditions": "", "jobTypes": "",
            "responsibility": _dedupe(responsibilities), "startDate": start_date, "endDate": None,
            "isPresentJob": bool(re.search(r"\b(ongoing|present|current)\b", date_text, re.I)),
        }
    }}


def extract_locally(text: str, blocks: list[dict]) -> tuple[ResumeInfo, dict]:
    lines = [line for line in (_clean_line(item) for item in text.splitlines()) if line]
    full_name, position = _name_position(lines)
    sections = _sections(lines)
    raw_sections = _extract_raw_sections(lines)
    emails = _dedupe(EMAIL.findall(text))
    phones = _dedupe([
        re.sub(r"\s+", " ", value).strip()
        for value in PHONE.findall(text)
        if len(re.sub(r"\D", "", value)) >= 10
    ])
    urls = _dedupe(URL.findall(text))
    summary_text = _join_wrapped(sections["summary"])
    education_lines = list(sections["education"])
    if not _education_records(education_lines):
        degree_signal = re.compile(
            r"\b(M\.?\s*Sc|B\.?\s*Sc|B\.?\s*Tech|M\.?\s*Tech|"
            r"BCA|MCA|BA|MA|BCom|MCom|BBA|MBA|Ph\.?D|Bachelor|Master|Diploma|"
            r"12th|10th|Higher Secondary|Senior Secondary|Matriculation|HSC|SSC)\b",
            re.I,
        )
        education_lines = [line for line in lines if degree_signal.search(line)]
    education = _parse_education(education_lines)
    skills_map = _parse_skills(sections)

    known_languages = {"english", "bengali", "hindi", "spanish", "french", "german", "mandarin", "chinese", "arabic", "urdu", "nepali", "tamil", "telugu", "marathi", "punjabi", "japanese", "korean", "portuguese", "russian"}
    language_names = _dedupe([_without_bullet(line).strip(" .") for line in sections["languages"] if _without_bullet(line).strip(" .").lower() in known_languages])
    language_items = [{"languageName": name, "proficiencyOutOfTen": 0} for name in language_names]
    experience = _parse_experience(sections["experience"])

    project_text = re.sub(r"^project\s*title\s*:\s*", "", _join_wrapped(sections["projects"]), flags=re.I).strip()
    projects = {"priority": 5, "sectionTitle": "Academic Project Undertaken", "projects": {project_text: {
        "priority": 1, "description": project_text, "startDate": "", "isWorking": False, "techStack": [], "skills": []
    }}} if project_text else None

    certification_text = _join_wrapped(sections["certifications"])
    certifications = None
    if certification_text:
        course_text = re.sub(r"^completed\s*:?[ \t]*", "", certification_text, flags=re.I).strip(" .")
        course_key = course_text[:120] or "Certification"
        certifications = {"priority": 6, "sectionTitle": "Certifications", "certificates": {course_key: {
            "priority": 1, "overview": certification_text, "skillLearned": [], "duration": ""
        }}}

    publication_text = _join_wrapped(sections["publications"])
    publications = None
    if publication_text:
        title_match = re.search(r"[\"“](.+?)[\"”]", publication_text)
        publication_title = title_match.group(1).strip() if title_match else publication_text[:160]
        publications = {"priority": 7, "sectionTitle": "Publications", "publications": {publication_title: {
            "priority": 1, "description": publication_text
        }}}

    payload: dict[str, Any] = {
        "basicInfo": {"fullName": full_name, "position": position},
        "address": {"streetName": "", "city": "", "district": "", "pincode": "", "country": ""},
        "contactInfo": {
            "primaryEmail": emails[0] if emails else "",
            "primaryMobile": phones[0] if phones else "",
            "linkedin": next((u for u in urls if "linkedin" in u.lower()), None),
            "github": next((u for u in urls if "github" in u.lower()), None),
            "portfolio": next((u for u in urls if "linkedin" not in u.lower() and "github" not in u.lower()), None),
        },
        "profileSummary": {"priority": 1, "subject": "Career Objective" if summary_text else "Professional Summary", "objective": summary_text or None},
        "educations": {"priority": 2, "sectionTitle": "Education", "qualifications": education},
        "skills": {"priority": 3, "sectionTitle": "Skills", "skills": skills_map},
        "languageProficiency": {"priority": 4, "sectionTitle": "Languages", "languageKnows": language_items},
        "templateName": "Imported Resume",
        "rawSections": raw_sections,
    }
    if experience: payload["workExperience"] = experience
    if projects: payload["projects"] = projects
    if certifications: payload["certifications"] = certifications
    if publications: payload["publications"] = publications
    resume = ResumeInfo.model_validate(payload)
    return resume, {
        "engine": "deterministic",
        "evidence": [{"fieldPath": "sourceText", "sourceChunkIds": [str(b.get("chunk_id", b.get("id", "source"))) for b in blocks], "confidence": 0.75}],
        "datesDetected": YEAR.findall(text),
        "notes": [],
    }