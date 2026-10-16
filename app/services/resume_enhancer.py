# app/services/resume_enhancer.py
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Tuple

import httpx
from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)

# Core list of high-impact ATS verbs recognized by parser dictionaries
STRONG_ACTION_VERBS = [
    "Spearheaded",
    "Architected",
    "Engineered",
    "Orchestrated",
    "Implemented",
    "Streamlined",
    "Optimized",
    "Deployed",
    "Developed",
    "Automated",
    "Accelerated",
    "Formulated",
    "Consolidated",
    "Maximized",
    "Delivered",
]

POLISH_SYSTEM_PROMPT = """You are an Elite Executive Resume Strategist and ATS Algorithm Specialist.
Your sole mission is to polish candidate resume descriptions so they score in the TOP 10% (85+) on Applicant Tracking Systems (ATS).

EVERY POLISHED BULLET AND DESCRIPTION MUST MEET THESE THREE RULES:
1. FORMULA: Begin immediately with a high-impact past-tense action verb (e.g., Spearheaded, Engineered, Architected, Optimized, Deployed, Streamlined).
2. METRICS & IMPACT: PRESERVE every existing number, percentage, and metric. If a bullet lacks a quantified metric, phrase it around business impact, efficiency, scalability, or latency reduction. NEVER delete existing numbers.
3. KEYWORDS: Retain all technical stack keywords, framework names, and tools.

ABSOLUTE CONSTRAINTS:
- DO NOT alter candidate names, personal details, dates, companies, or degrees.
- DO NOT invent false work experiences.
- Output MUST be 100% valid JSON matching the requested schema. No conversational markdown wrappers.
"""

ATS_SYSTEM_PROMPT = """You are an automated Applicant Tracking System (ATS) parser engine and recruiter scoring rubric.
Score the provided resume rigorously out of 100 based on:
1. Impact & Metrics (presence of %, numbers, latency, scale, or business outcomes)
2. Action Verbs (every bullet starting with an executive action verb)
3. ATS Clarity & Tech Keyword Alignment

OUTPUT REQUIREMENT:
Output MUST be 100% valid JSON matching the schema requested.
"""


def _get_gemini_client():
    settings = get_settings()
    api_key = getattr(settings, "gemini_api_key", None)
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


async def _call_openrouter(
    system_prompt: str, prompt: str, temperature: float = 0.2
) -> str | None:
    settings = get_settings()
    api_key = getattr(settings, "openrouter_api_key", None)
    if not api_key or not str(api_key).startswith("sk-or-"):
        return None

    model = getattr(
        settings, "openrouter_model", "meta-llama/llama-3.3-70b-instruct:free"
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://airesume.download",
        "X-Title": "AI Resume Builder",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"] or ""
            return None
    except Exception as exc:
        logger.warning("OpenRouter call failed: %s", exc)
        return None


async def _call_llm_resilient(
    system_prompt: str, prompt: str, temperature: float = 0.2
) -> str | None:
    settings = get_settings()
    client = _get_gemini_client()
    primary_model = getattr(settings, "gemini_model", "gemini-2.5-flash")
    fallback_model = "gemini-1.5-flash"

    if client:
        try:
            res = await client.aio.models.generate_content(
                model=primary_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                ),
            )
            if res.text:
                return res.text
        except Exception as exc:
            logger.warning("Gemini primary (%s) failed: %s", primary_model, exc)

        if fallback_model != primary_model:
            try:
                res = await client.aio.models.generate_content(
                    model=fallback_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                    ),
                )
                if res.text:
                    return res.text
            except Exception as exc:
                logger.warning("Gemini secondary (%s) failed: %s", fallback_model, exc)

    return await _call_openrouter(system_prompt, prompt, temperature)


def _clean_llm_json(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _ensure_list(data: Any) -> List[Any]:
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.values())
    return [data]


def _deterministic_bullet_polish(bullet: str) -> str:
    """Deterministic, high-scoring ATS polish when cloud LLMs are busy."""
    text = bullet.strip().rstrip(".")
    if not text:
        return ""

    # Replace weak / passive openers with high-ranking ATS action verbs
    passive_replacements = [
        (
            r"^(i was responsible for|responsible for|helped with|assisted in)\s+",
            "Spearheaded ",
        ),
        (r"^(worked on|worked with|involved in)\s+", "Engineered and delivered "),
        (r"^(did|made|built|created)\s+", "Architected and deployed "),
        (r"^(fixed|fixed bugs in|handled)\s+", "Streamlined stability and optimized "),
        (r"^(updated|maintained)\s+", "Enhanced performance and maintained "),
    ]
    for pattern, repl in passive_replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # If sentence does not start with an uppercase action verb, prepend an assertive verb
    first_word = text.split()[0].capitalize() if text.split() else ""
    if first_word.lower() not in [v.lower() for v in STRONG_ACTION_VERBS]:
        text = f"Optimized {text[0].lower() + text[1:]}" if len(text) > 1 else text
    else:
        text = first_word + text[len(first_word) :]

    return text + "."


def _calculate_ats_heuristics(
    token_friendly_resume: Dict[str, Any],
) -> Tuple[int, Dict[str, int], List[str], List[str], List[str]]:
    """Calculates deterministic ATS score with consistent weights."""
    bullets: List[str] = []
    for exp in token_friendly_resume.get("experiences", []):
        bullets.extend(exp.get("responsibilities", []))
    for proj in token_friendly_resume.get("projects", []):
        if proj.get("description"):
            bullets.append(proj["description"])

    total_bullets = max(len(bullets), 1)

    # 1. Metric presence (% or numeric indicators)
    metric_regex = re.compile(r"\b(\d+[%kKmM]?|\$\d+|[0-9]+x|[0-9]+\+)\b")
    metrics_count = sum(1 for b in bullets if metric_regex.search(b))
    metric_ratio = min(1.0, metrics_count / total_bullets)
    impact_score = int(65 + (metric_ratio * 30))

    # 2. Action verbs
    verbs_lower = {v.lower() for v in STRONG_ACTION_VERBS}
    action_verb_count = sum(
        1 for b in bullets if b.strip() and b.strip().split()[0].lower() in verbs_lower
    )
    action_ratio = min(1.0, action_verb_count / total_bullets)
    action_score = int(65 + (action_ratio * 30))

    # 3. Skills & Structure
    skills = token_friendly_resume.get("skills", [])
    has_summary = bool(str(token_friendly_resume.get("profileSummary", "")).strip())
    ats_clarity = 92 if has_summary else 75
    keyword_match = min(96, 65 + (len(skills) * 3))

    overall = int(
        (impact_score * 0.35)
        + (action_score * 0.30)
        + (ats_clarity * 0.20)
        + (keyword_match * 0.15)
    )

    strengths = []
    if action_ratio >= 0.7:
        strengths.append(
            f"Strong action verb density: {action_verb_count}/{total_bullets} bullets lead with high-impact executive verbs."
        )
    if metrics_count > 0:
        strengths.append(f"Identified {metrics_count} quantified performance metrics.")
    if has_summary:
        strengths.append("High-visibility professional profile summary present.")
    if not strengths:
        strengths.append("Standard ATS section headers and chronological structure.")

    improvements = []
    if metric_ratio < 0.5:
        improvements.append(
            "Add measurable outcomes (e.g., % latency drop, throughput, hours saved per sprint)."
        )
    if action_ratio < 0.7:
        improvements.append(
            "Start each bullet with action verbs (e.g., Spearheaded, Architected, Optimized)."
        )
    if len(skills) < 8:
        improvements.append(
            "Include more specific technical stack keywords to pass recruiter keyword filters."
        )

    suggested_keywords = [
        "CI/CD Pipelines",
        "System Scalability",
        "Performance Optimization",
        "Cloud Deployment",
    ]
    skills_lower = [str(s).lower() for s in skills]
    missing = [kw for kw in suggested_keywords if kw.lower() not in skills_lower]

    categories = {
        "impactAndMetrics": impact_score,
        "atsClarity": ats_clarity,
        "actionVerbs": action_score,
        "keywordMatch": keyword_match,
    }

    return overall, categories, strengths, improvements, missing


def extract_polishable_payload(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}

    basic_info = resume_data.get("basicInfo") or {}
    if isinstance(basic_info, dict) and basic_info.get("profileSummary"):
        payload["profileSummary"] = str(basic_info["profileSummary"])

    work_exp = resume_data.get("workExperience") or {}
    experiences = _ensure_list(
        work_exp.get("experiences") if isinstance(work_exp, dict) else []
    )
    clean_exp = []
    for idx, exp in enumerate(experiences):
        if isinstance(exp, dict):
            resp = exp.get("responsibilities")
            if resp:
                resp_list = resp if isinstance(resp, list) else [str(resp)]
                clean_exp.append(
                    {
                        "id": idx,
                        "title": str(exp.get("jobTitle") or ""),
                        "responsibilities": [str(r) for r in resp_list if r],
                    }
                )
    if clean_exp:
        payload["experiences"] = clean_exp

    projects_root = resume_data.get("projects") or {}
    projects = _ensure_list(
        projects_root.get("projects") if isinstance(projects_root, dict) else []
    )
    clean_proj = []
    for idx, proj in enumerate(projects):
        if isinstance(proj, dict):
            desc = proj.get("description")
            if desc:
                clean_proj.append(
                    {
                        "id": idx,
                        "title": str(proj.get("title") or proj.get("name") or ""),
                        "description": str(desc),
                    }
                )
        elif isinstance(proj, str) and proj.strip():
            clean_proj.append(
                {"id": idx, "title": f"Project {idx + 1}", "description": proj}
            )
    if clean_proj:
        payload["projects"] = clean_proj

    os_root = resume_data.get("openSource") or {}
    os_projects = _ensure_list(
        os_root.get("projects") if isinstance(os_root, dict) else []
    )
    clean_os = []
    for idx, os_item in enumerate(os_projects):
        if isinstance(os_item, dict):
            desc = os_item.get("description")
            if desc:
                clean_os.append(
                    {
                        "id": idx,
                        "name": str(os_item.get("projectName") or ""),
                        "description": str(desc),
                    }
                )
    if clean_os:
        payload["openSource"] = clean_os

    ach_root = resume_data.get("awardsAndAchievements") or {}
    achievements = _ensure_list(
        ach_root.get("achievements") if isinstance(ach_root, dict) else []
    )
    clean_ach = []
    for idx, ach in enumerate(achievements):
        if isinstance(ach, dict):
            desc = ach.get("description")
            if desc:
                clean_ach.append(
                    {
                        "id": idx,
                        "title": str(ach.get("title") or ""),
                        "description": str(desc),
                    }
                )
    if clean_ach:
        payload["achievements"] = clean_ach

    return payload


async def polish_resume_content(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    polishable = extract_polishable_payload(resume_data)
    if not polishable:
        return resume_data

    user_prompt = f"""Rewrite and polish the following resume sections for ATS compliance.
RULES:
1. Every responsibility bullet must start with an executive action verb from this list:
   {json.dumps(STRONG_ACTION_VERBS)}
2. Retain all existing numbers, %, technologies, and dates. Do not lose any metrics.
3. Make the tone active, concise, and focused on outcomes.

INPUT CONTENT:
{json.dumps(polishable, separators=(',', ':'))}

Respond ONLY with valid JSON using this structure:
{{
  "profileSummary": "polished summary",
  "experiences": [
    {{"id": 0, "responsibilities": ["Spearheaded...", "Engineered..."]}}
  ],
  "projects": [
    {{"id": 0, "description": "Architected and delivered..."}}
  ],
  "openSource": [
    {{"id": 0, "description": "Developed and maintained..."}}
  ],
  "achievements": [
    {{"id": 0, "description": "Awarded for..."}}
  ]
}}
"""

    raw_response = await _call_llm_resilient(
        system_prompt=POLISH_SYSTEM_PROMPT,
        prompt=user_prompt,
        temperature=0.15,
    )

    polished: Dict[str, Any] = {}
    if raw_response:
        try:
            polished = json.loads(_clean_llm_json(raw_response))
        except Exception as exc:
            logger.warning("Could not parse LLM JSON response during polish: %s", exc)

    updated = json.loads(json.dumps(resume_data))

    # 1. Update Profile Summary
    if isinstance(updated.get("basicInfo"), dict):
        new_summary = polished.get("profileSummary")
        if isinstance(new_summary, str) and new_summary.strip():
            updated["basicInfo"]["profileSummary"] = new_summary.strip()

    # 2. Update Experiences
    if isinstance(updated.get("workExperience"), dict):
        exp_target = updated["workExperience"].get("experiences")
        if isinstance(exp_target, list):
            llm_exp_map = {
                item["id"]: item.get("responsibilities", [])
                for item in polished.get("experiences", [])
                if isinstance(item, dict) and "id" in item
            }
            for idx, exp in enumerate(exp_target):
                if isinstance(exp, dict):
                    if idx in llm_exp_map and llm_exp_map[idx]:
                        exp["responsibilities"] = llm_exp_map[idx]
                    elif isinstance(exp.get("responsibilities"), list):
                        exp["responsibilities"] = [
                            _deterministic_bullet_polish(str(b))
                            for b in exp["responsibilities"]
                            if str(b).strip()
                        ]

    # 3. Update Projects
    if isinstance(updated.get("projects"), dict):
        proj_target = updated["projects"].get("projects")
        if isinstance(proj_target, list):
            llm_proj_map = {
                item["id"]: item.get("description", "")
                for item in polished.get("projects", [])
                if isinstance(item, dict) and "id" in item
            }
            for idx, proj in enumerate(proj_target):
                if isinstance(proj, dict):
                    if idx in llm_proj_map and llm_proj_map[idx]:
                        proj["description"] = llm_proj_map[idx]
                    elif isinstance(proj.get("description"), str):
                        proj["description"] = _deterministic_bullet_polish(
                            proj["description"]
                        )

    # 4. Update Open Source
    if isinstance(updated.get("openSource"), dict):
        os_target = updated["openSource"].get("projects")
        if isinstance(os_target, list):
            llm_os_map = {
                item["id"]: item.get("description", "")
                for item in polished.get("openSource", [])
                if isinstance(item, dict) and "id" in item
            }
            for idx, os_item in enumerate(os_target):
                if isinstance(os_item, dict):
                    if idx in llm_os_map and llm_os_map[idx]:
                        os_item["description"] = llm_os_map[idx]
                    elif isinstance(os_item.get("description"), str):
                        os_item["description"] = _deterministic_bullet_polish(
                            os_item["description"]
                        )

    # 5. Update Achievements
    if isinstance(updated.get("awardsAndAchievements"), dict):
        ach_target = updated["awardsAndAchievements"].get("achievements")
        if isinstance(ach_target, list):
            llm_ach_map = {
                item["id"]: item.get("description", "")
                for item in polished.get("achievements", [])
                if isinstance(item, dict) and "id" in item
            }
            for idx, ach in enumerate(ach_target):
                if isinstance(ach, dict):
                    if idx in llm_ach_map and llm_ach_map[idx]:
                        ach["description"] = llm_ach_map[idx]
                    elif isinstance(ach.get("description"), str):
                        ach["description"] = _deterministic_bullet_polish(
                            ach["description"]
                        )

    return updated


async def evaluate_ats_score(
    resume_data: Dict[str, Any], target_role: str = ""
) -> Dict[str, Any]:
    basic_info = (
        resume_data.get("basicInfo")
        if isinstance(resume_data.get("basicInfo"), dict)
        else {}
    )
    skills_root = (
        resume_data.get("skills") if isinstance(resume_data.get("skills"), dict) else {}
    )
    work_exp = (
        resume_data.get("workExperience")
        if isinstance(resume_data.get("workExperience"), dict)
        else {}
    )
    projects_root = (
        resume_data.get("projects")
        if isinstance(resume_data.get("projects"), dict)
        else {}
    )
    educations_root = (
        resume_data.get("educations")
        if isinstance(resume_data.get("educations"), dict)
        else {}
    )

    clean_experiences = []
    for exp in _ensure_list(work_exp.get("experiences")):
        if isinstance(exp, dict):
            resp = exp.get("responsibilities", [])
            resp_list = resp if isinstance(resp, list) else [str(resp)]
            clean_experiences.append(
                {
                    "title": str(exp.get("jobTitle") or ""),
                    "company": str(exp.get("company") or ""),
                    "responsibilities": [str(r) for r in resp_list if r],
                }
            )

    clean_projects = []
    for proj in _ensure_list(projects_root.get("projects")):
        if isinstance(proj, dict):
            tech = proj.get("techStack", [])
            tech_list = tech if isinstance(tech, list) else [str(tech)]
            clean_projects.append(
                {
                    "title": str(proj.get("title") or proj.get("name") or ""),
                    "techStack": [str(t) for t in tech_list if t],
                    "description": str(proj.get("description") or ""),
                }
            )
        elif isinstance(proj, str) and proj.strip():
            clean_projects.append(
                {
                    "title": "Project",
                    "techStack": [],
                    "description": proj.strip(),
                }
            )

    clean_educations = []
    for edu in _ensure_list(educations_root.get("education")):
        if isinstance(edu, dict):
            clean_educations.append(
                {
                    "degree": str(edu.get("degree") or ""),
                    "institution": str(edu.get("institution") or ""),
                }
            )

    raw_skills = skills_root.get("technicalSkills", [])
    clean_skills = [str(s) for s in _ensure_list(raw_skills) if s]

    token_friendly_resume = {
        "targetRole": target_role or basic_info.get("headline", "Professional"),
        "headline": str(basic_info.get("headline") or ""),
        "profileSummary": str(basic_info.get("profileSummary") or ""),
        "skills": clean_skills,
        "experiences": clean_experiences,
        "projects": clean_projects,
        "educations": clean_educations,
    }

    # Always compute deterministic baseline to ensure consistency
    base_score, base_cats, base_str, base_imp, base_kw = _calculate_ats_heuristics(
        token_friendly_resume
    )

    user_prompt = f"""Evaluate this candidate resume for ATS compatibility and recruiter score out of 100.
Reference baseline score: {base_score}/100.
Ensure your score reflects whether the bullets lead with strong action verbs and contain metrics/outcomes.

Resume Data:
{json.dumps(token_friendly_resume, separators=(',', ':'))}

Respond ONLY with valid JSON using this exact structure:
{{
  "overallScore": {base_score},
  "categoryScores": {{
    "impactAndMetrics": {base_cats["impactAndMetrics"]},
    "atsClarity": {base_cats["atsClarity"]},
    "actionVerbs": {base_cats["actionVerbs"]},
    "keywordMatch": {base_cats["keywordMatch"]}
  }},
  "strengths": [
    "Identified quantified metrics and active phrasing",
    "Clean section structure"
  ],
  "improvements": [
    "Recommended improvement 1",
    "Recommended improvement 2"
  ],
  "missingKeywords": [
    "CI/CD Pipelines", "System Scalability"
  ]
}}
"""

    raw_response = await _call_llm_resilient(
        system_prompt=ATS_SYSTEM_PROMPT,
        prompt=user_prompt,
        temperature=0.1,
    )

    if not raw_response:
        return {
            "overallScore": base_score,
            "categoryScores": base_cats,
            "strengths": base_str,
            "improvements": base_imp,
            "missingKeywords": base_kw,
        }

    try:
        data = json.loads(_clean_llm_json(raw_response))
        # Ensure LLM score does not drop below deterministic baseline
        reported_score = int(data.get("overallScore", base_score))
        data["overallScore"] = max(base_score, min(100, reported_score))
        return data
    except Exception as exc:
        logger.warning("Could not parse ATS evaluation JSON: %s. Using baseline.", exc)
        return {
            "overallScore": base_score,
            "categoryScores": base_cats,
            "strengths": base_str,
            "improvements": base_imp,
            "missingKeywords": base_kw,
        }
