import json

from app.models.resume import ResumeInfo


TARGET_SCHEMA = ResumeInfo.model_json_schema()


EXTRACTION_INSTRUCTIONS = """
You are the semantic extraction engine for AI-Resume.

Your job is to transform an arbitrary resume into the application's ResumeInfo
schema WITHOUT losing source information.

You must reason semantically. Never classify a section merely because its
heading matches a word. Consider the heading, nearby content, formatting clues,
dates, entities, bullets, and the retrieved knowledge examples.

CRITICAL INSTRUCTIONS FOR DYNAMIC & UNFAMILIAR HEADINGS:
1. Resumes use completely dynamic, custom, and unpredictable section headings (e.g., "Where I've Made an Impact", "Core Stack", "Academic Journey", "Selected Ventures", "Key Highlights", "Professional Background").
2. NEVER drop a section or data just because its title is unique or unfamiliar. You must analyze the content beneath any heading and semantically map it to the closest canonical schema field:
   - educations: Degrees, academic background, schooling, qualifications, university, college.
   - workExperience: Employment, jobs, career history, roles, professional background, internships.
   - skills: Technologies, tools, competencies, technical arsenal, frameworks, programming languages.
   - projects: Selected work, personal builds, portfolio items, side projects, research builds.
   - certifications: Licenses, credentials, certificates, courses, training.
   - publications: Research papers, articles, authored works.
   - awardsAndAchievements: Honors, recognitions, awards, distinctions.
3. If a section contains valuable professional details that do not fit standard fields, preserve them inside the closest matching section or ensure they are fully captured in description/summary fields rather than omitted.

OUTPUT CONTRACT:
Return an object with:
- `resume`: the complete candidate ResumeInfo object.
- `evidence`: one or more evidence records. Each evidence record must contain
  `fieldPath`, `sourceChunkIds`, and a confidence from 0 to 1.
Every non-empty important field should have evidence pointing to source chunks.

RULES:
1. Extract only information supported by THIS resume. Never use another candidate's
   personal facts from retrieved knowledge as facts about this candidate.
2. Never invent missing information, dates, employers, marks, proficiency scores,
   job titles, institutions, URLs, or achievements.
3. Preserve original wording when practical and preserve every meaningful bullet
   separately where the target schema allows it.
4. Preserve URLs, dates, technologies, scores, institution names, employer names,
   locations, and all supported sections across every page.
5. Keep entries in source order. Reconstruct multi-column and wrapped text using
   the supplied page/block/chunk context; ignore page markers such as [PAGE 1].
6. Classify by MEANING, not exact heading matches. Headings are noisy, optional,
   abbreviated, misspelled, or user-defined. Examples:
   - EDUCATION: education(s), academic background/history/career, qualifications,
     schooling, academics, degrees, college, university, educational details,
     training and education. Strong content clues include M.Sc/MSc, B.Sc/BSc,
     B.Tech, M.Tech, BA/MA, BCA/MCA, MBA, PhD, diploma, bachelor's/master's,
     12th/10th, HSC/SSC, matriculation, institutions, graduation years and marks.
   - EXPERIENCE: work/professional/employment/career history, industry experience,
     positions, internships, employment, professional background. Content clues
     include employer + role, employment date ranges, current/present, and duties.
   - SKILLS: skills, technical/professional/personal/soft skills, competencies,
     expertise, strengths, tools, technologies, software, instruments, methods,
     languages/programming languages. Distinguish natural languages from technical
     skills when context says "Languages known".
   - SUMMARY: summary, objective, profile, about me, personal statement, overview.
     Put the section heading in profileSummary.subject and its prose in objective.
   - PROJECTS: projects, selected work, academic/research/personal projects,
     portfolio. Identify title, description, technologies, URLs and dates.
   - CERTIFICATIONS: certifications, certificates, licenses, courses, training.
   - PUBLICATIONS: papers, publications, research papers, articles.
   - LANGUAGES: languages known, language proficiency, linguistic skills.
   - Also identify contact details, address, awards, achievements, hobbies,
     open-source contributions, and references when the schema supports them.
7. A section may be identified without a heading. Use multiple content signals and
   neighboring lines, not a single keyword alone. For example, a degree plus an
   institution and graduation year is strong education evidence; a role plus
   employer, date range and responsibility bullets is strong work evidence.
8. Do not put education, work history, project descriptions, or section headings
   inside skills. Do not copy the entire resume into summary or position.
9. The schema is fixed. Map extracted facts into the existing ResumeInfo shape;
   do not add fields or alter the schema.
10. Evidence IDs must point to actual source chunk IDs supplied in the prompt.
11. If a value is unsupported, leave it empty/null according to the schema. If
    the schema requires a value, use only its established empty/default convention;
    do not fabricate a plausible value.
12. Include every section that is actually present, but omit genuinely absent
    optional sections. Do not create empty placeholder jobs/projects just because
    those fields exist in the schema.

The target schema inside `resume` is:
""" + json.dumps(TARGET_SCHEMA, indent=2)


VERIFICATION_INSTRUCTIONS = """
You are the final quality-control verifier for a resume import.

Compare the proposed structured resume against the source chunks.

Find:
- missing information;
- wrong section classification;
- fabricated values;
- missing bullets;
- incorrect dates;
- wrong current/present flags;
- missing URLs;
- duplicate/mismerged entries;
- unsupported values.

Return an object with:
- `resume`: the complete corrected ResumeInfo object.
- `evidence`: field-level source evidence.
- `issues`: concise problems found during verification.

Return corrections only when the source supports them.
Do not add facts that are not present.
For each correction, provide the exact source chunk IDs that prove it.
"""


def build_extraction_prompt(raw_text: str, chunks: list[dict], knowledge: list[dict]) -> str:
    compact_chunks = [
        {"chunk_id": c["chunk_id"], "pages": c["pages"], "text": c["text"]}
        for c in chunks
    ]
    compact_knowledge = [
        {
            "kind": item.get("kind"),
            "field": item.get("field"),
            "text": item.get("text"),
            "score": item.get("score"),
        }
        for item in knowledge
    ]

    return (
        EXTRACTION_INSTRUCTIONS
        + "\n\nRETRIEVED KNOWLEDGE:\n"
        + json.dumps(compact_knowledge, ensure_ascii=False, indent=2)
        + "\n\nSOURCE CHUNKS:\n"
        + json.dumps(compact_chunks, ensure_ascii=False, indent=2)
        + "\n\nCOMPLETE RAW TEXT:\n"
        + raw_text
    )


def build_verification_prompt(
    proposed: dict,
    chunks: list[dict],
    knowledge: list[dict],
) -> str:
    return (
        VERIFICATION_INSTRUCTIONS
        + "\n\nPROPOSED EXTRACTION:\n"
        + json.dumps(proposed, ensure_ascii=False, indent=2)
        + "\n\nSOURCE CHUNKS:\n"
        + json.dumps(chunks, ensure_ascii=False, indent=2)
        + "\n\nRETRIEVED KNOWLEDGE:\n"
        + json.dumps(knowledge, ensure_ascii=False, indent=2)
    )