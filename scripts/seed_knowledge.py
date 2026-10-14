import asyncio

from app.services.knowledge import upsert_knowledge


KNOWLEDGE = [
    ("field_definition", "basicInfo", "Full name and professional/current title shown in the resume."),
    ("field_definition", "contactInfo", "Email, phone, LinkedIn, GitHub, portfolio and other direct contact details."),
    ("field_definition", "profileSummary", "Professional introduction, objective, profile, career summary or about section."),
    ("field_definition", "educations", "Degrees, qualifications, academic history, schools, universities, academic results."),
    ("field_definition", "workExperience", "Employment, jobs, professional experience, career history, internships, freelance roles."),
    ("field_definition", "skills", "Technical skills, competencies, expertise, tools, technologies, capabilities."),
    ("field_definition", "projects", "Projects, selected work, personal projects, portfolio projects and project experience."),
    ("field_definition", "certifications", "Certificates, licenses, credentials and professional certifications."),
    ("field_definition", "openSource", "Open-source contributions, community code contributions and public repositories."),
    ("field_definition", "publications", "Research papers, authored papers, articles, publications and research outputs."),
    ("field_definition", "awardsAndAchievements", "Awards, honors, accomplishments, distinctions and notable achievements."),
    ("field_definition", "languageProficiency", "Human languages and their stated fluency/proficiency."),
    ("field_definition", "hobbies", "Personal interests, hobbies and leisure activities."),
    ("synonym", "educations", "Qualification, Qualifications, Academic Background, Academic History, Educational Background, Academic Journey can describe education when the content contains degrees, schools, institutions or academic results."),
    ("synonym", "workExperience", "Employment, Professional Background, Career History, Career Experience, Work History and Experience can describe jobs when the content contains employers, positions, responsibilities and dates."),
    ("synonym", "projects", "Selected Work, Key Work, Portfolio, Personal Work, Project Experience and Selected Projects can describe projects when the content contains project names, descriptions, technologies or project links."),
    ("synonym", "skills", "Core Competencies, Technical Arsenal, Expertise, Technologies, Professional Skills, Key Skills and Competencies can describe skills when the content lists capabilities or technologies."),
    ("synonym", "profileSummary", "Career Objective, Professional Summary, Profile, About Me, Career Profile and Executive Summary can describe the professional introduction."),
    ("synonym", "publications", "Research Output, Papers, Articles, Research Publications and Scholarly Work can describe publications."),
    ("synonym", "awardsAndAchievements", "Honors, Distinctions, Accomplishments, Recognition and Awards can describe achievements."),
    ("edge_case", None, "A heading alone is insufficient evidence. Inspect the content below/near the heading before assigning a canonical section."),
    ("edge_case", None, "A section may contain multiple semantic types. Split entries when evidence clearly indicates different canonical sections."),
    ("edge_case", None, "Do not convert a person's job title into a project, and do not convert a project technology list into work experience."),
    ("edge_case", None, "Do not invent dates. Preserve human-readable source dates if exact normalization is uncertain."),
    ("edge_case", None, "Current, present, ongoing and till date indicate an active role when supported by context."),
    ("edge_case", None, "Preserve every responsibility bullet separately whenever the target schema supports a list."),
    ("edge_case", None, "If a URL is visible or embedded in the document, preserve it. Never fabricate a URL."),
]


async def main():
    for kind, field, text in KNOWLEDGE:
        await upsert_knowledge(
            text=text,
            kind=kind,
            field=field,
            source="seed",
        )
    print(f"Seeded {len(KNOWLEDGE)} knowledge documents.")


if __name__ == "__main__":
    asyncio.run(main())
