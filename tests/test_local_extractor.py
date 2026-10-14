from app.services.local_extractor import extract_locally


def test_local_extractor_preserves_contact_and_skills_without_an_api_key():
    resume, audit = extract_locally(
        """Jane Doe
Senior Python Engineer
jane@example.com | +1 555 123 4567

Skills
Python, FastAPI, MongoDB
""",
        [{"chunk_id": "source-1"}],
    )

    assert resume.basicInfo.fullName == "Jane Doe"
    assert resume.contactInfo.primaryEmail == "jane@example.com"
    assert resume.skills.skills["Imported skills"] == ["Python", "FastAPI", "MongoDB"]
    assert audit["engine"] == "deterministic"



def test_local_extractor_splits_comma_separated_skills():
    resume, _ = extract_locally(
        "Jane Doe\nSoftware Engineer\nSkills\nPython, FastAPI; MongoDB | Redis",
        [{"chunk_id": "skills"}],
    )
    assert resume.skills.skills["Imported skills"] == [
        "Python", "FastAPI", "MongoDB", "Redis"
    ]


def test_local_extractor_does_not_hardcode_certificate_names():
    resume, _ = extract_locally(
        "Jane Doe\nResearcher\nCertifications\nCompleted: Practical Data Analysis",
        [{"chunk_id": "certs"}],
    )
    assert "Practical Data Analysis" in resume.certifications.certificates


def test_functional_resume_preserves_all_distinct_sections_and_does_not_infer_phone():
    source = """Functional Resume Sample
John W. Smith
2002 Front Range Way Fort Collins, CO 80525
jwsmith@colostate.edu

Career Summary
Four years experience in early childhood development with a diverse background in the care of special needs children and adults.

Adult Care Experience
• Determined work placement for 150 special needs adult clients.
• Maintained client databases and records.
• Managed 25 volunteer workers.

Childcare Experience
• Coordinated service assignments for 20 part-time counselors and 100 client families.
• Oversaw daily activity and outing planning for 100 clients.

Employment History
1999-2002
Counseling Supervisor, The Wesley Center, Little Rock, Arkansas.
1997-1999
Client Specialist, Rainbow Special Care Center, Little Rock, Arkansas
1996-1997
Teacher's Assistant, Cowell Elementary, Conway, Arkansas

Education
University of Arkansas at Little Rock, Little Rock, AR
• BS in Early Childhood Development (1999)
• BA in Elementary Education (1998)
• GPA (4.0 Scale): Early Childhood Development – 3.8, Elementary Education – 3.5, Overall 3.4.
• Dean's List, Chancellor's List
"""
    resume, _ = extract_locally(source, [{"chunk_id": "functional-resume"}])
    sections = resume.model_dump(mode="json").get("rawSections", {})
    assert {"Career Summary", "Adult Care Experience", "Childcare Experience", "Employment History", "Education"}.issubset(sections)
    assert "150 special needs adult clients" in sections["Adult Care Experience"]
    assert "100 client families" in sections["Childcare Experience"]
    assert "1997-1999" in sections["Employment History"]
    assert "BA in Elementary Education" in sections["Education"]
    assert "Overall 3.4" in sections["Education"]
    assert resume.contactInfo.primaryMobile == ""
