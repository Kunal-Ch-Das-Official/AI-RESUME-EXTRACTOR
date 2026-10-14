"""Parity checks for the canonical src/schema/resume.schema.ts contract.

These assertions deliberately cover the JSON field names, required top-level
sections and file-reference payload used by the TypeScript HTTP schema.
"""
from app.models.resume import ResumeInfo


def test_python_resume_schema_has_typescript_required_sections():
    schema = ResumeInfo.model_json_schema()
    required = set(schema["required"])
    assert {
        "basicInfo",
        "address",
        "contactInfo",
        "profileSummary",
        "educations",
        "skills",
        "languageProficiency",
    } <= required


def test_file_reference_matches_typescript_multipart_metadata_shape():
    data = {
        "basicInfo": {"fullName": "Jane Doe", "position": "Engineer"},
        "address": {"streetName": "", "city": "", "district": "", "pincode": "", "country": ""},
        "contactInfo": {"primaryEmail": "jane@example.com", "primaryMobile": "1234567890"},
        "profileSummary": {"priority": 1, "subject": "Engineer"},
        "educations": {"priority": 1, "qualifications": {}},
        "skills": {"priority": 1, "skills": {}},
        "languageProficiency": {"priority": 1, "languageKnows": []},
        "certifications": {
            "priority": 1,
            "certificates": {
                "Example": {
                    "priority": 1,
                    "skillLearned": [],
                    "duration": "1 month",
                    "certificateContent": {"fileIndex": 0, "title": "certificate.pdf"},
                }
            },
        },
    }
    assert ResumeInfo.model_validate(data).certifications is not None
