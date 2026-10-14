from app.models.resume import ResumeInfo


def test_resume_schema_can_validate_minimum_shape():
    data = {
        "basicInfo": {"fullName": "Jane Doe", "position": "Engineer"},
        "address": {
            "streetName": "",
            "city": "",
            "district": "",
            "pincode": "",
            "country": "",
        },
        "contactInfo": {
            "primaryEmail": "jane@example.com",
            "primaryMobile": "1234567890",
        },
        "profileSummary": {"priority": 1, "subject": "Engineer"},
        "educations": {"priority": 1, "qualifications": {}},
        "skills": {"priority": 1, "skills": {}},
        "languageProficiency": {"priority": 1, "languageKnows": []},
    }
    resume = ResumeInfo.model_validate(data)
    assert resume.basicInfo.fullName == "Jane Doe"



def test_resume_accepts_missing_email_without_losing_other_fields():
    data = {
        "basicInfo": {"fullName": "Jane Doe", "position": "Engineer"},
        "address": {"streetName": "", "city": "", "district": "", "pincode": "", "country": ""},
        "contactInfo": {"primaryEmail": "", "primaryMobile": ""},
        "profileSummary": {"priority": 1, "subject": "Engineer"},
        "educations": {"priority": 1, "qualifications": {}},
        "skills": {"priority": 1, "skills": {}},
        "languageProficiency": {"priority": 1, "languageKnows": []},
    }
    resume = ResumeInfo.model_validate(data)
    assert resume.basicInfo.fullName == "Jane Doe"
    assert resume.contactInfo.primaryEmail == ""


def test_resume_normalizes_bare_linkedin_url():
    data = {
        "basicInfo": {"fullName": "Jane Doe", "position": "Engineer"},
        "address": {"streetName": "", "city": "", "district": "", "pincode": "", "country": ""},
        "contactInfo": {
            "primaryEmail": "jane@example.com",
            "primaryMobile": "",
            "linkedin": "www.linkedin.com/in/jane-doe",
        },
        "profileSummary": {"priority": 1, "subject": "Engineer"},
        "educations": {"priority": 1, "qualifications": {}},
        "skills": {"priority": 1, "skills": {}},
        "languageProficiency": {"priority": 1, "languageKnows": []},
    }
    resume = ResumeInfo.model_validate(data)
    assert resume.contactInfo.linkedin == "https://www.linkedin.com/in/jane-doe"
