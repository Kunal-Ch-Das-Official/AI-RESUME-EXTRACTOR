from app.models.resume import ResumeInfo
from app.services.validator import validate_semantics


def test_validator_returns_list():
    data = {
        "basicInfo": {"fullName": "Jane Doe", "position": "Engineer"},
        "address": {"streetName": "", "city": "", "district": "", "pincode": "", "country": ""},
        "contactInfo": {"primaryEmail": "jane@example.com", "primaryMobile": "1234567890"},
        "profileSummary": {"priority": 1, "subject": "Engineer"},
        "educations": {"priority": 1, "qualifications": {}},
        "skills": {"priority": 1, "skills": {}},
        "languageProficiency": {"priority": 1, "languageKnows": []},
    }
    assert isinstance(validate_semantics(ResumeInfo.model_validate(data)), list)
