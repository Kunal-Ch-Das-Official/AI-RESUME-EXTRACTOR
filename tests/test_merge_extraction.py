from app.services.merge import merge_extraction


def test_ai_records_replace_local_placeholders_but_fill_gaps():
    local = {
        "basicInfo": {"fullName": "Unknown", "position": "Imported Resume"},
        "address": {"city": "Kolkata", "country": ""},
        "contactInfo": {"primaryEmail": "jane@example.com", "primaryMobile": ""},
        "workExperience": {"companies": {"Employer not specified": {"jobTitle": "Not specified"}}},
        "educations": {"qualifications": {"BSc": {"institutionName": "College"}}},
        "skills": {"skills": {"Imported skills": ["Python"]}},
        "rawSections": {"Hobbies": "Chess"},
    }
    ai = {
        "basicInfo": {"fullName": "Jane Doe", "position": "Engineer"},
        "address": {"city": "", "country": "India"},
        "contactInfo": {"primaryEmail": "", "primaryMobile": "+91 98765 43210"},
        "workExperience": {"companies": {"Acme": {"jobTitle": "Engineer"}}},
        "educations": {"qualifications": {}},
        "skills": {"skills": {"Languages": ["Python", "Go"]}},
    }
    merged = merge_extraction(local, ai)
    assert merged["basicInfo"] == {"fullName": "Jane Doe", "position": "Engineer"}
    assert merged["address"] == {"city": "Kolkata", "country": "India"}
    assert merged["contactInfo"] == {
        "primaryEmail": "jane@example.com", "primaryMobile": "+91 98765 43210"
    }
    assert set(merged["workExperience"]["companies"]) == {"Acme"}  # no placeholder job
    assert set(merged["educations"]["qualifications"]) == {"BSc"}  # AI empty -> keep local
    assert merged["skills"]["skills"] == {"Languages": ["Python", "Go"]}
    assert merged["rawSections"] == {"Hobbies": "Chess"}


def test_empty_ai_result_keeps_local_untouched():
    local = {"basicInfo": {"fullName": "Jane", "position": "Dev"}, "skills": {"skills": {"a": ["x"]}}}
    assert merge_extraction(local, {})["skills"] == {"skills": {"a": ["x"]}}
