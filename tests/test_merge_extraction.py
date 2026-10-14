from app.services.merge import _merge_extraction_data


def test_merge_keeps_local_records_missing_from_ai():
    local = {
        "basicInfo": {"fullName": "Jane Doe", "position": "Engineer"},
        "educations": {"qualifications": {"BSc": {"institutionName": "College"}}},
        "skills": {"skills": {"Imported skills": ["Python", "MongoDB"]}},
    }
    ai = {
        "basicInfo": {"fullName": "Jane Doe", "position": "Senior Engineer"},
        "educations": {"qualifications": {"MSc": {"institutionName": "University"}}},
        "skills": {"skills": {"Imported skills": ["Python", "FastAPI"]}},
    }
    merged = _merge_extraction_data(local, ai)
    assert merged["basicInfo"]["position"] == "Senior Engineer"
    assert set(merged["educations"]["qualifications"]) == {"BSc", "MSc"}
    assert merged["skills"]["skills"]["Imported skills"] == ["Python", "FastAPI", "MongoDB"]


def test_merge_uses_local_value_when_ai_value_is_empty():
    assert _merge_extraction_data({"summary": "source summary"}, {"summary": ""}) == {
        "summary": "source summary"
    }
