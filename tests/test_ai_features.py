import asyncio

import pytest

from app.config import Settings
from app.services.suggestion import experience_responsibility as er
from app.services import llm
from app.services.ai_schema import parse_wire, wire_to_payload
from app.services.llm import Completion, LLMRouter, Provider
from app.services.prompts import prepare_resume_text
from app.services.suggestion import profile_summary as ps
from app.models.resume import ResumeInfo


class Canned(Provider):
    name = model = "canned"
    timeout = 2

    def __init__(self, text):
        self.text, self.prompts = text, []

    async def complete(self, prompt, *, system, max_tokens, temperature):
        self.prompts.append(prompt)
        return Completion(text=self.text)


@pytest.fixture
def use(monkeypatch):
    def install(text):
        provider = Canned(text)
        monkeypatch.setattr(llm, "_router", LLMRouter([provider], Settings()))
        monkeypatch.setattr(ps, "get_settings", lambda: Settings(GEMINI_API_KEY="x"))
        monkeypatch.setattr(er, "get_settings", lambda: Settings(GEMINI_API_KEY="x"))
        return provider
    return install


def test_profile_summary_contract_and_small_prompt(use):
    p = use('{"summaries": ["one", "two", "three", "four"]}')
    out = asyncio.run(ps.generate_profile_summaries("I build web apps with Python.", "Backend Engineer"))
    assert [o["title"] for o in out["options"]] == [s[0] for s in ps.STYLES]
    assert all(set(o) == {"title", "style", "summary"} for o in out["options"])
    assert len(p.prompts[0]) < 1200


def test_profile_summary_validates_input():
    with pytest.raises(ValueError):
        asyncio.run(ps.generate_profile_summaries("short", "Dev"))


def test_experience_contract(use):
    use('{"options": [["a1","a2","a3"],["b1","b2","b3","b4"],["c1","c2","c3"]]}')
    out = asyncio.run(er.generate_experience_responsibilities({
        "job_title": "Engineer", "company": "Acme", "job_conditions": "Remote",
        "job_type": "Full-time", "start_date": "Jan 2020", "end_date": "Mar 2022",
    }))
    assert len(out["options"]) == 3 and out["options"][1]["responsibilities"][3] == "b4"


def test_experience_tense_follows_end_date():
    base = {"job_title": "E", "company": "A", "job_conditions": "Remote", "job_type": "Full-time",
            "start_date": "2020", "location": None}
    assert "present tense" in er._build_prompt({**base, "end_date": None})
    assert "past tense" in er._build_prompt({**base, "end_date": "Mar 2022"})


SOURCE = "Jane Doe jane@x.com github.com/jane Acme Corp Engineer 2020 - Present"


def test_wire_conversion_validates_against_resume_schema_and_drops_fake_urls():
    wire = parse_wire({
        "name": "Jane Doe", "title": "Engineer", "email": "jane@x.com",
        "github": "https://github.com/jane", "linkedin": "https://linkedin.com/in/made-up",
        "experience": [{"company": "Acme Corp", "role": "Engineer", "start": "2020",
                        "current": "true", "bullets": ["Built APIs"]},
                       {"company": "Acme Corp", "role": "Lead"}],
        "education": [{"degree": "BSc", "institution": "Uni", "score": "8.5/10"}],
        "skills": [{"group": "", "items": "Python, Go"}],
        "languages": [{"name": "English", "level": 99}],
        "projects": None,
    })
    resume = ResumeInfo.model_validate(wire_to_payload(wire, SOURCE))
    assert resume.contactInfo.github == "https://github.com/jane"
    assert resume.contactInfo.linkedin is None  # not in source -> hallucination dropped
    assert set(resume.workExperience.companies) == {"Acme Corp", "Acme Corp (2)"}
    assert resume.workExperience.companies["Acme Corp"].isPresentJob is True
    assert resume.skills.skills == {"Skills": ["Python", "Go"]}
    assert resume.educations.qualifications["BSc"].achivedMarks == 8.5
    assert resume.languageProficiency.languageKnows[0].proficiencyOutOfTen == 10


def test_empty_model_output_is_rejected():
    with pytest.raises(ValueError):
        parse_wire({"name": "", "education": []})


def test_prepare_resume_text_shrinks_and_caps():
    raw = "[PAGE 1]\nJane   Doe\nJane   Doe\n\n[Embedded links]\nhttps://a.b\n[PAGE 2]\n" + "x\n" * 100
    out = prepare_resume_text(raw, 60)
    assert "[PAGE" not in out and out.count("Jane Doe") == 1 and len(out) <= 60
