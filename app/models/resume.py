from __future__ import annotations

import re
from datetime import date, datetime
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="allow")


def _validate_url(value: str | None) -> str | None:
    """Normalize common resume URLs before validating them."""
    if value is None:
        return None

    value = value.strip()
    if not value:
        return value

    parsed = urlparse(value)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        return value

    if value.startswith("//"):
        candidate = f"https:{value}"
        parsed_candidate = urlparse(candidate)
        if parsed_candidate.scheme == "https" and parsed_candidate.netloc:
            return candidate

    # Add https:// only when the value looks like a domain name.
    if not parsed.scheme and re.match(
        r"^(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/|:|$)",
        value,
        re.IGNORECASE,
    ):
        candidate = f"https://{value}"
        parsed_candidate = urlparse(candidate)
        if parsed_candidate.scheme == "https" and parsed_candidate.netloc:
            return candidate

    raise ValueError("must be an absolute HTTP(S) URL")


class BasicInfo(StrictModel):
    fullName: str
    position: str


class Avatar(StrictModel):
    """Matches the optional avatar file value in IResumeInfo."""
    url: str
    key: str


class Address(StrictModel):
    streetName: str
    city: str
    district: str
    pincode: str
    country: str


class ContactInfo(StrictModel):
    primaryEmail: str
    primaryMobile: str
    linkedin: str | None = None
    github: str | None = None
    portfolio: str | None = None

    @field_validator("primaryEmail")
    @classmethod
    def validate_email(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        # Empty values are permitted for resumes that omit email; absence of
        # optional contact data must not invalidate otherwise useful extraction.
        if not value:
            return value
        if "@" not in value or value.startswith("@") or value.endswith("@"):
            raise ValueError("invalid email address")
        return value

    @field_validator("linkedin", "github", "portfolio")
    @classmethod
    def validate_contact_url(cls, value: str | None) -> str | None:
        return _validate_url(value)


class ProfileSummary(StrictModel):
    priority: int
    subject: str
    objective: str | None = None


class ProjectItem(StrictModel):
    priority: int
    description: str
    projectUrl: str | None = None
    startDate: str | date | datetime
    endDate: str | date | datetime | None = None
    isWorking: bool
    techStack: list[str]
    skills: list[str]


class Projects(StrictModel):
    priority: int
    sectionTitle: str | None = None
    projects: dict[str, ProjectItem]


class QualificationItem(StrictModel):
    priority: int
    institutionName: str
    startedAt: str | date | datetime
    yearOfComplete: str | date | datetime | None = None
    pursuing: bool
    achivedMarks: float | None = None
    totalMarks: float | None = None
    percentage: str | None = None
    description: str | None = None


class Educations(StrictModel):
    priority: int
    sectionTitle: str | None = None
    qualifications: dict[str, QualificationItem]


class CompanyItem(StrictModel):
    priority: int
    jobTitle: str
    jobConditions: str
    jobTypes: str
    jobLocation: str | None = None
    responsibility: list[str] | None = None
    startDate: str | date | datetime
    endDate: str | date | datetime | None = None
    isPresentJob: bool


class WorkExperience(StrictModel):
    priority: int
    sectionTitle: str | None = None
    companies: dict[str, CompanyItem]


class SkillSet(StrictModel):
    priority: int
    sectionTitle: str | None = None
    skills: dict[str, list[str]]


class LanguageProficiencyItem(StrictModel):
    languageName: str
    proficiencyOutOfTen: float = Field(ge=0, le=10)


class LanguageProficiency(StrictModel):
    priority: int
    sectionTitle: str | None = None
    languageKnows: list[LanguageProficiencyItem]


class Hobbies(StrictModel):
    priority: int
    sectionTitle: str | None = None


class FileReference(StrictModel):
    fileIndex: int = Field(ge=0)
    title: str


class CertificateItem(StrictModel):
    priority: int
    overview: str | None = None
    skillLearned: list[str]
    certificateUrl: str | None = None
    certificateContent: FileReference | None = None
    duration: str

    @field_validator("certificateUrl")
    @classmethod
    def validate_certificate_url(cls, value: str | None) -> str | None:
        return _validate_url(value)


class Certificates(StrictModel):
    priority: int
    sectionTitle: str | None = None
    certificates: dict[str, CertificateItem]


class OpenSourceItem(StrictModel):
    priority: int
    githubUrl: str | None = None
    description: str | None = None
    duration: str | None = None

    @field_validator("githubUrl")
    @classmethod
    def validate_github_url(cls, value: str | None) -> str | None:
        return _validate_url(value)


class OpenSource(StrictModel):
    priority: int
    sectionTitle: str | None = None
    contributions: dict[str, OpenSourceItem]


class PublicationItem(StrictModel):
    priority: int
    description: str | None = None
    referenceUrl: str | None = None
    publicationReference: list[FileReference] | None = None

    @field_validator("referenceUrl")
    @classmethod
    def validate_reference_url(cls, value: str | None) -> str | None:
        return _validate_url(value)


class Publications(StrictModel):
    priority: int
    sectionTitle: str | None = None
    publications: dict[str, PublicationItem]


class AchievementItem(StrictModel):
    priority: int
    title: str | None = None
    description: str | None = None
    url: str | None = None
    documents: list[FileReference] | None = None

    @field_validator("url")
    @classmethod
    def validate_achievement_url(cls, value: str | None) -> str | None:
        return _validate_url(value)


class AwardsAndAchievements(StrictModel):
    priority: int
    sectionTitle: str | None = None
    achievements: list[AchievementItem]


class ResumeInfo(StrictModel):
    basicInfo: BasicInfo
    avatar: Avatar | None = None
    address: Address
    contactInfo: ContactInfo
    profileSummary: ProfileSummary
    educations: Educations
    workExperience: WorkExperience | None = None
    skills: SkillSet
    languageProficiency: LanguageProficiency
    hobbies: Hobbies | None = None
    projects: Projects | None = None
    certifications: Certificates | None = None
    openSource: OpenSource | None = None
    publications: Publications | None = None
    awardsAndAchievements: AwardsAndAchievements | None = None
    templateName: str = "Imported Resume"
