from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from bson import ObjectId

from app.config import get_settings
from app.db import get_collection
from app.models.resume import ResumeInfo

# Top-level keys defined by the TypeScript resume interface/schema. Mongoose
# adds _id and timestamps; dynamic maps remain ordinary JSON objects.
RESUME_FIELDS = {
    "basicInfo", "avatar", "address", "contactInfo", "profileSummary",
    "educations", "workExperience", "skills", "languageProficiency",
    "hobbies", "projects", "certifications", "openSource", "publications",
    "awardsAndAchievements", "templateName", "_id", "createdAt", "updatedAt", "__v",
}


def _local_connection() -> sqlite3.Connection:
    path = Path(get_settings().local_database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS resumes (id TEXT PRIMARY KEY, document_json TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    return connection


def _public_document(document: dict) -> dict:
    """Remove private/legacy top-level keys and serialize Mongoose metadata."""
    result = {key: value for key, value in document.items() if key in RESUME_FIELDS}
    if "_id" in result:
        result["_id"] = str(result["_id"])
    for key in ("createdAt", "updatedAt"):
        value = result.get(key)
        if isinstance(value, datetime):
            result[key] = value.isoformat()
    return result


async def insert_resume(
    resume: ResumeInfo,
    *,
    source_file_name: str,
    source_file_type: str,
    extraction_engine: str,
    audit: dict,
    source_text: str,
    chunks: list[dict],
) -> str:
    """Persist the resume in the same root-level shape as the Mongoose model.

    Ingestion diagnostics and source chunks are stored in the knowledge collection,
    not in the resume document. When MongoDB is configured, errors are surfaced
    instead of silently writing to SQLite where the Node/Mongoose app cannot see it.
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    document = resume.model_dump(mode="python", exclude_none=True)
    document["createdAt"] = now
    document["updatedAt"] = now

    if settings.database_mode != "local" and settings.mongodb_uri:
        result = await get_collection().insert_one(document)
        return str(result.inserted_id)

    document_id = str(uuid4())
    document["_id"] = document_id
    with _local_connection() as connection:
        connection.execute(
            "INSERT INTO resumes (id, document_json, created_at) VALUES (?, ?, ?)",
            (document_id, json.dumps(document, default=str), now.isoformat()),
        )
    return document_id


async def get_resume_by_id(resume_id: str) -> dict | None:
    settings = get_settings()
    if settings.database_mode == "local" or not settings.mongodb_uri:
        with _local_connection() as connection:
            row = connection.execute(
                "SELECT document_json FROM resumes WHERE id = ?", (resume_id,)
            ).fetchone()
        if row is None:
            return None
        return _public_document(json.loads(row[0]))

    if not ObjectId.is_valid(resume_id):
        return None
    document = await get_collection().find_one({"_id": ObjectId(resume_id)})
    if document is None:
        return None
    return _public_document(document)
