from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError

from app.config import get_settings
from app.models.resume import ResumeInfo
from app.services.ai_extractor import AIExtractor
from app.services.chunker import build_chunks
from app.services.confidence import calculate_confidence
from app.services.suggestion.experience_responsibility import (
    generate_experience_responsibilities,
)
from app.services.file_extractor import extract_document
from app.services.knowledge_ingestor import KnowledgeIngestor
from app.services.llm import LLMMisconfigured, LLMUnavailable
from app.services.local_extractor import extract_locally
from app.services.merge import merge_extraction
from app.services.repository import get_resume_by_id, insert_resume
from app.services.suggestion.profile_summary import generate_profile_summaries
from app.services.validator import validate_semantics
from app.services.suggestion.project_suggestion import generate_project_suggestions
from app.services.suggestion.achievement_suggestion import (
    generate_achievement_suggestions,
)
from app.services.suggestion.opensource_suggestion import (
    generate_opensource_suggestions,
)
from typing import Dict
from app.services.resume_enhancer import evaluate_ats_score, polish_resume_content

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resumes", tags=["resumes"])

# Finite limits so one request can never stay open forever. AI calls enforce
# their own (shorter) deadlines inside app.services.llm; the values below are
# only outer safety nets.
DOCUMENT_EXTRACTION_TIMEOUT_SECONDS = 45
KNOWLEDGE_INGESTION_TIMEOUT_SECONDS = 90
PERSISTENCE_TIMEOUT_SECONDS = 20
OUTER_MARGIN_SECONDS = 5

AI_BUSY_DETAIL = (
    "The AI providers are temporarily unavailable. Please try again shortly."
)


def _stage(name: str, filename: str, message: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value!r}" for key, value in fields.items())
    logger.info(
        "resume_import stage=%s filename=%r %s %s", name, filename, message, details
    )


AI_MISCONFIGURED_DETAIL = (
    "AI is not configured correctly (invalid API key or unknown model name). "
    "Check GEMINI_API_KEY / GEMINI_MODEL / OPENROUTER_API_KEY in .env and restart the server."
)


def _ai_unavailable(exc: Exception | None = None) -> HTTPException:
    if isinstance(exc, LLMMisconfigured):
        return HTTPException(status_code=503, detail=AI_MISCONFIGURED_DETAIL)
    return HTTPException(
        status_code=503, detail=AI_BUSY_DETAIL, headers={"Retry-After": "5"}
    )


async def _read_upload(file: UploadFile, allowed: set[str]) -> tuple[str, bytes]:
    settings = get_settings()
    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: .{extension}. Upload a PDF or DOCX resume.",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    # Read at most one byte over the limit; never buffer an arbitrarily large upload.
    content = await file.read(max_bytes + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is {settings.max_upload_mb} MB.",
        )
    return filename, content


async def _parse_document(
    content: bytes, filename: str, stage: str
) -> tuple[str, list[dict]]:
    _stage(stage, filename, "started", bytes=len(content))
    try:
        parsed = await asyncio.wait_for(
            asyncio.to_thread(extract_document, content, filename),
            timeout=DOCUMENT_EXTRACTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        logger.error("resume_import stage=%s timed_out filename=%r", stage, filename)
        raise HTTPException(
            status_code=408,
            detail=(
                "Document extraction took too long. If this is a scanned PDF "
                "or image, try a text-based PDF or DOCX."
            ),
        ) from exc

    raw_text = parsed.get("text", "") if isinstance(parsed, dict) else ""
    blocks = parsed.get("blocks", []) if isinstance(parsed, dict) else []
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("No readable text could be extracted from the uploaded file.")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError(
            "No readable document blocks were extracted from the uploaded file."
        )
    return raw_text, blocks


@router.post("/import", status_code=status.HTTP_201_CREATED)
async def import_resume(file: UploadFile = File(...)):
    settings = get_settings()
    filename = file.filename or ""

    try:
        filename, content = await _read_upload(file, {"pdf", "docx"})
        raw_text, blocks = await _parse_document(
            content, filename, "document_extraction"
        )

        chunks = build_chunks(blocks)
        if not chunks:
            raise ValueError(
                "No usable text chunks could be created from the uploaded file."
            )
        _stage(
            "chunking",
            filename,
            "completed",
            chunks=len(chunks),
            characters=len(raw_text),
        )

        _stage("local_extraction", filename, "started")
        verified_resume, local_audit = await asyncio.to_thread(
            extract_locally, raw_text, chunks
        )
        extraction_engine = "deterministic-layout-parser"
        ai_provider: str | None = None
        ai_error: str | None = None

        # AI enrichment is best effort: any failure keeps the deterministic result.
        if settings.ai_extraction_enabled and settings.any_ai_configured:
            _stage("ai_extraction", filename, "started")
            try:
                ai = await asyncio.wait_for(
                    AIExtractor().extract(raw_text),
                    timeout=settings.ai_extract_deadline_seconds + OUTER_MARGIN_SECONDS,
                )
                merged = merge_extraction(
                    verified_resume.model_dump(mode="json", exclude_none=True),
                    ai.payload,
                )
                verified_resume = ResumeInfo.model_validate(merged)
                extraction_engine = "gemini-with-openrouter-failover"
                ai_provider = f"{ai.provider}:{ai.model}"
                _stage(
                    "ai_extraction",
                    filename,
                    "completed",
                    provider=ai_provider,
                    cached=ai.cached,
                )
            except (LLMUnavailable, asyncio.TimeoutError) as exc:
                ai_error = type(exc).__name__
                logger.error(
                    "resume_import ai_extraction unavailable filename=%r: %s",
                    filename,
                    exc,
                )
            except (
                Exception
            ) as exc:  # noqa: BLE001 - never lose an otherwise good import
                ai_error = type(exc).__name__
                logger.error(
                    "resume_import ai_extraction failed filename=%r exception_type=%s",
                    filename,
                    type(exc).__name__,
                )
        else:
            _stage(
                "ai_extraction", filename, "skipped", reason="AI disabled or no API key"
            )

        warnings = validate_semantics(verified_resume)
        if ai_error:
            warnings.append(
                "AI extraction unavailable; saved deterministic extraction."
            )
        confidence = calculate_confidence(
            resume=verified_resume,
            knowledge=None,
            warnings=warnings,
            source_text=raw_text,
        )
        audit = {
            "version": "4.2-lean-ai-import",
            "confidence": confidence["score"],
            "confidenceReasons": confidence["reasons"],
            "needsReview": True,
            "hardReject": False,
            "warnings": warnings,
            "fieldEvidence": local_audit.get("evidence", []),
            "extraction": local_audit,
            "aiProvider": ai_provider,
            "aiProviderError": ai_error,
        }

        _stage("persistence", filename, "started")
        try:
            document_id = await asyncio.wait_for(
                insert_resume(
                    verified_resume,
                    source_file_name=filename,
                    source_file_type=file.content_type or "application/octet-stream",
                    extraction_engine=extraction_engine,
                    audit=audit,
                    source_text=raw_text,
                    chunks=chunks,
                ),
                timeout=PERSISTENCE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            logger.error(
                "resume_import stage=persistence timed_out filename=%r", filename
            )
            raise HTTPException(
                status_code=503,
                detail="Saving the imported resume timed out. Check database/storage availability.",
            ) from exc

        _stage("complete", filename, "completed", document_id=document_id)
        return {
            "success": True,
            "document_id": document_id,
            "filename": filename,
            "extraction_engine": extraction_engine,
            "ai_provider": ai_provider,
            "confidence": audit["confidence"],
            "needs_review": audit["needsReview"],
            "warnings": warnings,
            "resume": verified_resume.model_dump(mode="json", exclude_none=True),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "resume_import validation_failure filename=%r error=%s", filename, exc
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("resume_import runtime_failure filename=%r", filename)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("resume_import unexpected_failure filename=%r", filename)
        raise HTTPException(
            status_code=500,
            detail="Resume import failed due to an internal server error. Check the server log.",
        ) from exc
    finally:
        await file.close()


@router.post(
    "/knowledge/ingest", status_code=status.HTTP_201_CREATED, tags=["knowledge"]
)
async def ingest_knowledge(file: UploadFile = File(...)):
    """Ingest a trusted sample resume into the RAG knowledge base."""
    settings = get_settings()
    filename = file.filename or ""
    try:
        filename, content = await _read_upload(file, settings.allowed_extension_set)
        raw_text, blocks = await _parse_document(
            content, filename, "knowledge_document_extraction"
        )
        chunks = build_chunks(blocks)
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No usable chunks could be created from the uploaded file.",
            )
        try:
            result = await asyncio.wait_for(
                KnowledgeIngestor().ingest(
                    filename=filename, text=raw_text, chunks=chunks
                ),
                timeout=KNOWLEDGE_INGESTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            logger.error("knowledge_ingestion timed_out filename=%r", filename)
            raise HTTPException(
                status_code=504,
                detail="Knowledge ingestion timed out. Check embedding provider and database connectivity.",
            ) from exc
        return {
            "success": True,
            "filename": filename,
            "message": "Sample resume successfully added to the RAG knowledge base.",
            **result,
        }
    except HTTPException:
        raise
    except PyMongoError as exc:
        logger.exception("knowledge_ingestion MongoDB failure filename=%r", filename)
        raise HTTPException(
            status_code=503, detail="MongoDB operation failed."
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("knowledge_ingestion runtime_failure filename=%r", filename)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("knowledge_ingestion unexpected_failure filename=%r", filename)
        raise HTTPException(
            status_code=500, detail="Knowledge ingestion failed. Check the server log."
        ) from exc
    finally:
        await file.close()


@router.get("/{resume_id}")
async def fetch_resume_by_id(resume_id: str):
    # Local fallback records use UUIDs; Mongo records use ObjectIds.
    if len(resume_id) > 128 or not resume_id.strip():
        raise HTTPException(status_code=400, detail="Invalid resume ID.")
    try:
        resume = await get_resume_by_id(resume_id)
    except PyMongoError as exc:
        raise HTTPException(
            status_code=503, detail="MongoDB operation failed."
        ) from exc
    if resume is None:
        raise HTTPException(status_code=404, detail="Resume not found.")
    return {"success": True, "resume": resume}


class ProfileSummaryRequest(BaseModel):
    draft: str = Field(min_length=10, max_length=5000)
    target_role: str = Field(min_length=1, max_length=120)
    # Optional: true when the user presses "regenerate" so the short-lived
    # cache is bypassed and they get genuinely new wording.
    regenerate: bool = False


@router.post("/profile-summary/generate")
async def generate_profile_summary_route(payload: ProfileSummaryRequest):
    """Four profile-summary alternatives. The AI layer enforces the deadline."""
    settings = get_settings()
    try:
        return await asyncio.wait_for(
            generate_profile_summaries(
                payload.draft, payload.target_role, regenerate=payload.regenerate
            ),
            timeout=settings.ai_short_deadline_seconds + OUTER_MARGIN_SECONDS,
        )
    except LLMUnavailable as exc:
        logger.warning(
            "profile_summary unavailable: %s | %s", exc, "; ".join(exc.errors)[:400]
        )
        raise _ai_unavailable(exc) from exc
    except asyncio.TimeoutError as exc:
        logger.error("profile_summary outer timeout")
        raise HTTPException(
            status_code=504,
            detail="Summary generation took too long. Please try again.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("profile_summary unexpected failure")
        raise HTTPException(
            status_code=500, detail="Profile summary generation failed unexpectedly."
        ) from exc


class ExperienceResponsibilityRequest(BaseModel):
    job_title: str = Field(min_length=1, max_length=120)
    company: str = Field(min_length=1, max_length=160)
    location: str | None = Field(default=None, max_length=160)
    job_conditions: str = Field(min_length=1, max_length=40)
    job_type: str = Field(min_length=1, max_length=50)
    start_date: str = Field(min_length=1, max_length=40)
    end_date: str | None = Field(default=None, max_length=40)
    regenerate: bool = False


@router.post("/experience-responsibilities/generate")
async def generate_experience_responsibility_options(
    payload: ExperienceResponsibilityRequest,
):
    settings = get_settings()
    data = payload.model_dump(exclude={"regenerate"})
    try:
        return await asyncio.wait_for(
            generate_experience_responsibilities(data, regenerate=payload.regenerate),
            timeout=settings.ai_short_deadline_seconds + OUTER_MARGIN_SECONDS,
        )
    except LLMUnavailable as exc:
        logger.warning(
            "experience_responsibility unavailable: %s | %s",
            exc,
            "; ".join(exc.errors)[:400],
        )
        raise _ai_unavailable(exc) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Responsibility generation timed out. Please try again.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected experience responsibility generation error")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate responsibility suggestions right now.",
        ) from exc


# --- Add this schema and endpoint at the bottom of routes.py ---
class ProjectSuggestionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    description: str = Field(min_length=5, max_length=3000)
    tech_stack: list[str] = Field(default_factory=list)
    is_working: bool = False
    start_date: str | None = Field(default=None, max_length=40)
    end_date: str | None = Field(default=None, max_length=40)
    regenerate: bool = False


@router.post("/project-suggestions/generate")
async def generate_project_suggestion_options(payload: ProjectSuggestionRequest):
    settings = get_settings()
    data = payload.model_dump(exclude={"regenerate"})
    try:
        return await asyncio.wait_for(
            generate_project_suggestions(data, regenerate=payload.regenerate),
            timeout=settings.ai_short_deadline_seconds + OUTER_MARGIN_SECONDS,
        )
    except LLMUnavailable as exc:
        logger.warning(
            "project_suggestion unavailable: %s | %s", exc, "; ".join(exc.errors)[:400]
        )
        raise _ai_unavailable(exc) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Project suggestion generation timed out. Please try again.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected project suggestion generation error")
        raise HTTPException(
            status_code=500, detail="Unable to generate project suggestions right now."
        ) from exc


class AchievementSuggestionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(min_length=5, max_length=3000)
    regenerate: bool = False


@router.post("/achievement-suggestions/generate")
async def generate_achievement_suggestion_options(
    payload: AchievementSuggestionRequest,
):
    settings = get_settings()
    data = payload.model_dump(exclude={"regenerate"})
    try:
        return await asyncio.wait_for(
            generate_achievement_suggestions(data, regenerate=payload.regenerate),
            timeout=settings.ai_short_deadline_seconds + OUTER_MARGIN_SECONDS,
        )
    except LLMUnavailable as exc:
        logger.warning(
            "achievement_suggestion unavailable: %s | %s",
            exc,
            "; ".join(exc.errors)[:400],
        )
        raise _ai_unavailable(exc) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Achievement suggestion generation timed out. Please try again.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected achievement suggestion generation error")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate achievement suggestions right now.",
        ) from exc


class OpenSourceSuggestionRequest(BaseModel):
    project_name: str = Field(min_length=1, max_length=150)
    duration: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=5, max_length=3000)
    regenerate: bool = False


@router.post("/opensource-suggestions/generate")
async def generate_opensource_suggestion_options(payload: OpenSourceSuggestionRequest):
    settings = get_settings()
    data = payload.model_dump(exclude={"regenerate"})
    try:
        return await asyncio.wait_for(
            generate_opensource_suggestions(data, regenerate=payload.regenerate),
            timeout=settings.ai_short_deadline_seconds + OUTER_MARGIN_SECONDS,
        )
    except LLMUnavailable as exc:
        logger.warning(
            "opensource_suggestion unavailable: %s | %s",
            exc,
            "; ".join(exc.errors)[:400],
        )
        raise _ai_unavailable(exc) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Open source suggestion generation timed out. Please try again.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected open source suggestion generation error")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate open source suggestions right now.",
        ) from exc







class PolishResumeRequest(BaseModel):
    resume: Dict[str, Any]


class AtsScoreRequest(BaseModel):
    resume: Dict[str, Any]
    target_role: str = Field(default="", max_length=120)


@router.post("/polish")
async def polish_resume_endpoint(payload: PolishResumeRequest):
    """
    Polishes resume descriptions and summaries for ATS readability
    while preserving personal contact details, credentials, and institutions.
    """
    settings = get_settings()
    if not payload.resume:
        raise HTTPException(status_code=400, detail="Resume data is required.")

    try:
        polished_resume = await asyncio.wait_for(
            polish_resume_content(payload.resume),
            timeout=settings.ai_extract_deadline_seconds + OUTER_MARGIN_SECONDS,
        )
        return {"success": True, "resume": polished_resume}
    except LLMUnavailable as exc:
        logger.warning("Resume polishing LLM unavailable: %s", exc)
        raise _ai_unavailable(exc) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="Resume polishing timed out. Please try again.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during resume polishing")
        raise HTTPException(
            status_code=500, detail="Failed to polish resume content."
        ) from exc


@router.post("/ats-check")
async def check_ats_score_endpoint(payload: AtsScoreRequest):
    """
    Analyzes resume data and returns an accurate ATS score (0-100),
    category breakdowns, key strengths, missing keywords, and recommended actions.
    """
    settings = get_settings()
    if not payload.resume:
        raise HTTPException(status_code=400, detail="Resume data is required.")

    try:
        report = await asyncio.wait_for(
            evaluate_ats_score(payload.resume, target_role=payload.target_role),
            timeout=settings.ai_extract_deadline_seconds + OUTER_MARGIN_SECONDS,
        )
        return {"success": True, "report": report}
    except LLMUnavailable as exc:
        logger.warning("ATS check LLM unavailable: %s", exc)
        raise _ai_unavailable(exc) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail="ATS score evaluation timed out. Please try again.",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during ATS score check")
        raise HTTPException(
            status_code=500, detail="Failed to calculate ATS score."
        ) from exc