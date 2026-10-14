from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pymongo.errors import PyMongoError

from app.config import get_settings
from app.models.resume import ResumeInfo
from app.services.ai_extractor import AIExtractor
from app.services.chunker import build_chunks
from app.services.confidence import calculate_confidence
from app.services.file_extractor import extract_document
from app.services.knowledge_ingestor import KnowledgeIngestor
from app.services.local_extractor import extract_locally
from app.services.merge import _merge_extraction_data
from app.services.repository import get_resume_by_id, insert_resume
from app.services.validator import validate_semantics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/resumes", tags=["resumes"])


# These limits are deliberately finite so a single import cannot keep an HTTP
# request open forever. Adjust them only after measuring real documents.
DOCUMENT_EXTRACTION_TIMEOUT_SECONDS = 45
AI_EXTRACTION_TIMEOUT_SECONDS = 75
KNOWLEDGE_INGESTION_TIMEOUT_SECONDS = 90


def _stage(name: str, filename: str, message: str, **fields: Any) -> None:
    """Emit a concise, visible progress marker for every expensive stage."""
    details = " ".join(f"{key}={value!r}" for key, value in fields.items())
    logger.info(
        "resume_import stage=%s filename=%r %s %s",
        name,
        filename,
        message,
        details,
    )


@router.post("/import", status_code=status.HTTP_201_CREATED)
async def import_resume(file: UploadFile = File(...)):
    settings = get_settings()
    filename = file.filename or ""

    if not filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in {"pdf", "docx"}:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: .{extension}. Upload a PDF or DOCX resume.",
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024

    try:
        # Read at most one byte above the limit. Do not load an arbitrarily
        # large upload into memory before checking its size.
        content = await file.read(max_bytes + 1)
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large. Maximum size is {settings.max_upload_mb} MB.",
            )

        _stage("document_extraction", filename, "started", bytes=len(content))
        try:
            parsed = await asyncio.wait_for(
                asyncio.to_thread(extract_document, content, filename),
                timeout=DOCUMENT_EXTRACTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            logger.error(
                "resume_import stage=document_extraction timed_out filename=%r timeout_seconds=%s",
                filename,
                DOCUMENT_EXTRACTION_TIMEOUT_SECONDS,
            )
            raise HTTPException(
                status_code=408,
                detail=(
                    "Document extraction took too long. If this is a scanned PDF "
                    "or image, try a text-based PDF or DOCX."
                ),
            ) from exc

        if not isinstance(parsed, dict):
            raise ValueError("The document extractor returned an invalid result.")

        raw_text = parsed.get("text", "")
        blocks = parsed.get("blocks", [])
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError(
                "No readable text could be extracted from the uploaded file."
            )
        if not isinstance(blocks, list) or not blocks:
            raise ValueError(
                "No readable document blocks were extracted from the uploaded file."
            )

        _stage(
            "chunking",
            filename,
            "started",
            blocks=len(blocks),
            characters=len(raw_text),
        )
        chunks = build_chunks(blocks)
        if not chunks:
            raise ValueError(
                "No usable text chunks could be created from the uploaded file."
            )
        _stage("chunking", filename, "completed", chunks=len(chunks))

        _stage("local_extraction", filename, "started")
        verified_resume, local_audit = await asyncio.to_thread(
            extract_locally, raw_text, chunks
        )
        extraction_engine = "deterministic-layout-parser"
        ai_error: str | None = None
        ai_evidence: list[dict[str, Any]] = []

        # AI enrichment must never block the event loop or keep this request
        # open indefinitely. If it times out/fails, preserve the deterministic
        # result and continue to persistence.
        if settings.gemini_api_key or settings.openrouter_api_key:
            _stage("ai_extraction", filename, "started")
            try:
                candidate = await asyncio.wait_for(
                    asyncio.to_thread(
                        AIExtractor().extract,
                        raw_text=raw_text,
                        chunks=chunks,
                        knowledge=[],
                    ),
                    timeout=AI_EXTRACTION_TIMEOUT_SECONDS,
                )
                ai_payload = (
                    candidate.resume.model_dump(mode="json", exclude_none=True)
                    if hasattr(candidate.resume, "model_dump")
                    else candidate.resume
                )
                local_payload = verified_resume.model_dump(
                    mode="json", exclude_none=True
                )
                merged_payload = _merge_extraction_data(local_payload, ai_payload)
                verified_resume = ResumeInfo.model_validate(merged_payload)
                extraction_engine = "gemini-with-openrouter-failover"
                ai_evidence = [
                    {
                        "fieldPath": item.fieldPath,
                        "sourceChunkIds": item.sourceChunkIds,
                        "confidence": item.confidence,
                    }
                    for item in candidate.evidence
                ]
                _stage("ai_extraction", filename, "completed")
            except asyncio.TimeoutError:
                ai_error = (
                    f"AI extraction exceeded {AI_EXTRACTION_TIMEOUT_SECONDS} seconds; "
                    "saved deterministic extraction instead."
                )
                logger.error(
                    "resume_import stage=ai_extraction timed_out filename=%r timeout_seconds=%s",
                    filename,
                    AI_EXTRACTION_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                # Provider exceptions may contain request details. Persist only
                # the exception class; keep diagnostics useful without storing
                # raw provider payloads or possible sensitive response text.
                ai_error = type(exc).__name__
                logger.error(
                    "resume_import stage=ai_extraction failed; using deterministic extraction "
                    "filename=%r exception_type=%s",
                    filename,
                    type(exc).__name__,
                )
        else:
            _stage(
                "ai_extraction", filename, "skipped", reason="no AI API key configured"
            )

        _stage("validation", filename, "started")
        warnings = validate_semantics(verified_resume)
        if ai_error:
            warnings.append(
                "AI extraction unavailable; saved deterministic extraction."
            )
        confidence = calculate_confidence(
            resume=verified_resume,
            knowledge=[],
            warnings=warnings,
            source_text=raw_text,
        )

        audit = {
            "version": "4.1-time-bounded-import",
            "confidence": confidence["score"],
            "confidenceReasons": confidence["reasons"],
            "needsReview": True,
            "hardReject": False,
            "warnings": warnings,
            "fieldEvidence": ai_evidence or local_audit.get("evidence", []),
            "extraction": local_audit,
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
                timeout=20,
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
            "confidence": audit["confidence"],
            "needs_review": audit["needsReview"],
            "warnings": warnings,
            "resume": verified_resume.model_dump(mode="json", exclude_none=True),
        }

    except HTTPException:
        raise
    except ValueError as exc:
        logger.warning(
            "resume_import validation_failure filename=%r error=%s",
            filename,
            exc,
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("resume_import runtime_failure filename=%r", filename)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("resume_import unexpected_failure filename=%r", filename)
        raise HTTPException(
            status_code=500,
            detail=(
                "Resume import failed due to an internal server error. "
                "Check the FastAPI terminal for the stage and traceback."
            ),
        ) from exc
    finally:
        await file.close()


@router.post(
    "/knowledge/ingest",
    status_code=status.HTTP_201_CREATED,
    tags=["knowledge"],
)
async def ingest_knowledge(file: UploadFile = File(...)):
    """Ingest a trusted sample resume into the RAG knowledge base."""
    settings = get_settings()
    filename = file.filename or ""

    if not filename:
        raise HTTPException(status_code=400, detail="A filename is required.")

    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in settings.allowed_extension_set:
        raise HTTPException(
            status_code=415, detail=f"Unsupported file type: .{extension}"
        )

    max_bytes = settings.max_upload_mb * 1024 * 1024
    try:
        content = await file.read(max_bytes + 1)
        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        if len(content) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File is too large. Maximum size is {settings.max_upload_mb} MB.",
            )

        _stage("knowledge_document_extraction", filename, "started", bytes=len(content))
        try:
            parsed = await asyncio.wait_for(
                asyncio.to_thread(extract_document, content, filename),
                timeout=DOCUMENT_EXTRACTION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=408,
                detail="Knowledge document extraction exceeded the time limit.",
            ) from exc

        raw_text = parsed.get("text", "")
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise HTTPException(
                status_code=422,
                detail="No readable text could be extracted from the uploaded file.",
            )

        chunks = build_chunks(parsed.get("blocks", []))
        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No usable chunks could be created from the uploaded file.",
            )

        ingestor = KnowledgeIngestor()
        try:
            result = await asyncio.wait_for(
                ingestor.ingest(filename=filename, text=raw_text, chunks=chunks),
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
            status_code=503, detail=f"MongoDB operation failed: {exc}"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("knowledge_ingestion runtime_failure filename=%r", filename)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("knowledge_ingestion unexpected_failure filename=%r", filename)
        raise HTTPException(
            status_code=500,
            detail="Knowledge ingestion failed due to an internal server error. Check the FastAPI terminal.",
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
            status_code=503, detail=f"MongoDB operation failed: {exc}"
        ) from exc

    if resume is None:
        raise HTTPException(status_code=404, detail="Resume not found.")

    return {"success": True, "resume": resume}
