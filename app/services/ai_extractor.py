from __future__ import annotations

import copy
import time
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.models.resume import ResumeInfo
from app.services.prompts import (
    build_extraction_prompt,
    build_verification_prompt,
)
from app.services.openrouter_fallback import FallbackProviderError, OpenRouterFallback


class Evidence(BaseModel):
    fieldPath: str = Field(
        description="Canonical ResumeInfo field path."
    )
    sourceChunkIds: list[str] = Field(
        default_factory=list,
        description="IDs of chunks supporting this field."
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence from 0 to 1."
    )


class ExtractionEnvelope(BaseModel):
    """
    Gemini's first-pass extraction result.

    IMPORTANT:
    ResumeInfo is used directly here instead of dict[str, Any].

    This means Gemini receives the actual ResumeInfo schema and
    cannot simply return an empty arbitrary dictionary.
    """

    model_config = ConfigDict(extra="forbid")

    resume: ResumeInfo
    evidence: list[Evidence] = Field(default_factory=list)


class VerificationEnvelope(BaseModel):
    """
    Gemini's verification result.

    Again, ResumeInfo is used directly so the model must return
    the complete structured resume.
    """

    model_config = ConfigDict(extra="forbid")

    resume: ResumeInfo
    evidence: list[Evidence] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def _clean_gemini_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    Clean Pydantic JSON Schema before sending it to Gemini.

    Gemini supports only a subset of JSON Schema. Some Pydantic
    generated keywords are not accepted by Gemini.

    We remove unsupported keywords recursively.

    IMPORTANT:
    We only clean the schema sent to Gemini.

    The actual Gemini response is still validated against the
    original Pydantic models afterwards.
    """

    unsupported_keys = {
        "additionalProperties",
    }

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}

            for key, item in value.items():
                if key in unsupported_keys:
                    continue

                cleaned[key] = clean(item)

            return cleaned

        if isinstance(value, list):
            return [clean(item) for item in value]

        return value

    return clean(copy.deepcopy(schema))


class AIExtractor:
    """
    Gemini-based resume extraction and verification service.

    Pipeline:

        Uploaded Resume
              ↓
        Extracted Text
              ↓
        RAG Knowledge
              ↓
        Gemini
              ↓
        Structured ResumeInfo
              ↓
        Pydantic Validation
    """

    MAX_ATTEMPTS = 2
    INITIAL_RETRY_DELAY = 1.0
    MAX_RETRY_DELAY = 2.0

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.gemini_api_key and not settings.openrouter_api_key:
            raise RuntimeError("Configure GEMINI_API_KEY or OPENROUTER_API_KEY.")

        self.client = (
            genai.Client(
                api_key=settings.gemini_api_key,
                http_options=types.HttpOptions(timeout=20000),
            )
            if settings.gemini_api_key
            else None
        )

        self.model = settings.gemini_model

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        """
        Determine whether a Gemini failure is temporary.
        """

        if isinstance(exc, errors.ServerError):
            status_code = getattr(
                exc,
                "status_code",
                None,
            )

            return status_code in {
                500,
                502,
                503,
                504,
            }

        if isinstance(exc, errors.ClientError):
            status_code = getattr(
                exc,
                "status_code",
                None,
            )

            return status_code == 429

        return False

    @staticmethod
    def _format_gemini_error(exc: Exception) -> str:
        """
        Convert Gemini exceptions into useful application errors.
        """

        status_code = getattr(
            exc,
            "status_code",
            None,
        )

        if status_code == 429:
            return (
                "Gemini API rate limit reached. "
                "Please try again in a moment."
            )

        if status_code == 503:
            return (
                "Gemini is temporarily unavailable due to high demand. "
                "Please try again in a moment."
            )

        if status_code == 504:
            return (
                "Gemini took too long to respond. "
                "Please try again."
            )

        if status_code == 502:
            return (
                "Gemini is temporarily unavailable. "
                "Please try again."
            )

        if status_code == 500:
            return (
                "Gemini encountered a temporary server error. "
                "Please try again."
            )

        if status_code == 404:
            return (
                f"Gemini model '{getattr(exc, 'model', None) or 'configured model'}' "
                "is unavailable. Check GEMINI_MODEL."
            )

        if status_code in {401, 403}:
            return (
                "Gemini authentication or permission failed. "
                "Check GEMINI_API_KEY."
            )

        return f"Gemini API request failed: {exc}"

    def _generate(
        self,
        prompt: str,
        schema: type[BaseModel],
    ) -> BaseModel:
        """
        Generate structured JSON from Gemini.

        The supplied Pydantic schema is sent to Gemini as the
        response schema.

        The returned JSON is then validated again using the
        original Pydantic model.
        """

        if not prompt or not prompt.strip():
            raise ValueError(
                "Gemini prompt cannot be empty."
            )

        # ---------------------------------------------------------
        # Generate the REAL schema.
        #
        # For extraction this is ExtractionEnvelope, which contains:
        #
        #     resume: ResumeInfo
        #
        # NOT:
        #
        #     resume: dict[str, Any]
        #
        # This is the critical fix.
        # ---------------------------------------------------------

        pydantic_schema = schema.model_json_schema()

        gemini_schema = _clean_gemini_schema(
            pydantic_schema
        )

        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=gemini_schema,
        )

        last_exception: Exception | None = None

        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            if self.client is None:
                break
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )

                raw = (
                    response.text or ""
                ).strip()

                if not raw:
                    raise RuntimeError(
                        "Gemini returned an empty extraction response."
                    )

                # -------------------------------------------------
                # Validate Gemini's JSON against the ORIGINAL
                # Pydantic model.
                # -------------------------------------------------

                try:
                    result = schema.model_validate_json(
                        raw
                    )

                except Exception as exc:
                    raise RuntimeError(
                        "Gemini returned JSON that does not match "
                        f"the expected {schema.__name__} structure:\n"
                        f"{exc}\n\n"
                        f"Gemini response:\n{raw}"
                    ) from exc

                return result

            except RuntimeError as exc:
                last_exception = exc
                break

            except Exception as exc:
                last_exception = exc

                if not self._is_retryable_error(exc):
                    break

                if attempt >= self.MAX_ATTEMPTS:
                    break

                delay = min(
                    self.INITIAL_RETRY_DELAY
                    * (2 ** (attempt - 1)),
                    self.MAX_RETRY_DELAY,
                )

                time.sleep(delay)

        # Gemini errors (including 429/503) must not end the import.  Request
        # the exact same structured object from the independently hosted
        # OpenRouter free-model router, then validate it locally.
        try:
            raw = OpenRouterFallback().generate_json(prompt)
            return schema.model_validate_json(raw)
        except (FallbackProviderError, ValueError) as fallback_error:
            if last_exception is not None:
                raise RuntimeError(
                    "Gemini and OpenRouter both failed. Gemini: "
                    f"{self._format_gemini_error(last_exception)}; "
                    f"fallback: {fallback_error}"
                ) from fallback_error
            raise RuntimeError(str(fallback_error)) from fallback_error

        raise RuntimeError(
            "Gemini request failed without returning a response."
        )

    def extract(
        self,
        *,
        raw_text: str,
        chunks: list[dict],
        knowledge: list[dict],
    ) -> ExtractionEnvelope:
        """
        Perform the first-pass extraction.

        The uploaded resume text, chunks and retrieved RAG
        knowledge are passed to Gemini.

        Gemini must return a complete ResumeInfo object.
        """

        result = self._generate(
            build_extraction_prompt(
                raw_text,
                chunks,
                knowledge,
            ),
            ExtractionEnvelope,
        )

        return result

    def verify(
        self,
        *,
        proposed: ResumeInfo,
        chunks: list[dict],
        knowledge: list[dict],
    ) -> VerificationEnvelope:
        """
        Verify the extracted ResumeInfo against the original
        resume chunks and RAG knowledge.
        """

        result = self._generate(
            build_verification_prompt(
                proposed.model_dump(
                    mode="json"
                ),
                chunks,
                knowledge,
            ),
            VerificationEnvelope,
        )

        return result
