from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ------------------------------------------------------------------ storage
    # MongoDB is optional. A local SQLite store is used automatically when this
    # is not configured, so a missing cloud dependency cannot lose an import.
    mongodb_uri: str | None = Field(None, alias="MONGODB_URI")
    mongodb_database: str = Field("resume_craft", alias="MONGODB_DATABASE")
    # Must be the same collection used by src/models/resumeInfo.ts.
    mongodb_collection: str = Field(
        "fresh_resume_information", alias="MONGODB_COLLECTION"
    )
    mongodb_kb_collection: str = Field(
        "resume_import_knowledge", alias="MONGODB_KB_COLLECTION"
    )
    mongodb_vector_index: str = Field(
        "resume_knowledge_vector", alias="MONGODB_VECTOR_INDEX"
    )
    database_mode: str = Field("auto", alias="DATABASE_MODE")
    local_database_path: str = Field("data/resumes.sqlite3", alias="LOCAL_DATABASE_PATH")

    # ---------------------------------------------------------------- providers
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-3.5-flash", alias="GEMINI_MODEL")
    # Tried when the primary Gemini model is overloaded (503/429). Capacity is
    # per-model, so this is far more reliable than jumping straight to a free
    # OpenRouter model. Set to an empty string to disable.
    gemini_fallback_model: str | None = Field(
        "gemini-3.5-flash-lite", alias="GEMINI_FALLBACK_MODEL"
    )
    # Hidden "thinking" tokens are billed as output and are not needed for
    # rewriting/extraction. 0 = minimal thinking, -1 = model default.
    gemini_thinking_budget: int = Field(0, alias="GEMINI_THINKING_BUDGET")
    gemini_embedding_model: str = Field(
        "gemini-embedding-2", alias="GEMINI_EMBEDDING_MODEL"
    )
    gemini_embedding_dimensions: int = Field(768, alias="GEMINI_EMBEDDING_DIMENSIONS")

    openrouter_api_key: str | None = Field(None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field("openrouter/free", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    ai_fallback_enabled: bool = Field(True, alias="AI_FALLBACK_ENABLED")

    # ------------------------------------------------------- time / token budgets
    # Hard per-attempt ceilings (seconds) for short tasks. They are enforced with
    # asyncio cancellation, not just socket read timeouts.
    gemini_timeout_seconds: float = Field(10.0, alias="GEMINI_TIMEOUT_SECONDS")
    gemini_fallback_timeout_seconds: float = Field(
        8.0, alias="GEMINI_FALLBACK_TIMEOUT_SECONDS"
    )
    openrouter_timeout_seconds: float = Field(14.0, alias="OPENROUTER_TIMEOUT_SECONDS")
    # Total wall-clock budget across every provider for summaries/responsibilities.
    ai_short_deadline_seconds: float = Field(28.0, alias="AI_SHORT_DEADLINE_SECONDS")
    # Total budget for a full resume extraction.
    ai_extract_deadline_seconds: float = Field(
        75.0, alias="AI_EXTRACT_DEADLINE_SECONDS"
    )
    ai_extraction_enabled: bool = Field(True, alias="AI_EXTRACTION_ENABLED")
    # Resume text sent to the model is capped (~4 chars per token).
    ai_max_input_chars: int = Field(24000, alias="AI_MAX_INPUT_CHARS")
    ai_extract_max_output_tokens: int = Field(6000, alias="AI_EXTRACT_MAX_OUTPUT_TOKENS")
    ai_max_concurrency: int = Field(4, alias="AI_MAX_CONCURRENCY")
    # A provider that fails this many times in a row is skipped for a while so
    # users don't wait for a timeout on every request.
    ai_breaker_threshold: int = Field(3, alias="AI_BREAKER_THRESHOLD")
    ai_breaker_cooldown_seconds: float = Field(45.0, alias="AI_BREAKER_COOLDOWN_SECONDS")

    # ------------------------------------------------------------------- misc
    max_upload_mb: int = Field(10, alias="MAX_UPLOAD_MB")
    min_confidence: float = Field(0.78, alias="MIN_CONFIDENCE")
    review_confidence: float = Field(0.88, alias="REVIEW_CONFIDENCE")
    allowed_extensions: str = Field("pdf,docx", alias="ALLOWED_EXTENSIONS")
    cors_origins: str = Field(
        "http://localhost:3000,http://127.0.0.1:3000", alias="CORS_ORIGINS"
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @property
    def allowed_extension_set(self) -> set[str]:
        return {
            item.strip().lower().lstrip(".")
            for item in self.allowed_extensions.split(",")
            if item.strip()
        }

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.ai_fallback_enabled and self.openrouter_api_key)

    @property
    def any_ai_configured(self) -> bool:
        return bool(self.gemini_api_key or self.openrouter_enabled)


@lru_cache
def get_settings() -> Settings:
    return Settings()
