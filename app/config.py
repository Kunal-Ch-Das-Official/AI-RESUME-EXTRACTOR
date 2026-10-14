from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # MongoDB is optional.  A local SQLite store is used automatically when
    # this is not configured, so a missing cloud dependency cannot lose an
    # otherwise successful import.
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

    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    gemini_model: str = Field("gemini-2.5-flash", alias="GEMINI_MODEL")
    gemini_embedding_model: str = Field(
        "gemini-embedding-2", alias="GEMINI_EMBEDDING_MODEL"
    )
    gemini_embedding_dimensions: int = Field(
        768, alias="GEMINI_EMBEDDING_DIMENSIONS"
    )
    openrouter_api_key: str | None = Field(None, alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field("openrouter/free", alias="OPENROUTER_MODEL")
    openrouter_base_url: str = Field(
        "https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL"
    )
    ai_fallback_enabled: bool = Field(True, alias="AI_FALLBACK_ENABLED")

    max_upload_mb: int = Field(10, alias="MAX_UPLOAD_MB")
    min_confidence: float = Field(0.78, alias="MIN_CONFIDENCE")
    review_confidence: float = Field(0.88, alias="REVIEW_CONFIDENCE")
    allowed_extensions: str = Field(
        "pdf,docx",
        alias="ALLOWED_EXTENSIONS",
    )
    database_mode: str = Field("auto", alias="DATABASE_MODE")
    local_database_path: str = Field("data/resumes.sqlite3", alias="LOCAL_DATABASE_PATH")

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
