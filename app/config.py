"""Configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- embeddings ---
    # 'local' keeps the retrieval evaluation reproducible without an API key.
    embedding_provider: str = "local"
    embedding_model: str = ""
    voyage_api_key: str = ""
    openai_api_key: str = ""

    # --- generation ---
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-4-5"
    max_answer_tokens: int = 1024

    # --- chunking ---
    chunk_size: int = 1500
    chunk_overlap: int = 200

    # --- retrieval ---
    # Candidates fetched per retriever before fusion. Deliberately larger than
    # top_k: fusion can only reorder what it is given, so a chunk ranked 20th
    # by one method and 3rd by the other needs both lists to run deep.
    candidate_k: int = 30
    top_k: int = 8
    rrf_k: int = 60
    # Floor on the best retrieval score before the generator is called at all.
    # 0 disables the check; raise it to make the system refuse more readily.
    min_retrieval_score: float = 0.0

    # --- generation context ---
    max_context_chars: int = 12_000

    # --- storage ---
    store: str = "memory"                       # memory | pgvector
    database_url: str = "postgresql://engine:engine@localhost:5432/engine"

    # --- ingestion ---
    upload_dir: str = "data/uploads"
    max_upload_mb: int = 50
    max_concurrent_ingestions: int = 2
    embed_batch_size: int = 64

    log_level: str = "INFO"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def can_generate(self) -> bool:
        return bool(self.anthropic_api_key)


settings = Settings()
