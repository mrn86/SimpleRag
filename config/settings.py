"""Central configuration for Minimal RAG."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (OpenAI-compatible, e.g. DeepSeek)
    openai_api_key: str = "sk-xxx"
    openai_base_url: str = "https://api.deepseek.com/v1"
    llm_model: str = "deepseek-flash"

    # DashScope / 百炼（千问 Embedding & Rerank）
    dashscope_api_key: str = ""
    dashscope_rerank_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )

    # HuggingFace token（仅 local/hf BGE 时需要）
    hf_token: str = ""

    # Embedding: dashscope | openai | local | hf
    embedding_provider: str = "dashscope"
    embedding_model: str = "text-embedding-v3"
    embedding_api_key: str = ""
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Rerank: dashscope | local | hf
    reranker_provider: str = "dashscope"
    reranker_model: str = "qwen3-rerank"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 80
    semantic_similarity_threshold: float = 0.55

    # Retrieval
    retrieve_top_k: int = 8
    rerank_top_n: int = 5
    max_context_chars: int = 6000

    # Paths
    data_dir: str = "data"
    storage_dir: str = "storage"
    # BM25 inverted index SQLite DB (relative to storage_dir unless absolute)
    db_path: str = "rag.db"

    @property
    def root(self) -> Path:
        return ROOT_DIR

    @property
    def data_path(self) -> Path:
        return ROOT_DIR / self.data_dir

    @property
    def storage_path(self) -> Path:
        return ROOT_DIR / self.storage_dir

    @property
    def db_file(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else self.storage_path / p

    @property
    def uploads_path(self) -> Path:
        return self.data_path / "uploads"

    @property
    def sample_path(self) -> Path:
        return self.data_path / "sample"


settings = Settings()
