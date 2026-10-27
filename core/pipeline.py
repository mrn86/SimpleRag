"""End-to-end RAG pipeline: ingest → index → ask."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import settings
from core.bm25_index import BM25Index
from core.chunker import Chunker
from core.context_builder import ContextBuilder
from core.document_parser import parse_directory, parse_file
from core.embedder import Embedder
from core.llm_generator import LLMGenerator
from core.models import Chunk, Document
from core.query_rewrite import QueryRewriter
from core.reranker import BGEReranker
from core.retriever import HybridRetriever
from core.vector_store import VectorStore


@dataclass
class AskResult:
    answer: str
    sources: list[dict]
    rewritten_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.chunker = Chunker(embedder=self.embedder)
        self.bm25 = BM25Index()
        self.vector_store = VectorStore(embedder=self.embedder)
        self.rewriter = QueryRewriter()
        self.reranker = BGEReranker()
        self.retriever = HybridRetriever(
            bm25=self.bm25,
            vector_store=self.vector_store,
            rewriter=self.rewriter,
            reranker=self.reranker,
        )
        self.context_builder = ContextBuilder()
        self.llm = LLMGenerator()
        self.chunks: list[Chunk] = []

    # ---------- ingest ----------

    def ingest_paths(self, paths: list[str | Path]) -> list[Chunk]:
        docs: list[Document] = []
        for p in paths:
            p = Path(p)
            if p.is_dir():
                docs.extend(parse_directory(p))
            elif p.is_file():
                docs.append(parse_file(p))
        return self.ingest_documents(docs)

    def ingest_documents(self, documents: list[Document], rebuild: bool = True) -> list[Chunk]:
        new_chunks = self.chunker.chunk_documents(documents)
        if rebuild or not self.chunks:
            self.chunks = new_chunks
            self.bm25.build(self.chunks)
            self.vector_store.build(self.chunks)
        else:
            # Merge by id
            by_id = {c.id: c for c in self.chunks}
            for c in new_chunks:
                by_id[c.id] = c
            self.chunks = list(by_id.values())
            self.bm25.build(self.chunks)
            self.vector_store.build(self.chunks)
        return new_chunks

    def ingest_directory(self, directory: str | Path, rebuild: bool = True) -> list[Chunk]:
        docs = parse_directory(directory)
        return self.ingest_documents(docs, rebuild=rebuild)

    # ---------- persistence ----------

    @staticmethod
    def _bm25_db_path(storage_dir: Path) -> Path:
        p = Path(settings.db_path)
        return p if p.is_absolute() else storage_dir / p

    def save(self, storage_dir: str | Path | None = None) -> None:
        storage_dir = Path(storage_dir or settings.storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)
        # BM25 inverted index → SQLite; vectors → FAISS files
        self.bm25.save(self._bm25_db_path(storage_dir))
        self.vector_store.save(storage_dir / "faiss")

    def load(self, storage_dir: str | Path | None = None) -> None:
        storage_dir = Path(storage_dir or settings.storage_path)
        self.bm25.load(self._bm25_db_path(storage_dir))
        self.vector_store.load(storage_dir / "faiss")
        # Prefer FAISS chunk list as canonical
        self.chunks = list(self.vector_store._chunks)

    # ---------- QA ----------

    def ask(self, query: str, rewrite: bool = True) -> AskResult:
        scored, rewritten = self.retriever.retrieve(query, rewrite=rewrite)
        built = self.context_builder.build(query, scored)
        answer = self.llm.generate(built.messages)
        return AskResult(
            answer=answer,
            sources=built.sources,
            rewritten_queries=rewritten.queries,
            keywords=rewritten.keywords,
        )

    def retrieve_only(self, query: str, rewrite: bool = True) -> list[dict[str, Any]]:
        scored, _ = self.retriever.retrieve(query, rewrite=rewrite)
        return [
            {
                "chunk_id": s.chunk.id,
                "source": s.chunk.source,
                "score": s.score,
                "text": s.chunk.text,
                "title_path": s.chunk.title_path,
            }
            for s in scored
        ]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "num_chunks": len(self.chunks),
            "bm25_size": self.bm25.size,
            "vector_size": self.vector_store.size,
        }
