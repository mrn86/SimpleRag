"""FAISS vector store: IndexFlatIP + L2-normalized vectors ≈ cosine similarity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np

from core.embedder import Embedder
from core.models import Chunk, ScoredChunk


class VectorStore:
    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or Embedder()
        self._chunks: list[Chunk] = []
        self._id_to_pos: dict[str, int] = {}
        self._index: faiss.IndexFlatIP | None = None
        self._dim: int = 0

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._id_to_pos = {c.id: i for i, c in enumerate(self._chunks)}
        if not self._chunks:
            self._index = None
            self._dim = 0
            return

        vectors = self.embedder.embed([c.text for c in self._chunks])
        self._dim = int(vectors.shape[1])
        self._index = faiss.IndexFlatIP(self._dim)
        self._index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def add(self, chunks: list[Chunk]) -> None:
        """Append and rebuild (keeps local demo simple & consistent)."""
        if not chunks:
            return
        by_id = {c.id: c for c in self._chunks}
        for c in chunks:
            by_id[c.id] = c
        self.build(list(by_id.values()))

    def search(self, query: str, top_k: int = 8) -> list[ScoredChunk]:
        if self._index is None or not self._chunks:
            return []
        q = self.embedder.embed_query(query).reshape(1, -1).astype(np.float32)
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(np.ascontiguousarray(q), k)
        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                ScoredChunk(
                    chunk=self._chunks[int(idx)],
                    score=float(score),
                    channel="vector",
                )
            )
        return results

    def get(self, chunk_id: str) -> Chunk | None:
        pos = self._id_to_pos.get(chunk_id)
        if pos is None:
            return None
        return self._chunks[pos]

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "dim": self._dim,
            "chunks": [c.to_dict() for c in self._chunks],
        }
        (directory / "faiss_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if self._index is not None:
            faiss.write_index(self._index, str(directory / "faiss.index"))

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        meta_path = directory / "faiss_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"FAISS meta not found: {meta_path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk.from_dict(c) for c in meta["chunks"]]
        self._id_to_pos = {c.id: i for i, c in enumerate(self._chunks)}
        self._dim = int(meta.get("dim") or 0)

        index_path = directory / "faiss.index"
        if index_path.exists() and self._chunks:
            self._index = faiss.read_index(str(index_path))
            self._dim = self._index.d
        else:
            self._index = None

    def search_by_vector(self, vector: Sequence[float], top_k: int = 8) -> list[ScoredChunk]:
        if self._index is None or not self._chunks:
            return []
        q = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        # Ensure L2 norm
        norm = np.linalg.norm(q, axis=1, keepdims=True)
        q = q / np.clip(norm, 1e-12, None)
        k = min(top_k, len(self._chunks))
        scores, indices = self._index.search(np.ascontiguousarray(q), k)
        results: list[ScoredChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(
                ScoredChunk(chunk=self._chunks[int(idx)], score=float(score), channel="vector")
            )
        return results
