"""Hand-rolled BM25 index wrapping rank_bm25 with jieba tokenization.

We manage corpus, tokenized docs, and id mapping ourselves — not a black-box
retriever wrapper. BM25 handles keyword / proper-noun recall.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import Any

import jieba
from rank_bm25 import BM25Okapi

from core.models import Chunk, ScoredChunk


class BM25Index:
    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._id_to_pos: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokenized = [self.tokenize(c.text) for c in self._chunks]
        self._id_to_pos = {c.id: i for i, c in enumerate(self._chunks)}
        # BM25Okapi needs at least one doc; empty corpus → no index
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def add(self, chunks: list[Chunk]) -> None:
        """Append chunks and rebuild (simple strategy for local demo)."""
        if not chunks:
            return
        merged = self._chunks + list(chunks)
        # Deduplicate by id, keep latest
        by_id: dict[str, Chunk] = {c.id: c for c in merged}
        self.build(list(by_id.values()))

    def search(self, query: str, top_k: int = 8) -> list[ScoredChunk]:
        if not self._bm25 or not self._chunks:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Top-k indices by score
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[ScoredChunk] = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                break
            results.append(
                ScoredChunk(chunk=self._chunks[i], score=float(scores[i]), channel="bm25")
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
            "chunks": [c.to_dict() for c in self._chunks],
            "tokenized": self._tokenized,
        }
        (directory / "bm25_meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # Persist BM25 internal stats for faster load (optional pickle)
        with open(directory / "bm25_model.pkl", "wb") as f:
            pickle.dump(self._bm25, f)

    def load(self, directory: str | Path) -> None:
        directory = Path(directory)
        meta_path = directory / "bm25_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"BM25 meta not found: {meta_path}")

        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self._chunks = [Chunk.from_dict(c) for c in meta["chunks"]]
        self._tokenized = meta.get("tokenized") or [self.tokenize(c.text) for c in self._chunks]
        self._id_to_pos = {c.id: i for i, c in enumerate(self._chunks)}

        pkl = directory / "bm25_model.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                self._bm25 = pickle.load(f)
        else:
            self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Chinese-aware tokenization with light English normalization."""
        text = text.lower().strip()
        if not text:
            return []
        # Keep alphanumerics as tokens; jieba handles Chinese
        tokens: list[str] = []
        for tok in jieba.lcut(text):
            tok = tok.strip()
            if not tok or tok.isspace():
                continue
            # Drop pure punctuation
            if re.fullmatch(r"[\W_]+", tok, flags=re.UNICODE):
                continue
            tokens.append(tok)
        return tokens
