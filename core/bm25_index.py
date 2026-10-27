"""BM25 index: jieba tokenization + rank_bm25, inverted postings persisted in SQLite."""

from __future__ import annotations

import re
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from core.db import IndexDB
from core.models import Chunk, ScoredChunk


class BM25Index:
    def __init__(self, db: IndexDB | None = None) -> None:
        self._db = db
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._id_to_pos: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self._chunks)

    def _ensure_db(self) -> IndexDB:
        if self._db is None:
            self._db = IndexDB()
        return self._db

    def build(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokenized = [self.tokenize(c.text) for c in self._chunks]
        self._id_to_pos = {c.id: i for i, c in enumerate(self._chunks)}
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def add(self, chunks: list[Chunk]) -> None:
        """Append chunks and rebuild (simple strategy for local demo)."""
        if not chunks:
            return
        merged = self._chunks + list(chunks)
        by_id: dict[str, Chunk] = {c.id: c for c in merged}
        self.build(list(by_id.values()))

    def search(self, query: str, top_k: int = 8) -> list[ScoredChunk]:
        if not self._bm25 or not self._chunks:
            return []
        tokens = self.tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
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

    def save(self, db_path: str | Path | None = None) -> None:
        """Persist corpus + inverted index to SQLite."""
        db = IndexDB(db_path) if db_path is not None else self._ensure_db()
        self._db = db
        db.replace_corpus(self._chunks, self._tokenized)

    def load(self, db_path: str | Path | None = None) -> None:
        """Load chunks + inverted postings from SQLite and rebuild BM25Okapi."""
        db = IndexDB(db_path) if db_path is not None else self._ensure_db()
        self._db = db
        if not db.has_chunks():
            raise FileNotFoundError(f"BM25 SQLite has no chunks: {db.path}")

        self._chunks = db.load_chunks()
        self._tokenized = db.load_tokenized([c.id for c in self._chunks])
        self._id_to_pos = {c.id: i for i, c in enumerate(self._chunks)}
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Chinese-aware tokenization with light English normalization."""
        text = text.lower().strip()
        if not text:
            return []
        tokens: list[str] = []
        for tok in jieba.lcut(text):
            tok = tok.strip()
            if not tok or tok.isspace():
                continue
            if re.fullmatch(r"[\W_]+", tok, flags=re.UNICODE):
                continue
            tokens.append(tok)
        return tokens
