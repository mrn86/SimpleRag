"""SQLite persistence for BM25 inverted index and chunk corpus."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from config.settings import settings
from core.models import Chunk


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    title_path TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS inverted (
    term TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    tf INTEGER NOT NULL,
    PRIMARY KEY (term, chunk_id),
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_inverted_term ON inverted(term);

CREATE TABLE IF NOT EXISTS docs_stats (
    chunk_id TEXT PRIMARY KEY,
    doc_len INTEGER NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class IndexDB:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path or settings.db_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA_SQL)

    def clear_bm25(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM inverted")
            conn.execute("DELETE FROM docs_stats")
            conn.execute("DELETE FROM chunks")

    def has_chunks(self) -> bool:
        with self.connect() as conn:
            row = conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone()
            return row is not None

    def replace_corpus(
        self,
        chunks: list[Chunk],
        tokenized: list[list[str]],
    ) -> None:
        if len(chunks) != len(tokenized):
            raise ValueError("chunks and tokenized length mismatch")

        with self.connect() as conn:
            conn.execute("DELETE FROM inverted")
            conn.execute("DELETE FROM docs_stats")
            conn.execute("DELETE FROM chunks")

            for chunk, tokens in zip(chunks, tokenized):
                conn.execute(
                    """
                    INSERT INTO chunks (id, text, source, title_path, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.id,
                        chunk.text,
                        chunk.source,
                        chunk.title_path,
                        json.dumps(chunk.metadata, ensure_ascii=False),
                    ),
                )
                conn.execute(
                    "INSERT INTO docs_stats (chunk_id, doc_len) VALUES (?, ?)",
                    (chunk.id, len(tokens)),
                )
                tf = Counter(tokens)
                if tf:
                    conn.executemany(
                        "INSERT INTO inverted (term, chunk_id, tf) VALUES (?, ?, ?)",
                        [(term, chunk.id, count) for term, count in tf.items()],
                    )

    def load_chunks(self) -> list[Chunk]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, text, source, title_path, metadata_json FROM chunks"
            ).fetchall()
        chunks: list[Chunk] = []
        for row in rows:
            meta = json.loads(row["metadata_json"] or "{}")
            chunks.append(
                Chunk(
                    id=row["id"],
                    text=row["text"],
                    source=row["source"],
                    title_path=row["title_path"] or "",
                    metadata=meta,
                )
            )
        return chunks

    def load_tokenized(self, chunk_ids: list[str]) -> list[list[str]]:
        """Rebuild token lists from inverted postings (order follows chunk_ids)."""
        if not chunk_ids:
            return []

        with self.connect() as conn:
            placeholders = ",".join("?" * len(chunk_ids))
            rows = conn.execute(
                f"""
                SELECT chunk_id, term, tf
                FROM inverted
                WHERE chunk_id IN ({placeholders})
                """,
                chunk_ids,
            ).fetchall()

        by_id: dict[str, list[str]] = {cid: [] for cid in chunk_ids}
        for row in rows:
            by_id[row["chunk_id"]].extend([row["term"]] * int(row["tf"]))
        return [by_id[cid] for cid in chunk_ids]

    def postings_for_term(self, term: str) -> list[tuple[str, int]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT chunk_id, tf FROM inverted WHERE term = ?",
                (term,),
            ).fetchall()
        return [(r["chunk_id"], int(r["tf"])) for r in rows]
