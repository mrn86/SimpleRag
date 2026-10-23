"""Shared data models for documents and chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """Parsed document with optional markdown-style structure."""

    source: str
    text: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable text segment with provenance."""

    id: str
    text: str
    source: str
    title_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "title_path": self.title_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            source=data["source"],
            title_path=data.get("title_path", ""),
            metadata=data.get("metadata") or {},
        )


@dataclass
class ScoredChunk:
    """Chunk with retrieval / rerank score."""

    chunk: Chunk
    score: float
    channel: str = ""  # bm25 | vector | rerank

    @property
    def id(self) -> str:
        return self.chunk.id

    @property
    def text(self) -> str:
        return self.chunk.text
