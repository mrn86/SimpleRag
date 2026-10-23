"""Hierarchical + semantic chunker.

Strategy:
1. Prefer splitting by markdown / document headings to keep semantic boundaries.
2. If a section exceeds chunk_size, split by sentence embedding similarity valleys.
3. Keep overlap between adjacent chunks to avoid context fragmentation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from config.settings import settings
from core.models import Chunk, Document

if TYPE_CHECKING:
    from core.embedder import Embedder


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[。！？!?；;.\n])\s*"
)


@dataclass
class _Section:
    title_path: str
    text: str
    level: int


class Chunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        overlap: int | None = None,
        similarity_threshold: float | None = None,
        embedder: "Embedder | None" = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.overlap = overlap or settings.chunk_overlap
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.semantic_similarity_threshold
        )
        self.embedder = embedder

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(self.chunk_document(doc))
        return chunks

    def chunk_document(self, document: Document) -> list[Chunk]:
        sections = self._split_by_headings(document.text)
        if not sections:
            sections = [_Section(title_path=document.title or "", text=document.text, level=0)]

        raw_parts: list[tuple[str, str]] = []  # (title_path, text)
        for section in sections:
            text = section.text.strip()
            if not text:
                continue
            if len(text) <= self.chunk_size:
                raw_parts.append((section.title_path, text))
            else:
                for piece in self._semantic_split(text):
                    raw_parts.append((section.title_path, piece))

        # Apply overlap between consecutive pieces from same title path
        overlapped = self._apply_overlap(raw_parts)

        chunks: list[Chunk] = []
        for idx, (title_path, text) in enumerate(overlapped):
            chunk_id = self._make_id(document.source, idx, text)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source=document.source,
                    title_path=title_path,
                    metadata={"index": idx, "title": document.title},
                )
            )
        return chunks

    def _split_by_headings(self, text: str) -> list[_Section]:
        lines = text.splitlines()
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []  # (level, title)
        buffer: list[str] = []
        current_level = 0

        def flush() -> None:
            nonlocal buffer, current_level
            body = "\n".join(buffer).strip()
            if body or stack:
                title_path = " > ".join(t for _, t in stack)
                sections.append(_Section(title_path=title_path, text=body, level=current_level))
            buffer = []

        for line in lines:
            m = HEADING_RE.match(line.strip())
            if m:
                flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current_level = level
                # Keep heading text inside the section for context
                buffer.append(line.strip())
            else:
                buffer.append(line)

        flush()
        # Drop empty leading section without titles if everything is empty
        return [s for s in sections if s.text.strip()]

    def _split_sentences(self, text: str) -> list[str]:
        parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p and p.strip()]
        if not parts:
            return [text]
        return parts

    def _semantic_split(self, text: str) -> list[str]:
        sentences = self._split_sentences(text)
        if len(sentences) == 1:
            return self._hard_split(text)

        # Without embedder: fall back to size-based packing of sentences
        if self.embedder is None:
            return self._pack_sentences(sentences)

        try:
            embeddings = self.embedder.embed(sentences)
        except Exception:
            return self._pack_sentences(sentences)

        # Adjacent cosine similarities (already L2-normalized → dot product)
        sims = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        # Cut at local valleys below threshold, while respecting chunk_size
        cuts: list[int] = []
        current_len = len(sentences[0])
        for i, sim in enumerate(sims):
            next_len = len(sentences[i + 1])
            would_exceed = current_len + next_len + 1 > self.chunk_size
            is_valley = sim < self.similarity_threshold
            # Prefer cutting at semantic break when over size, or strong valley after min size
            if would_exceed and (is_valley or current_len >= self.chunk_size * 0.6):
                cuts.append(i)
                current_len = next_len
            elif is_valley and current_len >= self.chunk_size * 0.5:
                cuts.append(i)
                current_len = next_len
            else:
                current_len += next_len + 1

        return self._materialize_cuts(sentences, cuts)

    def _pack_sentences(self, sentences: list[str]) -> list[str]:
        chunks: list[str] = []
        buf: list[str] = []
        size = 0
        for s in sentences:
            add = len(s) + (1 if buf else 0)
            if buf and size + add > self.chunk_size:
                chunks.append("".join(buf) if self._is_cjk("".join(buf)) else " ".join(buf))
                buf = [s]
                size = len(s)
            else:
                buf.append(s)
                size += add
        if buf:
            chunks.append("".join(buf) if self._is_cjk("".join(buf)) else " ".join(buf))
        return chunks

    def _materialize_cuts(self, sentences: list[str], cut_after_indices: list[int]) -> list[str]:
        cut_set = set(cut_after_indices)
        groups: list[list[str]] = []
        current: list[str] = []
        for i, s in enumerate(sentences):
            current.append(s)
            if i in cut_set:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        result: list[str] = []
        for g in groups:
            joined = "".join(g) if self._is_cjk("".join(g)) else " ".join(g)
            if len(joined) > self.chunk_size * 1.5:
                result.extend(self._hard_split(joined))
            else:
                result.append(joined)
        return result

    def _hard_split(self, text: str) -> list[str]:
        """Last-resort fixed window with overlap (character-level)."""
        if len(text) <= self.chunk_size:
            return [text]
        step = max(1, self.chunk_size - self.overlap)
        parts: list[str] = []
        for start in range(0, len(text), step):
            piece = text[start : start + self.chunk_size]
            if piece.strip():
                parts.append(piece)
            if start + self.chunk_size >= len(text):
                break
        return parts

    def _apply_overlap(self, parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
        if self.overlap <= 0 or len(parts) <= 1:
            return parts

        result: list[tuple[str, str]] = [parts[0]]
        for i in range(1, len(parts)):
            title_path, text = parts[i]
            prev_text = parts[i - 1][1]
            # Only overlap within same section path
            if title_path == parts[i - 1][0]:
                prefix = prev_text[-self.overlap :] if len(prev_text) > self.overlap else prev_text
                # Avoid duplicating if already starts with prefix
                if not text.startswith(prefix.strip()[:20]):
                    text = prefix + ("\n" if not prefix.endswith("\n") else "") + text
            result.append((title_path, text))
        return result

    @staticmethod
    def _is_cjk(text: str) -> bool:
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        return cjk > len(text) * 0.2

    @staticmethod
    def _make_id(source: str, index: int, text: str) -> str:
        digest = hashlib.md5(f"{source}:{index}:{text[:64]}".encode("utf-8")).hexdigest()[:12]
        return f"{PathSafe(source)}-{index}-{digest}"


class PathSafe(str):
    """Sanitize source filename for use in chunk ids."""

    def __new__(cls, value: str) -> "PathSafe":
        safe = re.sub(r"[^\w.\-]+", "_", value)
        return str.__new__(cls, safe)
