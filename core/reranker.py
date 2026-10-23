"""Reranker: DashScope(千问) / local BGE / HF remote."""

from __future__ import annotations

from typing import Sequence

from config.settings import settings
from core.models import Chunk, ScoredChunk


class BGEReranker:
    """Name kept for compatibility; supports dashscope / local / hf backends."""

    def __init__(
        self,
        model_name: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self.provider = (provider or settings.reranker_provider or "dashscope").lower()
        self._local_model = None

    def _ensure_local(self):
        if self._local_model is None:
            from sentence_transformers import CrossEncoder

            from core.hf_auth import ensure_hf_token

            ensure_hf_token()
            self._local_model = CrossEncoder(self.model_name)
        return self._local_model

    def rerank(
        self,
        query: str,
        candidates: Sequence[Chunk | ScoredChunk],
        top_n: int | None = None,
    ) -> list[ScoredChunk]:
        top_n = top_n if top_n is not None else settings.rerank_top_n
        if not candidates:
            return []

        chunks: list[Chunk] = []
        for c in candidates:
            chunks.append(c.chunk if isinstance(c, ScoredChunk) else c)

        seen: set[str] = set()
        unique: list[Chunk] = []
        for ch in chunks:
            if ch.id in seen:
                continue
            seen.add(ch.id)
            unique.append(ch)

        if len(unique) == 1:
            return [ScoredChunk(chunk=unique[0], score=1.0, channel="rerank")]

        if self.provider in {"dashscope", "qwen"}:
            from core.dashscope_client import dashscope_rerank

            docs = [ch.text for ch in unique]
            ranked = dashscope_rerank(
                query, docs, model=self.model_name, top_n=top_n
            )
            return [
                ScoredChunk(chunk=unique[idx], score=score, channel="rerank")
                for idx, score in ranked
                if 0 <= idx < len(unique)
            ][:top_n]

        if self.provider in {"hf", "huggingface", "remote"}:
            from core.hf_auth import ensure_hf_token
            from core.hf_client import cross_encoder_scores

            ensure_hf_token()
            docs = [ch.text for ch in unique]
            scores = cross_encoder_scores(query, docs, model=self.model_name)
            scored = [
                ScoredChunk(chunk=ch, score=float(s), channel="rerank")
                for ch, s in zip(unique, scores)
            ]
        else:
            model = self._ensure_local()
            pairs = [(query, ch.text) for ch in unique]
            raw = model.predict(pairs)
            scored = [
                ScoredChunk(chunk=ch, score=float(s), channel="rerank")
                for ch, s in zip(unique, raw)
            ]

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_n]
