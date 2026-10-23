"""Embedding: DashScope(千问) / OpenAI-compatible / HF / local."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from config.settings import settings


class Embedder:
    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.provider = (provider or settings.embedding_provider or "dashscope").lower()
        self.model = model or settings.embedding_model
        # DashScope text-embedding-v3: batch size must be <= 10
        if batch_size is not None:
            self.batch_size = batch_size
        elif self.provider in {"dashscope", "qwen"}:
            self.batch_size = 10
        else:
            self.batch_size = 16
        if self.provider in {"dashscope", "qwen"}:
            self.batch_size = min(self.batch_size, 10)
        self._openai = None
        self._local_model = None

        if self.provider in {"openai", "dashscope", "qwen"}:
            from openai import OpenAI

            if self.provider in {"dashscope", "qwen"}:
                key = (
                    api_key
                    or settings.dashscope_api_key
                    or settings.embedding_api_key
                    or ""
                )
                url = (
                    base_url
                    or settings.embedding_base_url
                    or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                )
            else:
                key = (
                    api_key
                    or settings.embedding_api_key
                    or settings.openai_api_key
                )
                url = (
                    base_url
                    or settings.embedding_base_url
                    or settings.openai_base_url
                )

            if not key or key.startswith("sk-xxx"):
                raise RuntimeError(
                    "Embedding API Key 未配置。DashScope 请设置 DASHSCOPE_API_KEY。"
                )

            self._openai = OpenAI(api_key=key, base_url=url)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        cleaned = [t if t.strip() else " " for t in texts]

        if self.provider in {"hf", "huggingface", "remote"}:
            from core.hf_auth import ensure_hf_token
            from core.hf_client import feature_extraction

            ensure_hf_token()
            parts: list[np.ndarray] = []
            for i in range(0, len(cleaned), self.batch_size):
                batch = cleaned[i : i + self.batch_size]
                parts.append(feature_extraction(batch, model=self.model))
            return self._l2_normalize(np.vstack(parts).astype(np.float32))

        if self.provider == "local":
            model = self._ensure_local()
            vectors = model.encode(
                list(cleaned),
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return np.asarray(vectors, dtype=np.float32)

        # openai / dashscope compatible
        assert self._openai is not None
        vectors: list[list[float]] = []
        for i in range(0, len(cleaned), self.batch_size):
            batch = list(cleaned[i : i + self.batch_size])
            resp = self._openai.embeddings.create(model=self.model, input=batch)
            ordered = sorted(resp.data, key=lambda x: x.index)
            vectors.extend([item.embedding for item in ordered])
        return self._l2_normalize(np.asarray(vectors, dtype=np.float32))

    def _ensure_local(self):
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            from core.hf_auth import ensure_hf_token

            ensure_hf_token()
            token = settings.hf_token or None
            self._local_model = SentenceTransformer(
                self.model,
                token=token if token else None,
            )
        return self._local_model

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return vectors / norms
