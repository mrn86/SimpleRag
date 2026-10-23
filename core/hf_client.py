"""Shared HuggingFace Inference helpers (remote only, no local weights)."""

from __future__ import annotations

import time
from typing import Any

import httpx
import numpy as np

from config.settings import settings
from core.hf_auth import ensure_hf_token


def _token() -> str:
    ensure_hf_token()
    token = (settings.hf_token or "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for remote HuggingFace Inference.")
    return token


def get_inference_client():
    from huggingface_hub import InferenceClient

    return InferenceClient(
        provider="hf-inference",
        api_key=_token(),
    )


def feature_extraction(texts: list[str], model: str) -> np.ndarray:
    """Remote embeddings via HF Inference feature-extraction."""
    client = get_inference_client()
    vectors: list[np.ndarray] = []
    for text in texts:
        try:
            raw = _with_retry(lambda t=text: client.feature_extraction(t, model=model))
        except Exception as e:
            raise _friendly_hf_error(e) from e
        vectors.append(_to_sentence_vector(raw))
    return np.stack(vectors, axis=0).astype(np.float32)


def _friendly_hf_error(exc: Exception) -> RuntimeError:
    msg = str(exc)
    if "403" in msg or "sufficient permissions" in msg.lower():
        return RuntimeError(
            "HuggingFace Token 无 Inference Providers 权限，无法远程调用 BGE。\n"
            "请到 https://huggingface.co/settings/tokens 新建 Token，勾选：\n"
            "  - Read\n"
            "  - Make calls to Inference Providers（或 Serverless Inference）\n"
            "然后更新 .env 里的 HF_TOKEN 后重试。"
        )
    if "402" in msg or "no remaining credits" in msg.lower():
        return RuntimeError(
            "HuggingFace Inference Providers 额度已用完（402）。\n"
            "可：充值/开通 PRO，或把 .env 改为 EMBEDDING_PROVIDER=local 与 "
            "RERANKER_PROVIDER=local 本机跑 BGE，或换其他远程向量服务。"
        )
    return RuntimeError(f"HuggingFace Inference 调用失败: {msg[:500]}")


def _to_sentence_vector(raw: Any) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1]).mean(axis=0)
    return arr


def cross_encoder_scores(query: str, documents: list[str], model: str) -> list[float]:
    """Score (query, doc) pairs via HF Inference for BGE reranker."""
    if not documents:
        return []

    token = _token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"inputs": [[query, doc] for doc in documents]}
    urls = [
        f"https://router.huggingface.co/hf-inference/models/{model}",
        f"https://api-inference.huggingface.co/models/{model}",
    ]

    last_err: Exception | None = None
    for url in urls:
        try:
            data = _with_retry(lambda u=url: _http_post_json(u, headers, payload))
            return _parse_pair_scores(data, len(documents))
        except Exception as e:
            last_err = e
            continue
    if last_err is not None:
        raise _friendly_hf_error(last_err)
    raise RuntimeError("HF rerank failed with unknown error")


def _http_post_json(url: str, headers: dict, payload: dict) -> Any:
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        if resp.status_code == 503:
            raise RuntimeError(f"HF model loading (503): {resp.text[:300]}")
        if resp.status_code >= 400:
            raise RuntimeError(f"HF Inference HTTP {resp.status_code}: {resp.text[:400]}")
        return resp.json()


def _parse_pair_scores(data: Any, n: int) -> list[float]:
    if isinstance(data, list) and data and isinstance(data[0], list):
        scores: list[float] = []
        for item in data:
            if isinstance(item, list) and item:
                best = max(item, key=lambda x: float(x.get("score", 0.0)))
                scores.append(float(best["score"]))
            elif isinstance(item, dict) and "score" in item:
                scores.append(float(item["score"]))
            else:
                scores.append(0.0)
        return scores[:n]

    if isinstance(data, list) and data and isinstance(data[0], dict) and "score" in data[0]:
        return [float(x["score"]) for x in data][:n]

    if isinstance(data, list) and data and isinstance(data[0], (int, float)):
        return [float(x) for x in data][:n]

    raise RuntimeError(f"Unexpected HF rerank response: {str(data)[:400]}")


def _with_retry(fn, retries: int = 6, base_delay: float = 3.0):
    last: Exception | None = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            msg = str(e).lower()
            retryable = any(
                k in msg for k in ("503", "loading", "rate", "timeout", "429", "estimated_time")
            )
            if i < retries - 1 and retryable:
                time.sleep(base_delay * (i + 1))
                continue
            raise
    assert last is not None
    raise last
