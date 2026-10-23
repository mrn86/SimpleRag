"""Alibaba DashScope (千问/百炼) remote embedding helpers & rerank."""

from __future__ import annotations

from typing import Any

import httpx

from config.settings import settings


def dashscope_api_key() -> str:
    key = (
        settings.dashscope_api_key
        or settings.embedding_api_key
        or ""
    ).strip()
    if not key or key.startswith("sk-xxx"):
        raise RuntimeError(
            "缺少 DashScope/百炼 API Key。请在 .env 设置 DASHSCOPE_API_KEY=sk-..."
        )
    return key


def dashscope_rerank(
    query: str,
    documents: list[str],
    model: str,
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """
    Call DashScope text-rerank API.
    Returns list of (original_index, score) sorted by score desc.
    """
    if not documents:
        return []

    url = (
        settings.dashscope_rerank_url
        or "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    headers = {
        "Authorization": f"Bearer {dashscope_api_key()}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "input": {"query": query, "documents": documents},
        "parameters": {
            "return_documents": False,
            "top_n": top_n if top_n is not None else len(documents),
        },
    }

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"DashScope rerank HTTP {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()

    # Response: output.results = [{index, relevance_score}, ...]
    results = (
        data.get("output", {}).get("results")
        or data.get("results")
        or []
    )
    if not results:
        raise RuntimeError(f"DashScope rerank empty response: {str(data)[:400]}")

    scored: list[tuple[int, float]] = []
    for item in results:
        idx = int(item.get("index", 0))
        score = float(item.get("relevance_score", item.get("score", 0.0)))
        scored.append((idx, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
