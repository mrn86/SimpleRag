"""Query rewrite: expand short/vague questions into multiple retrieval queries."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from config.settings import settings


REWRITE_PROMPT = """你是查询改写助手，基于用户问题，生成2个语义等价、侧重点不同的查询，用于知识库检索。
同时提取3-8个关键词（专有名词、实体、核心术语优先）。

用户问题：{query}

严格输出 JSON，不要 markdown 代码块，格式：
{{"queries": ["改写1", "改写2"], "keywords": ["词1", "词2"]}}
"""


@dataclass
class RewrittenQuery:
    original: str
    queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)

    @property
    def all_queries(self) -> list[str]:
        """Original + rewrites, deduplicated, preserving order."""
        seen: set[str] = set()
        out: list[str] = []
        for q in [self.original, *self.queries]:
            key = q.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
        # Append keyword-joined query for BM25 boost if useful
        if self.keywords:
            kw = " ".join(self.keywords)
            if kw not in seen:
                out.append(kw)
        return out


class QueryRewriter:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.model = model or settings.llm_model
        self.enabled = enabled
        self._client = OpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url,
        )

    def rewrite(self, query: str) -> RewrittenQuery:
        query = query.strip()
        if not self.enabled or not query:
            return RewrittenQuery(original=query, queries=[], keywords=[])

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": REWRITE_PROMPT.format(query=query)},
                ],
                temperature=0.3,
            )
            content = (resp.choices[0].message.content or "").strip()
            data = _parse_json(content)
            queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
            keywords = [str(k).strip() for k in data.get("keywords", []) if str(k).strip()]
            return RewrittenQuery(original=query, queries=queries[:2], keywords=keywords)
        except Exception:
            # Fail open: still retrieve with original query
            return RewrittenQuery(original=query, queries=[], keywords=[])


def _parse_json(text: str) -> dict:
    text = text.strip()
    # Strip ```json fences if present
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find first {...}
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return json.loads(m.group(0))
        raise
