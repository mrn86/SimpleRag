"""Assemble context with length budget, provenance tags, and prompt templates."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import settings
from core.models import ScoredChunk


SYSTEM_PROMPT = """你是知识库问答助手，只能基于下面【参考文档片段】回答用户问题。
如果参考材料没有答案，直接回答无法从文档找到信息，不要编造。
回答时标注对应的来源编号（如 [1]、[2]）。"""


@dataclass
class BuiltContext:
    context_text: str
    sources: list[dict]
    messages: list[dict]


class ContextBuilder:
    def __init__(self, max_chars: int | None = None) -> None:
        self.max_chars = max_chars or settings.max_context_chars

    def build(self, query: str, scored_chunks: list[ScoredChunk]) -> BuiltContext:
        pieces: list[str] = []
        sources: list[dict] = []
        used = 0

        for i, sc in enumerate(scored_chunks, start=1):
            tag = f"[来源：{sc.chunk.source}，块id:{sc.chunk.id}]"
            title = f"（章节：{sc.chunk.title_path}）" if sc.chunk.title_path else ""
            block = f"[{i}] {tag}{title}\n{sc.chunk.text.strip()}"
            # Budget check (prefer higher-ranked chunks already ordered by rerank)
            if pieces and used + len(block) + 2 > self.max_chars:
                break
            if not pieces and len(block) > self.max_chars:
                block = block[: self.max_chars]
            pieces.append(block)
            used += len(block) + 2
            sources.append(
                {
                    "ref": i,
                    "chunk_id": sc.chunk.id,
                    "source": sc.chunk.source,
                    "title_path": sc.chunk.title_path,
                    "score": sc.score,
                    "text": sc.chunk.text,
                }
            )

        context_text = "\n\n".join(pieces) if pieces else "（无可用参考文档）"
        user_content = (
            f"【参考文档】\n{context_text}\n\n用户问题：{query}"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return BuiltContext(context_text=context_text, sources=sources, messages=messages)
