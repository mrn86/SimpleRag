"""Multi-channel retrieval: query rewrite → BM25 + vector → dedupe → BGE rerank."""

from __future__ import annotations

from config.settings import settings
from core.bm25_index import BM25Index
from core.models import ScoredChunk
from core.query_rewrite import QueryRewriter, RewrittenQuery
from core.reranker import BGEReranker
from core.vector_store import VectorStore


class HybridRetriever:
    def __init__(
        self,
        bm25: BM25Index,
        vector_store: VectorStore,
        rewriter: QueryRewriter | None = None,
        reranker: BGEReranker | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> None:
        self.bm25 = bm25
        self.vector_store = vector_store
        self.rewriter = rewriter or QueryRewriter()
        self.reranker = reranker or BGEReranker()
        self.top_k = top_k or settings.retrieve_top_k
        self.top_n = top_n or settings.rerank_top_n

    def retrieve(self, query: str, rewrite: bool = True) -> tuple[list[ScoredChunk], RewrittenQuery]:
        rewritten = self.rewriter.rewrite(query) if rewrite else RewrittenQuery(original=query)
        queries = rewritten.all_queries or [query]

        # Multi-query × dual-channel recall
        candidates: dict[str, ScoredChunk] = {}
        for q in queries:
            for hit in self.bm25.search(q, top_k=self.top_k):
                prev = candidates.get(hit.id)
                if prev is None or hit.score > prev.score:
                    candidates[hit.id] = hit
            for hit in self.vector_store.search(q, top_k=self.top_k):
                # Keep first-seen channel metadata; scores will be replaced by rerank
                if hit.id not in candidates:
                    candidates[hit.id] = hit

        pool = list(candidates.values())
        # Final order comes ONLY from reranker — not naive list concat
        ranked = self.reranker.rerank(query, pool, top_n=self.top_n)
        return ranked, rewritten
