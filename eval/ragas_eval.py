"""RAG evaluation: retrieval Hit Rate / Recall@k + optional Ragas generation metrics.

Usage:
  python eval/ragas_eval.py
  python eval/ragas_eval.py --skip-ragas
  python eval/ragas_eval.py --testset eval/sample_testset.json --k 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from core.pipeline import RAGPipeline


def _keyword_hit(chunk_text: str, keywords: list[str]) -> bool:
    text = chunk_text.lower()
    return any(kw.lower() in text for kw in keywords if kw)


def evaluate_retrieval(
    pipeline: RAGPipeline,
    testset: list[dict],
    k: int,
) -> dict:
    hits = 0
    recall_sum = 0.0
    details: list[dict] = []

    for item in testset:
        q = item["question"]
        keywords = item.get("relevant_keywords") or []
        # Temporarily use top_n = k for eval
        old_n = pipeline.retriever.top_n
        pipeline.retriever.top_n = k
        try:
            retrieved = pipeline.retrieve_only(q, rewrite=True)
        finally:
            pipeline.retriever.top_n = old_n

        matched = [
            r for r in retrieved if keywords and _keyword_hit(r["text"], keywords)
        ]
        hit = 1.0 if matched else 0.0
        # Soft recall: fraction of keywords appearing in top-k union
        if keywords:
            found = sum(
                1
                for kw in keywords
                if any(kw.lower() in r["text"].lower() for r in retrieved)
            )
            recall = found / len(keywords)
        else:
            recall = hit

        hits += hit
        recall_sum += recall
        details.append(
            {
                "question": q,
                "hit": hit,
                "recall": recall,
                "retrieved_ids": [r["chunk_id"] for r in retrieved],
            }
        )

    n = max(len(testset), 1)
    return {
        "hit_rate": hits / n,
        f"recall@{k}": recall_sum / n,
        "num_cases": len(testset),
        "details": details,
    }


def evaluate_generation_ragas(
    pipeline: RAGPipeline,
    testset: list[dict],
) -> dict:
    """Answer Relevancy + Faithfulness via Ragas."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError as e:
        return {"error": f"ragas/datasets not installed: {e}"}

    questions: list[str] = []
    answers: list[str] = []
    contexts: list[list[str]] = []
    ground_truths: list[str] = []

    for item in testset:
        q = item["question"]
        result = pipeline.ask(q, rewrite=True)
        questions.append(q)
        answers.append(result.answer)
        contexts.append([s["text"] for s in result.sources])
        ground_truths.append(item.get("ground_truth", ""))

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    # Ragas uses OPENAI_API_KEY / compatible settings via langchain/openai env
    import os

    os.environ.setdefault("OPENAI_API_KEY", settings.openai_api_key)
    if settings.openai_base_url:
        os.environ.setdefault("OPENAI_BASE_URL", settings.openai_base_url)

    result = evaluate(ds, metrics=[answer_relevancy, faithfulness])
    # result may be Dataset or dict-like
    try:
        summary = {k: float(v) for k, v in dict(result).items()}
    except Exception:
        summary = {"raw": str(result)}
    return summary


def ensure_index(pipeline: RAGPipeline, rebuild: bool) -> None:
    storage = settings.storage_path
    if (storage / "faiss" / "faiss_meta.json").exists() and not rebuild:
        pipeline.load(storage)
        return

    from core.document_parser import parse_directory

    documents = []
    for d in (settings.sample_path, settings.uploads_path):
        if d.exists():
            documents.extend(parse_directory(d))
    if documents:
        pipeline.ingest_documents(documents, rebuild=True)
        pipeline.save(storage)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Minimal RAG")
    parser.add_argument(
        "--testset",
        type=str,
        default=str(ROOT / "eval" / "sample_testset.json"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--skip-ragas", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    testset = json.loads(Path(args.testset).read_text(encoding="utf-8"))
    pipeline = RAGPipeline()
    ensure_index(pipeline, rebuild=args.rebuild)
    if pipeline.stats["num_chunks"] == 0:
        print("Empty index. Add documents under data/sample first.")
        sys.exit(1)

    print("=== Retrieval metrics ===")
    retrieval = evaluate_retrieval(pipeline, testset, k=args.k)
    print(f"Hit Rate:     {retrieval['hit_rate']:.4f}")
    print(f"Recall@{args.k}:  {retrieval[f'recall@{args.k}']:.4f}")

    gen_metrics = None
    if not args.skip_ragas:
        print("\n=== Generation metrics (Ragas) ===")
        gen_metrics = evaluate_generation_ragas(pipeline, testset)
        print(json.dumps(gen_metrics, ensure_ascii=False, indent=2))
    else:
        print("\n(Skipped Ragas generation metrics)")

    out = {
        "retrieval": {
            "hit_rate": retrieval["hit_rate"],
            f"recall@{args.k}": retrieval[f"recall@{args.k}"],
            "num_cases": retrieval["num_cases"],
        },
        "generation": gen_metrics,
    }
    out_path = settings.storage_path / "eval_report.json"
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
