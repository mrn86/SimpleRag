"""CLI demo: build index from data/ then interactive Q&A."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root on sys.path when run as script
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from core.pipeline import RAGPipeline


def _index_ready(storage: Path) -> bool:
    faiss_meta = storage / "faiss" / "faiss_meta.json"
    db_path = RAGPipeline._bm25_db_path(storage)
    return faiss_meta.exists() and db_path.exists()


def build_index(pipeline: RAGPipeline, data_dirs: list[Path], force: bool = False) -> None:
    storage = settings.storage_path

    if _index_ready(storage) and not force:
        print(f"Loading existing index from {storage} (SQLite BM25 + FAISS) ...")
        pipeline.load(storage)
        print(f"Loaded: {pipeline.stats}")
        return

    docs_found = False
    all_dirs = data_dirs
    for d in all_dirs:
        if d.exists() and any(d.rglob("*")):
            docs_found = True
            break

    if not docs_found:
        print("No documents found. Put .txt/.md/.pdf under data/sample or data/uploads.")
        sys.exit(1)

    print("Building hybrid index (BM25 + FAISS) ...")
    from core.document_parser import parse_directory

    documents = []
    for d in all_dirs:
        if d.exists():
            print(f"  ingesting {d} ...")
            documents.extend(parse_directory(d))
    pipeline.ingest_documents(documents, rebuild=True)
    pipeline.save(storage)
    print(f"Index saved to {storage}. Stats: {pipeline.stats}")


def interactive_qa(pipeline: RAGPipeline) -> None:
    print("\n=== Minimal RAG Demo ===")
    print("Type a question (or 'quit' / 'exit' to leave).\n")
    while True:
        try:
            q = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break
        try:
            result = pipeline.ask(q)
        except Exception as e:
            print(f"[Error] {e}")
            continue
        print("\nAssistant>")
        print(result.answer)
        if result.rewritten_queries:
            print(f"\n[rewrite] {result.rewritten_queries}")
        if result.keywords:
            print(f"[keywords] {result.keywords}")
        if result.sources:
            print("\n[sources]")
            for s in result.sources:
                preview = s["text"][:80].replace("\n", " ")
                print(f"  [{s['ref']}] {s['source']} (score={s['score']:.4f}) {preview}...")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal RAG CLI demo")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild index even if storage exists",
    )
    parser.add_argument(
        "--once",
        type=str,
        default="",
        help="Ask a single question and exit",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable query rewrite for --once",
    )
    args = parser.parse_args()

    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    settings.storage_path.mkdir(parents=True, exist_ok=True)

    pipeline = RAGPipeline()
    data_dirs = [settings.sample_path, settings.uploads_path]
    build_index(pipeline, data_dirs, force=args.rebuild)

    if args.once:
        result = pipeline.ask(args.once, rewrite=not args.no_rewrite)
        print(result.answer)
        for s in result.sources:
            print(f"- [{s['ref']}] {s['source']}#{s['chunk_id']} score={s['score']:.4f}")
        return

    interactive_qa(pipeline)


if __name__ == "__main__":
    main()
