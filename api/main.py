"""FastAPI: upload documents & ask questions."""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from core.document_parser import SUPPORTED_EXTENSIONS
from core.pipeline import RAGPipeline

app = FastAPI(title="Minimal RAG", version="0.1.0")

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
        storage = settings.storage_path
        if (storage / "faiss" / "faiss_meta.json").exists():
            _pipeline.load(storage)
        else:
            # Bootstrap from sample + uploads if any
            from core.document_parser import parse_directory

            settings.sample_path.mkdir(parents=True, exist_ok=True)
            settings.uploads_path.mkdir(parents=True, exist_ok=True)
            documents = []
            for d in (settings.sample_path, settings.uploads_path):
                if d.exists():
                    documents.extend(parse_directory(d))
            if documents:
                _pipeline.ingest_documents(documents, rebuild=True)
                _pipeline.save(storage)
    return _pipeline


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    rewrite: bool = True


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]
    rewritten_queries: list[str] = []
    keywords: list[str] = []


@app.on_event("startup")
def _startup() -> None:
    settings.data_path.mkdir(parents=True, exist_ok=True)
    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    settings.storage_path.mkdir(parents=True, exist_ok=True)
    # Lazy-load on first request to avoid long startup if models download
    # get_pipeline()  # uncomment to eager-load


@app.get("/health")
def health() -> dict:
    pipe = get_pipeline()
    return {"status": "ok", **pipe.stats}


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported type {suffix}. Allowed: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    settings.uploads_path.mkdir(parents=True, exist_ok=True)
    dest = settings.uploads_path / (file.filename or f"upload{suffix}")
    content = await file.read()
    dest.write_bytes(content)

    pipe = get_pipeline()
    # Re-ingest all uploads + sample (simple full rebuild for local demo)
    from core.document_parser import parse_directory

    documents = []
    for d in (settings.sample_path, settings.uploads_path):
        if d.exists():
            documents.extend(parse_directory(d))
    pipe.ingest_documents(documents, rebuild=True)
    pipe.save(settings.storage_path)

    return {
        "filename": dest.name,
        "saved_to": str(dest),
        "stats": pipe.stats,
    }


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    pipe = get_pipeline()
    if pipe.stats["num_chunks"] == 0:
        raise HTTPException(status_code=400, detail="Index is empty. Upload documents first.")
    try:
        result = pipe.ask(body.query, rewrite=body.rewrite)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    return AskResponse(
        answer=result.answer,
        sources=result.sources,
        rewritten_queries=result.rewritten_queries,
        keywords=result.keywords,
    )


# Run: uvicorn api.main:app --reload --port 8000
