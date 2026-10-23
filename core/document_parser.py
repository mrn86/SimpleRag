"""Document parser: txt / markdown / pdf (pypdfium2). No OCR."""

from __future__ import annotations

import re
from pathlib import Path

from core.models import Document


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def parse_file(path: str | Path) -> Document:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}. Supported: {SUPPORTED_EXTENSIONS}")

    if suffix == ".pdf":
        text = _parse_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="ignore")

    text = _normalize_text(text)
    title = _guess_title(text, path.stem)
    return Document(
        source=path.name,
        text=text,
        title=title,
        metadata={"path": str(path.resolve()), "suffix": suffix},
    )


def parse_directory(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    if not directory.exists():
        return []

    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append(parse_file(path))
    return docs


def _parse_pdf(path: Path) -> str:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    pages: list[str] = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            page_text = textpage.get_text_bounded()
            textpage.close()
            page.close()
            # Heuristic: promote short ALL-CAPS / title-like lines to markdown headers
            pages.append(_promote_pdf_headings(page_text))
    finally:
        pdf.close()
    return "\n\n".join(pages)


def _promote_pdf_headings(text: str) -> str:
    """Light heuristic to keep heading structure from PDFs without OCR."""
    lines = text.splitlines()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        # Short line that looks like a title → markdown H2
        if (
            len(stripped) <= 40
            and not stripped.endswith((".", "。", "，", ",", ";", "；"))
            and (stripped.isupper() or _looks_like_heading(stripped))
        ):
            out.append(f"## {stripped}")
        else:
            out.append(stripped)
    return "\n".join(out)


def _looks_like_heading(line: str) -> bool:
    # Numbered headings: 1. xxx / 1.1 xxx / 一、xxx
    if re.match(r"^(\d+(\.\d+)*|[一二三四五六七八九十]+)[\.、]\s*\S+", line):
        return True
    # Chapter-like Chinese headings
    if re.match(r"^(第[一二三四五六七八九十百千\d]+[章节篇部]|Chapter\s+\d+)", line, re.I):
        return True
    return False


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guess_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            return re.sub(r"^#+\s*", "", line).strip() or fallback
        if line:
            return line[:80]
    return fallback
