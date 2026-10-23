"""Apply HuggingFace Hub token from settings into process env."""

from __future__ import annotations

import os

from config.settings import settings


def ensure_hf_token() -> None:
    token = (settings.hf_token or "").strip()
    if not token:
        return
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
