"""LLM generator: OpenAI-compatible chat completions."""

from __future__ import annotations

from openai import OpenAI

from config.settings import settings


class LLMGenerator:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
    ) -> None:
        self.model = model or settings.llm_model
        self.temperature = temperature
        self._client = OpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url or settings.openai_base_url,
        )

    def generate(self, messages: list[dict], temperature: float | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
        )
        return (resp.choices[0].message.content or "").strip()
