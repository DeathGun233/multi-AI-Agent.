from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from app.config import Settings


class LLMService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: OpenAI | None = None
        if settings.llm_enabled:
            self._client = OpenAI(api_key=settings.api_key, base_url=settings.model_base_url)

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._client:
            return fallback
        try:
            response = self._client.chat.completions.create(
                model=self.settings.model_name,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content or ""
            return self._extract_json(content) or fallback
        except Exception:
            return fallback

    @staticmethod
    def _extract_json(content: str) -> dict[str, Any] | None:
        text = content.strip()
        if text.startswith("```"):
            parts = text.split("```")
            for part in parts:
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return None
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None
