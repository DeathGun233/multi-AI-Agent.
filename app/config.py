from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "FlowPilot"
    database_path: str = "flowpilot.db"
    model_name: str = "qwen3-max"
    model_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str | None = None
    disable_llm: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_AI_KEY")
        )
        disable_llm = bool(os.getenv("FLOWPILOT_DISABLE_LLM")) or bool(os.getenv("PYTEST_CURRENT_TEST"))
        raw_db = os.getenv("DATABASE_URL", "sqlite:///flowpilot.db")
        database_path = cls._normalize_database_path(raw_db)
        return cls(
            database_path=database_path,
            model_name=os.getenv("MODEL_NAME", "qwen3-max"),
            model_base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=api_key,
            disable_llm=disable_llm,
        )

    @staticmethod
    def _normalize_database_path(raw_value: str) -> str:
        if raw_value.startswith("sqlite:///"):
            return raw_value.replace("sqlite:///", "", 1)
        return raw_value

    @property
    def database_file(self) -> Path:
        return Path(self.database_path).resolve()

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key) and not self.disable_llm
