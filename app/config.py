from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str = "FlowPilot"
    database_url: str = "sqlite:///flowpilot.db"
    redis_url: str | None = None
    model_name: str = "qwen3-max"
    model_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str | None = None
    github_token: str | None = None
    disable_llm: bool = False
    monthly_budget_usd: float = 25.0
    http_timeout_seconds: float = 12.0
    secret_key: str = "flowpilot-demo-secret"
    session_cookie_name: str = "flowpilot_session"
    users_json: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = (
            os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPEN_AI_KEY")
        )
        pytest_mode = bool(os.getenv("PYTEST_CURRENT_TEST")) or ("pytest" in sys.modules)
        disable_llm = (
            bool(os.getenv("FLOWPILOT_DISABLE_LLM"))
            or pytest_mode
        )
        default_database_url = "sqlite:///flowpilot_test.db" if pytest_mode else "sqlite:///flowpilot.db"
        return cls(
            database_url=os.getenv("DATABASE_URL", default_database_url),
            redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
            model_name=os.getenv("MODEL_NAME", "qwen3-max"),
            model_base_url=os.getenv("MODEL_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=api_key,
            github_token=os.getenv("GITHUB_TOKEN"),
            disable_llm=disable_llm,
            monthly_budget_usd=float(os.getenv("FLOWPILOT_MONTHLY_BUDGET_USD", "25")),
            http_timeout_seconds=float(os.getenv("FLOWPILOT_HTTP_TIMEOUT_SECONDS", "12")),
            secret_key=os.getenv("FLOWPILOT_SECRET_KEY", "flowpilot-demo-secret"),
            session_cookie_name=os.getenv("FLOWPILOT_SESSION_COOKIE", "flowpilot_session"),
            users_json=os.getenv("FLOWPILOT_USERS_JSON"),
        )

    @property
    def database_file(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", "", 1)).resolve()
        return Path("non-sqlite-database")

    @property
    def database_backend(self) -> str:
        if self.database_url.startswith("mysql"):
            return "mysql"
        if self.database_url.startswith("sqlite"):
            return "sqlite"
        return "custom"

    @property
    def llm_enabled(self) -> bool:
        return bool(self.api_key) and not self.disable_llm
