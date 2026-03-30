from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.cache import CacheStore
from app.config import Settings
from app.db import Database
from app.models import WorkflowRequest
from app.repository import WorkflowRepository
from app.services import WorkflowEngine


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
settings = Settings.from_env()
database = Database(settings.database_url)
cache = CacheStore(settings.redis_url)

app = FastAPI(title="FlowPilot", version="0.1.0")
repository = WorkflowRepository(database, cache)
engine = WorkflowEngine(repository, settings)


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    sample_templates = engine.list_templates()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "templates": sample_templates,
        },
    )


@app.get("/api/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "database_backend": settings.database_backend,
        "database_path": str(settings.database_file) if settings.database_backend == "sqlite" else settings.database_url,
        "redis_enabled": cache.enabled,
        "redis_url": settings.redis_url or "",
        "llm_enabled": settings.llm_enabled,
        "model_name": settings.model_name,
    }


@app.get("/api/workflows/templates")
def list_templates() -> list[dict]:
    return engine.list_templates()


@app.get("/api/workflows")
def list_runs() -> list[dict]:
    return [run.model_dump(mode="json") for run in engine.list_runs()]


@app.get("/api/workflows/{run_id}")
def get_run(run_id: str) -> dict:
    run = engine.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="workflow run not found")
    return run.model_dump(mode="json")


@app.post("/api/workflows/run")
def run_workflow(request: WorkflowRequest) -> dict:
    run = engine.run_workflow(request)
    return run.model_dump(mode="json")
