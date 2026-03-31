from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.auth import (
    ROLE_ADMIN,
    ROLE_OPERATOR,
    ROLE_REVIEWER,
    AuthService,
    AuthUser,
)
from app.cache import CacheStore
from app.config import Settings
from app.db import Database
from app.models import ReviewSubmission, WorkflowRequest, WorkflowRun, WorkflowType
from app.repository import WorkflowRepository
from app.services import WorkflowEngine


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
settings = Settings.from_env()
database = Database(settings.database_url)
cache = CacheStore(settings.redis_url)

app = FastAPI(title="FlowPilot", version="0.2.0")
repository = WorkflowRepository(database, cache)
engine = WorkflowEngine(repository, settings)
auth_service = AuthService(settings)


def _safe_next_path(next_path: str | None) -> str:
    if next_path and next_path.startswith("/"):
        return next_path
    return "/dashboard"


def _redirect_to_login(request: Request) -> RedirectResponse:
    next_path = request.url.path
    if request.url.query:
        next_path = f"{next_path}?{request.url.query}"
    return RedirectResponse(url=f"/login?next={quote(next_path)}", status_code=303)


def _page_user(request: Request, roles: set[str] | None = None) -> AuthUser | None:
    user = auth_service.get_user_from_request(request)
    if user is None:
        return None
    if roles and user.role not in roles:
        raise HTTPException(status_code=403, detail="permission denied")
    return user


def _template_response(request: Request, template_name: str, context: dict, status_code: int = 200) -> HTMLResponse:
    return templates.TemplateResponse(request, template_name, context, status_code=status_code)


def _workflow_titles() -> dict[str, str]:
    return {item["workflow_type"]: item["title"] for item in engine.list_templates()}


def _summarize_runs(runs: list[WorkflowRun]) -> dict[str, int]:
    return {
        "total_runs": len(runs),
        "completed_runs": sum(1 for run in runs if run.status.value == "completed"),
        "waiting_review": sum(1 for run in runs if run.status.value == "waiting_human"),
        "failed_runs": sum(1 for run in runs if run.status.value == "failed"),
    }


def _build_timeline(run: WorkflowRun) -> list[dict]:
    if not run.logs:
        return []
    total_seconds = max((run.updated_at - run.created_at).total_seconds(), 1)
    points: list[dict] = []
    count = len(run.logs)
    for index, log in enumerate(run.logs):
        offset_seconds = max((log.timestamp - run.created_at).total_seconds(), 0)
        left = 50 if count == 1 else round((offset_seconds / total_seconds) * 100, 2)
        next_timestamp = run.logs[index + 1].timestamp if index + 1 < count else run.updated_at
        duration_seconds = max((next_timestamp - log.timestamp).total_seconds(), 0)
        points.append(
            {
                "index": index + 1,
                "agent": log.agent,
                "message": log.message,
                "timestamp": log.timestamp.astimezone().strftime("%H:%M:%S"),
                "left": min(max(left, 2), 98),
                "duration": f"{duration_seconds:.2f}s",
                "tool_name": log.tool_call.name if log.tool_call else "",
            }
        )
    return points


def _build_run_table(runs: list[WorkflowRun]) -> list[dict]:
    titles = _workflow_titles()
    return [
        {
            "id": run.id,
            "title": titles.get(run.workflow_type.value, run.workflow_type.value),
            "workflow_type": run.workflow_type.value,
            "status": run.status.value,
            "current_step": run.current_step,
            "objective": run.objective,
            "review_score": f"{run.review.score:.2f}" if run.review else "--",
            "updated_at": run.updated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "created_at": run.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        }
        for run in runs
    ]


def _common_context(request: Request, user: AuthUser, active_page: str, **kwargs: object) -> dict[str, object]:
    return {
        "request": request,
        "current_user": user,
        "capabilities": auth_service.capabilities_for(user),
        "active_page": active_page,
        **kwargs,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> RedirectResponse:
    if auth_service.get_user_from_request(request) is None:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/dashboard"):
    user = auth_service.get_user_from_request(request)
    if user is not None:
        return RedirectResponse(url=_safe_next_path(next), status_code=303)
    return _template_response(
        request,
        "login.html",
        {
            "request": request,
            "next": _safe_next_path(next),
            "error": "",
        },
    )


@app.post("/login", response_class=HTMLResponse)
def login_action(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    next: str = Form("/dashboard"),
):
    user = auth_service.authenticate(username, password)
    safe_next = _safe_next_path(next)
    if user is None:
        return _template_response(
            request,
            "login.html",
            {
                "request": request,
                "next": safe_next,
                "error": "用户名或密码错误，请重试。",
            },
            status_code=401,
        )
    response = RedirectResponse(url=safe_next, status_code=303)
    response.set_cookie(
        settings.session_cookie_name,
        auth_service.build_session_cookie(user),
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 8,
    )
    return response


@app.post("/logout")
def logout() -> RedirectResponse:
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(settings.session_cookie_name)
    return response


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    runs = engine.list_runs()
    queue = engine.list_review_queue() if user.role in {ROLE_REVIEWER, ROLE_ADMIN} else []
    return _template_response(
        request,
        "dashboard.html",
        _common_context(
            request,
            user,
            "dashboard",
            templates_data=engine.list_templates(),
            recent_runs=_build_run_table(runs[:6]),
            summary=_summarize_runs(runs),
            review_queue=_build_run_table(queue[:5]),
            graph=engine.get_graph_definition(),
        ),
    )


@app.post("/dashboard/run")
def dashboard_run_action(
    request: Request,
    workflow_type: str = Form(...),
    payload_json: str = Form("{}"),
) -> RedirectResponse:
    user = _page_user(request, {ROLE_OPERATOR, ROLE_ADMIN})
    if user is None:
        return _redirect_to_login(request)
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"payload_json is invalid JSON: {exc}") from exc
    run = engine.run_workflow(WorkflowRequest(workflow_type=WorkflowType(workflow_type), input_payload=payload))
    return RedirectResponse(url=f"/runs/{run.id}", status_code=303)


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    runs = engine.list_runs()
    return _template_response(
        request,
        "runs.html",
        _common_context(
            request,
            user,
            "runs",
            runs=_build_run_table(runs),
            summary=_summarize_runs(runs),
        ),
    )


@app.get("/reviews", response_class=HTMLResponse)
def reviews_page(request: Request):
    user = _page_user(request, {ROLE_REVIEWER, ROLE_ADMIN})
    if user is None:
        return _redirect_to_login(request)
    queue = engine.list_review_queue()
    return _template_response(
        request,
        "reviews.html",
        _common_context(
            request,
            user,
            "reviews",
            queue=_build_run_table(queue),
        ),
    )


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail_page(run_id: str, request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    run = engine.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="workflow run not found")
    return _template_response(
        request,
        "run_detail.html",
        _common_context(
            request,
            user,
            "runs",
            run=run,
            run_view={
                "title": _workflow_titles().get(run.workflow_type.value, run.workflow_type.value),
                "timeline": _build_timeline(run),
                "updated_at": run.updated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                "created_at": run.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            },
            graph=engine.get_graph_definition(),
        ),
    )


@app.post("/runs/{run_id}/review/form")
def review_form_action(
    run_id: str,
    request: Request,
    decision: str = Form(...),
    comment: str = Form(""),
    next_page: str = Form("detail"),
) -> RedirectResponse:
    user = _page_user(request, {ROLE_REVIEWER, ROLE_ADMIN})
    if user is None:
        return _redirect_to_login(request)
    run = engine.submit_review(
        run_id,
        approve=decision == "approve",
        comment=comment,
        reviewer_name=user.display_name,
    )
    if run is None:
        raise HTTPException(status_code=404, detail="workflow run not found")
    target = "/reviews" if next_page == "reviews" else f"/runs/{run_id}"
    return RedirectResponse(url=target, status_code=303)


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


@app.get("/api/session")
def session_info(request: Request) -> dict[str, object]:
    user = auth_service.require_user(request)
    return {
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "capabilities": auth_service.capabilities_for(user).__dict__,
    }


@app.get("/api/workflows/templates")
def list_templates(request: Request) -> list[dict]:
    auth_service.require_user(request)
    return engine.list_templates()


@app.get("/api/workflows")
def list_runs(request: Request) -> list[dict]:
    auth_service.require_user(request)
    return [run.model_dump(mode="json") for run in engine.list_runs()]


@app.get("/api/workflows/review-queue")
def get_review_queue(request: Request) -> list[dict]:
    auth_service.require_roles(request, ROLE_REVIEWER, ROLE_ADMIN)
    return [run.model_dump(mode="json") for run in engine.list_review_queue()]


@app.get("/api/workflows/graph")
def get_workflow_graph(request: Request) -> dict:
    auth_service.require_user(request)
    return engine.get_graph_definition()


@app.get("/api/workflows/{run_id}")
def get_run(run_id: str, request: Request) -> dict:
    auth_service.require_user(request)
    run = engine.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="workflow run not found")
    return run.model_dump(mode="json")


@app.post("/api/workflows/run")
def run_workflow(request: Request, body: WorkflowRequest) -> dict:
    auth_service.require_roles(request, ROLE_OPERATOR, ROLE_ADMIN)
    run = engine.run_workflow(body)
    return run.model_dump(mode="json")


@app.post("/api/workflows/{run_id}/review")
def submit_review(run_id: str, request: Request, submission: ReviewSubmission) -> dict:
    user = auth_service.require_roles(request, ROLE_REVIEWER, ROLE_ADMIN)
    run = engine.submit_review(run_id, submission.approve, submission.comment, reviewer_name=user.display_name)
    if run is None:
        raise HTTPException(status_code=404, detail="workflow run not found")
    return run.model_dump(mode="json")
