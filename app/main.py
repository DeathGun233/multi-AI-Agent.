from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.auth import ROLE_ADMIN, ROLE_OPERATOR, ROLE_REVIEWER, AuthService, AuthUser
from app.cache import CacheStore
from app.config import Settings
from app.db import Database
from app.models import BatchExperimentRequest, PromptProfileForm, ReviewSubmission, WorkflowRequest, WorkflowRun
from app.repository import WorkflowRepository
from app.services import BatchExperimentService, CostAnalyticsService, EvaluationService, WorkflowEngine


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
settings = Settings.from_env()


def status_label(value: str) -> str:
    return {
        "created": "已创建",
        "planning": "规划中",
        "executing": "执行中",
        "reviewing": "审核判断中",
        "waiting_human": "待人工审核",
        "completed": "已完成",
        "failed": "已失败",
    }.get(value, value)


def role_label(value: str) -> str:
    return {
        "viewer": "只读查看",
        "operator": "流程操作",
        "reviewer": "审核人员",
        "admin": "管理员",
    }.get(value, value)


def workflow_label(value: str) -> str:
    return {
        "sales_followup": "销售分析与跟进计划",
        "marketing_campaign": "营销内容工厂",
        "support_triage": "客服工单智能分流",
        "meeting_minutes": "会议纪要转执行系统",
    }.get(value, value)


templates.env.globals["settings"] = settings
templates.env.globals["status_label"] = status_label
templates.env.globals["role_label"] = role_label
templates.env.globals["workflow_label"] = workflow_label

database = Database(settings.database_url)
cache = CacheStore(settings.redis_url)

app = FastAPI(title="FlowPilot", version="1.0.0")
repository = WorkflowRepository(database, cache)
engine = WorkflowEngine(repository, settings)
evaluation_service = EvaluationService(repository, engine)
batch_service = BatchExperimentService(repository, engine)
cost_service = CostAnalyticsService(repository, settings)
auth_service = AuthService(settings, database)


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


def _pretty_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _workflow_titles() -> dict[str, str]:
    return {item.workflow_type.value: item.title for item in engine.list_templates()}


def _build_llm_summary(run: WorkflowRun) -> dict[str, object]:
    llm_calls = [log.llm_call for log in run.logs if log.llm_call is not None]
    if not llm_calls:
        return {
            "total_requests": 0,
            "model_names": [],
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "avg_latency_ms": 0,
            "total_cost_usd": 0.0,
            "fallback_requests": 0,
        }
    total_latency_ms = sum(call.latency_ms for call in llm_calls)
    return {
        "total_requests": len(llm_calls),
        "model_names": sorted({call.model_name for call in llm_calls}),
        "prompt_tokens": sum(call.prompt_tokens for call in llm_calls),
        "completion_tokens": sum(call.completion_tokens for call in llm_calls),
        "total_tokens": sum(call.total_tokens for call in llm_calls),
        "total_latency_ms": total_latency_ms,
        "avg_latency_ms": round(total_latency_ms / len(llm_calls), 1),
        "total_cost_usd": round(sum(call.estimated_cost_usd for call in llm_calls), 6),
        "fallback_requests": sum(1 for call in llm_calls if call.used_fallback),
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


def _build_run_rows(runs: list[WorkflowRun]) -> list[dict]:
    titles = _workflow_titles()
    rows = []
    for run in runs:
        execution_profile = run.result.get("execution_profile", {}) if isinstance(run.result, dict) else {}
        prompt_profile = execution_profile.get("prompt_profile", {})
        routing_policy = execution_profile.get("routing_policy", {})
        rows.append(
            {
                "id": run.id,
                "title": titles.get(run.workflow_type.value, workflow_label(run.workflow_type.value)),
                "status": run.status.value,
                "current_step": run.current_step,
                "objective": run.objective,
                "review_score": f"{run.review.score:.2f}" if run.review else "--",
                "updated_at": run.updated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": execution_profile.get("primary_model_name", settings.model_name),
                "prompt_profile_label": f"{prompt_profile.get('name', '未命名')} {prompt_profile.get('version', '')}".strip(),
                "routing_policy_name": routing_policy.get("name", "未配置"),
            }
        )
    return rows


def _compare_summary() -> dict[str, object]:
    rows = []
    grouped: dict[tuple[str, str, str], list[WorkflowRun]] = {}
    for run in engine.list_runs():
        execution_profile = run.result.get("execution_profile", {}) if isinstance(run.result, dict) else {}
        prompt_profile = execution_profile.get("prompt_profile", {})
        key = (
            run.workflow_type.value,
            execution_profile.get("primary_model_name", settings.model_name),
            prompt_profile.get("profile_id", "unknown"),
        )
        grouped.setdefault(key, []).append(run)

    for (workflow_type, model_name, prompt_id), group in grouped.items():
        llm_calls = [log.llm_call for run in group for log in run.logs if log.llm_call]
        prompt_profile = group[0].result.get("execution_profile", {}).get("prompt_profile", {})
        routing_policy = group[0].result.get("execution_profile", {}).get("routing_policy", {})
        rows.append(
            {
                "workflow_type": workflow_label(workflow_type),
                "model_name": model_name,
                "prompt_profile_id": prompt_id,
                "prompt_profile_label": f"{prompt_profile.get('name', '未命名')} {prompt_profile.get('version', '')}".strip(),
                "routing_policy_name": routing_policy.get("name", "未配置"),
                "run_count": len(group),
                "avg_score": round(sum(run.review.score for run in group if run.review) / max(sum(1 for run in group if run.review), 1), 3),
                "avg_latency_ms": round(sum(call.latency_ms for call in llm_calls) / max(len(llm_calls), 1), 1) if llm_calls else 0.0,
                "avg_cost_usd": round(sum(call.estimated_cost_usd for call in llm_calls) / max(len(llm_calls), 1), 6) if llm_calls else 0.0,
                "handoff_rate": round(sum(1 for run in group if run.status.value == "waiting_human") / len(group) * 100, 1),
            }
        )
    rows.sort(key=lambda item: (-item["run_count"], item["workflow_type"], item["model_name"]))
    return {"rows": rows, "run_count": sum(row["run_count"] for row in rows)}


def _common_context(request: Request, user: AuthUser, active_page: str, **kwargs: object) -> dict[str, object]:
    return {
        "request": request,
        "settings": settings,
        "current_user": user,
        "capabilities": auth_service.capabilities_for(user),
        "active_page": active_page,
        **kwargs,
    }


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    if auth_service.get_user_from_request(request) is None:
        return RedirectResponse(url="/login", status_code=303)
    return RedirectResponse(url="/dashboard", status_code=303)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/dashboard"):
    user = auth_service.get_user_from_request(request)
    if user is not None:
        return RedirectResponse(url=_safe_next_path(next), status_code=303)
    return _template_response(request, "login.html", {"request": request, "next": _safe_next_path(next), "error": ""})


@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...), next: str = Form("/dashboard")):
    user = auth_service.authenticate(username, password)
    if user is None:
        return _template_response(
            request,
            "login.html",
            {"request": request, "next": _safe_next_path(next), "error": "用户名或密码错误"},
            status_code=401,
        )
    response = RedirectResponse(url=_safe_next_path(next), status_code=303)
    response.set_cookie(
        key=settings.session_cookie_name,
        value=auth_service.build_session_cookie(user),
        httponly=True,
        samesite="lax",
    )
    return response


@app.post("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(settings.session_cookie_name)
    return response


@app.get("/api/session")
def session_info(request: Request):
    user = auth_service.get_user_from_request(request)
    if user is None:
        raise HTTPException(status_code=401, detail="not authenticated")
    return {
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "capabilities": auth_service.capabilities_for(user).__dict__,
    }


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "database_backend": settings.database_backend,
        "database_file": str(settings.database_file),
        "redis_enabled": cache.enabled,
        "llm_enabled": settings.llm_enabled,
        "model_name": settings.model_name,
        "monthly_budget_usd": settings.monthly_budget_usd,
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    runs = engine.list_runs()[:8]
    return _template_response(
        request,
        "dashboard.html",
        _common_context(
            request,
            user,
            "dashboard",
            templates_catalog=engine.list_templates(),
            recent_runs=_build_run_rows(runs),
            review_queue=_build_run_rows(engine.list_waiting_human()),
            models=engine.list_model_options(),
            prompt_profiles=engine.list_prompt_profiles(),
            routing_policies=engine.list_routing_policies(),
        ),
    )


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    return _template_response(request, "runs.html", _common_context(request, user, "runs", runs=_build_run_rows(engine.list_runs())))


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail_page(run_id: str, request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    run = engine.get(run_id)
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
            timeline=_build_timeline(run),
            llm_summary=_build_llm_summary(run),
            result_json=_pretty_json(run.result),
            graph=engine.graph_shape(),
        ),
    )


@app.get("/reviews", response_class=HTMLResponse)
def reviews_page(request: Request):
    user = _page_user(request, {ROLE_REVIEWER, ROLE_ADMIN})
    if user is None:
        return _redirect_to_login(request)
    return _template_response(
        request,
        "reviews.html",
        _common_context(request, user, "reviews", review_queue=_build_run_rows(engine.list_waiting_human())),
    )


@app.get("/compare", response_class=HTMLResponse)
def compare_page(request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    return _template_response(request, "compare.html", _common_context(request, user, "compare", summary=_compare_summary()))


@app.get("/evaluations", response_class=HTMLResponse)
def evaluations_page(request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    return _template_response(
        request,
        "evaluations.html",
        _common_context(
            request,
            user,
            "evaluations",
            datasets=evaluation_service.list_datasets(),
            evaluations=evaluation_service.list_runs(),
            models=engine.list_model_options(),
            prompt_profiles=engine.list_prompt_profiles(),
            routing_policies=engine.list_routing_policies(),
        ),
    )


@app.post("/evaluations/run")
def evaluation_run_submit(
    request: Request,
    dataset_id: str = Form(...),
    candidate_model_name: str = Form(...),
    candidate_prompt_profile_id: str = Form(...),
    candidate_routing_policy_id: str = Form(...),
    baseline_model_name: str = Form(...),
    baseline_prompt_profile_id: str = Form(...),
    baseline_routing_policy_id: str = Form(...),
):
    user = _page_user(request, {ROLE_OPERATOR, ROLE_ADMIN})
    if user is None:
        return _redirect_to_login(request)
    evaluation_service.run_evaluation(
        dataset_id=dataset_id,
        candidate_model_name=candidate_model_name,
        candidate_prompt_profile_id=candidate_prompt_profile_id,
        candidate_routing_policy_id=candidate_routing_policy_id,
        baseline_model_name=baseline_model_name,
        baseline_prompt_profile_id=baseline_prompt_profile_id,
        baseline_routing_policy_id=baseline_routing_policy_id,
    )
    return RedirectResponse(url="/evaluations", status_code=303)


@app.get("/prompts", response_class=HTMLResponse)
def prompts_page(request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    return _template_response(
        request,
        "prompts.html",
        _common_context(request, user, "prompts", prompt_profiles=engine.list_prompt_profiles(include_inactive=True)),
    )


@app.get("/costs", response_class=HTMLResponse)
def costs_page(request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    return _template_response(
        request,
        "costs.html",
        _common_context(request, user, "costs", summary=cost_service.build_summary()),
    )


@app.get("/batches", response_class=HTMLResponse)
def batches_page(request: Request):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    return _template_response(
        request,
        "batches.html",
        _common_context(
            request,
            user,
            "batches",
            batches=batch_service.list_runs(),
            templates_catalog=engine.list_templates(),
            models=engine.list_model_options(),
            prompt_profiles=engine.list_prompt_profiles(),
            routing_policies=engine.list_routing_policies(),
        ),
    )


@app.get("/api/experiments/catalog")
def experiments_catalog(request: Request):
    auth_service.require_user(request)
    return {
        "models": engine.list_model_options(),
        "prompt_profiles": [item.model_dump(mode="json") for item in engine.list_prompt_profiles()],
        "routing_policies": engine.list_routing_policies(),
        "datasets": evaluation_service.list_datasets(),
    }


@app.get("/api/workflows/graph")
def workflow_graph(request: Request):
    auth_service.require_user(request)
    return engine.graph_shape()


@app.post("/api/workflows/run")
def run_workflow(request: Request, payload: WorkflowRequest):
    auth_service.require_roles(request, ROLE_OPERATOR, ROLE_ADMIN)
    run = engine.run_workflow(payload)
    return run.model_dump(mode="json")


@app.get("/api/workflows")
def list_workflows(request: Request):
    auth_service.require_user(request)
    return [run.model_dump(mode="json") for run in engine.list_runs()]


@app.get("/api/workflows/review-queue")
def review_queue(request: Request):
    auth_service.require_roles(request, ROLE_REVIEWER, ROLE_ADMIN)
    return [run.model_dump(mode="json") for run in engine.list_waiting_human()]


@app.post("/api/workflows/{run_id}/review")
def submit_review(run_id: str, payload: ReviewSubmission, request: Request):
    reviewer = auth_service.require_roles(request, ROLE_REVIEWER, ROLE_ADMIN)
    run = engine.submit_review(run_id, payload, reviewer.display_name)
    return run.model_dump(mode="json")


@app.get("/api/experiments/compare")
def experiments_compare(request: Request):
    auth_service.require_user(request)
    return _compare_summary()


@app.get("/api/evaluations")
def list_evaluations(request: Request):
    auth_service.require_user(request)
    return [item.model_dump(mode="json") for item in evaluation_service.list_runs()]


@app.get("/api/feedback-samples")
def feedback_samples(request: Request):
    auth_service.require_user(request)
    return [item.model_dump(mode="json") for item in engine.feedback_service.list_samples()]


@app.get("/api/costs/summary")
def costs_summary(request: Request):
    auth_service.require_user(request)
    return cost_service.build_summary()


@app.get("/api/batches")
def list_batches(request: Request):
    auth_service.require_user(request)
    return [item.model_dump(mode="json") for item in batch_service.list_runs()]


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: str, request: Request):
    auth_service.require_user(request)
    batch = batch_service.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="batch not found")
    return batch.model_dump(mode="json")


@app.post("/api/batches")
def create_batch(request: Request, payload: BatchExperimentRequest):
    auth_service.require_roles(request, ROLE_OPERATOR, ROLE_ADMIN)
    batch = batch_service.run_batch(payload)
    return batch.model_dump(mode="json")


@app.post("/api/prompts")
def create_prompt(request: Request, payload: PromptProfileForm):
    auth_service.require_roles(request, ROLE_ADMIN)
    profile = engine.create_prompt_profile(payload)
    return profile.model_dump(mode="json")


@app.put("/api/prompts/{profile_id}")
def update_prompt(profile_id: str, request: Request, payload: PromptProfileForm):
    auth_service.require_roles(request, ROLE_ADMIN)
    profile = engine.update_prompt_profile(profile_id, payload)
    return profile.model_dump(mode="json")
