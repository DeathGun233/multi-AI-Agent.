from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from app.auth import ROLE_ADMIN, ROLE_OPERATOR, ROLE_REVIEWER, AuthService, AuthUser
from app.cache import CacheStore
from app.config import Settings
from app.db import Database
from app.models import (
    BatchExperimentRequest,
    BulkDeleteRequest,
    PromptProfileForm,
    ReviewSubmission,
    WorkflowRequest,
    WorkflowRun,
)
from app.reporting import (
    build_evaluation_html,
    build_evaluation_markdown,
    build_evaluation_pdf,
    build_workflow_html,
    build_workflow_markdown,
    build_workflow_pdf,
)
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
    points: list[dict] = []
    for index, log in enumerate(run.logs):
        next_timestamp = run.logs[index + 1].timestamp if index + 1 < len(run.logs) else run.updated_at
        duration_seconds = max((next_timestamp - log.timestamp).total_seconds(), 0)
        points.append(
            {
                "index": index + 1,
                "agent": log.agent,
                "message": log.message,
                "timestamp": log.timestamp.astimezone().strftime("%H:%M:%S"),
                "duration": f"{duration_seconds:.2f}s",
                "tool_name": log.tool_call.name if log.tool_call else "",
            }
        )
    return points


def _build_runtime_memory_sections(run: WorkflowRun) -> list[dict[str, object]]:
    if not isinstance(run.result, dict):
        return []

    sections: list[dict[str, object]] = []
    for key, agent_label in (
        ("analyst_context", "AnalystAgent"),
        ("content_context", "ContentAgent"),
        ("reviewer_context", "ReviewerAgent"),
    ):
        context = run.result.get(key)
        if not isinstance(context, dict):
            continue
        memory = context.get("memory", {})
        if not isinstance(memory, dict):
            memory = {}

        recent_runs = []
        for item in memory.get("recent_runs", [])[:3]:
            if not isinstance(item, dict):
                continue
            recent_runs.append(
                {
                    "title": item.get("objective", "") or item.get("run_id", ""),
                    "summary": item.get("summary", ""),
                    "meta": " / ".join(
                        str(value)
                        for value in (item.get("status"), item.get("updated_at"))
                        if str(value).strip()
                    ),
                    "tags": [str(value) for value in item.get("review_reasons", []) if str(value).strip()][:3],
                }
            )

        feedback_samples = []
        for item in memory.get("feedback_samples", [])[:3]:
            if not isinstance(item, dict):
                continue
            feedback_samples.append(
                {
                    "reviewer_name": item.get("reviewer_name", ""),
                    "comment": item.get("comment", ""),
                    "meta": " / ".join(
                        str(value)
                        for value in (item.get("expected_status"), item.get("created_at"))
                        if str(value).strip()
                    ),
                    "keywords": [str(value) for value in item.get("keywords", []) if str(value).strip()][:5],
                }
            )

        sections.append(
            {
                "agent": agent_label,
                "memory_hits": int(context.get("memory_hits", 0) or 0),
                "dominant_keywords": [str(value) for value in memory.get("dominant_keywords", []) if str(value).strip()][:6],
                "common_review_reasons": [
                    str(value) for value in memory.get("common_review_reasons", []) if str(value).strip()
                ][:4],
                "recent_runs": recent_runs,
                "feedback_samples": feedback_samples,
            }
        )
    return sections


def _build_route_trace_sections(run: WorkflowRun) -> dict[str, object]:
    decisions = run.result.get("route_decisions", []) if isinstance(run.result, dict) else []
    if not isinstance(decisions, list):
        decisions = []

    rows: list[dict[str, object]] = []
    fallback_count = 0
    cumulative_replan_count = 0
    for index, item in enumerate(decisions, start=1):
        if not isinstance(item, dict):
            continue

        to_node = str(item.get("final_route") or item.get("next_node") or item.get("route") or "")
        confidence = item.get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_display = f"{float(confidence):.2f}".rstrip("0").rstrip(".")
        else:
            confidence_display = "--"

        used_fallback = bool(item.get("used_fallback", False))
        if used_fallback:
            fallback_count += 1

        try:
            item_replan_count = int(item.get("replan_count", 0) or 0)
        except (TypeError, ValueError):
            item_replan_count = 0
        cumulative_replan_count = max(cumulative_replan_count, item_replan_count)
        is_replan = to_node == "planner" or item_replan_count > 0

        model_route = item.get("model_route")
        rows.append(
            {
                "step": index,
                "from_node": str(item.get("from_node") or ""),
                "to_node": to_node,
                "source": str(item.get("decision_source") or "rule").upper(),
                "model_route": str(model_route) if model_route else "--",
                "confidence": confidence_display,
                "used_fallback": used_fallback,
                "fallback_reason": str(item.get("fallback_reason") or ""),
                "reason": str(item.get("reason") or ""),
                "is_replan": is_replan,
                "replan_count": item_replan_count,
            }
        )

    inferred_replan_count = sum(1 for row in rows if row["is_replan"])
    replan_count = cumulative_replan_count or inferred_replan_count
    return {
        "rows": rows,
        "total_steps": len(rows),
        "fallback_count": fallback_count,
        "replan_count": replan_count,
        "has_replan": replan_count > 0,
    }


def _build_operator_context_section(run: WorkflowRun) -> dict[str, object]:
    if not isinstance(run.result, dict):
        return {"present": False}

    context = run.result.get("operator_context", {})
    if not isinstance(context, dict) or not context:
        return {"present": False}

    confidence = context.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_display = f"{float(confidence):.2f}".rstrip("0").rstrip(".")
    else:
        confidence_display = "--"

    tool_choices = context.get("tool_choices", [])
    if not isinstance(tool_choices, list):
        tool_choices = []

    return {
        "present": True,
        "source": str(context.get("decision_source") or "rule").upper(),
        "selected_tool": str(context.get("selected_tool") or "--"),
        "executed_tool": str(context.get("executed_tool") or "--"),
        "used_fallback": bool(context.get("used_fallback", False)),
        "fallback_reason": str(context.get("fallback_reason") or ""),
        "decision_reason": str(context.get("decision_reason") or ""),
        "confidence": confidence_display,
        "tool_choices": [str(item) for item in tool_choices if str(item).strip()],
    }


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
                "workflow_type": run.workflow_type.value,
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


def _build_evaluation_rows(evaluations: list[object]) -> list[dict[str, object]]:
    dimension_labels = {
        "status_match": "状态匹配",
        "keyword_coverage": "关键词覆盖",
        "result_completeness": "结果完整度",
        "review_alignment": "审核一致性",
    }
    status_labels = {
        "completed": "已完成",
        "waiting_human": "待人工审核",
        "failed": "已失败",
    }
    rows: list[dict[str, object]] = []
    for item in evaluations:
        candidate_dimensions = item.summary.get("candidate_dimensions", {})
        baseline_dimensions = item.summary.get("baseline_dimensions", {})
        dimension_rows = []
        for key in dict.fromkeys([*candidate_dimensions.keys(), *baseline_dimensions.keys()]):
            candidate_score = round(float(candidate_dimensions.get(key, 0.0)) * 100, 1)
            baseline_score = round(float(baseline_dimensions.get(key, 0.0)) * 100, 1)
            dimension_rows.append(
                {
                    "key": key,
                    "label": dimension_labels.get(key, key),
                    "candidate_score": candidate_score,
                    "baseline_score": baseline_score,
                    "delta": round(candidate_score - baseline_score, 1),
                }
            )
        case_rows = []
        for case in item.case_results:
            case_dimension_rows = []
            for key in dict.fromkeys([*case.get("candidate_dimensions", {}).keys(), *case.get("baseline_dimensions", {}).keys()]):
                case_dimension_rows.append(
                    {
                        "label": dimension_labels.get(key, key),
                        "candidate_score": round(float(case.get("candidate_dimensions", {}).get(key, 0.0)) * 100, 1),
                        "baseline_score": round(float(case.get("baseline_dimensions", {}).get(key, 0.0)) * 100, 1),
                    }
                )
            case_rows.append(
                {
                    "case_id": case["case_id"],
                    "title": case["title"],
                    "candidate_status_label": status_labels.get(case.get("candidate_status", "completed"), case.get("candidate_status", "completed")),
                    "baseline_status_label": status_labels.get(case.get("baseline_status", "completed"), case.get("baseline_status", "completed")),
                    "candidate_score": round(float(case.get("candidate_score", 0.0)) * 100, 1),
                    "baseline_score": round(float(case.get("baseline_score", 0.0)) * 100, 1),
                    "candidate_metrics": case.get("candidate_metrics", {}),
                    "baseline_metrics": case.get("baseline_metrics", {}),
                    "dimension_rows": case_dimension_rows,
                }
            )
        rows.append(
            {
                "id": item.id,
                "dataset_name": item.dataset_name,
                "candidate_label": f"{item.candidate_profile.primary_model_name} / {item.candidate_profile.prompt_profile.name}",
                "baseline_label": f"{item.baseline_profile.primary_model_name} / {item.baseline_profile.prompt_profile.name}",
                "candidate_avg_score": round(float(item.summary.get("candidate_avg_score", 0.0)) * 100, 1),
                "baseline_avg_score": round(float(item.summary.get("baseline_avg_score", 0.0)) * 100, 1),
                "score_delta": round(float(item.summary.get("score_delta", 0.0)) * 100, 1),
                "case_count": item.summary.get("case_count", len(item.case_results)),
                "dimension_rows": dimension_rows,
                "case_rows": case_rows,
                "created_at": item.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                "sort_key": item.created_at,
            }
        )
    rows.sort(key=lambda row: row["sort_key"], reverse=True)
    return rows


def _build_evaluation_trend_rows(evaluations: list[dict[str, object]]) -> list[dict[str, object]]:
    trend_rows = []
    history = list(reversed(evaluations[:8]))
    max_score = max(
        [max(float(item["candidate_avg_score"]), float(item["baseline_avg_score"])) for item in history],
        default=100.0,
    )
    scale = max(max_score, 100.0)
    for index, item in enumerate(history, start=1):
        trend_rows.append(
            {
                "label": f"第 {index} 次 / {item['dataset_name']}",
                "created_at": item["created_at"],
                "candidate_avg_score": item["candidate_avg_score"],
                "baseline_avg_score": item["baseline_avg_score"],
                "candidate_width": round(float(item["candidate_avg_score"]) / scale * 100, 1),
                "baseline_width": round(float(item["baseline_avg_score"]) / scale * 100, 1),
            }
        )
    return trend_rows


def _export_response(content: str | bytes, *, filename: str, media_type: str) -> Response:
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
            external_source_options=[
                {"provider": "github_issues", "label": "GitHub Issues", "description": "读取公开仓库 Issue 作为真实客服工单"},
                {"provider": "nyc_311", "label": "NYC 311", "description": "读取纽约 311 公共投诉数据作为真实工单"},
                {"provider": "stack_overflow", "label": "Stack Overflow", "description": "读取 Stack Overflow 公开问题作为技术支持工单"},
                {"provider": "hacker_news", "label": "Hacker News", "description": "读取 Hacker News 检索结果作为外部反馈样本"},
            ],
        ),
    )


@app.get("/runs", response_class=HTMLResponse)
def runs_page(request: Request, status_filter: str = "all", workflow_filter: str = "all"):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    all_runs = engine.list_runs()
    runs = all_runs
    if status_filter in {"waiting_human", "failed"}:
        runs = [run for run in runs if run.status.value == status_filter]
    if workflow_filter != "all":
        runs = [run for run in runs if run.workflow_type.value == workflow_filter]
    workflow_options = [{"value": "all", "label": "全部工作流"}] + [
        {"value": item.workflow_type.value, "label": item.title} for item in engine.list_templates()
    ]
    return _template_response(
        request,
        "runs.html",
        _common_context(
            request,
            user,
            "runs",
            runs=_build_run_rows(runs),
            status_filter=status_filter,
            workflow_filter=workflow_filter,
            workflow_options=workflow_options,
            filtered_count=len(runs),
            total_count=len(all_runs),
        ),
    )


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
            runtime_memory_sections=_build_runtime_memory_sections(run),
            route_trace=_build_route_trace_sections(run),
            operator_context=_build_operator_context_section(run),
            result_json=_pretty_json(run.result),
            graph=engine.graph_shape(),
        ),
    )


@app.get("/runs/{run_id}/export")
def export_run_report(
    run_id: str,
    request: Request,
    format: str = Query("markdown", pattern="^(markdown|html|pdf)$"),
):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    run = engine.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="workflow run not found")
    llm_summary = _build_llm_summary(run)
    result_json = _pretty_json(run.result)
    if format == "html":
        return _export_response(
            build_workflow_html(run, llm_summary, result_json),
            filename=f"workflow-{run.id}.html",
            media_type="text/html; charset=utf-8",
        )
    if format == "pdf":
        return _export_response(
            build_workflow_pdf(run, llm_summary, result_json),
            filename=f"workflow-{run.id}.pdf",
            media_type="application/pdf",
        )
    return _export_response(
        build_workflow_markdown(run, llm_summary, result_json),
        filename=f"workflow-{run.id}.md",
        media_type="text/markdown; charset=utf-8",
    )


@app.post("/runs/{run_id}/delete")
def delete_run_form(run_id: str, request: Request):
    user = _page_user(request, {ROLE_OPERATOR, ROLE_REVIEWER, ROLE_ADMIN})
    if user is None:
        return _redirect_to_login(request)
    deleted = engine.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="workflow run not found")
    return RedirectResponse(url="/runs", status_code=303)


@app.post("/runs/bulk-delete")
def bulk_delete_runs_form(request: Request, run_ids: list[str] = Form(default_factory=list)):
    user = _page_user(request, {ROLE_OPERATOR, ROLE_REVIEWER, ROLE_ADMIN})
    if user is None:
        return _redirect_to_login(request)
    engine.delete_runs(run_ids)
    return RedirectResponse(url="/runs", status_code=303)


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
    evaluation_rows = _build_evaluation_rows(evaluation_service.list_runs())
    return _template_response(
        request,
        "evaluations.html",
        _common_context(
            request,
            user,
            "evaluations",
            datasets=evaluation_service.list_datasets(),
            evaluations=evaluation_rows,
            latest_evaluation=evaluation_rows[0] if evaluation_rows else None,
            trend_rows=_build_evaluation_trend_rows(evaluation_rows),
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


@app.get("/evaluations/{evaluation_id}/export")
def export_evaluation_report(
    evaluation_id: str,
    request: Request,
    format: str = Query("markdown", pattern="^(markdown|html|pdf)$"),
):
    user = _page_user(request)
    if user is None:
        return _redirect_to_login(request)
    evaluation = repository.get_evaluation(evaluation_id)
    if evaluation is None:
        raise HTTPException(status_code=404, detail="evaluation not found")
    row = _build_evaluation_rows([evaluation])[0]
    if format == "html":
        return _export_response(
            build_evaluation_html(row),
            filename=f"evaluation-{evaluation.id}.html",
            media_type="text/html; charset=utf-8",
        )
    if format == "pdf":
        return _export_response(
            build_evaluation_pdf(row),
            filename=f"evaluation-{evaluation.id}.pdf",
            media_type="application/pdf",
        )
    return _export_response(
        build_evaluation_markdown(row),
        filename=f"evaluation-{evaluation.id}.md",
        media_type="text/markdown; charset=utf-8",
    )


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


@app.delete("/api/workflows/{run_id}")
def delete_workflow(run_id: str, request: Request):
    auth_service.require_roles(request, ROLE_OPERATOR, ROLE_REVIEWER, ROLE_ADMIN)
    deleted = engine.delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="workflow run not found")
    return {"ok": True, "run_id": run_id}


@app.post("/api/workflows/bulk-delete")
def bulk_delete_workflows(request: Request, payload: BulkDeleteRequest):
    auth_service.require_roles(request, ROLE_OPERATOR, ROLE_REVIEWER, ROLE_ADMIN)
    deleted_ids = engine.delete_runs(payload.run_ids)
    return {"ok": True, "deleted_count": len(deleted_ids), "deleted_run_ids": deleted_ids}


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
