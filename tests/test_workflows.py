from uuid import uuid4
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import UserAccountRecord
from app.external_data import ExternalDataService
from app.main import app, database, repository
from app.models import ReviewDecision, ReviewOutput, RunStatus, WorkflowRequest, WorkflowRun, WorkflowType
from app.services import AnalystAgent, OperatorAgent, ReviewerAgent, RouterAgent, ToolCenter


client = TestClient(app)


class FakeRouterLLM:
    def __init__(self, payload: dict | None = None, *, fail: bool = False) -> None:
        self.payload = payload or {}
        self.fail = fail

    def generate_json(self, **_: object) -> SimpleNamespace:
        if self.fail:
            raise RuntimeError("router unavailable")
        return SimpleNamespace(
            payload=self.payload,
            call=SimpleNamespace(used_fallback=False, error=None, validation_error=None),
        )


class FakeOperatorLLM:
    def __init__(self, payload: dict | None = None, *, fail: bool = False) -> None:
        self.payload = payload or {}
        self.fail = fail

    def generate_json(self, **_: object) -> SimpleNamespace:
        if self.fail:
            raise RuntimeError("operator unavailable")
        return SimpleNamespace(
            payload=self.payload,
            call=SimpleNamespace(used_fallback=False, error=None, validation_error=None),
        )


def login_as(username: str, password: str) -> None:
    response = client.post(
        "/login",
        data={"username": username, "password": password, "next": "/dashboard"},
        follow_redirects=False,
    )
    assert response.status_code == 303


def create_sales_run() -> dict:
    login_as("operator", "operator123")
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "sales_followup",
            "input_payload": {
                "period": "2026-W13",
                "region": "华东",
                "sales_reps": ["王晨", "李雪"],
                "focus_metric": "conversion_rate",
            },
            "model_name_override": "qwen-plus",
            "prompt_profile_id": "ops-deep-v1",
            "routing_policy_id": "balanced-router-v1",
        },
    )
    assert response.status_code == 200
    return response.json()


def create_support_run() -> dict:
    login_as("operator", "operator123")
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "support_triage",
            "input_payload": {
                "tickets": [
                    {
                        "customer": "示例客户",
                        "message": "生产接口持续报错，今天必须恢复，否则影响上线。",
                    }
                ]
            },
            "model_name_override": "qwen-turbo",
            "prompt_profile_id": "balanced-v1",
            "routing_policy_id": "strict-review-v1",
        },
    )
    assert response.status_code == 200
    return response.json()


def test_seeded_users_are_persisted_with_password_hash() -> None:
    with database.session() as session:
        record = session.get(UserAccountRecord, "admin")
        assert record is not None
        assert record.password_hash.startswith("pbkdf2_sha256$")
        assert record.password_hash != "admin123"


def test_root_redirects_to_login() -> None:
    anonymous = TestClient(app)
    response = anonymous.get("/", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"


def test_login_page_and_session_endpoint() -> None:
    response = client.get("/login")
    assert response.status_code == 200
    assert "FlowPilot" in response.text

    login_as("viewer", "viewer123")
    session = client.get("/api/session")
    assert session.status_code == 200
    assert session.json()["role"] == "viewer"
    assert session.json()["capabilities"]["can_view"] is True


def test_dashboard_renders_all_real_data_source_options() -> None:
    login_as("operator", "operator123")
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "新建工作流" in response.text
    assert "真实数据源配置" in response.text
    assert "GitHub Issues" in response.text
    assert "NYC 311" in response.text
    assert "Stack Overflow" in response.text
    assert "Hacker News" in response.text


def test_health_endpoint_exposes_backend_shape() -> None:
    response = client.get("/api/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["database_backend"] in {"sqlite", "mysql", "custom"}
    assert "redis_enabled" in body
    assert "monthly_budget_usd" in body


def test_catalog_endpoint_exposes_models_prompts_routing_and_datasets() -> None:
    login_as("viewer", "viewer123")
    response = client.get("/api/experiments/catalog")
    body = response.json()
    assert response.status_code == 200
    assert any(item["model_name"] == "qwen3-max" for item in body["models"])
    assert any(item["profile_id"] == "balanced-v1" for item in body["prompt_profiles"])
    assert any(item["policy_id"] == "balanced-router-v1" for item in body["routing_policies"])
    assert any(item["dataset_id"] == "ops-regression-v1" for item in body["datasets"])


def test_graph_endpoint_exposes_langgraph_shape() -> None:
    login_as("viewer", "viewer123")
    response = client.get("/api/workflows/graph")
    body = response.json()
    assert response.status_code == 200
    assert body["runtime"] == "langgraph"
    assert "router" in body["nodes"]
    assert any(edge["from"] == "planner" and edge["to"] == "router" for edge in body["edges"])
    assert any(edge["from"] == "router" and edge["to"] == "planner" for edge in body["edges"])
    assert any(edge["from"] == "router" and edge["to"] == "reviewer" for edge in body["edges"])


def test_review_page_requires_reviewer_role() -> None:
    login_as("operator", "operator123")
    forbidden = client.get("/reviews")
    assert forbidden.status_code == 403

    login_as("reviewer", "reviewer123")
    allowed = client.get("/reviews")
    assert allowed.status_code == 200


def test_sales_workflow_runs_with_selected_model_prompt_and_routing() -> None:
    body = create_sales_run()
    assert body["result"]["raw_result"]["lead_count"] > 0
    assert body["result"]["execution_profile"]["primary_model_name"] == "qwen-plus"
    assert body["result"]["execution_profile"]["prompt_profile"]["profile_id"] == "ops-deep-v1"
    assert body["result"]["execution_profile"]["routing_policy"]["policy_id"] == "balanced-router-v1"
    assert "planning_context" in body["result"]
    assert "operator_context" in body["result"]
    assert "memory" in body["result"]["planning_context"]
    assert body["result"]["operator_context"]["executed_tool"] == "sales_analytics_tool"
    planner_logs = [log for log in body["logs"] if log["agent"] == "PlannerAgent"]
    assert len(planner_logs) == 1
    assert planner_logs[0]["tool_call"]["name"] == "planning_context_tool"
    llm_logs = [log for log in body["logs"] if log.get("llm_call")]
    assert len(llm_logs) == 5
    assert {log["llm_call"]["route_target"] for log in llm_logs} == {"planner", "operator", "analyst", "content", "reviewer"}


def test_sales_workflow_with_no_matching_data_does_not_claim_metrics() -> None:
    login_as("operator", "operator123")
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "sales_followup",
            "input_payload": {
                "focus_metric": "conversion_rate",
                "period": "2024",
                "region": "广东深圳",
                "sales_reps": ["阿文达", "DeathGun"],
            },
        },
    )
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "waiting_human"
    assert body["result"]["raw_result"]["data_status"] == "no_match"
    assert body["result"]["raw_result"]["matched_rows"] == 0
    assert body["result"]["analysis"]["summary"] == "未找到匹配销售数据，无法生成可信销售分析。"
    assert "转化率为" not in body["result"]["analysis"]["summary"]
    assert body["result"]["deliverables"]["deliverables"]["data_gap"] == "未找到匹配销售数据"
    assert body["review"]["needs_human_review"] is True
    assert any("未找到匹配销售数据" in reason for reason in body["review"]["reasons"])


def test_router_requests_one_replan_when_content_is_missing() -> None:
    decision = RouterAgent().decide(
        last_node="content",
        state={"deliverables": {}, "replan_count": 0},
    )

    assert decision["next_node"] == "planner"
    assert decision["replan_count"] == 1
    assert decision["decision_source"] == "rule"
    assert decision["used_fallback"] is True
    assert decision["fallback_reason"] == "llm_unavailable"


def test_router_uses_confident_model_route_when_allowed() -> None:
    router = RouterAgent(
        llm_service=FakeRouterLLM(
            {
                "route": "reviewer",
                "reason": "content is complete enough for review",
                "confidence": 0.91,
                "fallback_required": False,
            }
        )
    )

    decision = router.decide(
        last_node="content",
        state={
            "raw_result": {"items": [1]},
            "analysis": {"summary": "ready"},
            "deliverables": {"deliverables": {"summary": "ready"}},
            "execution_profile": object(),
        },
    )

    assert decision["next_node"] == "reviewer"
    assert decision["decision_source"] == "model"
    assert decision["model_route"] == "reviewer"
    assert decision["final_route"] == "reviewer"
    assert decision["confidence"] == 0.91
    assert decision["used_fallback"] is False
    assert decision["fallback_reason"] is None


def test_router_falls_back_when_model_route_is_not_allowed() -> None:
    router = RouterAgent(
        llm_service=FakeRouterLLM(
            {
                "route": "archive_run",
                "reason": "unsupported target",
                "confidence": 0.95,
                "fallback_required": False,
            }
        )
    )

    decision = router.decide(last_node="planner", state={"execution_profile": object()})

    assert decision["next_node"] == "operator"
    assert decision["decision_source"] == "rule"
    assert decision["model_route"] == "archive_run"
    assert decision["final_route"] == "operator"
    assert decision["used_fallback"] is True
    assert decision["fallback_reason"] == "route_not_allowed"


def test_router_falls_back_when_model_route_skips_required_state() -> None:
    router = RouterAgent(
        llm_service=FakeRouterLLM(
            {
                "route": "content",
                "reason": "skip directly to content",
                "confidence": 0.95,
                "fallback_required": False,
            }
        )
    )

    decision = router.decide(last_node="planner", state={"execution_profile": object()})

    assert decision["next_node"] == "operator"
    assert decision["decision_source"] == "rule"
    assert decision["model_route"] == "content"
    assert decision["used_fallback"] is True
    assert decision["fallback_reason"] == "route_not_ready"


def test_router_falls_back_when_model_confidence_is_low() -> None:
    router = RouterAgent(
        llm_service=FakeRouterLLM(
            {
                "route": "reviewer",
                "reason": "maybe ready",
                "confidence": 0.69,
                "fallback_required": False,
            }
        )
    )

    decision = router.decide(last_node="analyst", state={"analysis": {"summary": "ok"}, "execution_profile": object()})

    assert decision["next_node"] == "content"
    assert decision["decision_source"] == "rule"
    assert decision["model_route"] == "reviewer"
    assert decision["confidence"] == 0.69
    assert decision["used_fallback"] is True
    assert decision["fallback_reason"] == "low_confidence"


def test_router_falls_back_when_model_is_unavailable() -> None:
    router = RouterAgent(llm_service=FakeRouterLLM(fail=True))

    decision = router.decide(
        last_node="content",
        state={"deliverables": {}, "replan_count": 0, "execution_profile": object()},
    )

    assert decision["next_node"] == "planner"
    assert decision["decision_source"] == "rule"
    assert decision["model_route"] is None
    assert decision["used_fallback"] is True
    assert decision["fallback_reason"] == "model_error"


def test_operator_uses_model_selected_tool_when_allowed() -> None:
    tool_center = ToolCenter(ExternalDataService(Settings()))
    operator = OperatorAgent(
        llm_service=FakeOperatorLLM(
            {
                "selected_tool": "sales_analytics_tool",
                "reason": "need funnel metrics first",
                "confidence": 0.93,
                "fallback_required": False,
            }
        ),
        tool_center=tool_center,
    )

    raw_result, tool_call, operator_context, llm_call = operator.execute(
        request=WorkflowRequest(
            workflow_type=WorkflowType.SALES_FOLLOWUP,
            input_payload={"period": "2026-W13", "region": "华东"},
        ),
        execution_profile=object(),
    )

    assert tool_call.name == "sales_analytics_tool"
    assert operator_context["decision_source"] == "model"
    assert operator_context["executed_tool"] == "sales_analytics_tool"
    assert operator_context["used_fallback"] is False
    assert llm_call is not None
    assert raw_result["lead_count"] > 0


def test_operator_respects_model_selected_support_triage_tool_with_data_source() -> None:
    class RejectingExternalData(ExternalDataService):
        def load_support_tickets(self, source: dict) -> object:
            raise AssertionError("external provider should not be called")

    tool_center = ToolCenter(RejectingExternalData(Settings()))
    operator = OperatorAgent(
        llm_service=FakeOperatorLLM(
            {
                "selected_tool": "support_triage_tool",
                "reason": "use local tickets from payload",
                "confidence": 0.9,
                "fallback_required": False,
            }
        ),
        tool_center=tool_center,
    )

    raw_result, tool_call, operator_context, _ = operator.execute(
        request=WorkflowRequest(
            workflow_type=WorkflowType.SUPPORT_TRIAGE,
            input_payload={
                "tickets": [{"customer": "ACME", "message": "invoice question"}],
                "data_source": {"provider": "github_issues"},
            },
        ),
        execution_profile=object(),
    )

    assert tool_call.name == "support_triage_tool"
    assert operator_context["decision_source"] == "model"
    assert operator_context["selected_tool"] == "support_triage_tool"
    assert operator_context["executed_tool"] == "support_triage_tool"
    assert raw_result["tickets"][0]["customer"] == "ACME"


def test_sales_tool_returns_no_match_instead_of_demo_rollup() -> None:
    tool_center = ToolCenter(ExternalDataService(Settings()))

    raw_result, tool_call = tool_center.run(
        WorkflowType.SALES_FOLLOWUP,
        {
            "focus_metric": "conversion_rate",
            "period": "2024",
            "region": "广东深圳",
            "sales_reps": ["阿文达", "DeathGun"],
        },
    )

    assert tool_call.name == "sales_analytics_tool"
    assert raw_result["data_status"] == "no_match"
    assert raw_result["matched_rows"] == 0
    assert raw_result["lead_count"] == 0
    assert raw_result["deals"] == 0
    assert raw_result["conversion_rate"] is None
    assert raw_result["fallback_reason"] == "no_sales_rows_matched_filters"
    assert raw_result["filters"]["region"] == "广东深圳"
    assert raw_result["filters"]["sales_reps"] == ["阿文达", "DeathGun"]


def test_sales_no_match_fallback_analysis_does_not_claim_metrics() -> None:
    raw_result = {
        "data_status": "no_match",
        "matched_rows": 0,
        "conversion_rate": None,
        "avg_cycle_days": None,
        "risk_customers": [],
        "filters": {
            "period": "2024",
            "region": "广东深圳",
            "sales_reps": ["阿文达", "DeathGun"],
        },
    }

    analysis = AnalystAgent._fallback_analysis(WorkflowType.SALES_FOLLOWUP, raw_result)

    assert "未找到匹配销售数据" in analysis["summary"]
    assert "转化率为" not in analysis["summary"]
    assert all("风险客户" not in item for item in analysis["insights"])
    assert any("补充或校正销售数据" in item for item in analysis["action_plan"])


def test_reviewer_requires_human_review_when_sales_data_is_missing() -> None:
    review = ReviewerAgent._rule_review(
        WorkflowType.SALES_FOLLOWUP,
        {
            "data_status": "no_match",
            "matched_rows": 0,
            "conversion_rate": None,
            "filters": {
                "period": "2024",
                "region": "广东深圳",
                "sales_reps": ["阿文达", "DeathGun"],
            },
        },
        {"summary": "未找到匹配销售数据，无法生成可信分析。"},
        {"deliverables": {"data_gap": "missing sales rows"}},
    )

    assert review["status"] == "waiting_human"
    assert review["needs_human_review"] is True
    assert review["score"] <= 0.5
    assert any("未找到匹配销售数据" in reason for reason in review["reasons"])


def test_operator_falls_back_when_selected_tool_is_not_allowed() -> None:
    tool_center = ToolCenter(ExternalDataService(Settings()))
    operator = OperatorAgent(
        llm_service=FakeOperatorLLM(
            {
                "selected_tool": "marketing_brief_tool",
                "reason": "wrong tool",
                "confidence": 0.95,
                "fallback_required": False,
            }
        ),
        tool_center=tool_center,
    )

    _, tool_call, operator_context, _ = operator.execute(
        request=WorkflowRequest(
            workflow_type=WorkflowType.SALES_FOLLOWUP,
            input_payload={"period": "2026-W13"},
        ),
        execution_profile=object(),
    )

    assert tool_call.name == "sales_analytics_tool"
    assert operator_context["decision_source"] == "rule"
    assert operator_context["used_fallback"] is True
    assert operator_context["fallback_reason"] == "tool_not_allowed"


def test_operator_falls_back_when_confidence_is_low() -> None:
    tool_center = ToolCenter(ExternalDataService(Settings()))
    operator = OperatorAgent(
        llm_service=FakeOperatorLLM(
            {
                "selected_tool": "sales_analytics_tool",
                "reason": "not sure",
                "confidence": 0.61,
                "fallback_required": False,
            }
        ),
        tool_center=tool_center,
    )

    _, tool_call, operator_context, _ = operator.execute(
        request=WorkflowRequest(
            workflow_type=WorkflowType.SALES_FOLLOWUP,
            input_payload={"period": "2026-W13"},
        ),
        execution_profile=object(),
    )

    assert tool_call.name == "sales_analytics_tool"
    assert operator_context["used_fallback"] is True
    assert operator_context["fallback_reason"] == "low_confidence"


def test_operator_falls_back_when_model_is_unavailable() -> None:
    tool_center = ToolCenter(ExternalDataService(Settings()))
    operator = OperatorAgent(
        llm_service=FakeOperatorLLM(fail=True),
        tool_center=tool_center,
    )

    _, tool_call, operator_context, llm_call = operator.execute(
        request=WorkflowRequest(
            workflow_type=WorkflowType.SALES_FOLLOWUP,
            input_payload={"period": "2026-W13"},
        ),
        execution_profile=object(),
    )

    assert tool_call.name == "sales_analytics_tool"
    assert operator_context["used_fallback"] is True
    assert operator_context["fallback_reason"] == "model_error"
    assert llm_call is None


def test_reviewer_output_can_request_content_correction() -> None:
    review = ReviewOutput(
        status="completed",
        needs_human_review=False,
        score=0.72,
        reasons=["交付物缺少负责人和截止时间，需要补充内容。"],
        correction_target="content",
        correction_reason="补充负责人、截止时间和风险说明。",
    )

    assert review.correction_target == "content"
    assert review.correction_reason == "补充负责人、截止时间和风险说明。"

    decision = ReviewDecision(
        status=RunStatus.COMPLETED,
        needs_human_review=False,
        score=review.score,
        reasons=review.reasons,
        correction_target=review.correction_target,
        correction_reason=review.correction_reason,
    )

    assert decision.correction_target == "content"


def test_reviewer_model_correction_does_not_override_rule_handoff() -> None:
    merged = ReviewerAgent._merge_review(
        {
            "status": "waiting_human",
            "needs_human_review": True,
            "score": 0.6,
            "reasons": ["疑似故障类工单必须进入人工接管。"],
            "correction_target": None,
            "correction_reason": None,
        },
        {
            "status": "completed",
            "needs_human_review": False,
            "score": 0.85,
            "reasons": ["可以先补充内容。"],
            "correction_target": "content",
            "correction_reason": "补充说明。",
        },
    )

    assert merged["status"] == "waiting_human"
    assert merged["needs_human_review"] is True
    assert merged["correction_target"] is None


def test_router_routes_reviewer_correction_to_content_once() -> None:
    decision = RouterAgent().decide(
        last_node="reviewer",
        state={
            "raw_result": {"items": [1]},
            "analysis": {"summary": "ready"},
            "deliverables": {"deliverables": {"summary": "missing owner"}},
            "review": {
                "status": "completed",
                "needs_human_review": False,
                "correction_target": "content",
                "correction_reason": "补充负责人和截止时间。",
            },
            "correction_count": 0,
        },
    )

    assert decision["next_node"] == "content"
    assert decision["correction_count"] == 1
    assert decision["correction_target"] == "content"
    assert decision["used_correction_loop"] is True
    assert "补充负责人和截止时间" in decision["reason"]


def test_router_routes_reviewer_correction_to_analyst_once() -> None:
    decision = RouterAgent().decide(
        last_node="reviewer",
        state={
            "raw_result": {"items": [1]},
            "analysis": {"summary": "thin analysis"},
            "deliverables": {"deliverables": {"summary": "ready"}},
            "review": {
                "status": "completed",
                "needs_human_review": False,
                "correction_target": "analyst",
                "correction_reason": "补充根因分析。",
            },
            "correction_count": 0,
        },
    )

    assert decision["next_node"] == "analyst"
    assert decision["correction_count"] == 1
    assert decision["correction_target"] == "analyst"
    assert decision["used_correction_loop"] is True


def test_router_hands_off_when_reviewer_correction_limit_is_reached() -> None:
    decision = RouterAgent().decide(
        last_node="reviewer",
        state={
            "raw_result": {"items": [1]},
            "analysis": {"summary": "ready"},
            "deliverables": {"deliverables": {"summary": "still incomplete"}},
            "review": {
                "status": "completed",
                "needs_human_review": False,
                "correction_target": "content",
                "correction_reason": "再次补充内容。",
            },
            "correction_count": 1,
        },
    )

    assert decision["next_node"] == "handoff_run"
    assert decision["correction_count"] == 1
    assert decision["correction_target"] == "content"
    assert decision["used_correction_loop"] is False
    assert decision["fallback_reason"] == "correction_limit_reached"


def test_workflow_result_records_route_decisions() -> None:
    body = create_sales_run()

    route_decisions = body["result"]["route_decisions"]

    assert route_decisions[0]["from_node"] == "planner"
    assert route_decisions[0]["next_node"] == "operator"
    assert route_decisions[0]["final_route"] == "operator"
    assert "decision_source" in route_decisions[0]
    assert "used_fallback" in route_decisions[0]
    assert any(item["next_node"] == "reviewer" for item in route_decisions)


def test_workflow_run_persists_operator_context() -> None:
    body = create_sales_run()

    assert "operator_context" in body["result"]
    assert "executed_tool" in body["result"]["operator_context"]
    assert "used_fallback" in body["result"]["operator_context"]


def test_waiting_human_reasons_do_not_include_auto_execute_copy() -> None:
    merged = ReviewerAgent._merge_review(
        {
            "status": "completed",
            "needs_human_review": False,
            "score": 0.92,
            "reasons": ["结果结构完整，可直接流转执行。"],
        },
        {
            "status": "waiting_human",
            "needs_human_review": True,
            "score": 0.65,
            "reasons": ["跟进策略需要管理者确认后才能执行。"],
        },
    )
    assert merged["status"] == "waiting_human"
    assert merged["needs_human_review"] is True
    assert all("直接流转" not in reason for reason in merged["reasons"])
    assert any("确认" in reason or "审核" in reason for reason in merged["reasons"])


def test_support_workflow_flags_human_review_and_can_be_approved() -> None:
    body = create_support_run()
    assert body["review"]["needs_human_review"] is True
    assert body["status"] == "waiting_human"

    login_as("reviewer", "reviewer123")
    queue = client.get("/api/workflows/review-queue").json()
    assert any(item["id"] == body["id"] for item in queue)

    approved = client.post(
        f"/api/workflows/{body['id']}/review",
        json={"approve": True, "comment": "值班负责人已确认回复策略与处理方案。"},
    ).json()
    assert approved["status"] == "completed"
    assert any("已通过" in log["message"] for log in approved["logs"])


def test_run_detail_page_contains_all_export_formats() -> None:
    created = create_sales_run()
    detail = client.get(f"/runs/{created['id']}")
    assert detail.status_code == 200
    assert "执行时间线" in detail.text
    assert "导出 Markdown" in detail.text
    assert "导出 HTML" in detail.text
    assert "导出 PDF" in detail.text


def test_run_detail_page_shows_route_trace_section() -> None:
    run = WorkflowRun(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        status=RunStatus.COMPLETED,
        current_step="completed",
        objective="Route trace phase 1 fixture",
        result={
            "route_decisions": [
                {
                    "from_node": "planner",
                    "next_node": "operator",
                    "reason": "plan is ready; collect or transform source data",
                    "replan_count": 0,
                },
                {
                    "from_node": "content",
                    "next_node": "planner",
                    "reason": "deliverables are missing; request one replanning pass",
                    "replan_count": 1,
                },
            ]
        },
    )
    repository.save(run)
    login_as("viewer", "viewer123")

    detail = client.get(f"/runs/{run.id}")

    assert detail.status_code == 200
    assert "路由轨迹" in detail.text
    assert "2 步路由" in detail.text
    assert "planner -> operator" in detail.text
    assert "content -> planner" in detail.text
    assert "发生重规划：是" in detail.text
    assert "REPLAN" in detail.text
    assert "plan is ready; collect or transform source data" in detail.text


def test_run_detail_page_handles_empty_route_trace_for_legacy_run() -> None:
    run = WorkflowRun(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        status=RunStatus.COMPLETED,
        current_step="completed",
        objective="Legacy run without route decisions",
        result={"analysis": {"summary": "legacy result"}},
    )
    repository.save(run)
    login_as("viewer", "viewer123")

    detail = client.get(f"/runs/{run.id}")

    assert detail.status_code == 200
    assert "路由轨迹" in detail.text
    assert "暂无路由轨迹（该 run 可能来自旧版本）" in detail.text
    assert "结果 JSON" in detail.text


def test_run_detail_page_shows_route_trace_audit_fields_when_present() -> None:
    run = WorkflowRun(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        status=RunStatus.COMPLETED,
        current_step="completed",
        objective="Route trace phase 2 fixture",
        result={
            "route_decisions": [
                {
                    "from_node": "content",
                    "next_node": "reviewer",
                    "final_route": "reviewer",
                    "decision_source": "model",
                    "model_route": "reviewer",
                    "confidence": 0.91,
                    "used_fallback": False,
                    "fallback_reason": None,
                    "reason": "content is complete enough for review",
                    "replan_count": 0,
                },
                {
                    "from_node": "planner",
                    "next_node": "operator",
                    "final_route": "operator",
                    "decision_source": "rule",
                    "model_route": "content",
                    "confidence": 0.95,
                    "used_fallback": True,
                    "fallback_reason": "route_not_ready",
                    "reason": "plan is ready; collect or transform source data",
                    "replan_count": 0,
                },
            ]
        },
    )
    repository.save(run)
    login_as("viewer", "viewer123")

    detail = client.get(f"/runs/{run.id}")

    assert detail.status_code == 200
    assert "MODEL" in detail.text
    assert "RULE" in detail.text
    assert "FALLBACK" in detail.text
    assert "fallback 次数：1" in detail.text
    assert "模型建议：content" in detail.text
    assert "置信度：0.95" in detail.text
    assert "route_not_ready" in detail.text


def test_run_detail_page_shows_operator_context_when_present() -> None:
    run = WorkflowRun(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        status=RunStatus.COMPLETED,
        current_step="completed",
        objective="Operator context fixture",
        result={
            "operator_context": {
                "decision_source": "model",
                "selected_tool": "sales_analytics_tool",
                "executed_tool": "sales_analytics_tool",
                "used_fallback": False,
                "fallback_reason": None,
                "tool_choices": ["sales_analytics_tool"],
                "decision_reason": "need funnel metrics first",
                "confidence": 0.93,
            }
        },
    )
    repository.save(run)
    login_as("viewer", "viewer123")

    detail = client.get(f"/runs/{run.id}")

    assert detail.status_code == 200
    assert "Operator Decision" in detail.text
    assert "MODEL" in detail.text
    assert "Selected Tool" in detail.text
    assert "Executed Tool" in detail.text
    assert "sales_analytics_tool" in detail.text
    assert "need funnel metrics first" in detail.text
    assert "0.93" in detail.text


def test_run_detail_page_handles_missing_operator_context() -> None:
    run = WorkflowRun(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        status=RunStatus.COMPLETED,
        current_step="completed",
        objective="Legacy run without operator context",
        result={"analysis": {"summary": "legacy result"}},
    )
    repository.save(run)
    login_as("viewer", "viewer123")

    detail = client.get(f"/runs/{run.id}")

    assert detail.status_code == 200
    assert "Operator Decision" in detail.text
    assert "No operator context recorded (legacy run)." in detail.text


def test_run_detail_page_shows_data_provenance_when_present() -> None:
    run = WorkflowRun(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        status=RunStatus.WAITING_HUMAN,
        current_step="reviewer",
        objective="No matching sales data fixture",
        result={
            "raw_result": {
                "data_status": "no_match",
                "matched_rows": 0,
                "source": "demo_sales_data",
                "fallback_reason": "no_sales_rows_matched_filters",
                "filters": {
                    "period": "2024",
                    "region": "广东深圳",
                    "sales_reps": ["阿文达", "DeathGun"],
                },
                "message": "未找到匹配销售数据，无法生成可信销售分析。",
            }
        },
    )
    repository.save(run)
    login_as("viewer", "viewer123")

    detail = client.get(f"/runs/{run.id}")

    assert detail.status_code == 200
    assert "Data Provenance" in detail.text
    assert "NO TRUSTED DATA" in detail.text
    assert "data_status" in detail.text
    assert "no_match" in detail.text
    assert "matched_rows" in detail.text
    assert "0" in detail.text
    assert "source" in detail.text
    assert "demo_sales_data" in detail.text
    assert "fallback_reason" in detail.text
    assert "no_sales_rows_matched_filters" in detail.text
    assert "filters" in detail.text
    assert "广东深圳" in detail.text
    assert "阿文达" in detail.text
    assert "DeathGun" in detail.text
    assert "未找到匹配销售数据" in detail.text


def test_run_detail_page_handles_missing_data_provenance_for_legacy_run() -> None:
    run = WorkflowRun(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        status=RunStatus.COMPLETED,
        current_step="completed",
        objective="Legacy run without data provenance",
        result={"raw_result": {"lead_count": 12}},
    )
    repository.save(run)
    login_as("viewer", "viewer123")

    detail = client.get(f"/runs/{run.id}")

    assert detail.status_code == 200
    assert "Data Provenance" in detail.text
    assert "No data provenance recorded (legacy run)." in detail.text
    assert "JSON" in detail.text


def test_workflow_export_endpoint_supports_markdown_html_and_pdf() -> None:
    created = create_sales_run()
    markdown = client.get(f"/runs/{created['id']}/export?format=markdown")
    html = client.get(f"/runs/{created['id']}/export?format=html")
    pdf = client.get(f"/runs/{created['id']}/export?format=pdf")
    assert markdown.status_code == 200
    assert html.status_code == 200
    assert pdf.status_code == 200
    assert "text/markdown" in markdown.headers["content-type"]
    assert "text/html" in html.headers["content-type"]
    assert "application/pdf" in pdf.headers["content-type"]


def test_runs_page_supports_status_and_workflow_filters() -> None:
    create_support_run()
    response = client.get("/runs?status_filter=waiting_human&workflow_filter=support_triage")
    assert response.status_code == 200
    assert "只看待审核" in response.text
    assert "客服工单智能分流" in response.text


def test_prompt_profile_can_be_created_and_updated() -> None:
    login_as("admin", "admin123")
    profile_id = f"ops-lab-{uuid4().hex[:8]}"
    created = client.post(
        "/api/prompts",
        json={
            "profile_id": profile_id,
            "base_profile_id": "ops-deep-v1",
            "name": "运营实验版",
            "version": "v1",
            "description": "用于评估更强运营语言风格的实验方案。",
            "analyst_instruction": "更强调根因分析。",
            "content_instruction": "更强调责任人与截止时间。",
            "reviewer_instruction": "风险不清晰时更偏向人工审核。",
        },
    )
    assert created.status_code == 200
    assert created.json()["profile_id"] == profile_id

    updated = client.put(
        f"/api/prompts/{profile_id}",
        json={
            "profile_id": profile_id,
            "base_profile_id": "ops-deep-v1",
            "name": "运营实验版",
            "version": "v2",
            "description": "用于评估更强运营语言风格的实验方案。",
            "analyst_instruction": "更强调根因分析和优先级。",
            "content_instruction": "更强调责任人与截止时间。",
            "reviewer_instruction": "风险不清晰时更偏向人工审核。",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == "v2"


def test_compare_page_and_api_show_routing_experiments() -> None:
    create_sales_run()
    login_as("viewer", "viewer123")
    page = client.get("/compare")
    assert page.status_code == 200

    compare_api = client.get("/api/experiments/compare")
    body = compare_api.json()
    assert compare_api.status_code == 200
    assert body["run_count"] >= 1
    assert isinstance(body["rows"], list)


def test_evaluation_run_page_supports_trend_and_drilldown_and_exports() -> None:
    login_as("operator", "operator123")
    page = client.get("/evaluations")
    assert page.status_code == 200
    assert "自动评测" in page.text

    response = client.post(
        "/evaluations/run",
        data={
            "dataset_id": "ops-regression-v1",
            "candidate_model_name": "qwen-plus",
            "candidate_prompt_profile_id": "balanced-v1",
            "candidate_routing_policy_id": "balanced-router-v1",
            "baseline_model_name": "qwen-turbo",
            "baseline_prompt_profile_id": "balanced-v1",
            "baseline_routing_policy_id": "single-model-v1",
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert "评测趋势" in response.text
    assert "查看单案例明细" in response.text

    evaluations = client.get("/api/evaluations").json()
    assert len(evaluations) >= 1
    evaluation_id = evaluations[0]["id"]
    markdown = client.get(f"/evaluations/{evaluation_id}/export?format=markdown")
    html = client.get(f"/evaluations/{evaluation_id}/export?format=html")
    pdf = client.get(f"/evaluations/{evaluation_id}/export?format=pdf")
    assert "text/markdown" in markdown.headers["content-type"]
    assert "text/html" in html.headers["content-type"]
    assert "application/pdf" in pdf.headers["content-type"]


def test_batch_experiment_run_and_listing_work() -> None:
    login_as("operator", "operator123")
    response = client.post(
        "/api/batches",
        json={
            "name": "Prompt AB 回归",
            "workflow_type": "sales_followup",
            "input_payload": {
                "period": "2026-W13",
                "region": "华东",
                "sales_reps": ["王晨", "李雪"],
                "focus_metric": "conversion_rate",
            },
            "repeats": 1,
            "variants": [
                {
                    "variant_id": "control",
                    "label": "对照组",
                    "model_name": "qwen-plus",
                    "prompt_profile_id": "balanced-v1",
                    "routing_policy_id": "balanced-router-v1",
                },
                {
                    "variant_id": "challenger",
                    "label": "挑战组",
                    "model_name": "qwen3-max",
                    "prompt_profile_id": "ops-deep-v1",
                    "routing_policy_id": "strict-review-v1",
                },
            ],
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["summary"]["variant_count"] == 2
    assert body["summary"]["champion"] is not None


def test_feedback_review_creates_feedback_sample() -> None:
    body = create_support_run()
    login_as("reviewer", "reviewer123")
    client.post(
        f"/api/workflows/{body['id']}/review",
        json={"approve": True, "comment": "需要保留负责人和风险说明。"},
    )
    feedback_samples = client.get("/api/feedback-samples")
    assert feedback_samples.status_code == 200
    assert len(feedback_samples.json()) >= 1


def test_runtime_memory_context_is_recorded_for_analyst_and_reviewer() -> None:
    first = create_support_run()
    login_as("reviewer", "reviewer123")
    reviewed = client.post(
        f"/api/workflows/{first['id']}/review",
        json={"approve": True, "comment": "keep owner and risk notes"},
    )
    assert reviewed.status_code == 200

    second = create_support_run()

    analyst_context = second["result"]["analyst_context"]
    reviewer_context = second["result"]["reviewer_context"]

    assert analyst_context["memory_hits"] >= 1
    assert reviewer_context["memory_hits"] >= 1
    assert len(analyst_context["memory"]["recent_runs"]) >= 1
    assert len(reviewer_context["memory"]["feedback_samples"]) >= 1
    assert any("历史记忆" in log["message"] for log in second["logs"] if log["agent"] == "AnalystAgent")
    assert any("历史记忆" in log["message"] for log in second["logs"] if log["agent"] == "ReviewerAgent")


def test_runtime_memory_context_is_recorded_for_content_agent() -> None:
    first = create_support_run()
    login_as("reviewer", "reviewer123")
    reviewed = client.post(
        f"/api/workflows/{first['id']}/review",
        json={"approve": True, "comment": "keep owner and risk notes"},
    )
    assert reviewed.status_code == 200

    second = create_support_run()

    content_context = second["result"]["content_context"]

    assert content_context["memory_hits"] >= 1
    assert len(content_context["memory"]["recent_runs"]) >= 1
    assert len(content_context["memory"]["feedback_samples"]) >= 1
    assert any("历史记忆" in log["message"] for log in second["logs"] if log["agent"] == "ContentAgent")


def test_run_detail_page_shows_runtime_memory_summary() -> None:
    first = create_support_run()
    login_as("reviewer", "reviewer123")
    reviewed = client.post(
        f"/api/workflows/{first['id']}/review",
        json={"approve": True, "comment": "keep owner and risk notes"},
    )
    assert reviewed.status_code == 200

    second = create_support_run()

    detail = client.get(f"/runs/{second['id']}")

    assert detail.status_code == 200
    assert "运行时记忆" in detail.text
    assert "AnalystAgent" in detail.text
    assert "ContentAgent" in detail.text
    assert "ReviewerAgent" in detail.text
    assert "memory hits" in detail.text


def test_settings_can_disable_runtime_memory(monkeypatch) -> None:
    monkeypatch.setenv("FLOWPILOT_ENABLE_RUNTIME_MEMORY", "false")
    settings = Settings.from_env()
    assert settings.enable_runtime_memory is False
