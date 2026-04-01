import json
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from app.config import Settings
from app.db import UserAccountRecord
from app.external_data import ExternalDataService, ExternalTicketBatch
from app.main import app, database
from app.models import ReviewDecision, RunStatus, WorkflowRun, WorkflowType
from app.prompt_catalog import BUILTIN_PROMPT_PROFILES, EvaluationCaseDefinition, resolve_execution_profile
from app.services import EvaluationService, ReviewerAgent, ToolCenter
from app.llm import LLMService
from app.models import AnalystOutput


client = TestClient(app)


def login_as(username: str, password: str) -> None:
    response = client.post(
        "/login",
        data={"username": username, "password": password, "next": "/dashboard"},
        follow_redirects=False,
    )
    assert response.status_code == 303


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


def test_dashboard_renders_prompt_and_routing_controls() -> None:
    login_as("operator", "operator123")
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "新建工作流" in response.text
    assert "routing-policy-id" in response.text
    assert "model-name" in response.text


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
    assert "planner" in body["nodes"]
    assert any(edge["from"] == "reviewer" for edge in body["edges"])


def test_review_page_requires_reviewer_role() -> None:
    login_as("operator", "operator123")
    forbidden = client.get("/reviews")
    assert forbidden.status_code == 403

    login_as("reviewer", "reviewer123")
    allowed = client.get("/reviews")
    assert allowed.status_code == 200
    assert "审核中心" in allowed.text


def test_sales_workflow_runs_with_selected_model_prompt_and_routing() -> None:
    login_as("operator", "operator123")
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "sales_followup",
            "input_payload": {
                "period": "2026-W13",
                "region": "华东",
                "sales_reps": ["王晨", "李雪"],
            },
            "model_name_override": "qwen-plus",
            "prompt_profile_id": "ops-deep-v1",
            "routing_policy_id": "balanced-router-v1",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["result"]["raw_result"]["lead_count"] > 0
    assert body["result"]["execution_profile"]["primary_model_name"] == "qwen-plus"
    assert body["result"]["execution_profile"]["prompt_profile"]["profile_id"] == "ops-deep-v1"
    assert body["result"]["execution_profile"]["routing_policy"]["policy_id"] == "balanced-router-v1"
    llm_logs = [log for log in body["logs"] if log.get("llm_call")]
    assert len(llm_logs) == 3
    assert {log["llm_call"]["route_target"] for log in llm_logs} == {"analyst", "content", "reviewer"}


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
    login_as("operator", "operator123")
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "support_triage",
            "input_payload": {
                "tickets": [{"customer": "示例客户", "message": "生产接口持续报错，今天必须恢复，否则影响上线。"}]
            },
            "model_name_override": "qwen-turbo",
            "prompt_profile_id": "balanced-v1",
            "routing_policy_id": "strict-review-v1",
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["review"]["needs_human_review"] is True
    assert body["status"] == "waiting_human"
    assert any("人工" in reason or "接管" in reason for reason in body["review"]["reasons"])

    login_as("reviewer", "reviewer123")
    queue = client.get("/api/workflows/review-queue").json()
    assert any(item["id"] == body["id"] for item in queue)

    approved = client.post(
        f"/api/workflows/{body['id']}/review",
        json={"approve": True, "comment": "值班负责人已确认回复策略与处置方案。"},
    ).json()
    assert approved["status"] == "completed"
    assert any("已通过" in log["message"] for log in approved["logs"])


def test_detail_page_contains_ai_metrics_and_execution_profile() -> None:
    login_as("operator", "operator123")
    created = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "meeting_minutes",
            "input_payload": {
                "meeting_title": "产品周会",
                "notes": "1. 张敏本周五前完成竞品复盘。2. 王晨今天下班前确认试点客户名单。",
            },
            "model_name_override": "qwen3-max",
            "prompt_profile_id": "exec-brief-v2",
            "routing_policy_id": "single-model-v1",
        },
    ).json()
    detail = client.get(f"/runs/{created['id']}")
    assert detail.status_code == 200
    assert "执行时间线" in detail.text
    assert "结果 JSON" in detail.text
    assert "审核与决策" in detail.text


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
            "content_instruction": "更强调责任人和截止时间。",
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
            "content_instruction": "更强调责任人和截止时间。",
            "reviewer_instruction": "风险不清晰时更偏向人工审核。",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == "v2"


def test_compare_page_and_api_show_routing_experiments() -> None:
    login_as("viewer", "viewer123")
    page = client.get("/compare")
    assert page.status_code == 200
    assert "方案对比" in page.text

    compare_api = client.get("/api/experiments/compare")
    body = compare_api.json()
    assert compare_api.status_code == 200
    assert body["run_count"] >= 1
    assert isinstance(body["rows"], list)


def test_evaluation_run_and_listing_work() -> None:
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
    evaluations = client.get("/api/evaluations").json()
    assert len(evaluations) >= 1
    assert evaluations[0]["dataset_id"] in {"ops-regression-v1", "feedback-loop-v1"}
    assert "candidate_avg_score" in evaluations[0]["summary"]
    assert "candidate_dimensions" in evaluations[0]["summary"]


def test_cost_summary_page_and_api_work() -> None:
    login_as("viewer", "viewer123")
    page = client.get("/costs")
    assert page.status_code == 200
    assert "成本看板" in page.text
    assert "预算进度" in page.text

    summary = client.get("/api/costs/summary")
    body = summary.json()
    assert summary.status_code == 200
    assert "total_cost_usd" in body
    assert "monthly_budget_usd" in body
    assert "model_rows" in body
    assert "alert_title" in body


def test_batch_experiment_run_and_listing_work() -> None:
    login_as("operator", "operator123")
    response = client.post(
        "/api/batches",
        json={
            "name": "Prompt AB regression",
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
    assert len(body["results"]) == 2
    assert body["summary"]["champion"] is not None
    assert any(row["is_champion"] for row in body["summary"]["rows"])

    listing = client.get("/api/batches").json()
    assert any(item["id"] == body["id"] for item in listing)
    page = client.get("/batches")
    assert page.status_code == 200
    assert "冠军方案" in page.text


def test_feedback_review_creates_feedback_sample_and_feedback_dataset() -> None:
    login_as("operator", "operator123")
    created = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "support_triage",
            "input_payload": {
                "tickets": [{"customer": "示例客户", "message": "生产接口持续报错，今天必须恢复，否则影响上线。"}]
            },
            "model_name_override": "qwen-turbo",
            "prompt_profile_id": "balanced-v1",
            "routing_policy_id": "strict-review-v1",
        },
    ).json()

    login_as("reviewer", "reviewer123")
    review_response = client.post(
        f"/api/workflows/{created['id']}/review",
        json={"approve": True, "comment": "人工复核通过，请保留升级与故障响应表述。"},
    )
    assert review_response.status_code == 200

    feedback_samples = client.get("/api/feedback-samples").json()
    assert any(item["source_run_id"] == created["id"] for item in feedback_samples)
    matched = next(item for item in feedback_samples if item["source_run_id"] == created["id"])
    assert "feedback_enrichment" in matched["output_snapshot"]
    assert "scoring_rubric" in matched["output_snapshot"]["feedback_enrichment"]

    catalog = client.get("/api/experiments/catalog").json()
    assert any(item["dataset_id"] == "feedback-loop-v1" for item in catalog["datasets"])


def test_run_can_be_deleted_and_related_feedback_samples_are_removed() -> None:
    login_as("operator", "operator123")
    created = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "support_triage",
            "input_payload": {
                "tickets": [{"customer": "删除测试客户", "message": "生产系统报错，需要尽快处理。"}]
            },
            "model_name_override": "qwen-turbo",
            "prompt_profile_id": "balanced-v1",
            "routing_policy_id": "strict-review-v1",
        },
    ).json()

    login_as("reviewer", "reviewer123")
    reviewed = client.post(
        f"/api/workflows/{created['id']}/review",
        json={"approve": True, "comment": "删除前先沉淀一条反馈样本。"},
    )
    assert reviewed.status_code == 200

    feedback_samples = client.get("/api/feedback-samples").json()
    assert any(item["source_run_id"] == created["id"] for item in feedback_samples)

    deleted = client.delete(f"/api/workflows/{created['id']}")
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True

    workflows = client.get("/api/workflows").json()
    assert all(item["id"] != created["id"] for item in workflows)
    feedback_samples_after = client.get("/api/feedback-samples").json()
    assert all(item["source_run_id"] != created["id"] for item in feedback_samples_after)


def test_runs_can_be_bulk_deleted_with_related_feedback_samples_removed() -> None:
    created_ids = []

    login_as("operator", "operator123")
    for customer in ["批量删除客户A", "批量删除客户B"]:
        created = client.post(
            "/api/workflows/run",
            json={
                "workflow_type": "support_triage",
                "input_payload": {
                    "tickets": [{"customer": customer, "message": "生产系统报错，需要尽快处理。"}]
                },
                "model_name_override": "qwen-turbo",
                "prompt_profile_id": "balanced-v1",
                "routing_policy_id": "strict-review-v1",
            },
        ).json()
        created_ids.append(created["id"])

    login_as("reviewer", "reviewer123")
    for run_id in created_ids:
        reviewed = client.post(
            f"/api/workflows/{run_id}/review",
            json={"approve": True, "comment": "用于批量删除测试的反馈样本。"},
        )
        assert reviewed.status_code == 200

    feedback_samples = client.get("/api/feedback-samples").json()
    assert all(any(item["source_run_id"] == run_id for item in feedback_samples) for run_id in created_ids)

    deleted = client.post("/api/workflows/bulk-delete", json={"run_ids": created_ids})
    assert deleted.status_code == 200
    assert deleted.json()["deleted_count"] == 2
    assert set(deleted.json()["deleted_run_ids"]) == set(created_ids)

    workflows = client.get("/api/workflows").json()
    assert all(item["id"] not in created_ids for item in workflows)
    feedback_samples_after = client.get("/api/feedback-samples").json()
    assert all(item["source_run_id"] not in created_ids for item in feedback_samples_after)


def test_external_data_service_normalizes_github_issues() -> None:
    service = ExternalDataService(Settings(disable_llm=True))
    service._fetch_json = lambda _url: [
        {
            "number": 101,
            "title": "生产接口报错",
            "body": "调用下游服务失败",
            "html_url": "https://github.com/example/repo/issues/101",
            "labels": [{"name": "bug"}],
        }
    ]
    batch = service.load_support_tickets({"provider": "github_issues", "repo": "example/repo", "per_page": 1})
    assert batch.provider == "github_issues"
    assert batch.summary["repo"] == "example/repo"
    assert batch.records[0]["customer"] == "example/repo"
    assert batch.records[0]["source_id"] == "issue#101"


def test_support_triage_can_use_external_ticket_source() -> None:
    class FakeExternalData:
        def load_support_tickets(self, source):
            assert source["provider"] == "github_issues"
            return ExternalTicketBatch(
                provider="github_issues",
                records=[
                    {
                        "customer": "example/repo",
                        "message": "生产环境 outage",
                        "body": "需要立即恢复",
                        "source_id": "issue#9",
                        "source_url": "https://example.invalid/9",
                    }
                ],
                summary={"repo": "example/repo", "ticket_count": 1},
            )

    tool_center = ToolCenter(FakeExternalData())
    result, call = tool_center.run(
        WorkflowType.SUPPORT_TRIAGE,
        {"data_source": {"provider": "github_issues", "repo": "example/repo"}},
    )
    assert call.name == "github_issues_tool"
    assert result["data_source_summary"]["provider"] == "github_issues"
    assert result["tickets"][0]["category"] == "incident"


def test_llm_service_retries_until_schema_valid() -> None:
    class FakeCompletions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                content = json.dumps({"summary": "", "insights": [], "action_plan": []}, ensure_ascii=False)
            else:
                content = json.dumps(
                    {
                        "summary": "这是有效总结",
                        "insights": ["发现一", "发现二"],
                        "action_plan": ["动作一", "动作二"],
                    },
                    ensure_ascii=False,
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                usage=SimpleNamespace(prompt_tokens=12, completion_tokens=8, total_tokens=20),
            )

    llm_service = LLMService(Settings(api_key="fake-key", disable_llm=False))
    llm_service._client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    profile = resolve_execution_profile(
        default_model_name="qwen3-max",
        prompt_profile=BUILTIN_PROMPT_PROFILES[0],
        model_name_override="qwen3-max",
        routing_policy_id="single-model-v1",
    )
    response = llm_service.generate_json(
        route_target="analyst",
        system_prompt="你是分析助手",
        user_prompt="请输出 JSON",
        fallback={"summary": "fallback", "insights": ["fallback"], "action_plan": ["fallback"]},
        execution_profile=profile,
        response_model=AnalystOutput,
    )
    assert response.payload["summary"] == "这是有效总结"
    assert response.call.retry_count == 1
    assert response.call.used_fallback is False


def test_evaluation_dimensions_are_calculated() -> None:
    run = WorkflowRun(
        workflow_type=WorkflowType.SUPPORT_TRIAGE,
        status=RunStatus.WAITING_HUMAN,
        result={"raw_result": {}, "analysis": {}, "deliverables": {}, "review": {}},
        review=ReviewDecision(
            status=RunStatus.WAITING_HUMAN,
            needs_human_review=True,
            score=0.7,
            reasons=["需要人工审核"],
        ),
    )
    case = EvaluationCaseDefinition(
        case_id="demo",
        title="demo",
        workflow_type="support_triage",
        input_payload={},
        expected_status="waiting_human",
        expected_keywords=("review",),
    )
    dimensions = EvaluationService._score_dimensions(run, case)
    assert set(dimensions) == {"status_match", "keyword_coverage", "result_completeness", "review_alignment"}
    assert dimensions["status_match"] == 1.0
    assert dimensions["result_completeness"] == 1.0
