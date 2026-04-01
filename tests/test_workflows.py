from uuid import uuid4

from fastapi.testclient import TestClient

from app.db import UserAccountRecord
from app.main import app, database
from app.services import ReviewerAgent


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
    assert "Launch workflow" in response.text
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
    assert "Review Queue" in allowed.text


def test_sales_workflow_runs_with_selected_model_prompt_and_routing() -> None:
    login_as("operator", "operator123")
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "sales_followup",
            "input_payload": {
                "period": "2026-W13",
                "region": "East",
                "sales_reps": ["Wang Chen", "Li Xue"],
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
            "reasons": ["Result is structured and can proceed automatically."],
        },
        {
            "status": "waiting_human",
            "needs_human_review": True,
            "score": 0.65,
            "reasons": ["A manager should confirm the follow-up strategy before execution."],
        },
    )
    assert merged["status"] == "waiting_human"
    assert merged["needs_human_review"] is True
    assert all("automatically" not in reason.lower() for reason in merged["reasons"])
    assert any("confirm" in reason.lower() or "review" in reason.lower() for reason in merged["reasons"])


def test_support_workflow_flags_human_review_and_can_be_approved() -> None:
    login_as("operator", "operator123")
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "support_triage",
            "input_payload": {
                "tickets": [{"customer": "Example Client", "message": "Production API errors are blocking a release. Need help today."}]
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
    assert any("human review" in " ".join(body["review"]["reasons"]).lower() or "handoff" in " ".join(body["review"]["reasons"]).lower() for _ in [1])

    login_as("reviewer", "reviewer123")
    queue = client.get("/api/workflows/review-queue").json()
    assert any(item["id"] == body["id"] for item in queue)

    approved = client.post(
        f"/api/workflows/{body['id']}/review",
        json={"approve": True, "comment": "On-call owner confirmed the response and rollout plan."},
    ).json()
    assert approved["status"] == "completed"
    assert any("approved" in log["message"] for log in approved["logs"])


def test_detail_page_contains_ai_metrics_and_execution_profile() -> None:
    login_as("operator", "operator123")
    created = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "meeting_minutes",
            "input_payload": {
                "meeting_title": "Product Weekly",
                "notes": "1. Zhang Min to finish competitor review by Friday. 2. Wang Chen to confirm pilot customers today.",
            },
            "model_name_override": "qwen3-max",
            "prompt_profile_id": "exec-brief-v2",
            "routing_policy_id": "single-model-v1",
        },
    ).json()
    detail = client.get(f"/runs/{created['id']}")
    assert detail.status_code == 200
    assert "Execution timeline" in detail.text
    assert "Result JSON" in detail.text
    assert "Review decision" in detail.text


def test_prompt_profile_can_be_created_and_updated() -> None:
    login_as("admin", "admin123")
    profile_id = f"ops-lab-{uuid4().hex[:8]}"
    created = client.post(
        "/api/prompts",
        json={
            "profile_id": profile_id,
            "base_profile_id": "ops-deep-v1",
            "name": "Ops Lab",
            "version": "v1",
            "description": "Experimental prompt profile for sharper operational language.",
            "analyst_instruction": "Stress root-cause analysis.",
            "content_instruction": "Stress owners and deadlines.",
            "reviewer_instruction": "Be conservative when risk is unclear.",
        },
    )
    assert created.status_code == 200
    assert created.json()["profile_id"] == profile_id

    updated = client.put(
        f"/api/prompts/{profile_id}",
        json={
            "profile_id": profile_id,
            "base_profile_id": "ops-deep-v1",
            "name": "Ops Lab",
            "version": "v2",
            "description": "Experimental prompt profile for sharper operational language.",
            "analyst_instruction": "Stress root-cause analysis and prioritization.",
            "content_instruction": "Stress owners and deadlines.",
            "reviewer_instruction": "Be conservative when risk is unclear.",
        },
    )
    assert updated.status_code == 200
    assert updated.json()["version"] == "v2"


def test_compare_page_and_api_show_routing_experiments() -> None:
    login_as("viewer", "viewer123")
    page = client.get("/compare")
    assert page.status_code == 200
    assert "Compare" in page.text

    compare_api = client.get("/api/experiments/compare")
    body = compare_api.json()
    assert compare_api.status_code == 200
    assert body["run_count"] >= 1
    assert isinstance(body["rows"], list)


def test_evaluation_run_and_listing_work() -> None:
    login_as("operator", "operator123")
    page = client.get("/evaluations")
    assert page.status_code == 200
    assert "Evaluations" in page.text

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


def test_cost_summary_page_and_api_work() -> None:
    login_as("viewer", "viewer123")
    page = client.get("/costs")
    assert page.status_code == 200
    assert "Cost Board" in page.text

    summary = client.get("/api/costs/summary")
    body = summary.json()
    assert summary.status_code == 200
    assert "total_cost_usd" in body
    assert "monthly_budget_usd" in body
    assert "model_rows" in body


def test_batch_experiment_run_and_listing_work() -> None:
    login_as("operator", "operator123")
    response = client.post(
        "/api/batches",
        json={
            "name": "Prompt AB regression",
            "workflow_type": "sales_followup",
            "input_payload": {
                "period": "2026-W13",
                "region": "East",
                "sales_reps": ["Wang Chen", "Li Xue"],
                "focus_metric": "conversion_rate",
            },
            "repeats": 1,
            "variants": [
                {
                    "variant_id": "control",
                    "label": "Control",
                    "model_name": "qwen-plus",
                    "prompt_profile_id": "balanced-v1",
                    "routing_policy_id": "balanced-router-v1",
                },
                {
                    "variant_id": "challenger",
                    "label": "Challenger",
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

    listing = client.get("/api/batches").json()
    assert any(item["id"] == body["id"] for item in listing)


def test_feedback_review_creates_feedback_sample_and_feedback_dataset() -> None:
    login_as("operator", "operator123")
    created = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "support_triage",
            "input_payload": {
                "tickets": [{"customer": "Example Client", "message": "Production API errors are blocking a release. Need help today."}]
            },
            "model_name_override": "qwen-turbo",
            "prompt_profile_id": "balanced-v1",
            "routing_policy_id": "strict-review-v1",
        },
    ).json()

    login_as("reviewer", "reviewer123")
    review_response = client.post(
        f"/api/workflows/{created['id']}/review",
        json={"approve": True, "comment": "Reviewed by human. Keep escalation and incident wording."},
    )
    assert review_response.status_code == 200

    feedback_samples = client.get("/api/feedback-samples").json()
    assert any(item["source_run_id"] == created["id"] for item in feedback_samples)

    catalog = client.get("/api/experiments/catalog").json()
    assert any(item["dataset_id"] == "feedback-loop-v1" for item in catalog["datasets"])
