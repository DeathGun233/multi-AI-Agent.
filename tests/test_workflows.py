from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_index_renders_console_shell() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "FlowPilot Console" in response.text
    assert "/api/workflows/review-queue" in response.text


def test_health_endpoint_exposes_backend_shape() -> None:
    response = client.get("/api/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["database_backend"] in {"sqlite", "mysql", "custom"}
    assert "redis_enabled" in body


def test_graph_endpoint_exposes_langgraph_shape() -> None:
    response = client.get("/api/workflows/graph")
    body = response.json()
    assert response.status_code == 200
    assert body["runtime"] == "langgraph"
    assert "planner" in body["nodes"]
    assert any(edge["from"] == "reviewer" for edge in body["edges"])


def test_templates_available() -> None:
    response = client.get("/api/workflows/templates")
    assert response.status_code == 200
    assert len(response.json()) == 4


def test_sales_workflow_runs() -> None:
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "sales_followup",
            "input_payload": {
                "period": "2026-W13",
                "region": "华东",
                "sales_reps": ["王晨", "李雪"],
            },
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["result"]["raw_result"]["summary"]["lead_count"] > 0
    assert body["review"]["status"] in {"completed", "waiting_human"}
    assert any("LangGraph 状态流" in log["message"] for log in body["logs"])


def test_support_workflow_flags_human_review_and_can_be_approved() -> None:
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "support_triage",
            "input_payload": {
                "tickets": [
                    {
                        "customer": "示例客户",
                        "message": "系统报错并影响上线，请尽快恢复。",
                    }
                ]
            },
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["review"]["needs_human_review"] is True
    assert body["status"] == "waiting_human"
    assert any("人工接管" in log["message"] for log in body["logs"])

    queue = client.get("/api/workflows/review-queue").json()
    assert any(item["id"] == body["id"] for item in queue)

    approved = client.post(
        f"/api/workflows/{body['id']}/review",
        json={"approve": True, "comment": "值班工程师已确认处理方案"},
    ).json()
    assert approved["status"] == "completed"
    assert any("人工审核通过" in log["message"] for log in approved["logs"])


def test_meeting_workflow_extracts_actions() -> None:
    response = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "meeting_minutes",
            "input_payload": {
                "meeting_title": "产品周会",
                "notes": "1. 张敏本周五前完成竞品复盘；2. 王晨今天下班前确认试点客户；",
            },
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert len(body["result"]["raw_result"]["actions"]) >= 2
