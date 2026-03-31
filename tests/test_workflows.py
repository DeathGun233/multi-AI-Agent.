from fastapi.testclient import TestClient

from app.db import UserAccountRecord
from app.main import app, database


client = TestClient(app)


def login_as(username: str, password: str) -> None:
    response = client.post(
        "/login",
        data={
            "username": username,
            "password": password,
            "next": "/dashboard",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_seeded_users_are_persisted_with_password_hash() -> None:
    with database.session() as session:
        record = session.get(UserAccountRecord, "admin")
        assert record is not None
        assert record.display_name == "系统管理员"
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
    assert "进入工作台" in response.text

    login_as("viewer", "viewer123")
    session = client.get("/api/session")
    assert session.status_code == 200
    assert session.json()["role"] == "viewer"
    assert session.json()["capabilities"]["can_view"] is True


def test_dashboard_renders_multi_page_shell() -> None:
    login_as("operator", "operator123")
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "新建工作流" in response.text
    assert "运行历史" in response.text
    assert "仪表盘" in response.text


def test_health_endpoint_exposes_backend_shape() -> None:
    response = client.get("/api/health")
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["database_backend"] in {"sqlite", "mysql", "custom"}
    assert "redis_enabled" in body


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


def test_sales_workflow_runs() -> None:
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
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert body["result"]["raw_result"]["summary"]["lead_count"] > 0
    assert body["review"]["status"] in {"completed", "waiting_human"}
    assert any("LangGraph 状态流" in log["message"] for log in body["logs"])


def test_support_workflow_flags_human_review_and_can_be_approved() -> None:
    login_as("operator", "operator123")
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

    login_as("reviewer", "reviewer123")
    queue = client.get("/api/workflows/review-queue").json()
    assert any(item["id"] == body["id"] for item in queue)

    approved = client.post(
        f"/api/workflows/{body['id']}/review",
        json={"approve": True, "comment": "值班工程师已确认处理方案"},
    ).json()
    assert approved["status"] == "completed"
    assert any("审核负责人" in log["message"] for log in approved["logs"])


def test_detail_page_contains_graphic_timeline() -> None:
    login_as("operator", "operator123")
    created = client.post(
        "/api/workflows/run",
        json={
            "workflow_type": "meeting_minutes",
            "input_payload": {
                "meeting_title": "产品周会",
                "notes": "1. 张敏本周五前完成竞品复盘。2. 王晨今天下班前确认试点客户。",
            },
        },
    ).json()
    detail = client.get(f"/runs/{created['id']}")
    assert detail.status_code == 200
    assert "图形化执行时间线" in detail.text
    assert "执行日志" in detail.text
    assert "结果 JSON" in detail.text
