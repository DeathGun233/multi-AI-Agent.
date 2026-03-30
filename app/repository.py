from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from app.models import ReviewDecision, WorkflowLog, WorkflowPlan, WorkflowRun


class WorkflowRepository:
    def __init__(self, database_path: str) -> None:
        self.database_path = Path(database_path).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS workflow_runs (
                    id TEXT PRIMARY KEY,
                    workflow_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_step TEXT NOT NULL,
                    objective TEXT NOT NULL,
                    input_payload TEXT NOT NULL,
                    plan_json TEXT,
                    result_json TEXT NOT NULL,
                    review_json TEXT,
                    logs_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save(self, run: WorkflowRun) -> WorkflowRun:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO workflow_runs (
                    id, workflow_type, status, current_step, objective,
                    input_payload, plan_json, result_json, review_json,
                    logs_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    workflow_type=excluded.workflow_type,
                    status=excluded.status,
                    current_step=excluded.current_step,
                    objective=excluded.objective,
                    input_payload=excluded.input_payload,
                    plan_json=excluded.plan_json,
                    result_json=excluded.result_json,
                    review_json=excluded.review_json,
                    logs_json=excluded.logs_json,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at
                """,
                (
                    run.id,
                    run.workflow_type.value,
                    run.status.value,
                    run.current_step,
                    run.objective,
                    json.dumps(run.input_payload, ensure_ascii=False),
                    json.dumps(run.plan.model_dump(mode="json"), ensure_ascii=False) if run.plan else None,
                    json.dumps(run.result, ensure_ascii=False),
                    json.dumps(run.review.model_dump(mode="json"), ensure_ascii=False) if run.review else None,
                    json.dumps([log.model_dump(mode="json") for log in run.logs], ensure_ascii=False),
                    run.created_at.isoformat(),
                    run.updated_at.isoformat(),
                ),
            )
        return run

    def get(self, run_id: str) -> WorkflowRun | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM workflow_runs WHERE id = ?", (run_id,)).fetchone()
        return self._deserialize(row) if row else None

    def list_all(self) -> list[WorkflowRun]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM workflow_runs ORDER BY datetime(created_at) DESC"
            ).fetchall()
        return [self._deserialize(row) for row in rows]

    @staticmethod
    def _deserialize(row: sqlite3.Row) -> WorkflowRun:
        plan_json = json.loads(row["plan_json"]) if row["plan_json"] else None
        review_json = json.loads(row["review_json"]) if row["review_json"] else None
        logs_json = json.loads(row["logs_json"]) if row["logs_json"] else []
        return WorkflowRun(
            id=row["id"],
            workflow_type=row["workflow_type"],
            status=row["status"],
            current_step=row["current_step"],
            objective=row["objective"],
            input_payload=json.loads(row["input_payload"]),
            plan=WorkflowPlan(**plan_json) if plan_json else None,
            result=json.loads(row["result_json"]),
            review=ReviewDecision(**review_json) if review_json else None,
            logs=[WorkflowLog(**item) for item in logs_json],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
