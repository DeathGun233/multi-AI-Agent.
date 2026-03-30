from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select

from app.cache import CacheStore
from app.db import Database, WorkflowRunRecord
from app.models import ReviewDecision, WorkflowLog, WorkflowPlan, WorkflowRun


class WorkflowRepository:
    def __init__(self, database: Database, cache: CacheStore | None = None) -> None:
        self.database = database
        self.cache = cache
        self.database.initialize()

    def save(self, run: WorkflowRun) -> WorkflowRun:
        with self.database.session() as session:
            record = session.get(WorkflowRunRecord, run.id)
            if record is None:
                record = WorkflowRunRecord(id=run.id)
                session.add(record)
            record.workflow_type = run.workflow_type.value
            record.status = run.status.value
            record.current_step = run.current_step
            record.objective = run.objective
            record.input_payload = json.dumps(run.input_payload, ensure_ascii=False)
            record.plan_json = json.dumps(run.plan.model_dump(mode="json"), ensure_ascii=False) if run.plan else None
            record.result_json = json.dumps(run.result, ensure_ascii=False)
            record.review_json = json.dumps(run.review.model_dump(mode="json"), ensure_ascii=False) if run.review else None
            record.logs_json = json.dumps([log.model_dump(mode="json") for log in run.logs], ensure_ascii=False)
            record.created_at = run.created_at
            record.updated_at = run.updated_at
        self._cache_run(run)
        return run

    def get(self, run_id: str) -> WorkflowRun | None:
        cached = self._get_cached_run(run_id)
        if cached is not None:
            return cached
        with self.database.session() as session:
            record = session.get(WorkflowRunRecord, run_id)
            if record is None:
                return None
            run = self._deserialize(record)
        self._cache_run(run)
        return run

    def list_all(self) -> list[WorkflowRun]:
        with self.database.session() as session:
            records = session.scalars(
                select(WorkflowRunRecord).order_by(WorkflowRunRecord.created_at.desc())
            ).all()
        return [self._deserialize(record) for record in records]

    def _cache_run(self, run: WorkflowRun) -> None:
        if not self.cache:
            return
        self.cache.set_json(f"workflow_run:{run.id}", run.model_dump(mode="json"))

    def _get_cached_run(self, run_id: str) -> WorkflowRun | None:
        if not self.cache:
            return None
        payload = self.cache.get_json(f"workflow_run:{run_id}")
        return WorkflowRun(**payload) if payload else None

    @staticmethod
    def _deserialize(record: WorkflowRunRecord) -> WorkflowRun:
        plan_json = json.loads(record.plan_json) if record.plan_json else None
        review_json = json.loads(record.review_json) if record.review_json else None
        logs_json = json.loads(record.logs_json) if record.logs_json else []
        return WorkflowRun(
            id=record.id,
            workflow_type=record.workflow_type,
            status=record.status,
            current_step=record.current_step,
            objective=record.objective,
            input_payload=json.loads(record.input_payload),
            plan=WorkflowPlan(**plan_json) if plan_json else None,
            result=json.loads(record.result_json),
            review=ReviewDecision(**review_json) if review_json else None,
            logs=[WorkflowLog(**item) for item in logs_json],
            created_at=WorkflowRepository._as_datetime(record.created_at),
            updated_at=WorkflowRepository._as_datetime(record.updated_at),
        )

    @staticmethod
    def _as_datetime(value: datetime | str) -> datetime:
        if isinstance(value, datetime):
            return value
        return datetime.fromisoformat(value)
