from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import delete, select

from app.cache import CacheStore
from app.db import (
    BatchExperimentRunRecord,
    Database,
    EvaluationRunRecord,
    FeedbackSampleRecord,
    PromptProfileRecord,
    WorkflowRunRecord,
)
from app.models import (
    BatchExperimentRun,
    BatchVariantSpec,
    EvaluationRun,
    ExecutionProfile,
    FeedbackSample,
    PromptProfile,
    ReviewDecision,
    WorkflowLog,
    WorkflowPlan,
    WorkflowRun,
)


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
            run = self._deserialize_run(record)
        self._cache_run(run)
        return run

    def list_all(self) -> list[WorkflowRun]:
        with self.database.session() as session:
            records = session.scalars(select(WorkflowRunRecord).order_by(WorkflowRunRecord.created_at.desc())).all()
        return [self._deserialize_run(record) for record in records]

    def list_waiting_human(self) -> list[WorkflowRun]:
        with self.database.session() as session:
            records = session.scalars(
                select(WorkflowRunRecord)
                .where(WorkflowRunRecord.status == "waiting_human")
                .order_by(WorkflowRunRecord.updated_at.desc())
            ).all()
        return [self._deserialize_run(record) for record in records]

    def delete_run(self, run_id: str) -> bool:
        deleted = False
        with self.database.session() as session:
            record = session.get(WorkflowRunRecord, run_id)
            if record is None:
                return False
            session.execute(
                delete(FeedbackSampleRecord).where(FeedbackSampleRecord.source_run_id == run_id)
            )
            session.delete(record)
            deleted = True
        if deleted and self.cache:
            self.cache.set_json(f"workflow_run:{run_id}", {})
        return deleted

    def delete_runs(self, run_ids: list[str]) -> list[str]:
        normalized_ids = [run_id for run_id in dict.fromkeys(run_ids) if run_id]
        if not normalized_ids:
            return []
        with self.database.session() as session:
            existing_records = session.scalars(
                select(WorkflowRunRecord).where(WorkflowRunRecord.id.in_(normalized_ids))
            ).all()
            deleted_ids = [record.id for record in existing_records]
            if not deleted_ids:
                return []
            session.execute(
                delete(FeedbackSampleRecord).where(FeedbackSampleRecord.source_run_id.in_(deleted_ids))
            )
            session.execute(
                delete(WorkflowRunRecord).where(WorkflowRunRecord.id.in_(deleted_ids))
            )
        if self.cache:
            for run_id in deleted_ids:
                self.cache.set_json(f"workflow_run:{run_id}", {})
        return deleted_ids

    def save_prompt_profile(self, prompt_profile: PromptProfile) -> PromptProfile:
        with self.database.session() as session:
            record = session.get(PromptProfileRecord, prompt_profile.profile_id)
            if record is None:
                record = PromptProfileRecord(profile_id=prompt_profile.profile_id)
                session.add(record)
                record.created_at = prompt_profile.created_at
            record.base_profile_id = prompt_profile.base_profile_id
            record.name = prompt_profile.name
            record.version = prompt_profile.version
            record.description = prompt_profile.description
            record.analyst_instruction = prompt_profile.analyst_instruction
            record.content_instruction = prompt_profile.content_instruction
            record.reviewer_instruction = prompt_profile.reviewer_instruction
            record.is_builtin = prompt_profile.is_builtin
            record.is_active = prompt_profile.is_active
            record.updated_at = prompt_profile.updated_at
        return prompt_profile

    def ensure_prompt_profiles(self, prompt_profiles: list[PromptProfile]) -> None:
        with self.database.session() as session:
            existing = {record.profile_id: record for record in session.scalars(select(PromptProfileRecord)).all()}
            for prompt_profile in prompt_profiles:
                record = existing.get(prompt_profile.profile_id)
                if record is None:
                    session.add(
                        PromptProfileRecord(
                            profile_id=prompt_profile.profile_id,
                            base_profile_id=prompt_profile.base_profile_id,
                            name=prompt_profile.name,
                            version=prompt_profile.version,
                            description=prompt_profile.description,
                            analyst_instruction=prompt_profile.analyst_instruction,
                            content_instruction=prompt_profile.content_instruction,
                            reviewer_instruction=prompt_profile.reviewer_instruction,
                            is_builtin=prompt_profile.is_builtin,
                            is_active=prompt_profile.is_active,
                            created_at=prompt_profile.created_at,
                            updated_at=prompt_profile.updated_at,
                        )
                    )
                    continue
                record.base_profile_id = prompt_profile.base_profile_id
                record.name = prompt_profile.name
                record.version = prompt_profile.version
                record.description = prompt_profile.description
                record.analyst_instruction = prompt_profile.analyst_instruction
                record.content_instruction = prompt_profile.content_instruction
                record.reviewer_instruction = prompt_profile.reviewer_instruction
                record.is_builtin = prompt_profile.is_builtin
                record.is_active = prompt_profile.is_active
                record.updated_at = prompt_profile.updated_at

    def get_prompt_profile(self, profile_id: str) -> PromptProfile | None:
        with self.database.session() as session:
            record = session.get(PromptProfileRecord, profile_id)
            return self._deserialize_prompt_profile(record) if record else None

    def list_prompt_profiles(self, include_inactive: bool = False) -> list[PromptProfile]:
        with self.database.session() as session:
            stmt = select(PromptProfileRecord).order_by(PromptProfileRecord.created_at.desc())
            if not include_inactive:
                stmt = stmt.where(PromptProfileRecord.is_active.is_(True))
            records = session.scalars(stmt).all()
        return [self._deserialize_prompt_profile(record) for record in records]

    def save_evaluation(self, evaluation_run: EvaluationRun) -> EvaluationRun:
        with self.database.session() as session:
            record = session.get(EvaluationRunRecord, evaluation_run.id)
            if record is None:
                record = EvaluationRunRecord(id=evaluation_run.id)
                session.add(record)
            record.dataset_id = evaluation_run.dataset_id
            record.dataset_name = evaluation_run.dataset_name
            record.candidate_profile_json = json.dumps(evaluation_run.candidate_profile.model_dump(mode="json"), ensure_ascii=False)
            record.baseline_profile_json = json.dumps(evaluation_run.baseline_profile.model_dump(mode="json"), ensure_ascii=False)
            record.summary_json = json.dumps(evaluation_run.summary, ensure_ascii=False)
            record.case_results_json = json.dumps(evaluation_run.case_results, ensure_ascii=False)
            record.created_at = evaluation_run.created_at
            record.updated_at = evaluation_run.updated_at
        return evaluation_run

    def get_evaluation(self, evaluation_id: str) -> EvaluationRun | None:
        with self.database.session() as session:
            record = session.get(EvaluationRunRecord, evaluation_id)
            return self._deserialize_evaluation_run(record) if record else None

    def list_evaluations(self) -> list[EvaluationRun]:
        with self.database.session() as session:
            records = session.scalars(select(EvaluationRunRecord).order_by(EvaluationRunRecord.created_at.desc())).all()
        return [self._deserialize_evaluation_run(record) for record in records]

    def save_batch_experiment(self, experiment: BatchExperimentRun) -> BatchExperimentRun:
        with self.database.session() as session:
            record = session.get(BatchExperimentRunRecord, experiment.id)
            if record is None:
                record = BatchExperimentRunRecord(id=experiment.id)
                session.add(record)
            record.name = experiment.name
            record.workflow_type = experiment.workflow_type.value
            record.input_payload_json = json.dumps(experiment.input_payload, ensure_ascii=False)
            record.variants_json = json.dumps([item.model_dump(mode="json") for item in experiment.variants], ensure_ascii=False)
            record.repeats = str(experiment.repeats)
            record.summary_json = json.dumps(experiment.summary, ensure_ascii=False)
            record.results_json = json.dumps(experiment.results, ensure_ascii=False)
            record.created_at = experiment.created_at
            record.updated_at = experiment.updated_at
        return experiment

    def get_batch_experiment(self, experiment_id: str) -> BatchExperimentRun | None:
        with self.database.session() as session:
            record = session.get(BatchExperimentRunRecord, experiment_id)
            return self._deserialize_batch_experiment(record) if record else None

    def list_batch_experiments(self) -> list[BatchExperimentRun]:
        with self.database.session() as session:
            records = session.scalars(select(BatchExperimentRunRecord).order_by(BatchExperimentRunRecord.created_at.desc())).all()
        return [self._deserialize_batch_experiment(record) for record in records]

    def save_feedback_sample(self, sample: FeedbackSample) -> FeedbackSample:
        with self.database.session() as session:
            record = session.get(FeedbackSampleRecord, sample.id)
            if record is None:
                record = FeedbackSampleRecord(id=sample.id)
                session.add(record)
            record.source_run_id = sample.source_run_id
            record.workflow_type = sample.workflow_type.value
            record.input_payload_json = json.dumps(sample.input_payload, ensure_ascii=False)
            record.expected_status = sample.expected_status.value
            record.reviewer_name = sample.reviewer_name
            record.reviewer_comment = sample.reviewer_comment
            record.review_score = str(sample.review_score)
            record.expected_keywords_json = json.dumps(sample.expected_keywords, ensure_ascii=False)
            record.output_snapshot_json = json.dumps(sample.output_snapshot, ensure_ascii=False)
            record.created_at = sample.created_at
        return sample

    def list_feedback_samples(self) -> list[FeedbackSample]:
        with self.database.session() as session:
            records = session.scalars(select(FeedbackSampleRecord).order_by(FeedbackSampleRecord.created_at.desc())).all()
        return [self._deserialize_feedback_sample(record) for record in records]

    def _cache_run(self, run: WorkflowRun) -> None:
        if self.cache:
            self.cache.set_json(f"workflow_run:{run.id}", run.model_dump(mode="json"))

    def _get_cached_run(self, run_id: str) -> WorkflowRun | None:
        if not self.cache:
            return None
        payload = self.cache.get_json(f"workflow_run:{run_id}")
        return WorkflowRun(**payload) if payload else None

    @staticmethod
    def _deserialize_run(record: WorkflowRunRecord) -> WorkflowRun:
        plan_json = json.loads(record.plan_json) if record.plan_json else None
        review_json = json.loads(record.review_json) if record.review_json else None
        logs_json = json.loads(record.logs_json) if record.logs_json else []
        for item in logs_json:
            llm_call = item.get("llm_call")
            if isinstance(llm_call, dict):
                llm_call.setdefault("route_target", "legacy")
                llm_call.setdefault("routing_policy_id", None)
                llm_call.setdefault("routing_policy_name", None)
                llm_call.setdefault("estimated_cost_usd", 0.0)
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
    def _deserialize_prompt_profile(record: PromptProfileRecord) -> PromptProfile:
        return PromptProfile(
            profile_id=record.profile_id,
            base_profile_id=record.base_profile_id,
            name=record.name,
            version=record.version,
            description=record.description,
            analyst_instruction=record.analyst_instruction,
            content_instruction=record.content_instruction,
            reviewer_instruction=record.reviewer_instruction,
            is_builtin=record.is_builtin,
            is_active=record.is_active,
            created_at=WorkflowRepository._as_datetime(record.created_at),
            updated_at=WorkflowRepository._as_datetime(record.updated_at),
        )

    @staticmethod
    def _deserialize_evaluation_run(record: EvaluationRunRecord) -> EvaluationRun:
        return EvaluationRun(
            id=record.id,
            dataset_id=record.dataset_id,
            dataset_name=record.dataset_name,
            candidate_profile=ExecutionProfile(**json.loads(record.candidate_profile_json)),
            baseline_profile=ExecutionProfile(**json.loads(record.baseline_profile_json)),
            summary=json.loads(record.summary_json),
            case_results=json.loads(record.case_results_json),
            created_at=WorkflowRepository._as_datetime(record.created_at),
            updated_at=WorkflowRepository._as_datetime(record.updated_at),
        )

    @staticmethod
    def _deserialize_batch_experiment(record: BatchExperimentRunRecord) -> BatchExperimentRun:
        return BatchExperimentRun(
            id=record.id,
            name=record.name,
            workflow_type=record.workflow_type,
            input_payload=json.loads(record.input_payload_json),
            variants=[BatchVariantSpec(**item) for item in json.loads(record.variants_json)],
            repeats=int(record.repeats),
            summary=json.loads(record.summary_json),
            results=json.loads(record.results_json),
            created_at=WorkflowRepository._as_datetime(record.created_at),
            updated_at=WorkflowRepository._as_datetime(record.updated_at),
        )

    @staticmethod
    def _deserialize_feedback_sample(record: FeedbackSampleRecord) -> FeedbackSample:
        return FeedbackSample(
            id=record.id,
            source_run_id=record.source_run_id,
            workflow_type=record.workflow_type,
            input_payload=json.loads(record.input_payload_json),
            expected_status=record.expected_status,
            reviewer_name=record.reviewer_name,
            reviewer_comment=record.reviewer_comment,
            review_score=float(record.review_score),
            expected_keywords=json.loads(record.expected_keywords_json),
            output_snapshot=json.loads(record.output_snapshot_json),
            created_at=WorkflowRepository._as_datetime(record.created_at),
        )

    @staticmethod
    def _as_datetime(value: datetime | str) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
