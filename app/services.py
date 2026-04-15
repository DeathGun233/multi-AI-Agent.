from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.data import RISK_CUSTOMERS, SALES_DATA, WORKFLOW_TEMPLATES
from app.external_data import ExternalDataError, ExternalDataService
from app.llm import LLMService
from app.models import (
    AnalystOutput,
    BatchExperimentRequest,
    BatchExperimentRun,
    ContentOutput,
    EvaluationRun,
    ExecutionProfile,
    FeedbackSample,
    PromptProfile,
    PromptProfileForm,
    ReviewDecision,
    ReviewOutput,
    ReviewSubmission,
    RunStatus,
    ToolCall,
    WorkflowPlan,
    WorkflowRequest,
    WorkflowRun,
    WorkflowTemplate,
    WorkflowType,
)
from app.prompt_catalog import (
    BUILTIN_PROMPT_PROFILES,
    DEFAULT_PROMPT_PROFILE_ID,
    EvaluationCaseDefinition,
    get_evaluation_dataset,
    list_evaluation_datasets,
    list_model_options,
    list_routing_policies,
    resolve_execution_profile,
)
from app.repository import WorkflowRepository


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowState(TypedDict, total=False):
    run: WorkflowRun
    request: WorkflowRequest
    execution_profile: ExecutionProfile
    prompt_profile: PromptProfile
    last_node: str
    next_node: str
    replan_count: int
    route_decisions: list[dict[str, Any]]
    planning_context: dict[str, Any]
    analyst_context: dict[str, Any]
    content_context: dict[str, Any]
    reviewer_context: dict[str, Any]
    raw_result: dict[str, Any]
    analysis: dict[str, Any]
    deliverables: dict[str, Any]
    review: dict[str, Any]
    persist: bool


def llm_calls_from_run(run: WorkflowRun) -> list[Any]:
    return [log.llm_call for log in run.logs if log.llm_call is not None]


def summarize_run_metrics(run: WorkflowRun) -> dict[str, float | int]:
    llm_calls = llm_calls_from_run(run)
    return {
        "cost_usd": round(sum(call.estimated_cost_usd for call in llm_calls), 6),
        "latency_ms": sum(call.latency_ms for call in llm_calls),
        "tokens": sum(call.total_tokens for call in llm_calls),
        "llm_calls": len(llm_calls),
    }


@dataclass(frozen=True)
class EvaluationDatasetRuntime:
    dataset_id: str
    name: str
    description: str
    cases: list[EvaluationCaseDefinition]


class AgentMemoryService:
    def __init__(self, repository: WorkflowRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings

    def _related_runs(self, workflow_type: WorkflowType, limit: int) -> list[WorkflowRun]:
        return [
            run for run in self.repository.list_all()
            if run.workflow_type == workflow_type and run.result
        ][:limit]

    def _related_feedback(self, workflow_type: WorkflowType, limit: int) -> list[FeedbackSample]:
        return [
            sample for sample in self.repository.list_feedback_samples()
            if sample.workflow_type == workflow_type
        ][:limit]

    @staticmethod
    def _execution_profile_id(payload: dict[str, Any]) -> str | None:
        execution_profile = payload.get("execution_profile", {})
        if not isinstance(execution_profile, dict):
            return None
        prompt_profile = execution_profile.get("prompt_profile", {})
        if not isinstance(prompt_profile, dict):
            return None
        profile_id = prompt_profile.get("profile_id")
        return str(profile_id) if profile_id else None

    def _matching_runs(
        self,
        workflow_type: WorkflowType,
        *,
        prompt_profile_id: str,
        limit: int,
    ) -> list[WorkflowRun]:
        preferred: list[WorkflowRun] = []
        fallback: list[WorkflowRun] = []
        for run in self.repository.list_all():
            if run.workflow_type != workflow_type or not run.result:
                continue
            result = run.result if isinstance(run.result, dict) else {}
            if self._execution_profile_id(result) == prompt_profile_id:
                preferred.append(run)
            else:
                fallback.append(run)
        return (preferred + fallback)[:limit]

    def _matching_feedback(
        self,
        workflow_type: WorkflowType,
        *,
        prompt_profile_id: str,
        limit: int,
    ) -> list[FeedbackSample]:
        preferred: list[FeedbackSample] = []
        fallback: list[FeedbackSample] = []
        for sample in self.repository.list_feedback_samples():
            if sample.workflow_type != workflow_type:
                continue
            if self._execution_profile_id(sample.output_snapshot) == prompt_profile_id:
                preferred.append(sample)
            else:
                fallback.append(sample)
        return (preferred + fallback)[:limit]

    @staticmethod
    def memory_hits(memory: dict[str, Any]) -> int:
        return len(memory.get("recent_runs", [])) + len(memory.get("feedback_samples", []))

    def planner_memory(self, request: WorkflowRequest, *, run_limit: int = 3, feedback_limit: int = 3) -> dict[str, Any]:
        if not self.settings.enable_runtime_memory:
            return {
                "enabled": False,
                "recent_runs": [],
                "feedback_samples": [],
                "dominant_keywords": [],
                "common_review_reasons": [],
            }
        related_runs = self._related_runs(request.workflow_type, run_limit)
        related_feedback = self._related_feedback(request.workflow_type, feedback_limit)

        keyword_counter: Counter[str] = Counter()
        review_reason_counter: Counter[str] = Counter()
        run_rows = []
        for run in related_runs:
            analysis = run.result.get("analysis", {}) if isinstance(run.result, dict) else {}
            review = run.result.get("review", {}) if isinstance(run.result, dict) else {}
            reasons = [str(item).strip() for item in review.get("reasons", []) if str(item).strip()]
            review_reason_counter.update(reasons)
            run_rows.append(
                {
                    "run_id": run.id,
                    "status": run.status.value,
                    "objective": run.objective,
                    "summary": analysis.get("summary", ""),
                    "review_reasons": reasons[:2],
                    "updated_at": run.updated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        feedback_rows = []
        for sample in related_feedback:
            keywords = [str(item).strip() for item in sample.expected_keywords if str(item).strip()]
            keyword_counter.update(keywords)
            feedback_rows.append(
                {
                    "sample_id": sample.id,
                    "expected_status": sample.expected_status.value,
                    "reviewer_name": sample.reviewer_name,
                    "comment": sample.reviewer_comment,
                    "keywords": keywords[:5],
                    "created_at": sample.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )

        return {
            "enabled": True,
            "recent_runs": run_rows,
            "feedback_samples": feedback_rows,
            "dominant_keywords": [item for item, _ in keyword_counter.most_common(6)],
            "common_review_reasons": [item for item, _ in review_reason_counter.most_common(4)],
        }

    def analyst_memory(
        self,
        request: WorkflowRequest,
        *,
        prompt_profile: PromptProfile,
        run_limit: int = 3,
        feedback_limit: int = 2,
    ) -> dict[str, Any]:
        if not self.settings.enable_runtime_memory:
            return {
                "enabled": False,
                "recent_runs": [],
                "feedback_samples": [],
                "prompt_profile_id": prompt_profile.profile_id,
                "highlight_keywords": [],
            }
        related_runs = self._matching_runs(
            request.workflow_type,
            prompt_profile_id=prompt_profile.profile_id,
            limit=run_limit,
        )
        related_feedback = self._matching_feedback(
            request.workflow_type,
            prompt_profile_id=prompt_profile.profile_id,
            limit=feedback_limit,
        )
        keyword_counter: Counter[str] = Counter()
        run_rows = []
        for run in related_runs:
            result = run.result if isinstance(run.result, dict) else {}
            analysis = result.get("analysis", {}) if isinstance(result.get("analysis", {}), dict) else {}
            review = result.get("review", {}) if isinstance(result.get("review", {}), dict) else {}
            run_rows.append(
                {
                    "run_id": run.id,
                    "status": run.status.value,
                    "summary": analysis.get("summary", ""),
                    "insights": [str(item) for item in analysis.get("insights", [])[:3]],
                    "review_reasons": [str(item) for item in review.get("reasons", [])[:2]],
                    "prompt_profile_id": self._execution_profile_id(result),
                    "updated_at": run.updated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        feedback_rows = []
        for sample in related_feedback:
            keywords = [str(item).strip() for item in sample.expected_keywords if str(item).strip()]
            keyword_counter.update(keywords)
            feedback_rows.append(
                {
                    "sample_id": sample.id,
                    "comment": sample.reviewer_comment,
                    "keywords": keywords[:5],
                    "expected_status": sample.expected_status.value,
                    "prompt_profile_id": self._execution_profile_id(sample.output_snapshot),
                }
            )
        return {
            "enabled": True,
            "prompt_profile_id": prompt_profile.profile_id,
            "recent_runs": run_rows,
            "feedback_samples": feedback_rows,
            "highlight_keywords": [item for item, _ in keyword_counter.most_common(5)],
        }

    def content_memory(
        self,
        request: WorkflowRequest,
        *,
        prompt_profile: PromptProfile,
        run_limit: int = 3,
        feedback_limit: int = 2,
    ) -> dict[str, Any]:
        if not self.settings.enable_runtime_memory:
            return {
                "enabled": False,
                "recent_runs": [],
                "feedback_samples": [],
                "prompt_profile_id": prompt_profile.profile_id,
                "common_output_keys": [],
            }
        related_runs = self._matching_runs(
            request.workflow_type,
            prompt_profile_id=prompt_profile.profile_id,
            limit=run_limit,
        )
        related_feedback = self._matching_feedback(
            request.workflow_type,
            prompt_profile_id=prompt_profile.profile_id,
            limit=feedback_limit,
        )
        output_key_counter: Counter[str] = Counter()
        run_rows = []
        for run in related_runs:
            result = run.result if isinstance(run.result, dict) else {}
            deliverables = result.get("deliverables", {}) if isinstance(result.get("deliverables", {}), dict) else {}
            review = result.get("review", {}) if isinstance(result.get("review", {}), dict) else {}
            output_keys = [str(key) for key in deliverables.keys()]
            output_key_counter.update(output_keys)
            run_rows.append(
                {
                    "run_id": run.id,
                    "status": run.status.value,
                    "deliverable_keys": output_keys[:6],
                    "review_reasons": [str(item) for item in review.get("reasons", [])[:2]],
                    "prompt_profile_id": self._execution_profile_id(result),
                    "updated_at": run.updated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        feedback_rows = []
        for sample in related_feedback:
            feedback_rows.append(
                {
                    "sample_id": sample.id,
                    "comment": sample.reviewer_comment,
                    "keywords": [str(item) for item in sample.expected_keywords[:5]],
                    "expected_status": sample.expected_status.value,
                    "prompt_profile_id": self._execution_profile_id(sample.output_snapshot),
                }
            )
        return {
            "enabled": True,
            "prompt_profile_id": prompt_profile.profile_id,
            "recent_runs": run_rows,
            "feedback_samples": feedback_rows,
            "common_output_keys": [item for item, _ in output_key_counter.most_common(6)],
        }

    def reviewer_memory(
        self,
        request: WorkflowRequest,
        *,
        prompt_profile: PromptProfile,
        run_limit: int = 3,
        feedback_limit: int = 3,
    ) -> dict[str, Any]:
        if not self.settings.enable_runtime_memory:
            return {
                "enabled": False,
                "recent_runs": [],
                "feedback_samples": [],
                "prompt_profile_id": prompt_profile.profile_id,
                "common_expected_statuses": [],
            }
        related_runs = self._matching_runs(
            request.workflow_type,
            prompt_profile_id=prompt_profile.profile_id,
            limit=run_limit,
        )
        related_feedback = self._matching_feedback(
            request.workflow_type,
            prompt_profile_id=prompt_profile.profile_id,
            limit=feedback_limit,
        )
        status_counter: Counter[str] = Counter()
        run_rows = []
        for run in related_runs:
            result = run.result if isinstance(run.result, dict) else {}
            review = result.get("review", {}) if isinstance(result.get("review", {}), dict) else {}
            status = str(review.get("status", run.status.value))
            status_counter.update([status])
            run_rows.append(
                {
                    "run_id": run.id,
                    "status": run.status.value,
                    "review_status": status,
                    "review_score": review.get("score"),
                    "review_reasons": [str(item) for item in review.get("reasons", [])[:3]],
                    "prompt_profile_id": self._execution_profile_id(result),
                    "updated_at": run.updated_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        feedback_rows = []
        for sample in related_feedback:
            status_counter.update([sample.expected_status.value])
            feedback_rows.append(
                {
                    "sample_id": sample.id,
                    "comment": sample.reviewer_comment,
                    "keywords": [str(item) for item in sample.expected_keywords[:5]],
                    "expected_status": sample.expected_status.value,
                    "review_score": sample.review_score,
                    "prompt_profile_id": self._execution_profile_id(sample.output_snapshot),
                }
            )
        return {
            "enabled": True,
            "prompt_profile_id": prompt_profile.profile_id,
            "recent_runs": run_rows,
            "feedback_samples": feedback_rows,
            "common_expected_statuses": [item for item, _ in status_counter.most_common(4)],
        }


class PlanningContextTool:
    def __init__(self, memory_service: AgentMemoryService) -> None:
        self.memory_service = memory_service

    def run(self, request: WorkflowRequest) -> tuple[dict[str, Any], ToolCall]:
        workflow_template = next(
            (item for item in WORKFLOW_TEMPLATES if item.workflow_type == request.workflow_type),
            None,
        )
        memory = self.memory_service.planner_memory(request)
        context = {
            "workflow_template": {
                "workflow_type": request.workflow_type.value,
                "title": workflow_template.title if workflow_template else request.workflow_type.value,
                "description": workflow_template.description if workflow_template else "",
                "sample_payload": workflow_template.sample_payload if workflow_template else {},
            },
            "input_payload_keys": sorted(request.input_payload.keys()),
            "memory": memory,
        }
        return context, ToolCall(
            name="planning_context_tool",
            input=request.model_dump(mode="json"),
            output=context,
        )


class PromptProfileService:
    def __init__(self, repository: WorkflowRepository) -> None:
        self.repository = repository
        self.repository.ensure_prompt_profiles(list(BUILTIN_PROMPT_PROFILES))

    def list_profiles(self, include_inactive: bool = False) -> list[PromptProfile]:
        return self.repository.list_prompt_profiles(include_inactive=include_inactive)

    def get_profile(self, profile_id: str | None) -> PromptProfile:
        resolved = self.repository.get_prompt_profile(profile_id or DEFAULT_PROMPT_PROFILE_ID)
        if resolved is not None:
            return resolved
        fallback = self.repository.get_prompt_profile(DEFAULT_PROMPT_PROFILE_ID)
        if fallback is None:
            raise ValueError("prompt profile not found")
        return fallback

    def create_profile(self, form: PromptProfileForm) -> PromptProfile:
        if self.repository.get_prompt_profile(form.profile_id):
            raise ValueError("prompt profile id already exists")
        profile = PromptProfile(
            profile_id=form.profile_id,
            base_profile_id=form.base_profile_id,
            name=form.name,
            version=form.version,
            description=form.description,
            analyst_instruction=form.analyst_instruction,
            content_instruction=form.content_instruction,
            reviewer_instruction=form.reviewer_instruction,
            is_builtin=False,
            is_active=True,
        )
        return self.repository.save_prompt_profile(profile)

    def update_profile(self, profile_id: str, form: PromptProfileForm) -> PromptProfile:
        existing = self.repository.get_prompt_profile(profile_id)
        if existing is None:
            raise ValueError("prompt profile not found")
        if existing.is_builtin:
            raise ValueError("builtin prompt profile cannot be edited directly")
        existing.base_profile_id = form.base_profile_id
        existing.name = form.name
        existing.version = form.version
        existing.description = form.description
        existing.analyst_instruction = form.analyst_instruction
        existing.content_instruction = form.content_instruction
        existing.reviewer_instruction = form.reviewer_instruction
        existing.updated_at = utc_now()
        return self.repository.save_prompt_profile(existing)


class PlannerAgent:
    def __init__(self, llm_service: LLMService, planning_context_tool: PlanningContextTool) -> None:
        self.llm_service = llm_service
        self.planning_context_tool = planning_context_tool

    def plan(
        self,
        *,
        request: WorkflowRequest,
        execution_profile: ExecutionProfile,
        prompt_profile: PromptProfile,
    ) -> tuple[WorkflowPlan, dict[str, Any], ToolCall, Any]:
        fallback = self._fallback_plan(request)
        planning_context, tool_call = self.planning_context_tool.run(request)
        system_prompt = (
            "你是企业 AI 工作流中的 PlannerAgent。"
            "请结合任务输入、工作流模板和历史经验，输出一个结构化执行计划。"
            "必须返回 JSON，键为 workflow_type、objective、steps、expected_outputs。"
            "steps 需要可执行且有顺序，expected_outputs 需要与后续 agent 的产出一致。"
        )
        user_prompt = (
            f"当前 Prompt 方案：{prompt_profile.name} {prompt_profile.version}\n"
            f"方案描述：{prompt_profile.description}\n"
            f"工作流类型：{request.workflow_type.value}\n"
            f"任务输入 JSON：\n{json.dumps(request.input_payload, ensure_ascii=False)}\n"
            f"规划上下文工具结果 JSON：\n{json.dumps(planning_context, ensure_ascii=False)}\n"
            "请生成适合当前任务的规划，若历史反馈里出现风险或缺口，请在 objective 或 steps 中体现。"
        )
        response = self.llm_service.generate_json(
            route_target="planner",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback.model_dump(mode="json"),
            execution_profile=execution_profile,
            response_model=WorkflowPlan,
        )
        payload = response.payload
        payload.setdefault("workflow_type", request.workflow_type.value)
        payload.setdefault("objective", fallback.objective)
        payload.setdefault("steps", fallback.steps)
        payload.setdefault("expected_outputs", fallback.expected_outputs)
        plan = WorkflowPlan(**payload)
        return plan, planning_context, tool_call, response.call

    @staticmethod
    def _fallback_plan(request: WorkflowRequest) -> WorkflowPlan:
        title = PlannerAgent._workflow_title(request.workflow_type)
        objective = PlannerAgent._objective(request.workflow_type, request.input_payload)
        steps = {
            WorkflowType.SALES_FOLLOWUP: [
                "Aggregate funnel metrics",
                "Spot risky customers and conversion gaps",
                "Draft follow-up actions and manager notes",
                "Decide whether human review is required",
            ],
            WorkflowType.MARKETING_CAMPAIGN: [
                "Extract audience, offer, and channels",
                "Produce multi-channel launch content",
                "Add compliance hints and publishing notes",
                "Decide whether manual review is required",
            ],
            WorkflowType.SUPPORT_TRIAGE: [
                "Classify tickets and assign priority",
                "Draft replies and escalation suggestions",
                "Identify outage or complaint risk",
                "Decide whether manual handoff is required",
            ],
            WorkflowType.MEETING_MINUTES: [
                "Extract owners, deadlines, and action items",
                "Create a compact execution summary",
                "Add reminders and next steps",
                "Decide whether it can auto-complete",
            ],
        }[request.workflow_type]
        expected_outputs = {
            WorkflowType.SALES_FOLLOWUP: ["analysis", "follow_up_plan", "manager_note"],
            WorkflowType.MARKETING_CAMPAIGN: ["content_assets", "launch_advice", "review_notes"],
            WorkflowType.SUPPORT_TRIAGE: ["ticket_labels", "reply_drafts", "escalation_decision"],
            WorkflowType.MEETING_MINUTES: ["action_items", "owners", "summary_mail"],
        }[request.workflow_type]
        return WorkflowPlan(
            workflow_type=request.workflow_type,
            objective=f"{title}: {objective}",
            steps=steps,
            expected_outputs=expected_outputs,
        )

    @staticmethod
    def _workflow_title(workflow_type: WorkflowType) -> str:
        for template in WORKFLOW_TEMPLATES:
            if template.workflow_type == workflow_type:
                return template.title
        return workflow_type.value

    @staticmethod
    def _objective(workflow_type: WorkflowType, payload: dict[str, Any]) -> str:
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            return f"分析 {payload.get('period', '当前周期')} 在 {payload.get('region', '全部区域')} 的销售表现"
        if workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            return f"为 {payload.get('product_name', '目标产品')} 生成投放内容"
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            return f"处理 {len(payload.get('tickets', []))} 条客服工单"
        return f"将 {payload.get('meeting_title', '会议')} 纪要转成执行动作"


class ToolCenter:
    def __init__(self, external_data: ExternalDataService) -> None:
        self.external_data = external_data

    def run(self, workflow_type: WorkflowType, payload: dict[str, Any]) -> tuple[dict[str, Any], ToolCall]:
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            result = self._sales_analytics(payload)
            return result, ToolCall(name="sales_analytics_tool", input=payload, output=result)
        if workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            result = self._marketing_brief(payload)
            return result, ToolCall(name="marketing_brief_tool", input=payload, output=result)
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            result, tool_name = self._support_triage(payload)
            return result, ToolCall(name=tool_name, input=payload, output=result)
        result = self._meeting_extract(payload)
        return result, ToolCall(name="meeting_minutes_tool", input=payload, output=result)

    def _sales_analytics(self, payload: dict[str, Any]) -> dict[str, Any]:
        region = payload.get("region")
        reps = set(payload.get("sales_reps", []))
        rows = [
            row for row in SALES_DATA
            if (not region or row["region"] == region) and (not reps or row["rep"] in reps)
        ]
        if not rows:
            rows = SALES_DATA[:]
        lead_count = sum(item["leads"] for item in rows)
        qualified = sum(item["qualified"] for item in rows)
        deals = sum(item["deals"] for item in rows)
        avg_cycle_days = round(mean(item["avg_cycle_days"] for item in rows), 1)
        conversion_rate = round(deals / lead_count, 2) if lead_count else 0.0
        return {
            "focus_metric": payload.get("focus_metric", "conversion_rate"),
            "period": payload.get("period", "current"),
            "region": region or "all",
            "lead_count": lead_count,
            "qualified_leads": qualified,
            "deals": deals,
            "conversion_rate": conversion_rate,
            "avg_cycle_days": avg_cycle_days,
            "risk_customers": [item for item in RISK_CUSTOMERS if not reps or item["owner"] in reps],
        }

    @staticmethod
    def _marketing_brief(payload: dict[str, Any]) -> dict[str, Any]:
        channels = payload.get("channels", ["xiaohongshu"])
        key_benefits = payload.get("key_benefits", [])
        return {
            "product_name": payload.get("product_name", "目标产品"),
            "audience": payload.get("audience", "目标人群"),
            "channels": channels,
            "key_benefits": key_benefits,
            "tone": payload.get("tone", "清晰直接"),
            "compliance_risks": [
                "避免使用绝对化效果承诺",
                "发布前复核产品能力表述",
            ],
            "channel_notes": {channel: f"为 {channel} 准备一个主卖点和一个行动号召" for channel in channels},
        }

    def _support_triage(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        tickets = payload.get("tickets", [])
        data_source = payload.get("data_source")
        tool_name = "support_triage_tool"
        source_summary: dict[str, Any] = {}
        if isinstance(data_source, dict) and data_source.get("provider"):
            try:
                batch = self.external_data.load_support_tickets(data_source)
                tickets = batch.records
                tool_name = f"{batch.provider}_tool"
                source_summary = {"provider": batch.provider, **batch.summary}
            except ExternalDataError as exc:
                source_summary = {
                    "provider": str(data_source.get("provider", "")),
                    "error": str(exc),
                    "fallback_to_sample": True,
                }
        enriched = []
        for index, ticket in enumerate(tickets, start=1):
            message = f"{ticket.get('message', '')} {ticket.get('body', '')}".lower()
            severity = "medium"
            category = "general_inquiry"
            if any(keyword in message for keyword in ["error", "outage", "release", "production", "blocked", "\u62a5\u9519", "\u6545\u969c", "\u4e0a\u7ebf", "\u751f\u4ea7", "\u6062\u590d"]):
                severity = "high"
                category = "incident"
            elif any(keyword in message for keyword in ["refund", "complaint", "angry", "\u6295\u8bc9", "\u9000\u6b3e"]):
                severity = "high"
                category = "complaint"
            elif any(keyword in message for keyword in ["invoice", "contract", "billing", "\u5f00\u7968", "\u5408\u540c", "\u8d26\u5355"]):
                severity = "low"
                category = "billing"
            enriched.append(
                {
                    "ticket_id": str(ticket.get("source_id") or f"T-{index:03d}"),
                    "customer": ticket.get("customer", f"客户{index}"),
                    "category": category,
                    "priority": severity,
                    "message": ticket.get("message", ""),
                    "body": ticket.get("body", ""),
                    "source_id": ticket.get("source_id", ""),
                    "source_url": ticket.get("source_url", ""),
                }
            )
        result = {
            "tickets": enriched,
            "high_priority_count": sum(1 for item in enriched if item["priority"] == "high"),
            "requires_incident_handoff": any(item["category"] == "incident" for item in enriched),
            "data_source_summary": source_summary,
        }
        return result, tool_name

    @staticmethod
    def _meeting_extract(payload: dict[str, Any]) -> dict[str, Any]:
        notes = str(payload.get("notes", ""))
        sentences = [item.strip(" .") for item in re.split(r"\d+\.\s*", notes) if item.strip()]
        actions = []
        for index, sentence in enumerate(sentences, start=1):
            owner = sentence.split(" ", 1)[0] if sentence else f"负责人{index}"
            actions.append(
                {
                    "id": f"A-{index:03d}",
                    "owner": owner,
                    "task": sentence,
                    "deadline": "下个检查点",
                }
            )
        return {
            "meeting_title": payload.get("meeting_title", "会议"),
            "action_items": actions,
            "owner_count": len({item["owner"] for item in actions}),
            "summary_points": sentences[:3],
        }


class AnalystAgent:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    def analyze(
        self,
        *,
        request: WorkflowRequest,
        raw_result: dict[str, Any],
        memory_context: dict[str, Any],
        execution_profile: ExecutionProfile,
        prompt_profile: PromptProfile,
    ) -> tuple[dict[str, Any], Any]:
        fallback = self._fallback_analysis(request.workflow_type, raw_result)
        system_prompt = (
            "你是企业 AI 工作流中的 AnalystAgent。请严格返回 JSON，键为 summary、insights、action_plan，全部使用中文。"
        )
        user_prompt = (
            f"Prompt 方案要求：{prompt_profile.analyst_instruction}\n"
            f"工作流类型：{request.workflow_type.value}\n"
            f"原始结果 JSON：\n{json.dumps(raw_result, ensure_ascii=False)}\n"
            f"历史记忆 JSON：\n{json.dumps(memory_context, ensure_ascii=False)}"
        )
        response = self.llm_service.generate_json(
            route_target="analyst",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            execution_profile=execution_profile,
            response_model=AnalystOutput,
        )
        payload = response.payload
        payload.setdefault("summary", fallback["summary"])
        payload.setdefault("insights", fallback["insights"])
        payload.setdefault("action_plan", fallback["action_plan"])
        return payload, response.call

    @staticmethod
    def _fallback_analysis(workflow_type: WorkflowType, raw_result: dict[str, Any]) -> dict[str, Any]:
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            risk_names = "、".join(item["name"] for item in raw_result.get("risk_customers", [])) or "暂无"
            return {
                "summary": (
                    f"当前转化率为 {raw_result.get('conversion_rate', 0):.0%}，平均销售周期 {raw_result.get('avg_cycle_days', 0)} 天。"
                ),
                "insights": [
                    "整体转化表现偏弱，风险客户需要重点跟进。",
                    f"当前风险客户包括：{risk_names}。",
                ],
                "action_plan": [
                    "为风险客户制定一对一跟进计划。",
                    "与区域负责人复盘停滞商机。",
                ],
            }
        if workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            return {
                "summary": f"已为 {raw_result.get('product_name', '产品')} 准备投放内容方案。",
                "insights": [
                    "不同渠道需要统一卖点但单独设计行动号召。",
                    "发布前需要复核合规表述。",
                ],
                "action_plan": [
                    "为每个渠道明确主卖点。",
                    "上线前增加一次合规检查。",
                ],
            }
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            return {
                "summary": f"当前有 {raw_result.get('high_priority_count', 0)} 条高优先级工单需要优先处理。",
                "insights": [
                    "疑似故障类工单应优先进入人工接管。",
                    "低风险账单类问题可自动生成回复草稿。",
                ],
                "action_plan": [
                    "将故障工单升级给值班负责人。",
                    "为低风险工单发送标准回复草稿。",
                ],
            }
        return {
            "summary": f"已抽取 {len(raw_result.get('action_items', []))} 条会议行动项。",
            "insights": [
                "每个行动项都需要明确责任人和检查节点。",
                "简洁总结更利于团队快速执行。",
            ],
            "action_plan": [
                "将行动项同步给对应负责人。",
                "自动完成前复核截止时间是否清晰。",
            ],
        }


class ContentAgent:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    def generate(
        self,
        *,
        request: WorkflowRequest,
        raw_result: dict[str, Any],
        analysis: dict[str, Any],
        memory_context: dict[str, Any],
        execution_profile: ExecutionProfile,
        prompt_profile: PromptProfile,
    ) -> tuple[dict[str, Any], Any]:
        fallback = self._fallback_deliverables(request.workflow_type, raw_result, analysis)
        system_prompt = (
            "你是企业 AI 工作流中的 ContentAgent。请严格返回 JSON，键为 deliverables 和 manager_note，全部使用中文。"
        )
        user_prompt = (
            f"Prompt 方案要求：{prompt_profile.content_instruction}\n"
            f"工作流类型：{request.workflow_type.value}\n"
            f"原始结果 JSON：\n{json.dumps(raw_result, ensure_ascii=False)}\n"
            f"分析结果 JSON：\n{json.dumps(analysis, ensure_ascii=False)}"
        )
        user_prompt += f"\n鍘嗗彶璁板繂 JSON锛歕n{json.dumps(memory_context, ensure_ascii=False)}"
        response = self.llm_service.generate_json(
            route_target="content",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            execution_profile=execution_profile,
            response_model=ContentOutput,
        )
        payload = response.payload
        payload.setdefault("deliverables", fallback["deliverables"])
        payload.setdefault("manager_note", fallback["manager_note"])
        return payload, response.call

    @staticmethod
    def _fallback_deliverables(
        workflow_type: WorkflowType,
        raw_result: dict[str, Any],
        analysis: dict[str, Any],
    ) -> dict[str, Any]:
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            deliverables = {
                "日报摘要": analysis.get("summary", ""),
                "follow_up_plan": [
                    {
                        "客户": item["name"],
                        "负责人": item["owner"],
                        "下一步动作": item["next_action"],
                    }
                    for item in raw_result.get("risk_customers", [])
                ],
            }
            manager_note = "需要管理者确认风险客户跟进节奏和较长销售周期的处理策略。"
        elif workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            deliverables = {
                "渠道内容资产": {
                    channel: f"{raw_result.get('product_name', '产品')} 面向 {raw_result.get('audience', '目标用户')} 的 {channel} 投放文案"
                    for channel in raw_result.get("channels", [])
                },
                "发布检查清单": raw_result.get("compliance_risks", []),
            }
            manager_note = "排期前建议先完成一次合规复核。"
        elif workflow_type == WorkflowType.SUPPORT_TRIAGE:
            deliverables = {
                "回复草稿": [
                    {
                        "工单编号": item["ticket_id"],
                        "草稿": f"我们已收到关于{item['category']}的问题，正在为您进一步处理，请稍候。",
                    }
                    for item in raw_result.get("tickets", [])
                ]
            }
            manager_note = "高优先级故障工单应由人工负责人复核。"
        else:
            deliverables = {
                "会后摘要邮件": analysis.get("summary", ""),
                "行动项": raw_result.get("action_items", []),
            }
            manager_note = "建议将行动摘要同步给负责人并确认截止时间。"
        return {"deliverables": deliverables, "manager_note": manager_note}


class ReviewerAgent:
    def __init__(self, llm_service: LLMService) -> None:
        self.llm_service = llm_service

    def review(
        self,
        *,
        request: WorkflowRequest,
        raw_result: dict[str, Any],
        analysis: dict[str, Any],
        deliverables: dict[str, Any],
        memory_context: dict[str, Any],
        execution_profile: ExecutionProfile,
        prompt_profile: PromptProfile,
    ) -> tuple[dict[str, Any], Any]:
        fallback = self._rule_review(request.workflow_type, raw_result, analysis, deliverables)
        system_prompt = (
            "你是企业 AI 工作流中的 ReviewerAgent。请严格返回 JSON，键为 status、needs_human_review、score、reasons，全部使用中文。"
        )
        user_prompt = (
            f"Prompt 方案要求：{prompt_profile.reviewer_instruction}\n"
            f"工作流类型：{request.workflow_type.value}\n"
            f"原始结果 JSON：\n{json.dumps(raw_result, ensure_ascii=False)}\n"
            f"分析结果 JSON：\n{json.dumps(analysis, ensure_ascii=False)}\n"
            f"交付结果 JSON：\n{json.dumps(deliverables, ensure_ascii=False)}\n"
            f"历史记忆 JSON：\n{json.dumps(memory_context, ensure_ascii=False)}"
        )
        response = self.llm_service.generate_json(
            route_target="reviewer",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            execution_profile=execution_profile,
            response_model=ReviewOutput,
        )
        merged = self._merge_review(fallback, response.payload)
        return merged, response.call

    @staticmethod
    def _rule_review(
        workflow_type: WorkflowType,
        raw_result: dict[str, Any],
        analysis: dict[str, Any],
        deliverables: dict[str, Any],
    ) -> dict[str, Any]:
        reasons: list[str] = []
        needs_human_review = False
        score = 0.88
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            if raw_result.get("conversion_rate", 0) <= 0.15:
                needs_human_review = True
                reasons.append("当前转化率偏低，跟进策略需要管理者确认。")
                score = 0.65
            if raw_result.get("risk_customers"):
                reasons.append("风险客户需要明确负责人和跟进节奏。")
        elif workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            needs_human_review = True
            reasons.append("营销内容需要人工复核品牌口径和合规表达。")
            score = 0.7
        elif workflow_type == WorkflowType.SUPPORT_TRIAGE:
            if raw_result.get("requires_incident_handoff"):
                needs_human_review = True
                reasons.append("疑似故障类工单必须进入人工接管。")
                score = 0.6
        elif workflow_type == WorkflowType.MEETING_MINUTES:
            if not raw_result.get("action_items"):
                needs_human_review = True
                reasons.append("当前未抽取到清晰的行动项，需要人工确认。")
                score = 0.55

        if not reasons:
            reasons.append("结果结构完整，可直接流转执行。")
        if not analysis or not deliverables:
            needs_human_review = True
            score = min(score, 0.5)
            reasons.append("核心分析或交付结果不完整，需要人工补充。")
        status = "waiting_human" if needs_human_review else "completed"
        return {
            "status": status,
            "needs_human_review": needs_human_review,
            "score": score,
            "reasons": reasons,
        }

    @staticmethod
    def _merge_review(rule_review: dict[str, Any], model_review: dict[str, Any]) -> dict[str, Any]:
        status = model_review.get("status") or rule_review.get("status", "completed")
        needs_human_review = bool(rule_review.get("needs_human_review", False)) or bool(
            model_review.get("needs_human_review", False)
        )
        if needs_human_review:
            status = "waiting_human"
        elif status not in {"completed", "waiting_human"}:
            status = "completed"

        rule_reasons = [str(item) for item in rule_review.get("reasons", []) if str(item).strip()]
        model_reasons = [str(item) for item in model_review.get("reasons", []) if str(item).strip()]
        reasons = []
        for reason in rule_reasons + model_reasons:
            if reason not in reasons:
                reasons.append(reason)

        if status == "waiting_human":
            reasons = [
                reason for reason in reasons
                if not any(
                    text in reason.lower()
                    for text in ["proceed automatically", "auto-complete", "can proceed directly", "直接流转", "自动流转"]
                )
            ]
            if not reasons:
                reasons = ["该任务需要人工审核后才能继续执行。"]
        else:
            reasons = [
                reason for reason in reasons
                if "human review" not in reason.lower() and "handoff" not in reason.lower()
            ]
            if not reasons:
                reasons = ["结果完整且风险可控，可以直接自动流转。"]

        score = float(model_review.get("score", rule_review.get("score", 0.8)))
        if status == "waiting_human":
            score = min(score, float(rule_review.get("score", score)))
        return {
            "status": status,
            "needs_human_review": status == "waiting_human",
            "score": round(score, 2),
            "reasons": reasons,
        }


class RouterAgent:
    def decide(self, *, last_node: str, state: dict[str, Any]) -> dict[str, Any]:
        replan_count = int(state.get("replan_count", 0) or 0)
        next_node = "complete_run"
        reason = "workflow state is complete"

        if last_node == "planner":
            next_node = "operator"
            reason = "plan is ready; collect or transform source data"
        elif last_node == "operator":
            next_node = "analyst"
            reason = "raw tool result is available for analysis"
        elif last_node == "analyst":
            next_node = "content"
            reason = "analysis is available for deliverable generation"
        elif last_node == "content":
            deliverables = state.get("deliverables", {})
            if self._missing_deliverables(deliverables) and replan_count < 1:
                next_node = "planner"
                replan_count += 1
                reason = "deliverables are missing; request one replanning pass"
            else:
                next_node = "reviewer"
                reason = "deliverables are ready for review"
        elif last_node == "reviewer":
            review = state.get("review", {})
            if isinstance(review, dict) and review.get("status") == "waiting_human":
                next_node = "handoff_run"
                reason = "review requires human handoff"
            else:
                next_node = "complete_run"
                reason = "review approved automatic completion"

        return {
            "from_node": last_node,
            "next_node": next_node,
            "reason": reason,
            "replan_count": replan_count,
        }

    @staticmethod
    def _missing_deliverables(deliverables: Any) -> bool:
        if not isinstance(deliverables, dict) or not deliverables:
            return True
        nested = deliverables.get("deliverables")
        if isinstance(nested, dict):
            return not bool(nested)
        return False


class FeedbackService:
    def __init__(self, repository: WorkflowRepository) -> None:
        self.repository = repository

    def create_from_review(self, run: WorkflowRun, reviewer_name: str, comment: str) -> FeedbackSample:
        expected_keywords = self._extract_keywords(comment, run.review.reasons if run.review else [])
        enriched_snapshot = dict(run.result)
        enriched_snapshot["feedback_enrichment"] = {
            "quality_tags": self._extract_quality_tags(run, comment),
            "scoring_rubric": self._build_scoring_rubric(run),
        }
        sample = FeedbackSample(
            source_run_id=run.id,
            workflow_type=run.workflow_type,
            input_payload=run.input_payload,
            expected_status=run.status,
            reviewer_name=reviewer_name,
            reviewer_comment=comment,
            review_score=run.review.score if run.review else 0.0,
            expected_keywords=expected_keywords,
            output_snapshot=enriched_snapshot,
        )
        return self.repository.save_feedback_sample(sample)

    def list_samples(self) -> list[FeedbackSample]:
        return self.repository.list_feedback_samples()

    @staticmethod
    def _extract_keywords(comment: str, reasons: list[str]) -> list[str]:
        text = " ".join([comment] + reasons)
        chinese_tokens = re.findall(r"[\u4e00-\u9fff]{2,6}", text)
        english_tokens = re.findall(r"[a-z]{4,}", text.lower())
        seen: list[str] = []
        for token in chinese_tokens + english_tokens:
            cleaned = token.strip()
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        if seen:
            return seen[:8]
        return ["人工审核", "质量复核", "执行清晰"]

    @staticmethod
    def _extract_quality_tags(run: WorkflowRun, comment: str) -> list[str]:
        tags = [run.workflow_type.value]
        text = f"{comment} {' '.join(run.review.reasons if run.review else [])}"
        if "风险" in text or "incident" in text.lower():
            tags.append("风险控制")
        if "负责人" in text or "owner" in text.lower():
            tags.append("责任清晰")
        if "合规" in text:
            tags.append("合规复核")
        if run.status == RunStatus.COMPLETED:
            tags.append("可执行")
        else:
            tags.append("需人工判断")
        return tags

    @staticmethod
    def _build_scoring_rubric(run: WorkflowRun) -> dict[str, str | int]:
        return {
            "clarity_weight": 35,
            "completeness_weight": 35,
            "risk_control_weight": 30,
            "workflow_type": run.workflow_type.value,
            "guide": "重点检查输出是否清晰、完整、可执行，并根据场景确认风险是否被正确识别。",
        }


class WorkflowEngine:
    def __init__(self, repository: WorkflowRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings
        self.prompt_profiles = PromptProfileService(repository)
        self.memory_service = AgentMemoryService(repository, settings)
        self.planning_context_tool = PlanningContextTool(self.memory_service)
        self.external_data = ExternalDataService(settings)
        self.tool_center = ToolCenter(self.external_data)
        self.llm_service = LLMService(settings)
        self.planner_agent = PlannerAgent(self.llm_service, self.planning_context_tool)
        self.analyst_agent = AnalystAgent(self.llm_service)
        self.content_agent = ContentAgent(self.llm_service)
        self.reviewer_agent = ReviewerAgent(self.llm_service)
        self.router_agent = RouterAgent()
        self.feedback_service = FeedbackService(repository)
        self.graph = self._build_graph()

    def list_templates(self) -> list[WorkflowTemplate]:
        return WORKFLOW_TEMPLATES

    def list_prompt_profiles(self, include_inactive: bool = False) -> list[PromptProfile]:
        return self.prompt_profiles.list_profiles(include_inactive=include_inactive)

    def list_model_options(self) -> list[dict[str, Any]]:
        return list_model_options()

    def list_routing_policies(self) -> list[dict[str, str]]:
        return list_routing_policies()

    def get(self, run_id: str) -> WorkflowRun | None:
        return self.repository.get(run_id)

    def list_runs(self) -> list[WorkflowRun]:
        return self.repository.list_all()

    def list_waiting_human(self) -> list[WorkflowRun]:
        return self.repository.list_waiting_human()

    def delete_run(self, run_id: str) -> bool:
        return self.repository.delete_run(run_id)

    def delete_runs(self, run_ids: list[str]) -> list[str]:
        return self.repository.delete_runs(run_ids)

    def create_prompt_profile(self, form: PromptProfileForm) -> PromptProfile:
        return self.prompt_profiles.create_profile(form)

    def update_prompt_profile(self, profile_id: str, form: PromptProfileForm) -> PromptProfile:
        return self.prompt_profiles.update_profile(profile_id, form)

    def run_workflow(self, request: WorkflowRequest, *, persist: bool = True) -> WorkflowRun:
        prompt_profile = self.prompt_profiles.get_profile(request.prompt_profile_id)
        execution_profile = resolve_execution_profile(
            default_model_name=self.settings.model_name,
            prompt_profile=prompt_profile,
            model_name_override=request.model_name_override,
            routing_policy_id=request.routing_policy_id,
        )
        run = WorkflowRun(
            workflow_type=request.workflow_type,
            input_payload=request.input_payload,
        )
        initial_state: WorkflowState = {
            "run": run,
            "request": request,
            "execution_profile": execution_profile,
            "prompt_profile": prompt_profile,
            "replan_count": 0,
            "route_decisions": [],
            "persist": persist,
        }
        try:
            final_state = self.graph.invoke(initial_state)
            final_run = final_state["run"]
            if persist:
                self.repository.save(final_run)
            return final_run
        except Exception as exc:
            run.touch(status=RunStatus.FAILED, current_step="failed")
            run.add_log("System", f"Workflow failed: {type(exc).__name__}: {exc}")
            if persist:
                self.repository.save(run)
            raise

    def submit_review(self, run_id: str, submission: ReviewSubmission, reviewer_name: str) -> WorkflowRun:
        run = self.repository.get(run_id)
        if run is None:
            raise ValueError("workflow run not found")
        if run.status != RunStatus.WAITING_HUMAN:
            raise ValueError("workflow run is not waiting for human review")
        run.touch(
            status=RunStatus.COMPLETED if submission.approve else RunStatus.FAILED,
            current_step="human_review",
        )
        run.add_log(
            "HumanReviewer",
            f"{reviewer_name}{'已通过' if submission.approve else '已驳回'}该任务。备注：{submission.comment or '无'}",
        )
        if run.review:
            run.review.status = run.status
            run.review.needs_human_review = False
            if submission.comment:
                run.review.reasons = [submission.comment]
        self.repository.save(run)
        self.feedback_service.create_from_review(run, reviewer_name, submission.comment)
        return run

    @staticmethod
    def graph_shape() -> dict[str, Any]:
        return {
            "runtime": "langgraph",
            "nodes": ["planner", "operator", "analyst", "content", "reviewer", "router", "complete_run", "handoff_run"],
            "edges": [
                {"from": "start", "to": "planner"},
                {"from": "planner", "to": "router"},
                {"from": "operator", "to": "router"},
                {"from": "analyst", "to": "router"},
                {"from": "content", "to": "router"},
                {"from": "reviewer", "to": "router"},
                {"from": "router", "to": "operator"},
                {"from": "router", "to": "analyst"},
                {"from": "router", "to": "content"},
                {"from": "router", "to": "planner"},
                {"from": "router", "to": "reviewer"},
                {"from": "router", "to": "complete_run"},
                {"from": "router", "to": "handoff_run"},
            ],
        }

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("planner", self._planner_step)
        graph.add_node("operator", self._operator_step)
        graph.add_node("analyst", self._analyst_step)
        graph.add_node("content", self._content_step)
        graph.add_node("reviewer", self._reviewer_step)
        graph.add_node("router", self._router_step)
        graph.add_node("complete_run", self._complete_step)
        graph.add_node("handoff_run", self._handoff_step)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "router")
        graph.add_edge("operator", "router")
        graph.add_edge("analyst", "router")
        graph.add_edge("content", "router")
        graph.add_edge("reviewer", "router")
        graph.add_conditional_edges(
            "router",
            self._route_from_router,
            {
                "planner": "planner",
                "operator": "operator",
                "analyst": "analyst",
                "content": "content",
                "reviewer": "reviewer",
                "complete_run": "complete_run",
                "handoff_run": "handoff_run",
            },
        )
        graph.add_edge("complete_run", END)
        graph.add_edge("handoff_run", END)
        return graph.compile()

    def _planner_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        request = state["request"]
        run.touch(status=RunStatus.PLANNING, current_step="planner")
        plan, planning_context, tool_call, llm_call = self.planner_agent.plan(
            request=request,
            execution_profile=state["execution_profile"],
            prompt_profile=state["prompt_profile"],
        )
        run.plan = plan
        run.objective = plan.objective
        state["planning_context"] = planning_context
        run.add_log(
            "PlannerAgent",
            f"已生成动态执行计划，共 {len(plan.steps)} 步，并注入规划上下文与历史记忆。",
            tool_call=tool_call,
            llm_call=llm_call,
        )
        state["last_node"] = "planner"
        return state

    def _operator_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        request = state["request"]
        run.touch(status=RunStatus.EXECUTING, current_step="operator")
        raw_result, tool_call = self.tool_center.run(request.workflow_type, request.input_payload)
        run.add_log("OperatorAgent", f"已完成工具调用：{tool_call.name}。", tool_call=tool_call)
        state["raw_result"] = raw_result
        state["last_node"] = "operator"
        return state

    def _analyst_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        request = state["request"]
        run.touch(status=RunStatus.EXECUTING, current_step="analyst")
        analyst_memory = self.memory_service.analyst_memory(
            request,
            prompt_profile=state["prompt_profile"],
        )
        analyst_context = {
            "memory": analyst_memory,
            "memory_hits": self.memory_service.memory_hits(analyst_memory),
        }
        analysis, llm_call = self.analyst_agent.analyze(
            request=request,
            raw_result=state["raw_result"],
            memory_context=analyst_context,
            execution_profile=state["execution_profile"],
            prompt_profile=state["prompt_profile"],
        )
        run.add_log(
            "AnalystAgent",
            f"已完成结果分析与行动建议整理，并注入 {analyst_context['memory_hits']} 条历史记忆。",
            llm_call=llm_call,
        )
        state["analysis"] = analysis
        state["analyst_context"] = analyst_context
        state["last_node"] = "analyst"
        return state

    def _content_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        request = state["request"]
        run.touch(status=RunStatus.EXECUTING, current_step="content")
        content_memory = self.memory_service.content_memory(
            request,
            prompt_profile=state["prompt_profile"],
        )
        content_context = {
            "memory": content_memory,
            "memory_hits": self.memory_service.memory_hits(content_memory),
        }
        deliverables, llm_call = self.content_agent.generate(
            request=request,
            raw_result=state["raw_result"],
            analysis=state["analysis"],
            memory_context=content_context,
            execution_profile=state["execution_profile"],
            prompt_profile=state["prompt_profile"],
        )
        run.add_log("ContentAgent", "已补充可直接使用的业务输出。", llm_call=llm_call)
        run.add_log(
            "ContentAgent",
            f"已注入 {content_context['memory_hits']} 条历史记忆，用于生成业务输出。",
        )
        state["deliverables"] = deliverables
        state["content_context"] = content_context
        state["last_node"] = "content"
        return state

    def _reviewer_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        request = state["request"]
        run.touch(status=RunStatus.REVIEWING, current_step="reviewer")
        reviewer_memory = self.memory_service.reviewer_memory(
            request,
            prompt_profile=state["prompt_profile"],
        )
        reviewer_context = {
            "memory": reviewer_memory,
            "memory_hits": self.memory_service.memory_hits(reviewer_memory),
        }
        review_payload, llm_call = self.reviewer_agent.review(
            request=request,
            raw_result=state["raw_result"],
            analysis=state["analysis"],
            deliverables=state["deliverables"],
            memory_context=reviewer_context,
            execution_profile=state["execution_profile"],
            prompt_profile=state["prompt_profile"],
        )
        review = ReviewDecision(**review_payload)
        run.review = review
        run.result = {
            "execution_profile": state["execution_profile"].model_dump(mode="json"),
            "planning_context": state.get("planning_context", {}),
            "analyst_context": state.get("analyst_context", {}),
            "content_context": state.get("content_context", {}),
            "reviewer_context": reviewer_context,
            "route_decisions": state.get("route_decisions", []),
            "raw_result": state["raw_result"],
            "analysis": state["analysis"],
            "deliverables": state["deliverables"],
            "review": review.model_dump(mode="json"),
            "metrics": summarize_run_metrics(run),
        }
        run.add_log(
            "ReviewerAgent",
            f"已完成审核判断，并注入 {reviewer_context['memory_hits']} 条历史记忆。",
            llm_call=llm_call,
        )
        state["review"] = review_payload
        state["reviewer_context"] = reviewer_context
        state["last_node"] = "reviewer"
        return state

    def _router_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        last_node = state.get("last_node", "planner")
        decision = self.router_agent.decide(last_node=last_node, state=dict(state))
        state["next_node"] = str(decision["next_node"])
        state["replan_count"] = int(decision["replan_count"])
        route_decisions = list(state.get("route_decisions", []))
        route_decisions.append(decision)
        state["route_decisions"] = route_decisions
        run.add_log(
            "RouterAgent",
            f"从 {decision['from_node']} 路由到 {decision['next_node']}：{decision['reason']}",
        )
        return state

    @staticmethod
    def _route_from_router(state: WorkflowState) -> str:
        return state.get("next_node", "complete_run")

    def _complete_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        run.touch(status=RunStatus.COMPLETED, current_step="completed")
        if run.review:
            run.review.status = RunStatus.COMPLETED
            run.review.needs_human_review = False
        run.add_log("System", "工作流已自动完成。")
        run.result["route_decisions"] = state.get("route_decisions", [])
        run.result["metrics"] = summarize_run_metrics(run)
        return state

    def _handoff_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        run.touch(status=RunStatus.WAITING_HUMAN, current_step="waiting_human")
        if run.review:
            run.review.status = RunStatus.WAITING_HUMAN
            run.review.needs_human_review = True
        run.add_log("System", "工作流已进入人工审核。")
        run.result["route_decisions"] = state.get("route_decisions", [])
        run.result["metrics"] = summarize_run_metrics(run)
        return state


class EvaluationService:
    def __init__(self, repository: WorkflowRepository, engine: WorkflowEngine) -> None:
        self.repository = repository
        self.engine = engine

    def list_datasets(self) -> list[dict[str, str]]:
        base = list_evaluation_datasets()
        samples = self.repository.list_feedback_samples()
        if samples:
            base.append(
                {
                    "dataset_id": "feedback-loop-v1",
                    "name": "人工反馈闭环集",
                    "description": "由人工审核结果自动沉淀生成的样本集。",
                    "case_count": str(len(samples)),
                }
            )
        return base

    def list_runs(self) -> list[EvaluationRun]:
        return self.repository.list_evaluations()

    def run_evaluation(
        self,
        *,
        dataset_id: str,
        candidate_model_name: str,
        candidate_prompt_profile_id: str,
        candidate_routing_policy_id: str,
        baseline_model_name: str,
        baseline_prompt_profile_id: str,
        baseline_routing_policy_id: str,
    ) -> EvaluationRun:
        dataset = self._resolve_dataset(dataset_id)
        candidate_profile = resolve_execution_profile(
            default_model_name=self.engine.settings.model_name,
            prompt_profile=self.engine.prompt_profiles.get_profile(candidate_prompt_profile_id),
            model_name_override=candidate_model_name,
            routing_policy_id=candidate_routing_policy_id,
        )
        baseline_profile = resolve_execution_profile(
            default_model_name=self.engine.settings.model_name,
            prompt_profile=self.engine.prompt_profiles.get_profile(baseline_prompt_profile_id),
            model_name_override=baseline_model_name,
            routing_policy_id=baseline_routing_policy_id,
        )
        case_results = []
        candidate_scores: list[float] = []
        baseline_scores: list[float] = []
        candidate_dimension_rollups: dict[str, list[float]] = {}
        baseline_dimension_rollups: dict[str, list[float]] = {}
        for case in dataset.cases:
            candidate_run = self.engine.run_workflow(
                WorkflowRequest(
                    workflow_type=WorkflowType(case.workflow_type),
                    input_payload=case.input_payload,
                    model_name_override=candidate_model_name,
                    prompt_profile_id=candidate_prompt_profile_id,
                    routing_policy_id=candidate_routing_policy_id,
                ),
                persist=False,
            )
            baseline_run = self.engine.run_workflow(
                WorkflowRequest(
                    workflow_type=WorkflowType(case.workflow_type),
                    input_payload=case.input_payload,
                    model_name_override=baseline_model_name,
                    prompt_profile_id=baseline_prompt_profile_id,
                    routing_policy_id=baseline_routing_policy_id,
                ),
                persist=False,
            )
            candidate_dimensions = self._score_dimensions(candidate_run, case)
            baseline_dimensions = self._score_dimensions(baseline_run, case)
            candidate_score = round(mean(candidate_dimensions.values()), 3)
            baseline_score = round(mean(baseline_dimensions.values()), 3)
            candidate_scores.append(candidate_score)
            baseline_scores.append(baseline_score)
            for key, value in candidate_dimensions.items():
                candidate_dimension_rollups.setdefault(key, []).append(value)
            for key, value in baseline_dimensions.items():
                baseline_dimension_rollups.setdefault(key, []).append(value)
            case_results.append(
                {
                    "case_id": case.case_id,
                    "title": case.title,
                    "candidate_status": candidate_run.status.value,
                    "baseline_status": baseline_run.status.value,
                    "candidate_score": candidate_score,
                    "baseline_score": baseline_score,
                    "candidate_dimensions": candidate_dimensions,
                    "baseline_dimensions": baseline_dimensions,
                    "candidate_metrics": summarize_run_metrics(candidate_run),
                    "baseline_metrics": summarize_run_metrics(baseline_run),
                }
            )
        evaluation_run = EvaluationRun(
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.name,
            candidate_profile=candidate_profile,
            baseline_profile=baseline_profile,
            summary={
                "case_count": len(dataset.cases),
                "candidate_avg_score": round(mean(candidate_scores), 3) if candidate_scores else 0.0,
                "baseline_avg_score": round(mean(baseline_scores), 3) if baseline_scores else 0.0,
                "score_delta": round((mean(candidate_scores) - mean(baseline_scores)), 3) if candidate_scores else 0.0,
                "candidate_dimensions": {
                    key: round(mean(values), 3) for key, values in candidate_dimension_rollups.items()
                },
                "baseline_dimensions": {
                    key: round(mean(values), 3) for key, values in baseline_dimension_rollups.items()
                },
            },
            case_results=case_results,
        )
        return self.repository.save_evaluation(evaluation_run)

    def _resolve_dataset(self, dataset_id: str) -> EvaluationDatasetRuntime:
        if dataset_id == "feedback-loop-v1":
            return self._build_feedback_dataset()
        source = get_evaluation_dataset(dataset_id)
        return EvaluationDatasetRuntime(
            dataset_id=source.dataset_id,
            name=source.name,
            description=source.description,
            cases=list(source.cases),
        )

    def _build_feedback_dataset(self) -> EvaluationDatasetRuntime:
        samples = self.repository.list_feedback_samples()
        cases = [
            EvaluationCaseDefinition(
                case_id=sample.id,
                title=f"{sample.reviewer_name} 的反馈样本",
                workflow_type=sample.workflow_type.value,
                input_payload=sample.input_payload,
                expected_status=sample.expected_status.value,
                expected_keywords=tuple(sample.expected_keywords),
            )
            for sample in samples
        ]
        return EvaluationDatasetRuntime(
            dataset_id="feedback-loop-v1",
            name="人工反馈闭环集",
            description="由人工审核反馈自动构建的评测样本集。",
            cases=cases,
        )

    @staticmethod
    def _score_dimensions(run: WorkflowRun, case: EvaluationCaseDefinition) -> dict[str, float]:
        status_score = 1.0 if run.status.value == case.expected_status else 0.0
        text = json.dumps(run.result, ensure_ascii=False).lower()
        if not case.expected_keywords:
            keyword_score = 1.0
        else:
            hits = sum(1 for keyword in case.expected_keywords if keyword.lower() in text)
            keyword_score = hits / len(case.expected_keywords)
        result = run.result if isinstance(run.result, dict) else {}
        completeness_score = 1.0 if all(key in result for key in ["raw_result", "analysis", "deliverables", "review"]) else 0.0
        review = run.review
        if review is None:
            review_alignment = 0.0
        else:
            review_alignment = 1.0 if review.needs_human_review == (run.status == RunStatus.WAITING_HUMAN) else 0.5
        return {
            "status_match": round(status_score, 3),
            "keyword_coverage": round(keyword_score, 3),
            "result_completeness": round(completeness_score, 3),
            "review_alignment": round(review_alignment, 3),
        }


class BatchExperimentService:
    def __init__(self, repository: WorkflowRepository, engine: WorkflowEngine) -> None:
        self.repository = repository
        self.engine = engine

    def list_runs(self) -> list[BatchExperimentRun]:
        return self.repository.list_batch_experiments()

    def get(self, experiment_id: str) -> BatchExperimentRun | None:
        return self.repository.get_batch_experiment(experiment_id)

    def run_batch(self, request: BatchExperimentRequest) -> BatchExperimentRun:
        results: list[dict[str, Any]] = []
        variant_rollups: dict[str, list[dict[str, Any]]] = {variant.variant_id: [] for variant in request.variants}
        for variant in request.variants:
            for repeat_index in range(request.repeats):
                run = self.engine.run_workflow(
                    WorkflowRequest(
                        workflow_type=request.workflow_type,
                        input_payload=request.input_payload,
                        model_name_override=variant.model_name,
                        prompt_profile_id=variant.prompt_profile_id,
                        routing_policy_id=variant.routing_policy_id,
                        metadata={"batch_name": request.name, "variant_id": variant.variant_id, "repeat_index": repeat_index + 1},
                    ),
                    persist=False,
                )
                metrics = summarize_run_metrics(run)
                row = {
                    "variant_id": variant.variant_id,
                    "variant_label": variant.label,
                    "repeat_index": repeat_index + 1,
                    "status": run.status.value,
                    "review_score": run.review.score if run.review else 0.0,
                    "cost_usd": metrics["cost_usd"],
                    "latency_ms": metrics["latency_ms"],
                    "tokens": metrics["tokens"],
                    "needs_human_review": run.review.needs_human_review if run.review else False,
                }
                results.append(row)
                variant_rollups[variant.variant_id].append(row)

        summary_rows = []
        for variant in request.variants:
            variant_rows = variant_rollups[variant.variant_id]
            summary_rows.append(
                {
                    "variant_id": variant.variant_id,
                    "variant_label": variant.label,
                    "model_name": variant.model_name,
                    "prompt_profile_id": variant.prompt_profile_id,
                    "routing_policy_id": variant.routing_policy_id,
                    "run_count": len(variant_rows),
                    "avg_score": round(mean(row["review_score"] for row in variant_rows), 3) if variant_rows else 0.0,
                    "avg_cost_usd": round(mean(row["cost_usd"] for row in variant_rows), 6) if variant_rows else 0.0,
                    "avg_latency_ms": round(mean(row["latency_ms"] for row in variant_rows), 1) if variant_rows else 0.0,
                    "avg_tokens": round(mean(row["tokens"] for row in variant_rows), 1) if variant_rows else 0.0,
                    "handoff_rate": round(
                        sum(1 for row in variant_rows if row["needs_human_review"]) / len(variant_rows) * 100,
                        1,
                    ) if variant_rows else 0.0,
                }
            )

        summary_rows.sort(
            key=lambda row: (-row["avg_score"], row["avg_cost_usd"], row["avg_latency_ms"], row["handoff_rate"])
        )
        champion = None
        for index, row in enumerate(summary_rows, start=1):
            row["rank"] = index
            row["is_champion"] = index == 1
            row["champion_reason"] = ""
            if index == 1:
                row["champion_reason"] = "综合得分最高，且成本与时延表现更优。"
                champion = {
                    "variant_id": row["variant_id"],
                    "variant_label": row["variant_label"],
                    "avg_score": row["avg_score"],
                    "avg_cost_usd": row["avg_cost_usd"],
                    "avg_latency_ms": row["avg_latency_ms"],
                    "handoff_rate": row["handoff_rate"],
                    "reason": row["champion_reason"],
                }

        batch_run = BatchExperimentRun(
            name=request.name,
            workflow_type=request.workflow_type,
            input_payload=request.input_payload,
            variants=request.variants,
            repeats=request.repeats,
            summary={
                "variant_count": len(request.variants),
                "run_count": len(results),
                "rows": summary_rows,
                "champion": champion,
            },
            results=results,
        )
        return self.repository.save_batch_experiment(batch_run)


class CostAnalyticsService:
    def __init__(self, repository: WorkflowRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings

    def build_summary(self) -> dict[str, Any]:
        runs = self.repository.list_all()
        now = utc_now()
        monthly_runs = [
            run for run in runs
            if run.created_at.year == now.year and run.created_at.month == now.month
        ]
        month_calls = [call for run in monthly_runs for call in llm_calls_from_run(run)]
        total_cost = round(sum(call.estimated_cost_usd for call in month_calls), 6)
        total_tokens = sum(call.total_tokens for call in month_calls)
        total_latency_ms = sum(call.latency_ms for call in month_calls)
        budget = self.settings.monthly_budget_usd
        spend_ratio = round((total_cost / budget) * 100, 2) if budget else 0.0
        days_in_month = 30
        elapsed_days = max(now.day, 1)
        projected_cost = round(total_cost / elapsed_days * days_in_month, 6) if total_cost else 0.0

        alert_level = "healthy"
        alert_title = "预算运行正常"
        alert_message = "当前模型消耗低于预警阈值，可以继续推进实验。"
        if spend_ratio >= 100:
            alert_level = "danger"
            alert_title = "预算已超支"
            alert_message = "当前月度模型成本已超过预算，请立即收敛高成本实验。"
        elif spend_ratio >= 80:
            alert_level = "warning"
            alert_title = "预算接近上限"
            alert_message = "当前成本已超过 80% 预算，建议优先使用批量实验对比后再扩大运行。"
        elif spend_ratio >= 50:
            alert_level = "notice"
            alert_title = "预算进入观察区"
            alert_message = "当前成本已超过 50% 预算，建议开始关注模型成本趋势。"

        by_model: dict[str, dict[str, Any]] = {}
        for call in month_calls:
            bucket = by_model.setdefault(
                call.model_name,
                {"model_name": call.model_name, "requests": 0, "cost_usd": 0.0, "tokens": 0, "latency_ms": 0},
            )
            bucket["requests"] += 1
            bucket["cost_usd"] += call.estimated_cost_usd
            bucket["tokens"] += call.total_tokens
            bucket["latency_ms"] += call.latency_ms
        model_rows = sorted(
            [
                {
                    **row,
                    "cost_usd": round(row["cost_usd"], 6),
                    "avg_latency_ms": round(row["latency_ms"] / row["requests"], 1) if row["requests"] else 0.0,
                }
                for row in by_model.values()
            ],
            key=lambda item: item["cost_usd"],
            reverse=True,
        )

        by_day: dict[str, float] = {}
        for run in monthly_runs:
            day_key = run.created_at.astimezone(timezone.utc).strftime("%Y-%m-%d")
            by_day.setdefault(day_key, 0.0)
            by_day[day_key] += summarize_run_metrics(run)["cost_usd"]

        return {
            "month": now.strftime("%Y-%m"),
            "monthly_budget_usd": budget,
            "total_cost_usd": total_cost,
            "total_tokens": total_tokens,
            "avg_latency_ms": round(total_latency_ms / len(month_calls), 1) if month_calls else 0.0,
            "spend_ratio_percent": spend_ratio,
            "projected_month_end_cost_usd": projected_cost,
            "budget_remaining_usd": round(max(budget - total_cost, 0.0), 6),
            "model_rows": model_rows,
            "daily_cost_rows": [{"day": key, "cost_usd": round(value, 6)} for key, value in sorted(by_day.items())],
            "run_count": len(monthly_runs),
            "llm_call_count": len(month_calls),
            "alert_level": alert_level,
            "alert_title": alert_title,
            "alert_message": alert_message,
            "thresholds": [
                {"label": "观察线", "ratio_percent": 50},
                {"label": "预警线", "ratio_percent": 80},
                {"label": "超支线", "ratio_percent": 100},
            ],
        }
