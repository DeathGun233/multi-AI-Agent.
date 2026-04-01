from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.data import RISK_CUSTOMERS, SALES_DATA, WORKFLOW_TEMPLATES
from app.llm import LLMService
from app.models import (
    BatchExperimentRequest,
    BatchExperimentRun,
    EvaluationRun,
    ExecutionProfile,
    FeedbackSample,
    PromptProfile,
    PromptProfileForm,
    ReviewDecision,
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
    def plan(self, request: WorkflowRequest) -> WorkflowPlan:
        title = self._workflow_title(request.workflow_type)
        objective = self._objective(request.workflow_type, request.input_payload)
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
            return f"Analyze {payload.get('period', 'current period')} performance in {payload.get('region', 'all regions')}"
        if workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            return f"Launch content for {payload.get('product_name', 'target product')}"
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            return f"Process {len(payload.get('tickets', []))} support tickets"
        return f"Convert {payload.get('meeting_title', 'meeting')} notes into actions"


class ToolCenter:
    def run(self, workflow_type: WorkflowType, payload: dict[str, Any]) -> tuple[dict[str, Any], ToolCall]:
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            result = self._sales_analytics(payload)
            return result, ToolCall(name="sales_analytics_tool", input=payload, output=result)
        if workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            result = self._marketing_brief(payload)
            return result, ToolCall(name="marketing_brief_tool", input=payload, output=result)
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            result = self._support_triage(payload)
            return result, ToolCall(name="support_triage_tool", input=payload, output=result)
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
            "product_name": payload.get("product_name", "Target product"),
            "audience": payload.get("audience", "Target audience"),
            "channels": channels,
            "key_benefits": key_benefits,
            "tone": payload.get("tone", "clear and useful"),
            "compliance_risks": [
                "Avoid hard performance guarantees",
                "Review product claims before publishing",
            ],
            "channel_notes": {channel: f"Create one hero angle and one CTA for {channel}" for channel in channels},
        }

    @staticmethod
    def _support_triage(payload: dict[str, Any]) -> dict[str, Any]:
        tickets = payload.get("tickets", [])
        enriched = []
        for index, ticket in enumerate(tickets, start=1):
            message = str(ticket.get("message", "")).lower()
            severity = "medium"
            category = "general"
            if any(keyword in message for keyword in ["error", "outage", "release", "production", "blocked"]):
                severity = "high"
                category = "incident"
            elif any(keyword in message for keyword in ["refund", "complaint", "angry"]):
                severity = "high"
                category = "complaint"
            elif any(keyword in message for keyword in ["invoice", "contract", "billing"]):
                severity = "low"
                category = "billing"
            enriched.append(
                {
                    "ticket_id": f"T-{index:03d}",
                    "customer": ticket.get("customer", f"Customer {index}"),
                    "category": category,
                    "priority": severity,
                    "message": ticket.get("message", ""),
                }
            )
        return {
            "tickets": enriched,
            "high_priority_count": sum(1 for item in enriched if item["priority"] == "high"),
            "requires_incident_handoff": any(item["category"] == "incident" for item in enriched),
        }

    @staticmethod
    def _meeting_extract(payload: dict[str, Any]) -> dict[str, Any]:
        notes = str(payload.get("notes", ""))
        sentences = [item.strip(" .") for item in re.split(r"\d+\.\s*", notes) if item.strip()]
        actions = []
        for index, sentence in enumerate(sentences, start=1):
            owner = sentence.split(" ", 1)[0] if sentence else f"Owner {index}"
            actions.append(
                {
                    "id": f"A-{index:03d}",
                    "owner": owner,
                    "task": sentence,
                    "deadline": "next checkpoint",
                }
            )
        return {
            "meeting_title": payload.get("meeting_title", "Meeting"),
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
        execution_profile: ExecutionProfile,
        prompt_profile: PromptProfile,
    ) -> tuple[dict[str, Any], Any]:
        fallback = self._fallback_analysis(request.workflow_type, raw_result)
        system_prompt = (
            "You are the AnalystAgent in an enterprise AI workflow. "
            "Return strict JSON with keys: summary, insights, action_plan."
        )
        user_prompt = (
            f"Prompt profile instruction: {prompt_profile.analyst_instruction}\n"
            f"Workflow type: {request.workflow_type.value}\n"
            f"Raw result JSON:\n{json.dumps(raw_result, ensure_ascii=False)}"
        )
        response = self.llm_service.generate_json(
            route_target="analyst",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            execution_profile=execution_profile,
        )
        payload = response.payload
        payload.setdefault("summary", fallback["summary"])
        payload.setdefault("insights", fallback["insights"])
        payload.setdefault("action_plan", fallback["action_plan"])
        return payload, response.call

    @staticmethod
    def _fallback_analysis(workflow_type: WorkflowType, raw_result: dict[str, Any]) -> dict[str, Any]:
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            risk_names = ", ".join(item["name"] for item in raw_result.get("risk_customers", [])) or "none"
            return {
                "summary": (
                    f"Conversion is {raw_result.get('conversion_rate', 0):.0%} "
                    f"with average cycle {raw_result.get('avg_cycle_days', 0)} days."
                ),
                "insights": [
                    "Overall conversion is below target and risky accounts need manual follow-up.",
                    f"Risk accounts: {risk_names}.",
                ],
                "action_plan": [
                    "Create a one-to-one follow-up plan for risky accounts.",
                    "Review stalled deals with the regional manager.",
                ],
            }
        if workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            return {
                "summary": f"Prepare launch content for {raw_result.get('product_name', 'product')}.",
                "insights": [
                    "Keep channel tone consistent while adapting CTA by platform.",
                    "Review compliance language before publishing.",
                ],
                "action_plan": [
                    "Create a hero message for each channel.",
                    "Add one compliance check before release.",
                ],
            }
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            return {
                "summary": f"{raw_result.get('high_priority_count', 0)} tickets need priority handling.",
                "insights": [
                    "Incident-like tickets should trigger a human handoff.",
                    "Low-risk billing questions can auto-draft a reply.",
                ],
                "action_plan": [
                    "Escalate incident tickets to the on-call owner.",
                    "Send draft responses for low-risk tickets.",
                ],
            }
        return {
            "summary": f"Extracted {len(raw_result.get('action_items', []))} meeting action items.",
            "insights": [
                "Each action should have a clear owner and checkpoint.",
                "Short summaries help the team execute faster.",
            ],
            "action_plan": [
                "Share the action list with owners.",
                "Review deadline clarity before auto-closing the workflow.",
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
        execution_profile: ExecutionProfile,
        prompt_profile: PromptProfile,
    ) -> tuple[dict[str, Any], Any]:
        fallback = self._fallback_deliverables(request.workflow_type, raw_result, analysis)
        system_prompt = (
            "You are the ContentAgent in an enterprise AI workflow. "
            "Return strict JSON with keys: deliverables and manager_note."
        )
        user_prompt = (
            f"Prompt profile instruction: {prompt_profile.content_instruction}\n"
            f"Workflow type: {request.workflow_type.value}\n"
            f"Raw result JSON:\n{json.dumps(raw_result, ensure_ascii=False)}\n"
            f"Analysis JSON:\n{json.dumps(analysis, ensure_ascii=False)}"
        )
        response = self.llm_service.generate_json(
            route_target="content",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            execution_profile=execution_profile,
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
                "daily_brief": analysis.get("summary", ""),
                "follow_up_plan": [
                    {
                        "account": item["name"],
                        "owner": item["owner"],
                        "next_action": item["next_action"],
                    }
                    for item in raw_result.get("risk_customers", [])
                ],
            }
            manager_note = "Need manager alignment on risky accounts and longer sales cycles."
        elif workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            deliverables = {
                "channel_assets": {
                    channel: f"{raw_result.get('product_name', 'Product')} for {raw_result.get('audience', 'target users')} on {channel}"
                    for channel in raw_result.get("channels", [])
                },
                "launch_checklist": raw_result.get("compliance_risks", []),
            }
            manager_note = "Run compliance review before scheduling posts."
        elif workflow_type == WorkflowType.SUPPORT_TRIAGE:
            deliverables = {
                "reply_drafts": [
                    {
                        "ticket_id": item["ticket_id"],
                        "draft": f"We received your request about {item['category']} and are reviewing it now.",
                    }
                    for item in raw_result.get("tickets", [])
                ]
            }
            manager_note = "High priority incident tickets should be reviewed by a human owner."
        else:
            deliverables = {
                "summary_mail": analysis.get("summary", ""),
                "action_items": raw_result.get("action_items", []),
            }
            manager_note = "Send the action summary to owners and confirm deadlines."
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
        execution_profile: ExecutionProfile,
        prompt_profile: PromptProfile,
    ) -> tuple[dict[str, Any], Any]:
        fallback = self._rule_review(request.workflow_type, raw_result, analysis, deliverables)
        system_prompt = (
            "You are the ReviewerAgent in an enterprise AI workflow. "
            "Return strict JSON with keys: status, needs_human_review, score, reasons."
        )
        user_prompt = (
            f"Prompt profile instruction: {prompt_profile.reviewer_instruction}\n"
            f"Workflow type: {request.workflow_type.value}\n"
            f"Raw result JSON:\n{json.dumps(raw_result, ensure_ascii=False)}\n"
            f"Analysis JSON:\n{json.dumps(analysis, ensure_ascii=False)}\n"
            f"Deliverables JSON:\n{json.dumps(deliverables, ensure_ascii=False)}"
        )
        response = self.llm_service.generate_json(
            route_target="reviewer",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            fallback=fallback,
            execution_profile=execution_profile,
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
                reasons.append("Conversion is low and follow-up strategy should be confirmed by a manager.")
                score = 0.65
            if raw_result.get("risk_customers"):
                reasons.append("Risk accounts require explicit ownership and follow-up cadence.")
        elif workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            needs_human_review = True
            reasons.append("Marketing copy should be reviewed for brand and compliance consistency.")
            score = 0.7
        elif workflow_type == WorkflowType.SUPPORT_TRIAGE:
            if raw_result.get("requires_incident_handoff"):
                needs_human_review = True
                reasons.append("Incident-like tickets require a human handoff.")
                score = 0.6
        elif workflow_type == WorkflowType.MEETING_MINUTES:
            if not raw_result.get("action_items"):
                needs_human_review = True
                reasons.append("No clear action items were extracted.")
                score = 0.55

        if not reasons:
            reasons.append("Result is structured and can proceed automatically.")
        if not analysis or not deliverables:
            needs_human_review = True
            score = min(score, 0.5)
            reasons.append("Core analysis or deliverables are incomplete.")
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
                if not any(text in reason.lower() for text in ["proceed automatically", "auto-complete", "can proceed directly"])
            ]
            if not reasons:
                reasons = ["This run needs a human review before execution can continue."]
        else:
            reasons = [
                reason for reason in reasons
                if "human review" not in reason.lower() and "handoff" not in reason.lower()
            ]
            if not reasons:
                reasons = ["Result is complete enough to proceed automatically."]

        score = float(model_review.get("score", rule_review.get("score", 0.8)))
        if status == "waiting_human":
            score = min(score, float(rule_review.get("score", score)))
        return {
            "status": status,
            "needs_human_review": status == "waiting_human",
            "score": round(score, 2),
            "reasons": reasons,
        }


class FeedbackService:
    def __init__(self, repository: WorkflowRepository) -> None:
        self.repository = repository

    def create_from_review(self, run: WorkflowRun, reviewer_name: str, comment: str) -> FeedbackSample:
        expected_keywords = self._extract_keywords(comment, run.review.reasons if run.review else [])
        sample = FeedbackSample(
            source_run_id=run.id,
            workflow_type=run.workflow_type,
            input_payload=run.input_payload,
            expected_status=run.status,
            reviewer_name=reviewer_name,
            reviewer_comment=comment,
            review_score=run.review.score if run.review else 0.0,
            expected_keywords=expected_keywords,
            output_snapshot=run.result,
        )
        return self.repository.save_feedback_sample(sample)

    def list_samples(self) -> list[FeedbackSample]:
        return self.repository.list_feedback_samples()

    @staticmethod
    def _extract_keywords(comment: str, reasons: list[str]) -> list[str]:
        text = " ".join([comment] + reasons).lower()
        tokens = re.findall(r"[a-z]{4,}", text)
        seen: list[str] = []
        for token in tokens:
            if token not in seen:
                seen.append(token)
        if seen:
            return seen[:6]
        fallback = ["review", "quality"]
        for reason in reasons:
            if "risk" in reason.lower():
                fallback.append("risk")
        return fallback


class WorkflowEngine:
    def __init__(self, repository: WorkflowRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings
        self.prompt_profiles = PromptProfileService(repository)
        self.tool_center = ToolCenter()
        self.llm_service = LLMService(settings)
        self.planner_agent = PlannerAgent()
        self.analyst_agent = AnalystAgent(self.llm_service)
        self.content_agent = ContentAgent(self.llm_service)
        self.reviewer_agent = ReviewerAgent(self.llm_service)
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
        plan = self.planner_agent.plan(request)
        run = WorkflowRun(
            workflow_type=request.workflow_type,
            input_payload=request.input_payload,
            plan=plan,
            objective=plan.objective,
        )
        initial_state: WorkflowState = {
            "run": run,
            "request": request,
            "execution_profile": execution_profile,
            "prompt_profile": prompt_profile,
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
            f"{reviewer_name} {'approved' if submission.approve else 'rejected'} the run. Comment: {submission.comment or 'no comment'}",
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
            "nodes": ["planner", "operator", "analyst", "content", "reviewer", "complete_run", "handoff_run"],
            "edges": [
                {"from": "start", "to": "planner"},
                {"from": "planner", "to": "operator"},
                {"from": "operator", "to": "analyst"},
                {"from": "analyst", "to": "content"},
                {"from": "content", "to": "reviewer"},
                {"from": "reviewer", "to": "complete_run"},
                {"from": "reviewer", "to": "handoff_run"},
            ],
        }

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("planner", self._planner_step)
        graph.add_node("operator", self._operator_step)
        graph.add_node("analyst", self._analyst_step)
        graph.add_node("content", self._content_step)
        graph.add_node("reviewer", self._reviewer_step)
        graph.add_node("complete_run", self._complete_step)
        graph.add_node("handoff_run", self._handoff_step)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "operator")
        graph.add_edge("operator", "analyst")
        graph.add_edge("analyst", "content")
        graph.add_edge("content", "reviewer")
        graph.add_conditional_edges(
            "reviewer",
            self._route_after_review,
            {"complete_run": "complete_run", "handoff_run": "handoff_run"},
        )
        graph.add_edge("complete_run", END)
        graph.add_edge("handoff_run", END)
        return graph.compile()

    def _planner_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        run.touch(status=RunStatus.PLANNING, current_step="planner")
        run.add_log("PlannerAgent", f"Generated execution plan with {len(run.plan.steps if run.plan else [])} steps.")
        return state

    def _operator_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        request = state["request"]
        run.touch(status=RunStatus.EXECUTING, current_step="operator")
        raw_result, tool_call = self.tool_center.run(request.workflow_type, request.input_payload)
        run.add_log("OperatorAgent", f"Completed tool execution: {tool_call.name}.", tool_call=tool_call)
        state["raw_result"] = raw_result
        return state

    def _analyst_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        request = state["request"]
        run.touch(status=RunStatus.EXECUTING, current_step="analyst")
        analysis, llm_call = self.analyst_agent.analyze(
            request=request,
            raw_result=state["raw_result"],
            execution_profile=state["execution_profile"],
            prompt_profile=state["prompt_profile"],
        )
        run.add_log("AnalystAgent", "Completed analysis and action recommendations.", llm_call=llm_call)
        state["analysis"] = analysis
        return state

    def _content_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        request = state["request"]
        run.touch(status=RunStatus.EXECUTING, current_step="content")
        deliverables, llm_call = self.content_agent.generate(
            request=request,
            raw_result=state["raw_result"],
            analysis=state["analysis"],
            execution_profile=state["execution_profile"],
            prompt_profile=state["prompt_profile"],
        )
        run.add_log("ContentAgent", "Prepared business-ready deliverables.", llm_call=llm_call)
        state["deliverables"] = deliverables
        return state

    def _reviewer_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        request = state["request"]
        run.touch(status=RunStatus.REVIEWING, current_step="reviewer")
        review_payload, llm_call = self.reviewer_agent.review(
            request=request,
            raw_result=state["raw_result"],
            analysis=state["analysis"],
            deliverables=state["deliverables"],
            execution_profile=state["execution_profile"],
            prompt_profile=state["prompt_profile"],
        )
        review = ReviewDecision(**review_payload)
        run.review = review
        run.result = {
            "execution_profile": state["execution_profile"].model_dump(mode="json"),
            "raw_result": state["raw_result"],
            "analysis": state["analysis"],
            "deliverables": state["deliverables"],
            "review": review.model_dump(mode="json"),
            "metrics": summarize_run_metrics(run),
        }
        run.add_log("ReviewerAgent", "Completed review decision.", llm_call=llm_call)
        state["review"] = review_payload
        return state

    @staticmethod
    def _route_after_review(state: WorkflowState) -> str:
        review = state.get("review") or {}
        if review.get("status") == "waiting_human":
            return "handoff_run"
        return "complete_run"

    def _complete_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        run.touch(status=RunStatus.COMPLETED, current_step="completed")
        if run.review:
            run.review.status = RunStatus.COMPLETED
            run.review.needs_human_review = False
        run.add_log("System", "Workflow completed automatically.")
        run.result["metrics"] = summarize_run_metrics(run)
        return state

    def _handoff_step(self, state: WorkflowState) -> WorkflowState:
        run = state["run"]
        run.touch(status=RunStatus.WAITING_HUMAN, current_step="waiting_human")
        if run.review:
            run.review.status = RunStatus.WAITING_HUMAN
            run.review.needs_human_review = True
        run.add_log("System", "Workflow routed to human review.")
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
                    "name": "Human feedback loop",
                    "description": "Samples generated from manual reviews.",
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
            candidate_score = self._score_run(candidate_run, case)
            baseline_score = self._score_run(baseline_run, case)
            candidate_scores.append(candidate_score)
            baseline_scores.append(baseline_score)
            case_results.append(
                {
                    "case_id": case.case_id,
                    "title": case.title,
                    "candidate_status": candidate_run.status.value,
                    "baseline_status": baseline_run.status.value,
                    "candidate_score": candidate_score,
                    "baseline_score": baseline_score,
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
                title=f"Feedback sample from {sample.reviewer_name}",
                workflow_type=sample.workflow_type.value,
                input_payload=sample.input_payload,
                expected_status=sample.expected_status.value,
                expected_keywords=tuple(sample.expected_keywords),
            )
            for sample in samples
        ]
        return EvaluationDatasetRuntime(
            dataset_id="feedback-loop-v1",
            name="Human feedback loop",
            description="Evaluation cases generated from manual review feedback.",
            cases=cases,
        )

    @staticmethod
    def _score_run(run: WorkflowRun, case: EvaluationCaseDefinition) -> float:
        status_score = 1.0 if run.status.value == case.expected_status else 0.0
        text = json.dumps(run.result, ensure_ascii=False).lower()
        if not case.expected_keywords:
            keyword_score = 1.0
        else:
            hits = sum(1 for keyword in case.expected_keywords if keyword.lower() in text)
            keyword_score = hits / len(case.expected_keywords)
        return round(status_score * 0.5 + keyword_score * 0.5, 3)


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
        }
