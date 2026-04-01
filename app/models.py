from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class WorkflowType(str, Enum):
    SALES_FOLLOWUP = "sales_followup"
    MARKETING_CAMPAIGN = "marketing_campaign"
    SUPPORT_TRIAGE = "support_triage"
    MEETING_MINUTES = "meeting_minutes"


class RunStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVIEWING = "reviewing"
    WAITING_HUMAN = "waiting_human"
    COMPLETED = "completed"
    FAILED = "failed"


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: dict[str, Any] = Field(default_factory=dict)
    success: bool = True


class PromptProfileRef(BaseModel):
    profile_id: str
    name: str
    version: str
    description: str


class RoutingPolicyRef(BaseModel):
    policy_id: str
    name: str
    description: str


class ExecutionProfile(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    primary_model_name: str
    primary_model_label: str
    prompt_profile: PromptProfileRef
    routing_policy: RoutingPolicyRef
    model_routes: dict[str, str] = Field(default_factory=dict)


class LLMCall(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    provider: str = "openai_compatible"
    model_name: str
    route_target: str
    prompt_profile_id: str | None = None
    prompt_profile_name: str | None = None
    prompt_profile_version: str | None = None
    routing_policy_id: str | None = None
    routing_policy_name: str | None = None
    system_prompt: str
    user_prompt: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: int = 0
    estimated_cost_usd: float = 0.0
    used_fallback: bool = False
    error: str | None = None


class WorkflowLog(BaseModel):
    timestamp: datetime = Field(default_factory=utc_now)
    agent: str
    message: str
    tool_call: ToolCall | None = None
    llm_call: LLMCall | None = None


class ReviewDecision(BaseModel):
    status: RunStatus
    needs_human_review: bool
    score: float
    reasons: list[str] = Field(default_factory=list)


class WorkflowRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    workflow_type: WorkflowType
    input_payload: dict[str, Any] = Field(default_factory=dict)
    model_name_override: str | None = None
    prompt_profile_id: str | None = None
    routing_policy_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewSubmission(BaseModel):
    approve: bool
    comment: str = ""


class WorkflowPlan(BaseModel):
    workflow_type: WorkflowType
    objective: str
    steps: list[str]
    expected_outputs: list[str]


class WorkflowRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    workflow_type: WorkflowType
    input_payload: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus = RunStatus.CREATED
    current_step: str = "created"
    objective: str = ""
    plan: WorkflowPlan | None = None
    result: dict[str, Any] = Field(default_factory=dict)
    review: ReviewDecision | None = None
    logs: list[WorkflowLog] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def add_log(
        self,
        agent: str,
        message: str,
        tool_call: ToolCall | None = None,
        llm_call: LLMCall | None = None,
    ) -> None:
        self.logs.append(WorkflowLog(agent=agent, message=message, tool_call=tool_call, llm_call=llm_call))
        self.updated_at = utc_now()

    def touch(self, status: RunStatus | None = None, current_step: str | None = None) -> None:
        if status is not None:
            self.status = status
        if current_step is not None:
            self.current_step = current_step
        self.updated_at = utc_now()


class WorkflowTemplate(BaseModel):
    workflow_type: WorkflowType
    title: str
    description: str
    sample_payload: dict[str, Any]


class PromptProfile(BaseModel):
    profile_id: str
    base_profile_id: str | None = None
    name: str
    version: str
    description: str
    analyst_instruction: str
    content_instruction: str
    reviewer_instruction: str
    is_builtin: bool = False
    is_active: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def as_ref(self) -> PromptProfileRef:
        return PromptProfileRef(
            profile_id=self.profile_id,
            name=self.name,
            version=self.version,
            description=self.description,
        )


class PromptProfileForm(BaseModel):
    profile_id: str
    base_profile_id: str | None = None
    name: str
    version: str
    description: str
    analyst_instruction: str
    content_instruction: str
    reviewer_instruction: str


class EvaluationRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    dataset_id: str
    dataset_name: str
    candidate_profile: ExecutionProfile
    baseline_profile: ExecutionProfile
    summary: dict[str, Any] = Field(default_factory=dict)
    case_results: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class BatchVariantSpec(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    variant_id: str
    label: str
    model_name: str
    prompt_profile_id: str
    routing_policy_id: str


class BatchExperimentRequest(BaseModel):
    name: str
    workflow_type: WorkflowType
    input_payload: dict[str, Any]
    variants: list[BatchVariantSpec]
    repeats: int = 1
    metadata: dict[str, Any] = Field(default_factory=dict)


class BatchExperimentRun(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    workflow_type: WorkflowType
    input_payload: dict[str, Any]
    variants: list[BatchVariantSpec] = Field(default_factory=list)
    repeats: int = 1
    summary: dict[str, Any] = Field(default_factory=dict)
    results: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class FeedbackSample(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    source_run_id: str
    workflow_type: WorkflowType
    input_payload: dict[str, Any] = Field(default_factory=dict)
    expected_status: RunStatus
    reviewer_name: str
    reviewer_comment: str
    review_score: float
    expected_keywords: list[str] = Field(default_factory=list)
    output_snapshot: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
