from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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
    retry_count: int = 0
    used_fallback: bool = False
    error: str | None = None
    validation_error: str | None = None


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


class BulkDeleteRequest(BaseModel):
    run_ids: list[str] = Field(default_factory=list)


class WorkflowPlan(BaseModel):
    workflow_type: WorkflowType
    objective: str
    steps: list[str]
    expected_outputs: list[str]


class OperatorDecision(BaseModel):
    action: Literal["use_tool", "finish"]
    selected_tool: str | None = None
    tool_input: dict[str, Any] = Field(default_factory=dict)
    reason: str

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("reason cannot be empty")
        return normalized

    @model_validator(mode="after")
    def validate_selected_tool(self) -> "OperatorDecision":
        if self.action == "use_tool" and not (self.selected_tool and self.selected_tool.strip()):
            raise ValueError("selected_tool is required when action is use_tool")
        if self.action == "finish":
            self.selected_tool = None
            self.tool_input = {}
        return self


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


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise ValueError("must be a list of strings")
    normalized = [str(item).strip() for item in value if str(item).strip()]
    if not normalized:
        raise ValueError("must contain at least one non-empty item")
    return normalized


class AnalystOutput(BaseModel):
    summary: str
    insights: list[str]
    action_plan: list[str]

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("summary cannot be empty")
        return normalized

    @field_validator("insights", "action_plan", mode="before")
    @classmethod
    def validate_text_list(cls, value: Any) -> list[str]:
        return _normalize_text_list(value)


class ContentOutput(BaseModel):
    deliverables: dict[str, Any]
    manager_note: str

    @field_validator("deliverables")
    @classmethod
    def validate_deliverables(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict) or not value:
            raise ValueError("deliverables cannot be empty")
        return value

    @field_validator("manager_note")
    @classmethod
    def validate_manager_note(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("manager_note cannot be empty")
        return normalized


class ReviewOutput(BaseModel):
    status: Literal["completed", "waiting_human"]
    needs_human_review: bool
    score: float
    reasons: list[str]

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("score must be between 0 and 1")
        return numeric

    @field_validator("reasons", mode="before")
    @classmethod
    def validate_reasons(cls, value: Any) -> list[str]:
        return _normalize_text_list(value)

    @model_validator(mode="after")
    def align_status(self) -> "ReviewOutput":
        if self.status == "waiting_human":
            self.needs_human_review = True
        else:
            self.needs_human_review = False
        return self
