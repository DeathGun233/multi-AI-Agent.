from __future__ import annotations

from dataclasses import dataclass

from app.models import ExecutionProfile, PromptProfile, RoutingPolicyRef


@dataclass(frozen=True)
class ModelOption:
    model_name: str
    label: str
    description: str
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float


@dataclass(frozen=True)
class RoutingPolicyDefinition:
    policy_id: str
    name: str
    description: str
    route_templates: dict[str, str]

    def resolve_routes(self, primary_model_name: str) -> dict[str, str]:
        return {
            agent: (primary_model_name if template == "{primary}" else template)
            for agent, template in self.route_templates.items()
        }


@dataclass(frozen=True)
class EvaluationCaseDefinition:
    case_id: str
    title: str
    workflow_type: str
    input_payload: dict
    expected_status: str
    expected_keywords: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationDatasetDefinition:
    dataset_id: str
    name: str
    description: str
    cases: tuple[EvaluationCaseDefinition, ...]


AVAILABLE_MODELS = (
    ModelOption(
        model_name="qwen3-max",
        label="Qwen3 Max",
        description="Quality first. Best for complex analysis and strict review.",
        input_cost_per_1k_tokens=0.02,
        output_cost_per_1k_tokens=0.06,
    ),
    ModelOption(
        model_name="qwen-plus",
        label="Qwen Plus",
        description="Balanced quality and cost for everyday workflow runs.",
        input_cost_per_1k_tokens=0.008,
        output_cost_per_1k_tokens=0.024,
    ),
    ModelOption(
        model_name="qwen-turbo",
        label="Qwen Turbo",
        description="Speed first. Good for light content generation and high frequency runs.",
        input_cost_per_1k_tokens=0.002,
        output_cost_per_1k_tokens=0.006,
    ),
)


ROUTING_POLICIES = (
    RoutingPolicyDefinition(
        policy_id="single-model-v1",
        name="Single model",
        description="Use the same primary model for all agent routes.",
        route_templates={"planner": "{primary}", "analyst": "{primary}", "content": "{primary}", "reviewer": "{primary}"},
    ),
    RoutingPolicyDefinition(
        policy_id="balanced-router-v1",
        name="Balanced router",
        description="Keep analysis on the primary model, send content to Turbo, and review to Max.",
        route_templates={"planner": "{primary}", "analyst": "{primary}", "content": "qwen-turbo", "reviewer": "qwen3-max"},
    ),
    RoutingPolicyDefinition(
        policy_id="speed-router-v1",
        name="Speed router",
        description="Use Turbo for analysis and content, reserve the primary model for planning and review.",
        route_templates={"planner": "{primary}", "analyst": "qwen-turbo", "content": "qwen-turbo", "reviewer": "{primary}"},
    ),
    RoutingPolicyDefinition(
        policy_id="strict-review-v1",
        name="Strict review router",
        description="Always review with Qwen3 Max for safer decisions.",
        route_templates={"planner": "{primary}", "analyst": "{primary}", "content": "{primary}", "reviewer": "qwen3-max"},
    ),
)


BUILTIN_PROMPT_PROFILES = (
    PromptProfile(
        profile_id="balanced-v1",
        name="Balanced",
        version="v1",
        description="Default prompt profile that balances depth, execution clarity, and stable output.",
        analyst_instruction="Lead with the conclusion, then explain key findings and actions in a concise way.",
        content_instruction="Generate business-ready output that can be reused directly by operators or reviewers.",
        reviewer_instruction="Allow auto-complete only when the result is complete, low risk, and operationally clear.",
        is_builtin=True,
    ),
    PromptProfile(
        profile_id="ops-deep-v1",
        name="Ops Deep Dive",
        version="v1",
        description="Stronger analysis and clearer prioritization for operational workflows.",
        analyst_instruction="Highlight root causes, bottlenecks, priority order, and execution tradeoffs.",
        content_instruction="Produce clearer owners, deadlines, and next steps with less generic filler.",
        reviewer_instruction="Be conservative about approval whenever the output has gaps or operational risk.",
        is_builtin=True,
    ),
    PromptProfile(
        profile_id="exec-brief-v2",
        name="Exec Brief",
        version="v2",
        description="Short, high-signal summaries for managers and weekly reviews.",
        analyst_instruction="Focus on top-line conclusions, major risks, and the two or three most important actions.",
        content_instruction="Keep the tone brief and presentation-ready for summaries and management updates.",
        reviewer_instruction="Check whether the result is decision-ready and easy to hand off upward.",
        is_builtin=True,
    ),
)


EVALUATION_DATASETS = (
    EvaluationDatasetDefinition(
        dataset_id="ops-regression-v1",
        name="Operations regression set",
        description="Covers sales, marketing, support, and meeting workflows for a stable baseline.",
        cases=(
            EvaluationCaseDefinition(
                case_id="sales-conversion",
                title="Sales funnel conversion analysis",
                workflow_type="sales_followup",
                input_payload={
                    "period": "2026-W13",
                    "region": "East",
                    "sales_reps": ["Wang Chen", "Li Xue"],
                    "focus_metric": "conversion_rate",
                },
                expected_status="waiting_human",
                expected_keywords=("conversion", "risk customers", "follow-up"),
            ),
            EvaluationCaseDefinition(
                case_id="marketing-assets",
                title="Marketing multi-channel content",
                workflow_type="marketing_campaign",
                input_payload={
                    "product_name": "FlowPilot AI",
                    "audience": "B2B operations leads",
                    "channels": ["xiaohongshu", "douyin", "wechat"],
                    "key_benefits": ["multi-agent execution", "human handoff", "traceable workflow"],
                    "tone": "professional and action-oriented",
                },
                expected_status="waiting_human",
                expected_keywords=("launch", "compliance", "content"),
            ),
            EvaluationCaseDefinition(
                case_id="support-handoff",
                title="Support escalation decision",
                workflow_type="support_triage",
                input_payload={
                    "tickets": [
                        {"customer": "Example Client", "message": "Production API errors are blocking a release. Need help today."},
                        {"customer": "Xingyun Education", "message": "Can you share the invoicing flow and contract template?"},
                    ]
                },
                expected_status="waiting_human",
                expected_keywords=("urgent", "escalation", "reply"),
            ),
            EvaluationCaseDefinition(
                case_id="meeting-followup",
                title="Meeting note action extraction",
                workflow_type="meeting_minutes",
                input_payload={
                    "meeting_title": "AI Growth Weekly",
                    "notes": (
                        "1. Zhang Min to finish competitor review by Friday. "
                        "2. Chen Tao to propose lead scoring plan by next Tuesday. "
                        "3. Wang Chen to confirm pilot customers before end of day."
                    ),
                },
                expected_status="completed",
                expected_keywords=("action items", "owners", "summary"),
            ),
        ),
    ),
)


DEFAULT_PROMPT_PROFILE_ID = BUILTIN_PROMPT_PROFILES[0].profile_id
DEFAULT_ROUTING_POLICY_ID = ROUTING_POLICIES[1].policy_id

_MODEL_INDEX = {item.model_name: item for item in AVAILABLE_MODELS}
_ROUTING_INDEX = {item.policy_id: item for item in ROUTING_POLICIES}
_DATASET_INDEX = {item.dataset_id: item for item in EVALUATION_DATASETS}


def list_model_options() -> list[dict[str, str | float]]:
    return [
        {
            "model_name": item.model_name,
            "label": item.label,
            "description": item.description,
            "input_cost_per_1k_tokens": item.input_cost_per_1k_tokens,
            "output_cost_per_1k_tokens": item.output_cost_per_1k_tokens,
        }
        for item in AVAILABLE_MODELS
    ]


def list_routing_policies() -> list[dict[str, str]]:
    return [{"policy_id": item.policy_id, "name": item.name, "description": item.description} for item in ROUTING_POLICIES]


def list_evaluation_datasets() -> list[dict[str, str]]:
    return [
        {
            "dataset_id": item.dataset_id,
            "name": item.name,
            "description": item.description,
            "case_count": str(len(item.cases)),
        }
        for item in EVALUATION_DATASETS
    ]


def get_model_option(model_name: str) -> ModelOption:
    return _MODEL_INDEX[model_name]


def get_routing_policy(policy_id: str | None = None) -> RoutingPolicyDefinition:
    return _ROUTING_INDEX.get(policy_id or DEFAULT_ROUTING_POLICY_ID) or _ROUTING_INDEX[DEFAULT_ROUTING_POLICY_ID]


def get_evaluation_dataset(dataset_id: str) -> EvaluationDatasetDefinition:
    return _DATASET_INDEX[dataset_id]


def resolve_execution_profile(
    *,
    default_model_name: str,
    prompt_profile: PromptProfile,
    model_name_override: str | None = None,
    routing_policy_id: str | None = None,
) -> ExecutionProfile:
    primary_model_name = model_name_override if model_name_override in _MODEL_INDEX else default_model_name
    primary_model = _MODEL_INDEX.get(primary_model_name) or _MODEL_INDEX[default_model_name]
    routing_policy = get_routing_policy(routing_policy_id)
    return ExecutionProfile(
        primary_model_name=primary_model.model_name,
        primary_model_label=primary_model.label,
        prompt_profile=prompt_profile.as_ref(),
        routing_policy=RoutingPolicyRef(
            policy_id=routing_policy.policy_id,
            name=routing_policy.name,
            description=routing_policy.description,
        ),
        model_routes=routing_policy.resolve_routes(primary_model.model_name),
    )
