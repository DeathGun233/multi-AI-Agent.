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
        label="千问3 Max",
        description="质量优先，适合复杂分析和严格审核。",
        input_cost_per_1k_tokens=0.02,
        output_cost_per_1k_tokens=0.06,
    ),
    ModelOption(
        model_name="qwen-plus",
        label="千问 Plus",
        description="质量与成本均衡，适合日常工作流运行。",
        input_cost_per_1k_tokens=0.008,
        output_cost_per_1k_tokens=0.024,
    ),
    ModelOption(
        model_name="qwen-turbo",
        label="千问 Turbo",
        description="速度优先，适合轻量内容生成和高频调用。",
        input_cost_per_1k_tokens=0.002,
        output_cost_per_1k_tokens=0.006,
    ),
)


ROUTING_POLICIES = (
    RoutingPolicyDefinition(
        policy_id="single-model-v1",
        name="单模型直连",
        description="所有 Agent 共用同一个主模型，适合基础对比。",
        route_templates={"planner": "{primary}", "operator": "{primary}", "analyst": "{primary}", "content": "{primary}", "reviewer": "{primary}"},
    ),
    RoutingPolicyDefinition(
        policy_id="balanced-router-v1",
        name="均衡路由",
        description="分析使用主模型，内容生成走更快模型，审核固定走高质量模型。",
        route_templates={"planner": "{primary}", "operator": "qwen-turbo", "analyst": "{primary}", "content": "qwen-turbo", "reviewer": "qwen3-max"},
    ),
    RoutingPolicyDefinition(
        policy_id="speed-router-v1",
        name="速度优先路由",
        description="分析和内容都走更快模型，规划与审核保留主模型。",
        route_templates={"planner": "{primary}", "operator": "qwen-turbo", "analyst": "qwen-turbo", "content": "qwen-turbo", "reviewer": "{primary}"},
    ),
    RoutingPolicyDefinition(
        policy_id="strict-review-v1",
        name="严格审核路由",
        description="审核固定使用高质量模型，适合高风险场景。",
        route_templates={"planner": "{primary}", "operator": "{primary}", "analyst": "{primary}", "content": "{primary}", "reviewer": "qwen3-max"},
    ),
)


BUILTIN_PROMPT_PROFILES = (
    PromptProfile(
        profile_id="balanced-v1",
        name="平衡版",
        version="v1",
        description="默认方案，兼顾分析深度、执行清晰度和输出稳定性。",
        analyst_instruction="请先给结论，再用简洁中文补充关键洞察和行动建议。",
        content_instruction="请输出能被业务同学直接使用的中文内容，强调结构清晰和可执行性。",
        reviewer_instruction="仅在结果完整、风险可控且执行路径清晰时允许自动流转，请使用中文。",
        is_builtin=True,
    ),
    PromptProfile(
        profile_id="ops-deep-v1",
        name="运营深挖版",
        version="v1",
        description="更强调异常定位、优先级和运营执行建议。",
        analyst_instruction="请突出根因、瓶颈、优先级和执行取舍，用中文输出。",
        content_instruction="请明确责任人、时间节点和后续动作，减少空泛表述。",
        reviewer_instruction="只要结果存在信息缺口或执行风险，就偏向人工审核，并用中文说明理由。",
        is_builtin=True,
    ),
    PromptProfile(
        profile_id="exec-brief-v2",
        name="管理摘要版",
        version="v2",
        description="适合管理层快速阅读的简洁摘要版。",
        analyst_instruction="请优先输出高层结论、关键风险和最重要的 2 到 3 个动作，使用中文。",
        content_instruction="请采用适合汇报的简洁中文风格，方便周报和管理看板直接引用。",
        reviewer_instruction="请检查是否具备决策信息、是否可直接向上汇报，并用中文说明。",
        is_builtin=True,
    ),
)


EVALUATION_DATASETS = (
    EvaluationDatasetDefinition(
        dataset_id="ops-regression-v1",
        name="运营回归集",
        description="覆盖销售、营销、客服和会议纪要四类工作流，作为稳定基线。",
        cases=(
            EvaluationCaseDefinition(
                case_id="sales-conversion",
                title="销售漏斗转化分析",
                workflow_type="sales_followup",
                input_payload={
                    "period": "2026-W13",
                    "region": "华东",
                    "sales_reps": ["王晨", "李雪"],
                    "focus_metric": "conversion_rate",
                },
                expected_status="waiting_human",
                expected_keywords=("转化", "风险客户", "跟进"),
            ),
            EvaluationCaseDefinition(
                case_id="marketing-assets",
                title="营销多渠道内容生成",
                workflow_type="marketing_campaign",
                input_payload={
                    "product_name": "FlowPilot AI 平台",
                    "audience": "B2B 企业运营负责人",
                    "channels": ["xiaohongshu", "douyin", "wechat"],
                    "key_benefits": ["多智能体执行", "人工接管", "可追踪流程"],
                    "tone": "专业且有行动感",
                },
                expected_status="waiting_human",
                expected_keywords=("投放", "合规", "内容"),
            ),
            EvaluationCaseDefinition(
                case_id="support-handoff",
                title="客服升级与接管判断",
                workflow_type="support_triage",
                input_payload={
                    "tickets": [
                        {"customer": "示例客户", "message": "生产接口持续报错，今天必须恢复，否则影响上线。"},
                        {"customer": "星云教育", "message": "请问可以提供开票流程和合同模板吗？"},
                    ]
                },
                expected_status="waiting_human",
                expected_keywords=("紧急", "升级", "回复"),
            ),
            EvaluationCaseDefinition(
                case_id="meeting-followup",
                title="会议纪要行动项抽取",
                workflow_type="meeting_minutes",
                input_payload={
                    "meeting_title": "AI 增长周会",
                    "notes": (
                        "1. 张敏本周五前完成竞品复盘。"
                        "2. 陈涛下周二前提交线索分层方案。"
                        "3. 王晨今天下班前确认试点客户名单。"
                    ),
                },
                expected_status="completed",
                expected_keywords=("行动项", "负责人", "总结"),
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
