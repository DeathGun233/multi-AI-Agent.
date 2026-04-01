from __future__ import annotations

from app.models import WorkflowTemplate, WorkflowType


WORKFLOW_TEMPLATES = [
    WorkflowTemplate(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        title="销售分析与跟进计划",
        description="分析销售漏斗表现，识别风险客户，并生成可执行的跟进动作。",
        sample_payload={
            "period": "2026-W13",
            "region": "华东",
            "sales_reps": ["王晨", "李雪"],
            "focus_metric": "conversion_rate",
        },
    ),
    WorkflowTemplate(
        workflow_type=WorkflowType.MARKETING_CAMPAIGN,
        title="营销内容工厂",
        description="生成多渠道内容资产、投放建议和审核提示。",
        sample_payload={
            "product_name": "FlowPilot AI 平台",
            "audience": "B2B 企业运营负责人",
            "channels": ["xiaohongshu", "douyin", "wechat"],
            "key_benefits": ["多智能体执行", "人工接管", "可追踪流程"],
            "tone": "专业且有行动感",
        },
    ),
    WorkflowTemplate(
        workflow_type=WorkflowType.SUPPORT_TRIAGE,
        title="客服工单智能分流",
        description="完成工单分类、优先级判断、回复草稿生成和人工接管决策。",
        sample_payload={
            "tickets": [
                {
                    "customer": "嘉恒医疗",
                    "message": "生产接口持续报错，今天必须恢复，否则影响上线。",
                },
                {
                    "customer": "星云教育",
                    "message": "请问可以提供开票流程和合同模板吗？",
                },
            ]
        },
    ),
    WorkflowTemplate(
        workflow_type=WorkflowType.MEETING_MINUTES,
        title="会议纪要转执行系统",
        description="提取负责人、截止时间、待办项和执行摘要。",
        sample_payload={
            "meeting_title": "AI 增长周会",
            "notes": (
                "1. 张敏本周五前完成竞品复盘。"
                "2. 陈涛下周二前提交线索分层方案。"
                "3. 王晨今天下班前确认试点客户名单。"
                "4. 市场组补一版小红书脚本。"
            ),
        },
    ),
]


SALES_DATA = [
    {"rep": "王晨", "region": "华东", "leads": 48, "qualified": 21, "deals": 7, "avg_cycle_days": 11},
    {"rep": "李雪", "region": "华东", "leads": 35, "qualified": 15, "deals": 3, "avg_cycle_days": 17},
    {"rep": "顾林", "region": "华南", "leads": 52, "qualified": 23, "deals": 8, "avg_cycle_days": 10},
]


RISK_CUSTOMERS = [
    {"name": "华庭医药", "risk": "10 天未跟进", "owner": "李雪", "next_action": "安排方案演示复盘"},
    {"name": "聚能科技", "risk": "POC 尚未启动", "owner": "王晨", "next_action": "对齐技术实施范围"},
    {"name": "深海物流", "risk": "预算审批卡住", "owner": "李雪", "next_action": "补充 ROI 材料并推动决策会"},
]
