from __future__ import annotations

from app.models import WorkflowTemplate, WorkflowType


WORKFLOW_TEMPLATES = [
    WorkflowTemplate(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        title="销售分析与跟进计划",
        description="分析销售漏斗、识别风险客户并生成跟进策略。",
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
        description="为多个渠道生成内容资产和投放建议。",
        sample_payload={
            "product_name": "FlowPilot AI 自动化平台",
            "audience": "B2B 企业运营负责人",
            "channels": ["xiaohongshu", "douyin", "wechat"],
            "key_benefits": ["多智能体执行", "人工接管", "流程可观测"],
            "tone": "专业但有行动感",
        },
    ),
    WorkflowTemplate(
        workflow_type=WorkflowType.SUPPORT_TRIAGE,
        title="客服工单智能分流",
        description="工单分类、优先级排序、回复草稿和升级建议。",
        sample_payload={
            "tickets": [
                {
                    "customer": "上海嘉禾医疗",
                    "message": "部署后接口连续报错，今天必须恢复，不然影响上线。",
                },
                {
                    "customer": "星云教育",
                    "message": "请问你们支持开票和合同模板下载吗？",
                },
            ]
        },
    ),
    WorkflowTemplate(
        workflow_type=WorkflowType.MEETING_MINUTES,
        title="会议纪要转执行系统",
        description="提取负责人、截止时间、待办项和会后邮件。",
        sample_payload={
            "meeting_title": "AI 增长周会",
            "notes": (
                "1. 张敏负责本周五前整理竞品投放复盘；"
                "2. 陈涛下周二前提交销售线索分层方案；"
                "3. 王晨今天下班前确认客户试点名单；"
                "4. 市场组需要补一版小红书脚本。"
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
    {"name": "华启制造", "risk": "沉默 10 天", "owner": "李雪", "next_action": "安排产品演示复盘"},
    {"name": "聚能科技", "risk": "POC 迟迟未启动", "owner": "王晨", "next_action": "拉技术负责人对齐范围"},
    {"name": "深海物流", "risk": "预算审批卡住", "owner": "李雪", "next_action": "补 ROI 材料并推动决策人会议"},
]
