from __future__ import annotations

from app.models import WorkflowTemplate, WorkflowType


WORKFLOW_TEMPLATES = [
    WorkflowTemplate(
        workflow_type=WorkflowType.SALES_FOLLOWUP,
        title="Sales follow-up plan",
        description="Analyze funnel performance, find risky accounts, and produce follow-up actions.",
        sample_payload={
            "period": "2026-W13",
            "region": "East",
            "sales_reps": ["Wang Chen", "Li Xue"],
            "focus_metric": "conversion_rate",
        },
    ),
    WorkflowTemplate(
        workflow_type=WorkflowType.MARKETING_CAMPAIGN,
        title="Marketing content factory",
        description="Generate multi-channel content assets, launch ideas, and review notes.",
        sample_payload={
            "product_name": "FlowPilot AI",
            "audience": "B2B operations leads",
            "channels": ["xiaohongshu", "douyin", "wechat"],
            "key_benefits": ["multi-agent execution", "human handoff", "traceable workflow"],
            "tone": "professional and action-oriented",
        },
    ),
    WorkflowTemplate(
        workflow_type=WorkflowType.SUPPORT_TRIAGE,
        title="Support ticket triage",
        description="Classify tickets, assign priority, draft replies, and decide handoff.",
        sample_payload={
            "tickets": [
                {
                    "customer": "Jiaheng Medical",
                    "message": "Production API errors are blocking a release. Need help today.",
                },
                {
                    "customer": "Xingyun Education",
                    "message": "Can you share the invoicing flow and contract template?",
                },
            ]
        },
    ),
    WorkflowTemplate(
        workflow_type=WorkflowType.MEETING_MINUTES,
        title="Meeting notes to action items",
        description="Extract owners, deadlines, action items, and an execution summary.",
        sample_payload={
            "meeting_title": "AI Growth Weekly",
            "notes": (
                "1. Zhang Min to finish competitor review by Friday. "
                "2. Chen Tao to propose lead scoring plan by next Tuesday. "
                "3. Wang Chen to confirm pilot customers before end of day. "
                "4. Marketing team needs a refreshed social script."
            ),
        },
    ),
]


SALES_DATA = [
    {"rep": "Wang Chen", "region": "East", "leads": 48, "qualified": 21, "deals": 7, "avg_cycle_days": 11},
    {"rep": "Li Xue", "region": "East", "leads": 35, "qualified": 15, "deals": 3, "avg_cycle_days": 17},
    {"rep": "Gu Lin", "region": "South", "leads": 52, "qualified": 23, "deals": 8, "avg_cycle_days": 10},
]


RISK_CUSTOMERS = [
    {"name": "Huating Pharma", "risk": "silent for 10 days", "owner": "Li Xue", "next_action": "schedule a demo review"},
    {"name": "Juneng Tech", "risk": "POC not started", "owner": "Wang Chen", "next_action": "align technical scope"},
    {"name": "DeepSea Logistics", "risk": "budget approval blocked", "owner": "Li Xue", "next_action": "send ROI deck and push decision meeting"},
]
