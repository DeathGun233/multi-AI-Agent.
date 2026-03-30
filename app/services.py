from __future__ import annotations

import json
import re
from collections import Counter
from statistics import mean
from typing import Any

from app.config import Settings
from app.data import RISK_CUSTOMERS, SALES_DATA, WORKFLOW_TEMPLATES
from app.llm import LLMService
from app.models import ReviewDecision, RunStatus, ToolCall, WorkflowPlan, WorkflowRequest, WorkflowRun, WorkflowType
from app.repository import WorkflowRepository


WORKFLOW_TITLES = {
    WorkflowType.SALES_FOLLOWUP: "销售分析与跟进计划",
    WorkflowType.MARKETING_CAMPAIGN: "营销内容工厂",
    WorkflowType.SUPPORT_TRIAGE: "客服工单智能分流",
    WorkflowType.MEETING_MINUTES: "会议纪要转执行系统",
}


class PlannerAgent:
    def build_plan(self, request: WorkflowRequest) -> WorkflowPlan:
        objective = WORKFLOW_TITLES[request.workflow_type]
        step_map = {
            WorkflowType.SALES_FOLLOWUP: [
                "解析分析范围和关注指标",
                "调用销售分析工具生成漏斗与风险客户",
                "提炼异常原因并生成跟进计划",
                "执行质量审核与人工接管判断",
            ],
            WorkflowType.MARKETING_CAMPAIGN: [
                "解析产品、受众和渠道要求",
                "调用内容工厂工具生成多渠道内容资产",
                "补充投放建议和 A/B 版本",
                "执行品牌与合规审核",
            ],
            WorkflowType.SUPPORT_TRIAGE: [
                "读取工单并进行优先级判断",
                "调用分流工具完成分类与回复草稿",
                "识别高风险工单并安排升级",
                "执行人工接管判断",
            ],
            WorkflowType.MEETING_MINUTES: [
                "解析会议纪要与决策信息",
                "调用行动项提取工具整理负责人和截止时间",
                "生成会后总结与待办清单",
                "执行完整性审核",
            ],
        }
        expected_outputs = {
            WorkflowType.SALES_FOLLOWUP: ["漏斗分析", "风险客户", "跟进建议", "日报摘要"],
            WorkflowType.MARKETING_CAMPAIGN: ["多渠道文案", "投放建议", "A/B 版本", "审核结论"],
            WorkflowType.SUPPORT_TRIAGE: ["工单分类", "优先级", "回复草稿", "升级建议"],
            WorkflowType.MEETING_MINUTES: ["行动项列表", "负责人", "截止时间", "会后邮件"],
        }
        return WorkflowPlan(
            workflow_type=request.workflow_type,
            objective=objective,
            steps=step_map[request.workflow_type],
            expected_outputs=expected_outputs[request.workflow_type],
        )


class ToolCenter:
    def analyze_sales(self, payload: dict[str, Any]) -> dict[str, Any]:
        reps = set(payload.get("sales_reps") or [])
        filtered = [row for row in SALES_DATA if not reps or row["rep"] in reps]
        if not filtered:
            filtered = SALES_DATA
        total_leads = sum(item["leads"] for item in filtered)
        total_qualified = sum(item["qualified"] for item in filtered)
        total_deals = sum(item["deals"] for item in filtered)
        conversion = round(total_deals / total_leads, 3) if total_leads else 0.0
        qualification = round(total_qualified / total_leads, 3) if total_leads else 0.0
        avg_cycle = round(mean(item["avg_cycle_days"] for item in filtered), 1)
        weak_rep = min(filtered, key=lambda item: item["deals"] / item["leads"] if item["leads"] else 0)
        risks = [item for item in RISK_CUSTOMERS if not reps or item["owner"] in reps]
        return {
            "period": payload.get("period", "最近一周"),
            "focus_metric": payload.get("focus_metric", "conversion_rate"),
            "summary": {
                "lead_count": total_leads,
                "qualified_count": total_qualified,
                "deal_count": total_deals,
                "qualification_rate": qualification,
                "conversion_rate": conversion,
                "avg_cycle_days": avg_cycle,
            },
            "weak_point": {
                "rep": weak_rep["rep"],
                "issue": "成交转化偏低且销售周期偏长",
            },
            "risk_customers": risks,
        }

    def generate_marketing_assets(self, payload: dict[str, Any]) -> dict[str, Any]:
        product = payload.get("product_name", "企业 AI 平台")
        audience = payload.get("audience", "企业负责人")
        benefits = payload.get("key_benefits") or ["效率提升", "可控落地"]
        tone = payload.get("tone", "专业")
        channels = payload.get("channels") or ["xiaohongshu"]
        assets: dict[str, Any] = {}
        if "xiaohongshu" in channels:
            assets["xiaohongshu"] = {
                "title": f"{product} 如何把 {benefits[0]} 真正落到团队里",
                "body": f"给 {audience} 的一套实战方法：先梳理流程瓶颈，再让多智能体接管重复任务，最后用人工审核保证结果可落地。",
            }
        if "douyin" in channels:
            assets["douyin"] = {
                "hook": f"如果你的团队还在人工追任务，{product} 可以把流程自动跑起来。",
                "script": [
                    "第一幕：展示流程断点和协作混乱。",
                    f"第二幕：演示 {product} 自动拆解任务、调工具、产出结果。",
                    "第三幕：突出人工接管、日志可追踪和 ROI 提升。",
                ],
            }
        if "wechat" in channels:
            assets["wechat"] = {
                "headline": f"{product}：企业流程自动化的下一步",
                "summary": f"围绕 {', '.join(benefits)} 构建可审核、可追踪的 AI 执行体系。",
            }
        return {
            "product": product,
            "audience": audience,
            "tone": tone,
            "assets": assets,
            "ab_test": [
                {"variant": "A", "angle": "效率提升", "cta": "申请试点演示"},
                {"variant": "B", "angle": "流程可控", "cta": "预约业务诊断"},
            ],
        }

    def triage_tickets(self, payload: dict[str, Any]) -> dict[str, Any]:
        tickets = payload.get("tickets") or []
        processed = []
        for ticket in tickets:
            text = ticket.get("message", "")
            priority = "medium"
            category = "咨询"
            escalate = False
            if any(keyword in text for keyword in ["报错", "恢复", "上线", "故障", "宕机"]):
                category = "技术故障"
                priority = "urgent"
                escalate = True
            elif any(keyword in text for keyword in ["退款", "投诉"]):
                category = "投诉退款"
                priority = "high"
                escalate = True
            elif any(keyword in text for keyword in ["合同", "开票", "发票"]):
                category = "商务咨询"
                priority = "medium"
            reply = {
                "技术故障": "我们已将问题升级给值班工程师，15 分钟内同步初步排查结果。",
                "投诉退款": "已安排专人核实订单与服务记录，今天内给到处理方案。",
                "商务咨询": "支持开票与合同模板下载，我这边同步发你标准流程和所需资料。",
                "咨询": "已收到你的问题，我们会尽快整理答案并回复。",
            }[category]
            processed.append(
                {
                    "customer": ticket.get("customer", "未知客户"),
                    "category": category,
                    "priority": priority,
                    "escalate": escalate,
                    "draft_reply": reply,
                }
            )
        return {
            "ticket_count": len(processed),
            "processed": processed,
            "priority_breakdown": dict(Counter(item["priority"] for item in processed)),
        }

    def extract_meeting_actions(self, payload: dict[str, Any]) -> dict[str, Any]:
        notes = payload.get("notes", "")
        segments = [segment.strip(" ；;。.") for segment in re.split(r"[；;\n]+", notes) if segment.strip()]
        actions = []
        for segment in segments:
            match = re.search(
                r"(?P<owner>[\u4e00-\u9fa5]{2,4}).{0,4}(?P<deadline>本周五前|下周二前|今天下班前|本周内|下周内)?(?P<task>.+)",
                segment,
            )
            if match:
                owner = match.group("owner")
                deadline = match.group("deadline") or "待确认"
                task = match.group("task").strip("负责需要将")
                actions.append({"owner": owner, "deadline": deadline, "task": task})
            else:
                actions.append({"owner": "待分配", "deadline": "待确认", "task": segment})
        return {
            "meeting_title": payload.get("meeting_title", "会议纪要"),
            "actions": actions,
            "summary_email": (
                f"会议《{payload.get('meeting_title', '会议纪要')}》已整理完成，"
                f"共提取 {len(actions)} 项待办，请各负责人按时推进。"
            ),
        }


class AnalystAgent:
    def __init__(self, llm: LLMService) -> None:
        self.llm = llm

    def analyze(self, workflow_type: WorkflowType, raw_result: dict[str, Any]) -> dict[str, Any]:
        fallback = self._fallback_analysis(workflow_type, raw_result)
        prompt = {
            "workflow_type": workflow_type.value,
            "raw_result": raw_result,
            "required_keys": ["insights", "action_plan"],
        }
        return self.llm.generate_json(
            system_prompt=(
                "你是企业工作流平台中的 Analyst Agent。"
                "请根据输入数据输出纯 JSON，必须包含 insights 和 action_plan 两个数组，"
                "每个数组至少 2 条，用简洁专业的中文表达，不要输出 markdown。"
            ),
            user_prompt=json.dumps(prompt, ensure_ascii=False),
            fallback=fallback,
        )

    @staticmethod
    def _fallback_analysis(workflow_type: WorkflowType, raw_result: dict[str, Any]) -> dict[str, Any]:
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            summary = raw_result["summary"]
            return {
                "insights": [
                    f"当前线索转化率为 {summary['conversion_rate']:.1%}，低于稳态目标。",
                    f"资格转化率为 {summary['qualification_rate']:.1%}，说明筛选阶段仍有优化空间。",
                    f"平均销售周期 {summary['avg_cycle_days']} 天，需优先缩短中后段推进时长。",
                ],
                "action_plan": [
                    "针对高风险客户安排 24 小时内跟进。",
                    "对转化偏低销售补一轮演示复盘与异议处理脚本。",
                    "按行业整理 ROI 物料，推动预算审批场景。",
                ],
            }
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            processed = raw_result["processed"]
            urgent = [item for item in processed if item["priority"] == "urgent"]
            return {
                "insights": [
                    f"本批工单共 {len(processed)} 条，其中紧急工单 {len(urgent)} 条。",
                    "技术故障和投诉退款类问题需要优先人工跟进。",
                ],
                "action_plan": [
                    "紧急工单进入值班群和客户成功经理双通知。",
                    "商务咨询统一走标准资料包，提高首响效率。",
                ],
            }
        if workflow_type == WorkflowType.MEETING_MINUTES:
            actions = raw_result["actions"]
            return {
                "insights": [
                    f"共抽取 {len(actions)} 个执行项，适合直接同步到任务系统。",
                    "纪要中出现了缺少明确截止时间的事项，建议补齐。",
                ],
                "action_plan": [
                    "会后 30 分钟内发出总结邮件。",
                    "对截止时间缺失事项追加确认。",
                ],
            }
        return {
            "insights": [
                "内容已按渠道拆分，可直接进入投放准备。",
                "建议保留两套版本做 A/B 测试。",
            ],
            "action_plan": [
                "先投放小红书版本验证信息密度。",
                "将高反馈卖点同步到销售话术。",
            ],
        }


class ContentAgent:
    def __init__(self, llm: LLMService) -> None:
        self.llm = llm

    def refine(self, workflow_type: WorkflowType, raw_result: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
        fallback = self._fallback_content(workflow_type, raw_result)
        prompt = {
            "workflow_type": workflow_type.value,
            "raw_result": raw_result,
            "analysis": analysis,
            "fallback_shape": fallback,
        }
        return self.llm.generate_json(
            system_prompt=(
                "你是企业工作流平台中的 Content Agent。"
                "请根据业务分析结果输出纯 JSON，字段结构尽量贴合 fallback_shape，"
                "内容必须是中文，便于业务同学直接使用。"
            ),
            user_prompt=json.dumps(prompt, ensure_ascii=False),
            fallback=fallback,
        )

    @staticmethod
    def _fallback_content(workflow_type: WorkflowType, raw_result: dict[str, Any]) -> dict[str, Any]:
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            return {
                "daily_brief": "销售漏斗存在中段转化偏弱，建议围绕高风险客户和演示复盘做集中推进。",
                "manager_note": "将风险客户推进动作纳入明早站会追踪。",
            }
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            return {"handoff_note": "紧急故障类工单已建议升级，商务咨询类可自动回复后转销售支持。"}
        if workflow_type == WorkflowType.MEETING_MINUTES:
            return {"followup_message": raw_result["summary_email"]}
        return {
            "campaign_note": "建议先用效率提升角度开首轮投放，再根据互动率决定是否强化流程可控卖点。"
        }


class ReviewerAgent:
    def __init__(self, llm: LLMService) -> None:
        self.llm = llm

    def review(self, workflow_type: WorkflowType, raw_result: dict[str, Any], analysis: dict[str, Any]) -> ReviewDecision:
        fallback = self._fallback_review(workflow_type, raw_result)
        prompt = {
            "workflow_type": workflow_type.value,
            "raw_result": raw_result,
            "analysis": analysis,
            "allowed_status": ["completed", "waiting_human"],
            "required_keys": ["status", "needs_human_review", "score", "reasons"],
        }
        llm_payload = self.llm.generate_json(
            system_prompt=(
                "你是企业工作流平台中的 Reviewer Agent。"
                "请输出纯 JSON。status 只能是 completed 或 waiting_human，"
                "score 取值 0 到 1，reasons 是中文数组。"
            ),
            user_prompt=json.dumps(prompt, ensure_ascii=False),
            fallback=fallback,
        )
        return ReviewDecision(**self._merge_review(fallback, llm_payload))

    @staticmethod
    def _fallback_review(workflow_type: WorkflowType, raw_result: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        score = 0.92
        needs_human_review = False
        status = RunStatus.COMPLETED.value
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            urgent_count = raw_result["priority_breakdown"].get("urgent", 0)
            if urgent_count:
                reasons.append("存在紧急工单，建议人工确认升级路径。")
                needs_human_review = True
                status = RunStatus.WAITING_HUMAN.value
                score = 0.81
        elif workflow_type == WorkflowType.MEETING_MINUTES:
            if any(action["deadline"] == "待确认" for action in raw_result["actions"]):
                reasons.append("部分行动项缺少明确截止时间。")
                needs_human_review = True
                status = RunStatus.WAITING_HUMAN.value
                score = 0.84
        elif workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            reasons.append("建议投放前做品牌与合规复核。")
            needs_human_review = True
            status = RunStatus.WAITING_HUMAN.value
            score = 0.86
        else:
            reasons.append("结果结构完整，可直接流转执行。")
        return {
            "status": status,
            "needs_human_review": needs_human_review,
            "score": score,
            "reasons": reasons,
        }

    @staticmethod
    def _merge_review(rule_payload: dict[str, Any], llm_payload: dict[str, Any]) -> dict[str, Any]:
        merged_reasons = []
        for reason in [*(rule_payload.get("reasons") or []), *(llm_payload.get("reasons") or [])]:
            if reason and reason not in merged_reasons:
                merged_reasons.append(reason)

        # Hard safety gates always win. The model can make the review stricter, but not looser.
        if rule_payload.get("needs_human_review"):
            return {
                "status": RunStatus.WAITING_HUMAN.value,
                "needs_human_review": True,
                "score": min(float(rule_payload.get("score", 1.0)), float(llm_payload.get("score", 1.0))),
                "reasons": merged_reasons,
            }

        llm_requests_handoff = bool(llm_payload.get("needs_human_review")) or llm_payload.get("status") == RunStatus.WAITING_HUMAN.value
        return {
            "status": RunStatus.WAITING_HUMAN.value if llm_requests_handoff else RunStatus.COMPLETED.value,
            "needs_human_review": llm_requests_handoff,
            "score": float(llm_payload.get("score", rule_payload.get("score", 0.9))),
            "reasons": merged_reasons or rule_payload.get("reasons", []),
        }


class WorkflowEngine:
    def __init__(self, repository: WorkflowRepository, settings: Settings) -> None:
        self.repository = repository
        self.settings = settings
        self.llm = LLMService(settings)
        self.planner = PlannerAgent()
        self.tools = ToolCenter()
        self.analyst = AnalystAgent(self.llm)
        self.content = ContentAgent(self.llm)
        self.reviewer = ReviewerAgent(self.llm)

    def list_templates(self) -> list[dict[str, Any]]:
        return [template.model_dump(mode="json") for template in WORKFLOW_TEMPLATES]

    def list_runs(self) -> list[WorkflowRun]:
        return self.repository.list_all()

    def get_run(self, run_id: str) -> WorkflowRun | None:
        return self.repository.get(run_id)

    def run_workflow(self, request: WorkflowRequest) -> WorkflowRun:
        run = WorkflowRun(workflow_type=request.workflow_type, input_payload=request.input_payload)
        self.repository.save(run)

        run.touch(status=RunStatus.PLANNING, current_step="planning")
        plan = self.planner.build_plan(request)
        run.plan = plan
        run.objective = plan.objective
        run.add_log("PlannerAgent", f"已生成执行计划，共 {len(plan.steps)} 步。")

        run.touch(status=RunStatus.EXECUTING, current_step="tool_execution")
        raw_result, tool_name = self._execute_tool(request.workflow_type, request.input_payload)
        run.add_log(
            "OperatorAgent",
            f"完成工具调用：{tool_name}",
            tool_call=ToolCall(name=tool_name, input=request.input_payload, output=raw_result),
        )

        analysis = self.analyst.analyze(request.workflow_type, raw_result)
        run.add_log(
            "AnalystAgent",
            "已完成结果分析与行动建议整理。"
            + ("（已调用真实模型）" if self.llm.enabled else "（当前为本地回退策略）"),
        )

        content = self.content.refine(request.workflow_type, raw_result, analysis)
        run.add_log(
            "ContentAgent",
            "已补充业务可直接使用的输出内容。"
            + ("（已调用真实模型）" if self.llm.enabled else "（当前为本地回退策略）"),
        )

        run.touch(status=RunStatus.REVIEWING, current_step="review")
        review = self.reviewer.review(request.workflow_type, raw_result, analysis)
        run.review = review
        run.add_log(
            "ReviewerAgent",
            "已完成质量审核与人工接管判断。"
            + ("（已调用真实模型）" if self.llm.enabled else "（当前为本地回退策略）"),
        )

        run.result = {
            "raw_result": raw_result,
            "analysis": analysis,
            "deliverables": content,
        }
        run.touch(status=review.status, current_step="done")
        self.repository.save(run)
        return run

    def _execute_tool(self, workflow_type: WorkflowType, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if workflow_type == WorkflowType.SALES_FOLLOWUP:
            return self.tools.analyze_sales(payload), "sales_analytics_tool"
        if workflow_type == WorkflowType.MARKETING_CAMPAIGN:
            return self.tools.generate_marketing_assets(payload), "content_factory_tool"
        if workflow_type == WorkflowType.SUPPORT_TRIAGE:
            return self.tools.triage_tickets(payload), "ticket_triage_tool"
        return self.tools.extract_meeting_actions(payload), "meeting_action_extractor"
