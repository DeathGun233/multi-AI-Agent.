from __future__ import annotations

import html
from io import BytesIO
from pathlib import Path
from textwrap import wrap
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from app.models import WorkflowRun


_FONT_NAME = "MicrosoftYaHei"
_FONT_REGISTERED = False


def _status_label(value: str) -> str:
    return {
        "created": "已创建",
        "planning": "规划中",
        "executing": "执行中",
        "reviewing": "审核判断中",
        "waiting_human": "待人工审核",
        "completed": "已完成",
        "failed": "已失败",
    }.get(value, value)


def _workflow_label(value: str) -> str:
    return {
        "sales_followup": "销售分析与跟进计划",
        "marketing_campaign": "营销内容工厂",
        "support_triage": "客服工单智能分流",
        "meeting_minutes": "会议纪要转执行系统",
    }.get(value, value)


def _ensure_font_registered() -> str:
    global _FONT_REGISTERED
    if not _FONT_REGISTERED:
        font_path = Path("C:/Windows/Fonts/msyh.ttc")
        pdfmetrics.registerFont(TTFont(_FONT_NAME, str(font_path), subfontIndex=0))
        _FONT_REGISTERED = True
    return _FONT_NAME


def build_workflow_markdown(run: WorkflowRun, llm_summary: dict[str, Any], result_json: str) -> str:
    lines = [
        f"# {_workflow_label(run.workflow_type.value)}",
        "",
        "## 基本信息",
        f"- 运行 ID：{run.id}",
        f"- 状态：{_status_label(run.status.value)}",
        f"- 当前节点：{run.current_step}",
        f"- 目标：{run.objective}",
        f"- 创建时间：{run.created_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 更新时间：{run.updated_at.astimezone().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## AI 指标",
        f"- 模型调用次数：{llm_summary['total_requests']}",
        f"- 累计 Tokens：{llm_summary['total_tokens']}",
        f"- 平均耗时：{llm_summary['avg_latency_ms']} 毫秒",
        f"- 累计成本：${llm_summary['total_cost_usd']}",
        "",
        "## 审核与决策",
    ]
    if run.review:
        lines.extend(
            [
                f"- 审核状态：{_status_label(run.review.status.value)}",
                f"- 是否需要人工审核：{'是' if run.review.needs_human_review else '否'}",
                f"- 评分：{run.review.score:.2f}",
                "- 审核理由：",
            ]
        )
        lines.extend([f"  - {item}" for item in run.review.reasons])
    else:
        lines.append("- 暂无审核结果")
    lines.extend(["", "## 结果 JSON", "```json", result_json, "```", "", "## 执行日志"])
    for log in run.logs:
        lines.append(f"- [{log.timestamp.astimezone().strftime('%H:%M:%S')}] {log.agent}：{log.message}")
    return "\n".join(lines)


def build_workflow_html(run: WorkflowRun, llm_summary: dict[str, Any], result_json: str) -> str:
    reasons = "".join(f"<li>{html.escape(item)}</li>" for item in (run.review.reasons if run.review else []))
    logs = "".join(
        f"<li><strong>{html.escape(log.agent)}</strong> [{log.timestamp.astimezone().strftime('%H:%M:%S')}] {html.escape(log.message)}</li>"
        for log in run.logs
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(_workflow_label(run.workflow_type.value))} 报告</title>
  <style>
    body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 32px; color: #222018; }}
    h1, h2 {{ color: #1d1b18; }}
    .card {{ border: 1px solid #d6c9b3; border-radius: 16px; padding: 16px; margin-bottom: 16px; background: #fffaf2; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    pre {{ background: #171612; color: #f6f3eb; padding: 16px; border-radius: 12px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>{html.escape(_workflow_label(run.workflow_type.value))}</h1>
  <div class="card">{html.escape(run.objective)}</div>
  <div class="grid">
    <div class="card">
      <h2>基本信息</h2>
      <ul>
        <li>运行 ID：{html.escape(run.id)}</li>
        <li>状态：{html.escape(_status_label(run.status.value))}</li>
        <li>当前节点：{html.escape(run.current_step)}</li>
      </ul>
    </div>
    <div class="card">
      <h2>AI 指标</h2>
      <ul>
        <li>模型调用次数：{llm_summary['total_requests']}</li>
        <li>累计 Tokens：{llm_summary['total_tokens']}</li>
        <li>平均耗时：{llm_summary['avg_latency_ms']} 毫秒</li>
        <li>累计成本：${llm_summary['total_cost_usd']}</li>
      </ul>
    </div>
  </div>
  <div class="card">
    <h2>审核与决策</h2>
    <ul>{reasons or '<li>暂无审核结果</li>'}</ul>
  </div>
  <div class="card">
    <h2>结果 JSON</h2>
    <pre>{html.escape(result_json)}</pre>
  </div>
  <div class="card">
    <h2>执行日志</h2>
    <ul>{logs}</ul>
  </div>
</body>
</html>"""


def build_workflow_pdf(run: WorkflowRun, llm_summary: dict[str, Any], result_json: str) -> bytes:
    lines = [
        _workflow_label(run.workflow_type.value),
        "",
        f"目标：{run.objective}",
        f"状态：{_status_label(run.status.value)}",
        f"当前节点：{run.current_step}",
        f"模型调用次数：{llm_summary['total_requests']}",
        f"累计 Tokens：{llm_summary['total_tokens']}",
        f"平均耗时：{llm_summary['avg_latency_ms']} 毫秒",
        f"累计成本：${llm_summary['total_cost_usd']}",
        "",
        "审核与决策：",
    ]
    if run.review:
        lines.append(f"审核状态：{_status_label(run.review.status.value)}")
        lines.extend([f"- {item}" for item in run.review.reasons])
    else:
        lines.append("暂无审核结果")
    lines.extend(["", "执行日志："])
    lines.extend([f"- {log.agent}：{log.message}" for log in run.logs])
    lines.extend(["", "结果 JSON：", result_json])
    return _render_pdf(lines)


def build_evaluation_markdown(item: dict[str, Any]) -> str:
    lines = [
        f"# {item['dataset_name']} 评测报告",
        "",
        "## 方案信息",
        f"- 候选方案：{item['candidate_label']}",
        f"- 基线方案：{item['baseline_label']}",
        f"- 样本数：{item['case_count']}",
        "",
        "## 总体得分",
        f"- 候选平均分：{item['candidate_avg_score']}",
        f"- 基线平均分：{item['baseline_avg_score']}",
        f"- 分数差值：{item['score_delta']}",
        "",
        "## 多维评分",
    ]
    for row in item["dimension_rows"]:
        lines.append(
            f"- {row['label']}：候选 {row['candidate_score']} / 基线 {row['baseline_score']} / 差值 {row['delta']}"
        )
    if item.get("case_rows"):
        lines.extend(["", "## 单案例明细"])
        for case in item["case_rows"]:
            lines.append(
                f"- {case['title']}：候选 {case['candidate_score']} / 基线 {case['baseline_score']} / 状态 {case['candidate_status_label']}"
            )
    return "\n".join(lines)


def build_evaluation_html(item: dict[str, Any]) -> str:
    dimension_rows = "".join(
        f"<li>{html.escape(row['label'])}：候选 {row['candidate_score']} / 基线 {row['baseline_score']} / 差值 {row['delta']}</li>"
        for row in item["dimension_rows"]
    )
    case_rows = "".join(
        f"<li><strong>{html.escape(case['title'])}</strong>：候选 {case['candidate_score']} / 基线 {case['baseline_score']} / 状态 {html.escape(case['candidate_status_label'])}</li>"
        for case in item.get("case_rows", [])
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(item['dataset_name'])} 评测报告</title>
  <style>
    body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 32px; color: #222018; }}
    .card {{ border: 1px solid #d6c9b3; border-radius: 16px; padding: 16px; margin-bottom: 16px; background: #fffaf2; }}
  </style>
</head>
<body>
  <h1>{html.escape(item['dataset_name'])} 评测报告</h1>
  <div class="card">
    <ul>
      <li>候选方案：{html.escape(item['candidate_label'])}</li>
      <li>基线方案：{html.escape(item['baseline_label'])}</li>
      <li>样本数：{item['case_count']}</li>
    </ul>
  </div>
  <div class="card">
    <h2>总体得分</h2>
    <ul>
      <li>候选平均分：{item['candidate_avg_score']}</li>
      <li>基线平均分：{item['baseline_avg_score']}</li>
      <li>分数差值：{item['score_delta']}</li>
    </ul>
  </div>
  <div class="card">
    <h2>多维评分</h2>
    <ul>{dimension_rows}</ul>
  </div>
  <div class="card">
    <h2>单案例明细</h2>
    <ul>{case_rows or '<li>暂无单案例明细</li>'}</ul>
  </div>
</body>
</html>"""


def build_evaluation_pdf(item: dict[str, Any]) -> bytes:
    lines = [
        f"{item['dataset_name']} 评测报告",
        "",
        f"候选方案：{item['candidate_label']}",
        f"基线方案：{item['baseline_label']}",
        f"样本数：{item['case_count']}",
        f"候选平均分：{item['candidate_avg_score']}",
        f"基线平均分：{item['baseline_avg_score']}",
        f"分数差值：{item['score_delta']}",
        "",
        "多维评分：",
    ]
    lines.extend(
        [f"- {row['label']}：候选 {row['candidate_score']} / 基线 {row['baseline_score']} / 差值 {row['delta']}" for row in item["dimension_rows"]]
    )
    if item.get("case_rows"):
        lines.extend(["", "单案例明细："])
        lines.extend(
            [f"- {case['title']}：候选 {case['candidate_score']} / 基线 {case['baseline_score']}" for case in item["case_rows"]]
        )
    return _render_pdf(lines)


def _render_pdf(lines: list[str]) -> bytes:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    font_name = _ensure_font_registered()
    width, height = A4
    text = pdf.beginText(40, height - 48)
    text.setFont(font_name, 10.5)
    for raw_line in lines:
        wrapped = wrap(str(raw_line), width=46, break_long_words=True, drop_whitespace=False) or [""]
        for line in wrapped:
            if text.getY() < 48:
                pdf.drawText(text)
                pdf.showPage()
                text = pdf.beginText(40, height - 48)
                text.setFont(font_name, 10.5)
            text.textLine(line)
    pdf.drawText(text)
    pdf.save()
    return buffer.getvalue()
