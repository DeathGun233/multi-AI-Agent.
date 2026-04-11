from __future__ import annotations

from collections import Counter
from typing import Any

from app.models import WorkflowRun, WorkflowType


def _all_calls_are_llm_disabled_fallback(run: WorkflowRun) -> bool:
    llm_calls = [log.llm_call for log in run.logs if log.llm_call is not None]
    return bool(llm_calls) and all(call.used_fallback and call.error == "llm_disabled" for call in llm_calls)


def classify_run_pollution_reason(run: WorkflowRun) -> str | None:
    if not _all_calls_are_llm_disabled_fallback(run):
        return None

    payload = run.input_payload or {}
    if run.workflow_type == WorkflowType.SALES_FOLLOWUP:
        if (
            payload.get("period") == "2026-W13"
            and payload.get("region") == "华东"
            and payload.get("sales_reps") == ["王晨", "李雪"]
            and payload.get("focus_metric") == "conversion_rate"
        ):
            return "pytest_sales_fixture"

    if run.workflow_type == WorkflowType.SUPPORT_TRIAGE:
        tickets = payload.get("tickets")
        if isinstance(tickets, list) and len(tickets) == 1 and isinstance(tickets[0], dict):
            ticket = tickets[0]
            if ticket.get("customer") == "示例客户" and ticket.get("message") == "生产接口持续报错，今天必须恢复，否则影响上线。":
                return "pytest_support_fixture"

    return None


def find_test_pollution_candidates(runs: list[WorkflowRun]) -> list[WorkflowRun]:
    return [run for run in runs if classify_run_pollution_reason(run) is not None]


def summarize_test_pollution_candidates(runs: list[WorkflowRun]) -> dict[str, Any]:
    candidates = find_test_pollution_candidates(runs)
    reason_counter = Counter(classify_run_pollution_reason(run) for run in candidates)
    return {
        "candidate_count": len(candidates),
        "reason_counts": dict(reason_counter),
        "candidate_ids": [run.id for run in candidates],
    }
