# Issue #2 Minimal Tool Calling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal model-guided `OperatorAgent` that chooses one allowed tool, executes it once, falls back safely, and persists `operator_context`.

**Architecture:** Keep the existing workflow graph and downstream agent contracts intact. Add a small `OperatorDecision` schema plus an `OperatorAgent` wrapper around the current `ToolCenter`, and only refactor `ToolCenter` enough to expose tool allowlists, deterministic defaults, and named execution. Persist `operator_context` beside `raw_result` so operator behavior becomes auditable without introducing multi-turn tool loops.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, LangGraph, pytest

---

### Task 1: Add failing operator decision tests

**Files:**
- Modify: `tests/test_workflows.py`
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Write the failing unit tests for operator model selection and fallback**

```python
class FakeOperatorLLM:
    def __init__(self, payload: dict | None = None, *, fail: bool = False) -> None:
        self.payload = payload or {}
        self.fail = fail

    def generate_json(self, **_: object) -> SimpleNamespace:
        if self.fail:
            raise RuntimeError("operator unavailable")
        return SimpleNamespace(
            payload=self.payload,
            call=SimpleNamespace(used_fallback=False, error=None, validation_error=None),
        )


def test_operator_uses_model_selected_tool_when_allowed() -> None:
    tool_center = ToolCenter(ExternalDataService(Settings()))
    operator = OperatorAgent(llm_service=FakeOperatorLLM(
        {
            "selected_tool": "sales_analytics_tool",
            "reason": "need funnel metrics first",
            "confidence": 0.93,
            "fallback_required": False,
        }
    ), tool_center=tool_center)

    raw_result, tool_call, operator_context, llm_call = operator.execute(
        request=WorkflowRequest(
            workflow_type=WorkflowType.SALES_FOLLOWUP,
            input_payload={"period": "2026-W13", "region": "华东"},
        ),
        execution_profile=object(),
    )

    assert tool_call.name == "sales_analytics_tool"
    assert operator_context["decision_source"] == "model"
    assert operator_context["executed_tool"] == "sales_analytics_tool"
    assert operator_context["used_fallback"] is False
    assert llm_call is not None
    assert raw_result["lead_count"] > 0


def test_operator_falls_back_when_selected_tool_is_not_allowed() -> None:
    tool_center = ToolCenter(ExternalDataService(Settings()))
    operator = OperatorAgent(llm_service=FakeOperatorLLM(
        {
            "selected_tool": "marketing_brief_tool",
            "reason": "wrong tool",
            "confidence": 0.95,
            "fallback_required": False,
        }
    ), tool_center=tool_center)

    _, tool_call, operator_context, _ = operator.execute(
        request=WorkflowRequest(
            workflow_type=WorkflowType.SALES_FOLLOWUP,
            input_payload={"period": "2026-W13"},
        ),
        execution_profile=object(),
    )

    assert tool_call.name == "sales_analytics_tool"
    assert operator_context["decision_source"] == "rule"
    assert operator_context["used_fallback"] is True
    assert operator_context["fallback_reason"] == "tool_not_allowed"


def test_operator_falls_back_when_confidence_is_low() -> None:
    tool_center = ToolCenter(ExternalDataService(Settings()))
    operator = OperatorAgent(llm_service=FakeOperatorLLM(
        {
            "selected_tool": "sales_analytics_tool",
            "reason": "not sure",
            "confidence": 0.61,
            "fallback_required": False,
        }
    ), tool_center=tool_center)

    _, tool_call, operator_context, _ = operator.execute(
        request=WorkflowRequest(
            workflow_type=WorkflowType.SALES_FOLLOWUP,
            input_payload={"period": "2026-W13"},
        ),
        execution_profile=object(),
    )

    assert tool_call.name == "sales_analytics_tool"
    assert operator_context["used_fallback"] is True
    assert operator_context["fallback_reason"] == "low_confidence"


def test_operator_falls_back_when_model_is_unavailable() -> None:
    tool_center = ToolCenter(ExternalDataService(Settings()))
    operator = OperatorAgent(llm_service=FakeOperatorLLM(fail=True), tool_center=tool_center)

    _, tool_call, operator_context, llm_call = operator.execute(
        request=WorkflowRequest(
            workflow_type=WorkflowType.SALES_FOLLOWUP,
            input_payload={"period": "2026-W13"},
        ),
        execution_profile=object(),
    )

    assert tool_call.name == "sales_analytics_tool"
    assert operator_context["used_fallback"] is True
    assert operator_context["fallback_reason"] == "model_error"
    assert llm_call is None
```

- [ ] **Step 2: Run the operator-focused tests to verify RED**

Run: `python -m pytest tests/test_workflows.py -k "operator_uses_model_selected_tool_when_allowed or operator_falls_back_when_selected_tool_is_not_allowed or operator_falls_back_when_confidence_is_low or operator_falls_back_when_model_is_unavailable" -q`

Expected: FAIL because `OperatorAgent` and `OperatorDecision` do not exist yet.

- [ ] **Step 3: Add a failing workflow-level persistence test**

```python
def test_workflow_run_persists_operator_context() -> None:
    body = create_sales_run()

    assert "operator_context" in body["result"]
    assert "executed_tool" in body["result"]["operator_context"]
    assert "used_fallback" in body["result"]["operator_context"]
```

- [ ] **Step 4: Run the persistence test to verify RED**

Run: `python -m pytest tests/test_workflows.py -k "workflow_run_persists_operator_context" -q`

Expected: FAIL because `operator_context` is not persisted yet.

- [ ] **Step 5: Commit the red tests**

```bash
git add tests/test_workflows.py
git commit -m "test: add operator tool selection red tests"
```

### Task 2: Implement OperatorDecision and minimal ToolCenter registry support

**Files:**
- Modify: `app/models.py`
- Modify: `app/services.py`
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Add `OperatorDecision` to `app/models.py`**

```python
class OperatorDecision(BaseModel):
    selected_tool: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    fallback_required: bool = False
```

- [ ] **Step 2: Add minimal tool registry helpers to `ToolCenter`**

```python
def tool_choices_for(self, workflow_type: WorkflowType, payload: dict[str, Any]) -> list[dict[str, str]]:
    if workflow_type == WorkflowType.SALES_FOLLOWUP:
        return [{"name": "sales_analytics_tool", "description": "aggregate funnel and risk metrics"}]
    if workflow_type == WorkflowType.MARKETING_CAMPAIGN:
        return [{"name": "marketing_brief_tool", "description": "prepare campaign brief and channel notes"}]
    if workflow_type == WorkflowType.SUPPORT_TRIAGE:
        choices = [{"name": "support_triage_tool", "description": "classify tickets and escalation risk"}]
        data_source = payload.get("data_source", {})
        provider = data_source.get("provider") if isinstance(data_source, dict) else None
        if provider:
            choices.append({"name": f"{provider}_tool", "description": f"load support tickets from {provider}"})
        return choices
    return [{"name": "meeting_minutes_tool", "description": "extract action items from meeting notes"}]


def default_tool_for(self, workflow_type: WorkflowType, payload: dict[str, Any]) -> str:
    choices = self.tool_choices_for(workflow_type, payload)
    if workflow_type == WorkflowType.SUPPORT_TRIAGE:
        for item in choices:
            if item["name"] != "support_triage_tool":
                return item["name"]
    return choices[0]["name"]
```

- [ ] **Step 3: Add named execution entrypoint to `ToolCenter`**

```python
def run_named(self, tool_name: str, payload: dict[str, Any]) -> tuple[dict[str, Any], ToolCall]:
    if tool_name == "sales_analytics_tool":
        result = self._sales_analytics(payload)
        return result, ToolCall(name=tool_name, input=payload, output=result)
    if tool_name == "marketing_brief_tool":
        result = self._marketing_brief(payload)
        return result, ToolCall(name=tool_name, input=payload, output=result)
    if tool_name in {"support_triage_tool", "github_issues_tool", "nyc_311_tool", "stack_overflow_tool", "hacker_news_tool"}:
        result, actual_name = self._support_triage(payload)
        return result, ToolCall(name=actual_name, input=payload, output=result)
    if tool_name == "meeting_minutes_tool":
        result = self._meeting_extract(payload)
        return result, ToolCall(name=tool_name, input=payload, output=result)
    raise ValueError(f"unknown tool: {tool_name}")
```

- [ ] **Step 4: Run the focused tests and keep them red only on missing operator execution logic**

Run: `python -m pytest tests/test_workflows.py -k "operator or workflow_run_persists_operator_context" -q`

Expected: FAIL because the schema exists but `OperatorAgent` and workflow persistence still do not.

- [ ] **Step 5: Commit the registry/schema groundwork**

```bash
git add app/models.py app/services.py
git commit -m "feat: add operator decision schema and tool registry helpers"
```

### Task 3: Implement OperatorAgent and wire operator_context into workflow results

**Files:**
- Modify: `app/services.py`
- Modify: `tests/test_workflows.py`
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Add `operator_context` to workflow state and result payload shape**

```python
class WorkflowState(TypedDict, total=False):
    ...
    operator_context: dict[str, Any]
```

And include it when building `run.result`:

```python
"operator_context": state.get("operator_context", {}),
```

- [ ] **Step 2: Implement minimal `OperatorAgent`**

```python
class OperatorAgent:
    MIN_CONFIDENCE = 0.7

    def __init__(self, tool_center: ToolCenter, llm_service: LLMService | None = None) -> None:
        self.tool_center = tool_center
        self.llm_service = llm_service

    def execute(self, *, request: WorkflowRequest, execution_profile: ExecutionProfile) -> tuple[dict[str, Any], ToolCall, dict[str, Any], Any]:
        choices = self.tool_center.tool_choices_for(request.workflow_type, request.input_payload)
        default_tool = self.tool_center.default_tool_for(request.workflow_type, request.input_payload)
        model_decision, fallback_reason, llm_call = self._select_tool(
            request=request,
            execution_profile=execution_profile,
            choices=choices,
            default_tool=default_tool,
        )
        if model_decision is not None and fallback_reason is None:
            raw_result, tool_call = self.tool_center.run_named(model_decision.selected_tool, request.input_payload)
            return raw_result, tool_call, {
                "decision_source": "model",
                "selected_tool": model_decision.selected_tool,
                "executed_tool": tool_call.name,
                "used_fallback": False,
                "fallback_reason": None,
                "tool_choices": [item["name"] for item in choices],
                "decision_reason": model_decision.reason,
                "confidence": model_decision.confidence,
            }, llm_call
        raw_result, tool_call = self.tool_center.run_named(default_tool, request.input_payload)
        return raw_result, tool_call, {
            "decision_source": "rule",
            "selected_tool": model_decision.selected_tool if model_decision is not None else None,
            "executed_tool": tool_call.name,
            "used_fallback": True,
            "fallback_reason": fallback_reason or "llm_unavailable",
            "tool_choices": [item["name"] for item in choices],
            "decision_reason": model_decision.reason if model_decision is not None else "use deterministic operator tool",
            "confidence": model_decision.confidence if model_decision is not None else None,
        }, llm_call
```

- [ ] **Step 3: Add `_select_tool()` validation inside `OperatorAgent`**

```python
def _select_tool(...):
    if self.llm_service is None or execution_profile is None:
        return None, "llm_unavailable", None
    fallback_payload = {
        "selected_tool": default_tool,
        "reason": "use deterministic operator tool",
        "confidence": 0.0,
        "fallback_required": True,
    }
    try:
        response = self.llm_service.generate_json(
            route_target="operator",
            system_prompt="...",
            user_prompt=json.dumps({...}, ensure_ascii=False),
            fallback=fallback_payload,
            execution_profile=execution_profile,
            response_model=OperatorDecision,
        )
    except Exception:
        return None, "model_error", None

    call = getattr(response, "call", None)
    try:
        decision = OperatorDecision.model_validate(response.payload)
    except ValidationError:
        return None, "schema_validation_failed", call
    allowed_tools = {item["name"] for item in choices}
    if decision.selected_tool not in allowed_tools:
        return decision, "tool_not_allowed", call
    if decision.confidence < self.MIN_CONFIDENCE:
        return decision, "low_confidence", call
    if decision.fallback_required:
        return decision, "fallback_required", call
    return decision, None, call
```

- [ ] **Step 4: Wire `_operator_step()` through `OperatorAgent`**

```python
def _operator_step(self, state: WorkflowState) -> WorkflowState:
    run = state["run"]
    request = state["request"]
    run.touch(status=RunStatus.EXECUTING, current_step="operator")
    raw_result, tool_call, operator_context, llm_call = self.operator_agent.execute(
        request=request,
        execution_profile=state["execution_profile"],
    )
    run.add_log(
        "OperatorAgent",
        f"已执行工具：{tool_call.name}（source={operator_context['decision_source']}）。",
        tool_call=tool_call,
        llm_call=llm_call,
    )
    state["raw_result"] = raw_result
    state["operator_context"] = operator_context
    state["last_node"] = "operator"
    return state
```

- [ ] **Step 5: Update engine initialization to construct `OperatorAgent`**

```python
self.operator_agent = OperatorAgent(self.tool_center, self.llm_service)
```

- [ ] **Step 6: Run operator-focused tests to verify GREEN**

Run: `python -m pytest tests/test_workflows.py -k "operator or workflow_run_persists_operator_context" -q`

Expected: PASS

- [ ] **Step 7: Commit the operator minimal loop**

```bash
git add app/services.py tests/test_workflows.py
git commit -m "feat: add minimal operator tool selection loop"
```

### Task 4: Verify compatibility and full suite stability

**Files:**
- Modify: `tests/test_workflows.py` (only if compatibility assertions need minor updates)
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Add or tighten a compatibility assertion on workflow results**

```python
def test_sales_workflow_runs_with_selected_model_prompt_and_routing() -> None:
    body = create_sales_run()
    ...
    assert "operator_context" in body["result"]
    assert body["result"]["operator_context"]["executed_tool"] == "sales_analytics_tool"
```

- [ ] **Step 2: Run focused workflow/operator verification**

Run: `python -m pytest tests/test_workflows.py -k "operator or workflow" -q`

Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest -q`

Expected: PASS

- [ ] **Step 4: Run diff hygiene check**

Run: `git diff --check`

Expected: no whitespace or patch-format errors

- [ ] **Step 5: Commit the verification-compatible final state**

```bash
git add app/services.py tests/test_workflows.py
git commit -m "test: verify operator tool calling compatibility"
```
