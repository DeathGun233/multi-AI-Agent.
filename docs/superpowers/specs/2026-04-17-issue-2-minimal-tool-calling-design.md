# Issue #2 Minimal Tool Calling Design

## Goal

Upgrade `OperatorAgent` from a fixed `ToolCenter.run()` dispatcher into a minimal model-guided tool selection step that still preserves the current workflow contract.

This round is intentionally limited to a single tool decision and a single tool execution. It does not attempt to build a multi-turn observe -> replan -> tool loop.

## Problem Statement

Current operator behavior is hard-coded in `WorkflowEngine._operator_step()`:

- it does not ask the model whether a tool should be used
- it does not let the model choose among multiple tools
- it does not persist operator decision metadata beyond the raw `ToolCall`
- it keeps tool selection logic concentrated inside `ToolCenter.run(workflow_type, payload)`

This leaves `OperatorAgent` as a fixed execution node rather than an agentic tool user.

## Scope

This design only includes:

- one structured `OperatorDecision` model
- one model decision attempt per workflow run
- one tool execution per operator step
- deterministic fallback to the current workflow-type-based tool path
- persisted operator audit context in run results
- focused tests for success and fallback paths

## Out of Scope

This design explicitly does not include:

- multi-round tool calling
- observe -> replan -> tool re-entry
- dynamic agent registration
- changes to `RouterAgent` behavior
- cleanup scripts, pollution repair, or unrelated observability work
- UI redesign beyond existing JSON/result visibility
- tool-calling changes for `AnalystAgent`, `ContentAgent`, or `ReviewerAgent`

## Recommended Approach

Use a minimal layered design:

1. Introduce a small `OperatorDecision` schema for model output.
2. Refactor `ToolCenter` just enough to expose an allowed tool list and a unified execution entrypoint.
3. Add an `OperatorAgent` that chooses a tool with LLM structured output, then executes exactly one tool.
4. Fall back to the current deterministic tool path whenever the model is unavailable, invalid, low-confidence, or asks for fallback.
5. Persist `operator_context` so the decision path is auditable.

This keeps the blast radius small and avoids repeating the oversized scope of PR #6.

## Architecture

### 1. OperatorDecision

Add a local structured model for operator routing:

- `selected_tool: str`
- `reason: str`
- `confidence: float`
- `fallback_required: bool`

Validation rules:

- `selected_tool` must be in the tool allowlist for the current workflow
- `confidence` must be between `0.0` and `1.0`
- if `confidence < 0.7`, the engine must fall back
- if `fallback_required == true`, the engine must fall back

### 2. Tool Registry Shape

Do not introduce a large generic framework yet. Keep `ToolCenter` as the owner of tool execution, but split responsibilities:

- `tool_choices_for(workflow_type, payload) -> list[dict[str, str]]`
- `default_tool_for(workflow_type, payload) -> str`
- `run_named(tool_name, payload) -> tuple[dict[str, Any], ToolCall]`

Current concrete tools remain the same logical tools already embedded in `ToolCenter`:

- `sales_analytics_tool`
- `marketing_brief_tool`
- `support_triage_tool` and provider-backed support variants
- `meeting_minutes_tool`

For this round, each workflow may still expose only one or a small fixed set of tools. The important change is that the operator selects from an explicit allowlist instead of `WorkflowEngine` directly dispatching by workflow type.

### 3. OperatorAgent

Add a dedicated `OperatorAgent` with a method shaped roughly like:

- input: `WorkflowRequest`, optional memory/context, execution profile
- output: `raw_result`, `ToolCall`, `operator_context`, optional `llm_call`

Decision flow:

1. Build the allowed tool list for the current workflow.
2. Ask the model for a structured `OperatorDecision`.
3. Validate the tool selection.
4. If valid, execute the named tool exactly once.
5. If invalid or unavailable, execute the deterministic default tool.

### 4. operator_context

Persist a minimal operator audit structure in `run.result`:

- `decision_source`: `model` or `rule`
- `selected_tool`: tool requested by model, or `null`
- `executed_tool`: tool actually run
- `used_fallback`: bool
- `fallback_reason`: string or `null`
- `tool_choices`: allowed tool names for this request
- `decision_reason`: operator reasoning text
- `confidence`: model confidence, if present

This is enough for auditability now and provides a stable base for future multi-turn operator work.

## Workflow Integration

`WorkflowEngine._operator_step()` should stop calling `ToolCenter.run()` directly.

Instead it should:

1. call `OperatorAgent.execute(...)`
2. store `state["raw_result"]`
3. store `state["operator_context"]`
4. log both the operator LLM call and the concrete tool call

`run.result` should include `operator_context` alongside the existing execution/profile/result fields.

## Error Handling

Fallback must occur in any of these conditions:

- LLM disabled
- operator model call raises
- JSON parsing or schema validation fails
- `selected_tool` not in current allowlist
- `confidence < 0.7`
- `fallback_required == true`

Fallback behavior must preserve current user-visible workflow compatibility by running the same deterministic tool path used today.

## Testing Strategy

Use TDD and keep tests independent from planner/analyst/content/reviewer call-count assertions.

Minimum tests:

1. operator uses model-selected tool when selection is valid
2. operator falls back when selected tool is not allowed
3. operator falls back when confidence is below threshold
4. operator falls back when model is unavailable
5. workflow run persists `operator_context`
6. fallback path preserves current raw result shape for existing workflow tests

Full test command for this round remains:

- `python -m pytest tests/test_workflows.py -k "operator or workflow" -q`
- `python -m pytest -q`

The exact focused selector can be adjusted once the final test names exist, but the new tests should stay narrow and operator-specific.

## Low-Risk Follow-Ons

These are acceptable to include only if they remain tightly scoped and do not broaden the PR:

- add one helper for operator tool choice normalization to keep `services.py` readable
- add one compact operator result summary string in logs

These should not be included in this round:

- cleanup database scripts
- review queue UI changes
- observability dashboards
- additional agent memory expansions

## Implementation Boundaries

The implementation should prefer:

- adding `OperatorDecision` near existing local structured models
- adding `OperatorAgent` beside the other agents in `app/services.py`
- minimizing changes to template files
- avoiding broad refactors to unrelated agent code

If `app/services.py` starts growing awkwardly during implementation, the only acceptable structural extraction in this round is a narrowly-scoped helper for operator tool resolution. No general framework refactor.

## Success Criteria

This phase is complete when:

- `OperatorAgent` performs one structured tool selection attempt
- invalid model output safely falls back to the current deterministic path
- `operator_context` is persisted and auditable
- existing workflow behavior remains compatible
- targeted tests and full test suite pass
