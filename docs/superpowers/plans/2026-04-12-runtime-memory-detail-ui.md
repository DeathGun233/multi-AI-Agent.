# Runtime Memory Detail UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make runtime memory usage visible on the run detail page and fix the unreadable ContentAgent memory log copy.

**Architecture:** Reuse the existing `run.result` payload as the view model. Add one focused page regression test first, then minimally update the template and the single bad log string in the workflow engine.

**Tech Stack:** FastAPI, Jinja2 templates, pytest, TestClient

---

### Task 1: Lock the UI expectation with a failing page test

**Files:**
- Modify: `tests/test_workflows.py`
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Write the failing test**

```python
def test_run_detail_page_shows_runtime_memory_summary() -> None:
    first = create_support_run()
    login_as("reviewer", "reviewer123")
    approved = client.post(
        f"/api/workflows/{first['id']}/review",
        json={"approve": True, "comment": "keep owner and risk notes"},
    )
    assert approved.status_code == 200

    second = create_support_run()

    detail = client.get(f"/runs/{second['id']}")

    assert detail.status_code == 200
    assert "运行时记忆" in detail.text
    assert "AnalystAgent" in detail.text
    assert "ContentAgent" in detail.text
    assert "ReviewerAgent" in detail.text
    assert "memory hits" in detail.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_workflows.py -k run_detail_page_shows_runtime_memory_summary -q`
Expected: FAIL because the current template does not render the runtime memory section.

- [ ] **Step 3: Write minimal implementation**

Update the run detail template to render a runtime memory section from `run.result`, and fix the `ContentAgent` log string in `app/services.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_workflows.py -k run_detail_page_shows_runtime_memory_summary -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-04-12-runtime-memory-detail-ui-design.md docs/superpowers/plans/2026-04-12-runtime-memory-detail-ui.md tests/test_workflows.py app/templates/run_detail.html app/services.py
git commit -m "feat: show runtime memory on run detail page"
```

### Task 2: Verify the integrated flow

**Files:**
- Modify: `app/templates/run_detail.html`
- Modify: `app/services.py`
- Test: `tests/test_workflows.py`

- [ ] **Step 1: Run the focused workflow test slice**

Run: `python -m pytest tests/test_workflows.py -k "runtime_memory or run_detail_page" -q`
Expected: PASS

- [ ] **Step 2: Run the full suite**

Run: `python -m pytest -q`
Expected: PASS

- [ ] **Step 3: Inspect git diff**

Run: `git diff -- app/templates/run_detail.html app/services.py tests/test_workflows.py`
Expected: only the detail UI, the log copy, and test coverage changed.

