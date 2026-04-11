from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.cleanup import classify_run_pollution_reason, find_test_pollution_candidates, summarize_test_pollution_candidates
from app.config import Settings
from app.db import Database
from app.repository import WorkflowRepository


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect or clean test-polluted workflow runs from the real database.")
    parser.add_argument("--apply", action="store_true", help="Actually delete matched candidates.")
    parser.add_argument("--database-url", default=None, help="Override database url. Defaults to current app setting.")
    parser.add_argument("--report", default="docs/cleanup/2026-04-12-test-run-cleanup-report.json", help="Path to write the dry-run report.")
    args = parser.parse_args()

    settings = Settings.from_env()
    database_url = args.database_url or settings.database_url
    repo = WorkflowRepository(Database(database_url))
    runs = repo.list_all()
    candidates = find_test_pollution_candidates(runs)
    summary = summarize_test_pollution_candidates(runs)
    report = {
        "database_url": database_url,
        "mode": "apply" if args.apply else "dry-run",
        **summary,
        "rows": [
            {
                "id": run.id,
                "workflow_type": run.workflow_type.value,
                "status": run.status.value,
                "current_step": run.current_step,
                "created_at": run.created_at.isoformat(),
                "reason": classify_run_pollution_reason(run),
            }
            for run in candidates
        ],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"mode={report['mode']}")
    print(f"database_url={database_url}")
    print(f"candidate_count={report['candidate_count']}")
    print(f"reason_counts={report['reason_counts']}")
    print(f"report={report_path}")

    if args.apply:
        deleted_ids = repo.delete_runs(report["candidate_ids"])
        print(f"deleted_count={len(deleted_ids)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
