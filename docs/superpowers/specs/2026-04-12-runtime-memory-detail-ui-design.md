# Runtime Memory Detail UI Design

## Goal

Expose runtime memory usage on the run detail page so operators and reviewers can see which agents used historical context, how many memory hits were injected, and a compact summary of the related runs and feedback samples without digging through raw JSON.

## Scope

- Add a dedicated runtime memory section to the run detail page.
- Show `analyst_context`, `content_context`, and `reviewer_context` when present.
- Surface memory hit counts and compact rows for recent runs and feedback samples.
- Fix the garbled `ContentAgent` memory log message so timeline text is readable.
- Add page-level regression coverage.

## Non-Goals

- No new memory retrieval logic.
- No export/report format changes.
- No `MemoryService` refactor in this slice.

## Design

The page will keep using `run.result` as the source of truth. The template will render a new runtime memory section only when at least one agent context exists. Each agent gets a compact card with:

- memory hit count
- dominant keywords / common review reasons when available
- recent run summaries
- recent feedback summaries

The UI stays read-only and follows the existing card/grid pattern in the current run detail template. This keeps the change small and avoids new route or serializer logic.

## Test Strategy

- Add a failing page test that creates memory-bearing workflow runs and asserts the run detail page shows the runtime memory section plus agent labels and hit counts.
- Keep existing workflow memory tests intact.

