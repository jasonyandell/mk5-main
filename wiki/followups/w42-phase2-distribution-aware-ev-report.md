Reviewed against code on 2026-07-07 — no issues found.

- Note: summary.json's own `walkthrough_example` is sample_001_decl_1/move 18; the page's walkthrough (sample_002_decl_2/move 16) comes from example_decisions.json and every number matches — no correction needed, but a reader diffing against summary.json alone might be confused.
- Cheap next probe: 81 of 140 decisions have multiple legal PDF actions (summary.json `decisions_with_multiple_legal_pdf_actions`) — the flag rates could be re-expressed per multi-action decision rather than per row.
