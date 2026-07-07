# w42-champion-teaching-battery — audit

Reviewed against code on 2026-07-07 — no issues found.

All headline numbers (7,168 decisions; 19,264 action rows; 1,484 labeled rows; all six receipt rows and label counts) match `w42/book_validation_v1/wave5/champion_teaching_battery/summary.json`, `receipts.csv`, and `action_rows.csv`. Checkpoint, probe script, and artifact paths exist; the ten utility-lens columns are present in `action_rows.csv`; manifest confirms the exact reproduction command and that the central ledger was not touched.

## Follow-ups
- The within-CI safe-donation result (N_paired=150) might resolve with more seeds — a 512-seed rerun of just the ch04 slice is cheap (~40 min at the observed 574s/128 seeds).
- The page's hypothesis that safe donation is "context-limited" is testable: split the N=150 pairs by whether the actor holds another non-count safe play, using the existing action_rows.csv — no new games needed.
- The ledger note cites `phase4_claim_completion_board/completion_board.csv`; the actual file is at `w42/phase4_claim_completion_board/completion_board.csv` (path resolves, just repo-root-relative — no correction needed, but worth normalizing if the wiki adopts a path convention).
