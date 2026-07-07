Reviewed against code on 2026-07-07 — no issues found.

- Commit f578bfa exists and matches the description (local STaR runner, llama.cpp CPU, Q4_K_M GGUF, reuses parse_play/grade_k1 from star_harness.py).
- `grade_k1` in `lem/gemma_star/star_harness.py` implements exactly the stated criterion (`gemma_eq >= bot_eq`, with illegal/parse_fail buckets).
- Results table (60/30/10/0 on n=10) matches the source digest `wiki/sources/f578bfa.md`; the raw run log lives only in the session transcript, so the counts are verifiable only against the digest, not a repo artifact.
- `lem/data/narrations_train.jsonl` is gitignored generated data, absent from the worktree — expected, not a stale path.

Follow-ups:
- A cheap next probe: rerun with a fixed `--seed` and n=50 so the baseline has a repo-committed JSON artifact instead of living only in a digest.
- Could quantify the "structural ceiling" claim directly: fraction of trick-6 decisions where >1 legal action ties the argmax E[Q] (computable offline from `all_eq`, no inference needed).
