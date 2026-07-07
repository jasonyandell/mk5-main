# w42-plateau-probe — audit

Reviewed against code on 2026-07-07 — no issues found.

Verified: head_8 +0.38 [+0.09, +0.67] and +0.42 [+0.12, +0.72] (both 287/512) against `champion/evidence/jud_v0/ab_definitive_512_r8_summary.json` and `ab_replication_9M_512_r8_summary.json`; head_12 +0.21/+0.37 against the r12 summaries; full r0–r12 table, offense/made/notrump shares, and ECE 0.053→0.010 against `loop_metrics.json`; SP_GAMES=1000 / AB_GAMES=256 against `scratch/jud-v0/loop/run_loop.py`; commits 68fda7b/0bdd4d5/2c652f0 exist and match descriptions; `champion/margin_net_r8.pt` exists.

## Follow-ups

- `scratch/jud-v0/loop/run_loop.py` is gitignored (exists only on this machine); consider copying the loop driver into `champion/evidence/jud_v0/` so the recipe survives a scratch cleanup.
- A cheap next probe before jud v1: one round at 2–3k SP games with a slightly larger MLP head, to distinguish "capacity" from "mechanism" as the next binding constraint.
