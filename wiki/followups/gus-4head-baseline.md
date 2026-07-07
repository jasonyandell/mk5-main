# gus-4head-baseline — audit 2026-07-07

## Corrections

- Page said the ~3400× Q-supervision density comes from "M sampled worlds × 7 actions vs 84 belief slots"; code says the factor is avg M≈3400 sampled worlds per decision itself (evidence: gus/model/dataset_seq_world.py docstring, "avg M=3400" and "3400×-denser supervision signal"). The page's formula would give ~283×, not 3400×.

## Verified

- Metrics table (π_me 57.9%, belief top-1 34.5%, V MAE 11.9, Q MAE 18.4), per-decision π_me, no-overfit finding, and chunked-generation/INT_MAX note all match commit da21f52's message verbatim.
- Referenced modules exist: gus/model/student.py, gus/model/dataset_seq_world.py, gus/train/train_v1_full.py.
- Seed convention (0-99 train, 900000+ eval) confirmed in gus/BUILD_PLAN.md and gus/GEN_FLEET.md.

## Follow-ups

- Metrics are transcribed from the commit message only; no training-log/JSON artifact for this run exists in-repo, so the 57.9% etc. are unverifiable beyond the commit text. A cheap probe: re-run train_v1_full.py on the 100g corpus to confirm reproducibility.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. Re-derived the 3400× correction from gus/model/dataset_seq_world.py (lines 20, 53: "avg M=3400", one random world per item) and commit da21f52 ("3400× signal per decision vs belief-alone"); the original page's M×7/84 formula would give ~283×, so the fix is right and the diff touched nothing else. Metrics, per-decision table, seed convention (gus/GEN_FLEET.md:113,119), and module paths all check out; the reproducibility follow-up remains valid (no in-repo artifact contains these metrics).
