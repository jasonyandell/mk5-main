Reviewed against code on 2026-07-07 — no issues found.

All headline numbers (257/300/43 contrasts, mean −1.95, CI [−2.94, −1.07], count-in-trick N=86 mean −0.02 CI [−2.16, +2.15], phase slices, oracle-choice rates) match `summary.json` and `slice_breakdown.csv` in `w42/book_validation_v1/wave2/probes/t42-jysl_low_trump_trap/`. Script mechanism (dominant-trump-global definition, within-snapshot paired contrast, bootstrap 1000-iter CI) matches `run_analysis.py`.

Notes:
- Source corpus dir `gus/data/` (corpus_train_chunk_*.pt) is not present in this worktree (gitignored data); unverifiable here, but the builder script does reference it.
- Follow-ups:
  - A cheap next probe: segment count-in-trick cases by opponent trump-holding inference (belief-conditioned) to see if the trap effect emerges in the sub-configuration the book actually describes.
  - Pool count-in-trick cases across wave2 probes sharing the corpus to push N past 100 for the underpowered subgroup.
