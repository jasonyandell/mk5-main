Reviewed against code on 2026-07-07 — no issues found.

Verified: all artifact paths exist under `w42/v0_strategy_tags_baseline/`; commit `3a5575f` exists; headline metrics (tagged best 2.000 regret / 63.93% match, raw prior best 2.471 / 59.64%, reloaded raw final 2.834 exactly matching prior final, deltas -0.471 / -0.734), training history, and bucket-slice table all match `metrics.json` / `run.json`; tag dims (68 global, 32 action-local) and model shape confirmed in `w42/v0_strategy_tags_baseline.py` and `gus/model/strategy_features.py`.

- The `bucket_comparison_raw_final_vs_tagged_best` slices show several action groups (hand_shape, identity, pip_pressure, slot) with identical n=381 stats — those groups fire on essentially every decision, so a cheap next probe is to drop always-on groups from the diagnostic table.
- Multi-seed repeat (already listed in Next Checks) remains the cheapest way to know if the -0.47 best-epoch delta is real; epoch 6 vs 7 disagreement on match rate hints at noise.
