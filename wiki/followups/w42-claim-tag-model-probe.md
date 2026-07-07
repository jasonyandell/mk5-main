Reviewed against code on 2026-07-07 — no issues found.

All headline numbers verified against artifacts: primary table (1.483/61.70% baseline, 1.319/63.64% tags, ablation deltas +0.098 / +0.053) matches `w42/claim_tag_model_probe/ablation_results.csv`; coordinator rerun (1.384→1.295 regret, 64.52%→66.04% match, actual-policy 0.122) matches `model_metrics.csv`; slice regrets (pounce 1.299→1.031, closure 1.312→1.064, no-trump 1.627→1.478, doubles 1.356→1.260) match `metrics.json`. Script mechanism (seed split at train-seed-max=79, is_best_mean target, per-decision argmax eval) matches `run_claim_tag_model_probe.py`. W&B run `gn7xxk14` is external and unverified.

- Minor observation worth a footnote someday: in the coordinator rerun, `drop_seat_position_closure` (1.290) slightly *beats* the full tag model (1.295) — the "hurts less but still weakens" reading holds only in the primary run, so the seat/closure family contribution is within noise.
- A cheap next probe: the dynamic-84 OOD eval in `metrics.json` shows baseline and tag model producing identical scores (0.488 regret both) — worth confirming the tag features are truly all-zero on that fixture rather than silently dropped.
