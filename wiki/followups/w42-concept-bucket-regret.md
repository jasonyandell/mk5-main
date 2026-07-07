Reviewed against code on 2026-07-07 — no issues found.

- All headline metrics verified against artifacts: raw final 2.834/56.61%, tagged final 2.099/63.75%, tagged best 2.000/63.93% (epoch 7) match `w42/raw_public_state_baseline/metrics.json` and `w42/v0_strategy_tags_baseline/metrics.json`; the full bucket table matches `w42/concept_bucket_regret_report/concept_bucket_regret.csv` exactly.
- Note: source corpora `gus/data/corpus_train_100.pt` / `corpus_eval_20.pt` are gitignored local data files, so the manifest paths are unverifiable in-repo (expected, not an error).
- Cheap next probe: the CSV already carries raw-final-vs-tagged-best deltas per bucket; the page only surfaces them in prose for the secondary column — a small table row could make the best-epoch comparison first-class.
