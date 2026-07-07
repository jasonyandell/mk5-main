Reviewed against code on 2026-07-07 — no issues found.

Verified: `gus/model/strategy_features.py` and `gus/eval/strategy_probe.py` exist; feature dims match the page (STRATEGY_FEATURE_DIM = 68, STRATEGY_ACTION_FEATURE_DIM = 32 per action across 7 actions); `include_strategy_features` defaults to False in `gus/model/dataset_seq_world.py`; probe defaults match the "100-game" command shape (`--train gus/data/corpus_train_100.pt`, `--eval gus/data/corpus_eval_20.pt`, epochs 8, d_model 96, etc.); commits `a417295` ("Add Winning 42 strategy measurement probes") and `2d3e9d7` ("Promote w42 to top-level workstream") exist as described. Headline numbers (2.012 → 1.181, boss 0.167) corroborated independently in `wiki/log.md` (~line 1532).

Not verifiable in-repo: the corpus `.pt` files (`gus/data/` is not checked in), so the run tables themselves rest on log.md corroboration only.

## Follow-ups

- The "Runs" tables have no raw artifact in-repo; a cheap next step would be to check in the probe's stdout/JSON summaries under a small results dir so future audits don't rely on log.md.
- The "estimate the boss's variance" item (multiple E[Q] N=10 samples per decision) is a cheap probe worth confirming actually landed in the w42 family pages.
