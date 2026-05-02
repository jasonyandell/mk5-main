# w42

w42 is the Winning 42 empirical strategy-validation workstream.

It lives at the project root because it is no longer throwaway scratch work. Its
job is to turn book-derived Texas 42 strategy concepts into measurable
detectors, report buckets, model features, W&B-tracked probes, and conservative
claim-ledger evidence.

The boundary is intentionally research-shaped:

- w42 may reuse Gus corpus formats, forge E[Q] labels, and prior strategy-tag
  lessons.
- w42 does not change Gus core training, Burl behavior, or forge oracle
  semantics unless a later promotion decision explicitly scopes that work.
- w42 reports treat the book as a hypothesis source, not as ground truth.

Main entry points:

- `raw_public_state_baseline.py` - raw public-state tiny baseline
- `v0_strategy_tags_baseline.py` - raw plus v0 strategy tags
- `rich_tag_many_signal_probe.py` - richer chapter-derived tag probe
- `multi_seed_larger_eval_replication.py` - five-seed larger-eval replication
- `strategy_tags_v0.py` and `strategy_tags_v1_map/` - detector surfaces
- `*_claim_validation/` directories - claim-family validation reports
- `branch_atlas_v1/` - powered joint-world E[Q] branch atlas with hidden-holder
  impact rows and W&B progress series
- `branch_atlas_scaled_v0/` - all-declarations N=1000 branch atlas scale-up
  with bid-aware threshold plumbing and W&B progress series
- `wandb_utils.py` - shared W&B helper with failure-visible and series logging

Canonical wiki pages:

- `wiki/entities/w42.md`
- `wiki/experiments/w42-final-empirical-strategy-report.md`
- `wiki/decisions/w42-next-model-decision.md`
- `wiki/decisions/w42-promote-or-retire.md`
