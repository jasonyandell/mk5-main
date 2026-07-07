Reviewed against code on 2026-07-07 — no issues found.

Verified: `seed % 1000` train/val/test buckets (900/50/50) match `forge/ml/tokenize.py`; eval range 900000–909999 matches `wiki/decisions/eval-seed-holdout.md`; `forge.eq.generate` (package with `__main__.py`, `--save-joint-worlds` and adaptive flags) and `forge/cli/generate_eq_continuous.py` (`--adaptive`, `--min-samples`, `--max-samples`, `--sem-threshold`) exist with the documented flags; checkpoint `forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt` exists. `gus/data/` is absent in this checkout, as the page itself already states.

- Follow-up: `lem/data/narrations_eval.jsonl` is also absent locally; the leakage-exclusion list could note it as provenance-only, like the Gus corpora.
- Follow-up: the page references beads (`bd show t42-csw6.*`), but beads were retired to GitHub issues in 2026-06; the `owner_bead` manifest field may want a successor convention.
