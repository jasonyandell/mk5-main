# w42-strategy-tags-v0 — audit 2026-07-07

## Corrections

- Page said run commit at artifact generation was `8df0c3b1f03893cc4c059762de1fa1dd0c84ca06`; the committed artifact records `e3bfefa7f3f01b32c39b74545badbb6d1b27a642` (evidence: `w42/strategy_tags_v0/report.json` `repo_commit` field). `8df0c3b` exists but is the earlier data-smoke path fix, not the run commit the artifact records.

## Verified

- All tag dimensions (68 global / 32 per action slot x 7 slots), group widths, sample-row values, seeds, command, and bead id match `w42/strategy_tags_v0/report.json`, `summary.csv`, and `example_row.json`.
- `w42/strategy_tags_v0.py` does import `gus.model.strategy_features` constants and `JointWorldFullDataset(include_strategy_features=True)`, with a fixture fallback as described.

## Follow-ups

- `gus/data/corpus_train_100.pt` is absent in this worktree (expected — corpus lives on the main checkout); the page's fixture-fallback note covers this.
- Cheap next probe: run with `--limit` > 1 across the chunked corpus globs to check tag stability beyond a single deterministic row. (Caveat: the wrapper consumes only the first available declared path, so on the main checkout this exercises more rows of `corpus_train_100.pt`, not the chunk globs.)

## Review (second pass, 2026-07-07)

- Verified — corrections stand. The commit fix is confirmed at one level deeper than the auditor cited: the original fixture-mode run (`git show 2418c42f:scratch/w42/strategy_tags_v0/report.json`) recorded `repo_commit: 8df0c3b1...`, and `fe083b87` ("Verify w42 strategy tags on real corpus") regenerated the artifact with `repo_commit: e3bfefa7...`. The page describes the real-corpus run (its sample-row values match the regenerated `w42/strategy_tags_v0/example_row.json` and `summary.csv`), so `e3bfefa7` is the right commit and the old `8df0c3b1` was a stale carry-over from the fixture run.
- Independently re-checked group widths (68 global / 32x7 action), seeds (all 42), bead `t42-csw6.7`, command, and sample-row fields against `w42/strategy_tags_v0/report.json`, `summary.csv`, `example_row.json`; and the imports/fallback in `w42/strategy_tags_v0.py` (lines 190, 220-232, 416-418). All match the page.
- Added a caveat to the `--limit` follow-up above: the wrapper only consumes the first available declared corpus path, so the probe would not span the chunk globs as originally phrased.
- `last_updated: afd4802` was not bumped by the audit edit; left as-is since no page in the 45a7e35 batch bumped it (consistent batch convention).
