# w42-strategy-tags-v0 — audit 2026-07-07

## Corrections

- Page said run commit at artifact generation was `8df0c3b1f03893cc4c059762de1fa1dd0c84ca06`; the committed artifact records `e3bfefa7f3f01b32c39b74545badbb6d1b27a642` (evidence: `w42/strategy_tags_v0/report.json` `repo_commit` field). `8df0c3b` exists but is the earlier data-smoke path fix, not the run commit the artifact records.

## Verified

- All tag dimensions (68 global / 32 per action slot x 7 slots), group widths, sample-row values, seeds, command, and bead id match `w42/strategy_tags_v0/report.json`, `summary.csv`, and `example_row.json`.
- `w42/strategy_tags_v0.py` does import `gus.model.strategy_features` constants and `JointWorldFullDataset(include_strategy_features=True)`, with a fixture fallback as described.

## Follow-ups

- `gus/data/corpus_train_100.pt` is absent in this worktree (expected — corpus lives on the main checkout); the page's fixture-fallback note covers this.
- Cheap next probe: run with `--limit` > 1 across the chunked corpus globs to check tag stability beyond a single deterministic row.
