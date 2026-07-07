## Corrections

- Page said the eval set was "10 held-out decisions from `burl/eval/decision_dataset.py`"; the dataset is 50 held-out decisions, of which 10 were evaluated with `--n 10` (evidence: commit 4b3ba3d message and wiki/sources/4b3ba3d.md line 42). Minor clarification only — all metrics correctly refer to the 10 evaluated.

## Follow-ups

- The move3 results directory (`burl/eval/results/move3_<timestamp>/`) is not in the repo; metrics were verified only against the commit message and source digest. Committing or archiving summary.json would make the numbers independently checkable.
- `burl/eval/data/move3_decisions.jsonl` (the dataset path in run_move3.py's usage string) is absent from the worktree — regenerating or noting it as generated-on-demand would help reproduction.
