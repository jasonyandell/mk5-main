# w42-data-adapter-smoke — audit 2026-07-07

## Corrections

- Page said run commit `fe8500155f4a3dd003feb6e3a2a92c35db28ea4d`; the recorded run commit is `68e6ee3afb659e26239f72264e5c94515ef947f9` (evidence: w42/data_adapter_smoke/report.json and manifest.json `repo_commit`).
- Page's example e_q was `[-14.435, -15.853, -15.045, -14.06, -13.798, -12.936, -14.917]`; the actual example row has `[20.909, 15.15, 18.93, 16.764, 14.965, 21.372, 12.356]` (evidence: w42/data_adapter_smoke/example_row.json). The stale values look like fixture-mode output; argmax of the real values is index 5, consistent with `oracle_best_action: 5`.

## Verified

- All four referenced w42 paths exist; report/manifest confirm `source_mode: real-corpus`, seed 42, batch size 1, and every batch shape in the table matches report.json exactly.
- `w42/data_adapter_smoke.py` does load `gus/data/corpus_train_100.pt` via `JointWorldFullDataset` with a deterministic fixture fallback, as described. The corpus .pt is gitignored but present in the main checkout.

## Follow-ups

- A cheap next probe: assert in the smoke that `oracle_best_action == argmax(e_q)` on the loaded row, so a fixture/corpus mixup like the stale e_q values would fail loudly.
