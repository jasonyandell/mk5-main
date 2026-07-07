# w42-data-adapter-smoke — audit 2026-07-07

## Corrections

- Page said run commit `fe8500155f4a3dd003feb6e3a2a92c35db28ea4d`; the recorded run commit is `68e6ee3afb659e26239f72264e5c94515ef947f9` (evidence: w42/data_adapter_smoke/report.json and manifest.json `repo_commit`).
- Page's example e_q was `[-14.435, -15.853, -15.045, -14.06, -13.798, -12.936, -14.917]`; the actual example row has `[20.909, 15.15, 18.93, 16.764, 14.965, 21.372, 12.356]` (evidence: w42/data_adapter_smoke/example_row.json). The stale values are not fixture-mode output (`fixture_batch(42)` yields `[3.25, 6.5, 5.75, -99.0, 4.0, -99.0, -99.0]`; same values in the fixture-run artifact at `255ed592:scratch/w42/data_adapter_smoke/example_row.json`); their origin is an unidentified draft-time paste. Argmax of the real values is index 5, consistent with `oracle_best_action: 5`.

## Verified

- All four referenced w42 paths exist; report/manifest confirm `source_mode: real-corpus`, seed 42, batch size 1, and every batch shape in the table matches report.json exactly.
- `w42/data_adapter_smoke.py` does load `gus/data/corpus_train_100.pt` via `JointWorldFullDataset` with a deterministic fixture fallback, as described. The corpus .pt is gitignored but present in the main checkout.

## Follow-ups

- None. (First pass suggested asserting `oracle_best_action == argmax(e_q)` in the smoke; dropped on review — the script already computes `oracle_best_action` as the argmax of legality-masked `e_q` (w42/data_adapter_smoke.py:275), so the assert is a tautology and cannot detect the page-vs-artifact drift that actually occurred.)

## Review (second pass, 2026-07-07)

- Both page corrections verified against primary artifacts: run commit `68e6ee3a` (w42/data_adapter_smoke/report.json + manifest.json `repo_commit`) and the e_q vector (w42/data_adapter_smoke/example_row.json; rounds exactly to the page's values). Page needed no further edits; the two-line diff damaged nothing nearby.
- Amended this followup's provenance claim: the stale e_q values were NOT fixture-mode output. Evidence: running `fixture_batch(42)` from w42/data_adapter_smoke.py gives `[3.25, 6.5, 5.75, -99.0, 4.0, -99.0, -99.0]`, matching the committed fixture-run artifact `255ed592:scratch/w42/data_adapter_smoke/example_row.json` (that run's report records `source_mode: fixture`, `repo_commit: fe850015...` — which also explains where the page's stale commit hash came from).
- Dropped the suggested `oracle_best_action == argmax(e_q)` assert: `oracle_best_action` is derived from `e_q` by that exact argmax at w42/data_adapter_smoke.py:275, so the assert can never fail.
- Spot-verified the first pass's "Verified" bullets: all four w42 paths exist; report.json shapes match the page's batch-shape table field-for-field; JointWorldFullDataset load + fixture fallback confirmed at w42/data_adapter_smoke.py:157-159 and :79/:267-268; gus/data/corpus_train_100.pt present in the main checkout and gitignored (.gitignore:103).
