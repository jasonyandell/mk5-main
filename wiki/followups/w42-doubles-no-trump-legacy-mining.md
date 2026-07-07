Reviewed against code on 2026-07-07 — no issues found.

- All headline deltas verified against `w42/doubles_no_trump_legacy_mining/paired_contrasts.csv` (+5.76 [5.46, 6.05] early NT spend, +3.03 [2.65, 3.45] late, +12.80 [11.18, 14.46] defender weapon, -2.10 low-double lead, +0.01 vs high-double with -0.027 threshold mass, -1.62 for 6-5) and label counts / coverage (56000 decisions, 149415 action rows, 25906 labeled, 100 files, 164.75 s wall) against `summary.json`.
- The input `gus/data/corpus_train_chunk_*.pt` files are no longer present in this worktree; the run is reproducible only from the recorded manifest/summary, not from raw data on disk.
- Cheap next probe: the "hold a live double when a tempting spend is legal" preservation label could be built from the same labeled slice without regenerating data — only the labeler needs to change.
