Reviewed against code on 2026-07-07 — no issues found.

- All headline numbers (feature regrets, paired deltas, E[Q] N=10 baseline 0.1747/87.00%/92.89%) match `w42/multi_seed_larger_eval_replication/pilot_2seed_e2800/{feature_summary.csv,paired_deltas.csv,summary.json}` exactly; W&B run URLs match summary.json.
- The per-run manifests listed in Artifacts (`{raw,v0,rich}_s*/manifest.json`) are written by the script but not committed to the repo — only the four aggregate files are; a one-word "local-only" durability note would prevent future confusion.
- Cheap next probe: the page's own "8 epochs to match the original baseline horizon" rerun is a one-flag change (`--epochs 8 --output-dir .../five_seed_e2800_ep8`) and would settle whether the rich-over-v0 edge grows or washes out with training.
