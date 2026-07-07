Reviewed against code on 2026-07-07 — no issues found.

Verified: 114 files / family counts / GiB / inferred games+decisions match `w42/claim_data_inventory/summary.json`; 113 inspected with only `gus/data/_len_cache_train_10k.pt` uninspected; sample counts {200..6400} and legacy-vs-v2 field presence (bid_value, per-seat fields on the 11 v2+eval-ish files) match `corpus_files.csv`; `gus/data/` still holds exactly 114 `.pt` files.

- Follow-up: beads are retired (bd → GitHub issues); the `t42-0b4l.*` route table still routes to bead IDs — a cheap pass could annotate their gh-issue successors or note them as historical IDs.
- The 100 legacy chunks' `q_per_world` sample counts vary widely (200–6400); a small probe quantifying how sample count correlates with E[Q] variance would sharpen the "improves power" claim.
