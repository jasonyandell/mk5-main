Reviewed against code on 2026-07-07 — no issues found.

Verified: metrics.json matches all four table rows (2.471/59.64% best epoch 3, 2.834/56.61% final, E[Q] N=10 0.118/90.00%, n=560); run.json history matches the per-epoch table; model architecture in `w42/raw_public_state_baseline.py` matches the Model Shape table (d_model 64, 4 heads, 1 layer, ff 128, action_hidden 96, Linear+GELU+LayerNorm+Linear scorer, Embedding(7,64)); manifest.json confirms features, leakage exclusions, seeds, command, and commit 8df0c3b (exists in repo).

Follow-ups:
- Source corpora live at absolute paths in the main checkout (`gus/data/corpus_*.pt`), not this worktree — the page already flags this, but a repo-relative or hashed reference would make the run reproducible from any worktree.
- The best-checkpoint save/restore gap the page names is still the cheapest next fix before any variant comparison.
- Bead `t42-csw6.10` is now historical (beads retired in favor of GitHub issues); harmless as provenance but not resolvable via `bd`.
