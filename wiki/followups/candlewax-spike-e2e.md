Reviewed against code on 2026-07-07 — no issues found.

Verified: all listed scripts exist under `burl/candlewax_spike/` at commit 0545342; model is Qwen3.6-35B-A3B via mlx-vlm as stated (qwen_local.py); star_filter.py implements the K1 soft-margin (default epsilon 1.0) + prediction-winner gate; post_commit_sim.py fills remaining trick seats with oracle E[Q] argmax exactly as described; all wiki link targets exist.

Not verifiable from the repo: the numeric receipts (Haiku d000 13→21 flip, v7 adapter +15% bot-match on 33 examples, LR collapse thresholds) — result artifacts lived in gitignored scratch/local runs and are not committed.

- A cheap next probe if this thread revives: commit the v7 eval JSON (or a digest) so the +15% receipt is reproducible from the repo.
