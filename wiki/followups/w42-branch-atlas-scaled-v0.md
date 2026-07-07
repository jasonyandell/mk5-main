Reviewed against code on 2026-07-07 — no issues found.

- All metrics table values match `w42/branch_atlas_scaled_v0/summary.json` exactly (147/280 omission, 230/280 multi-peak, 457/392/642 tags, max 56.82, mean top 18.46, 251/234/216 selector alignment, 773 action rows, 5955 hidden-threat rows).
- Bid-aware threshold description (offense Q >= 2B - 42, defense bin 85 - 2B preserving Q >= -17 at bid 30, 84 -> 42 contract points) matches `forge/eq/generate/actions.py`.
- W&B run id 7fwi2zwn confirmed in summary.json; the wandb.ai URL itself is external and not verified.
