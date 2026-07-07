# w42-strategy-tag-family-ablations — audit 2026-07-07

## Corrections

- page said commit SHA at run time was `1830e5f379ba090786015cf37d4859d0e7b2bbd6`; manifest and git say `1830e5f3bd7fedaf24e186765d709e868618c900` (evidence: `w42/strategy_tag_family_ablations/manifest.json`; the page's SHA does not exist in the repo). The short prefix `1830e5f` used in the W&B run name was correct.

Everything else verified: ablation matrix numbers (all 11 rows) match `w42/strategy_tag_family_ablations/ablation_matrix.csv`; baselines (raw 2.834, v0 2.000, rich 1.970, E[Q] N=10 0.118, n=560) match `metrics.json`; script exists and masks rich global/action columns before projection via registered buffers as described; W&B run id `qvq73ix6` matches `run.json`/`manifest.json`.

## Follow-ups

- The W&B run itself was not fetched (external); numbers verified against local artifacts only.
- A cheap next probe would be re-running the two edge families (no-trump/doubles +0.107, off-protection -0.051) at 3 seeds only, rather than the whole matrix, before spending on the full multi-seed repeat.

## Review (second pass, 2026-07-07)

- Verified — corrections stand. Independently re-derived: manifest SHA `1830e5f3bd7fedaf24e186765d709e868618c900` is a real commit (`git cat-file -t` = commit), the page's original SHA does not exist in the repo, all 11 ablation rows match `w42/strategy_tag_family_ablations/ablation_matrix.csv`, all baselines (raw 2.834, v0 2.000, rich 1.970, E[Q] N=10 0.118, n=560) match `metrics.json`, run id `qvq73ix6` matches `run.json`/`manifest.json`, and the script masks rich columns via `register_buffer` before projection. The single edit damaged nothing nearby.
