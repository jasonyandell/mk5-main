# Otis v0 — W3 paired arena gates (P4)

Generated 2026-07-15. Data-only receipts; NO interpretation. P4 grading happens in W7.

## Protocol

Stage-0 paired-marks convention (`wiki/experiments/stage-0-closure.md`,
`wiki/experiments/w42-plateau-probe.md`). Each gate:

```
.venv/bin/python -u -m arena.cli --n-games 512 --n-samples 10 --device mps \
  --team-a <A> --team-b <B> --base-seed <seed> \
  --emit-decisions scratch/otis-night/gates/<name>/decisions.jsonl \
  --out-dir scratch/otis-night/gates/<name>
```

Bidder specs (all play = `+lens:ev`):

- **control**   = `margin:wp,model=otis/models/otis_v0_control.pt`
- **treatment** = `margin:wp,model=otis/models/otis_v0_treatment.pt`
- **incumbent** = `margin:wp,model=champion/margin_net_r8.pt`

Seed blocks: A=8000000, B=8500000 (fresh, no overlap with 7M/9M reserved or 41–43M
corpus seeds); C,D=7000000 (reserved stage-0 block, comparable with the incumbent's
own promotion receipts). All runs exit 0; `n_samples=10`, `marks_to_win=7`,
`max_redeals=3`, `fast_batching=true`.

Mark margin = **side_a − side_b** marks/game; 95% CI is the paired bootstrap emitted in
`summary.json`.

## Results

| Gate | Side A | Side B | Seed | N games (hands) | A wins / rate | Mark margin (A−B) /game | 95% CI | CI excl 0 | Pt margin /hand | CI favors |
|------|--------|--------|------|-----------------|---------------|-------------------------|--------|-----------|-----------------|-----------|
| A | treatment | control   | 8000000 | 512 (5720) | 268 / 0.5234 | +0.1426 | [-0.1230, +0.4219] | no  | +0.4357 | neither (CI spans 0) |
| B | treatment | control   | 8500000 | 512 (5658) | 265 / 0.5176 | +0.1719 | [-0.1094, +0.4571] | no  | +0.1905 | neither (CI spans 0) |
| C | treatment | incumbent | 7000000 | 512 (5689) | 243 / 0.4746 | -0.0703 | [-0.3574, +0.2148] | no  | -1.0466 | neither (CI spans 0) |
| D | control   | incumbent | 7000000 | 512 (5640) | 216 / 0.4219 | -0.5664 | [-0.8654, -0.2793] | yes | -1.6103 | incumbent |

## Artifacts

Per gate under `scratch/otis-night/gates/<name>/`: `summary.json`, `per_hand.csv`,
`per_game.csv`, `decisions.jsonl` (+ manifest), `run.log`.

- A: `scratch/otis-night/gates/A_treat_vs_ctrl/`
- B: `scratch/otis-night/gates/B_treat_vs_ctrl/`
- C: `scratch/otis-night/gates/C_treat_vs_inc/`
- D: `scratch/otis-night/gates/D_ctrl_vs_inc/`

Machine-readable rollup: `scratch/otis-night/w3_gates.json`.
