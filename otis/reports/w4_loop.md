# Otis v0 — W4 self-improvement loop (verified report)

Five on-policy self-play rounds per arm (treatment = trunk + pricing + fate/trick/consistency
organs; control = trunk + pricing only), each round = regen (1024 games, MPS, incumbent-style
self-play under the arm's own bidder) → parse through the W2 referee stack → retrain from
scratch on the cumulative corpus (seed 42, W3 hyperparameters, fixed W2 val for early stop) →
bit-exact MarginNet export → 512-game paired gate vs the incumbent
(`margin:wp`(r8)+`lens:ev`, base-seed 7,000,000, `--emit-decisions`).

Every number below was independently re-verified against the primary artifacts on
2026-07-15 (see Verification appendix). All 61 otis tests pass, no skips.

## Round table

Regen games = 1024 per arm-round; hands is what the parser accepted (all with 100%
P1 / recorded-points / ledger referee pass). Corpus rows = cumulative training N
(shared W2 base = 102,207 rows + this arm's rounds 1..N; val is the fixed W2 val,
5,557 rows, untouched). Gate delta = mean paired mark margin vs incumbent, 512 games.

### Treatment

| Round | Regen seed | Hands | Corpus rows | Val CE | Epochs | Gate delta [95% CI] | pt/hand |
|---|---|---|---|---|---|---|---|
| 0 (W3) | — | — | 102,207 | 2.6407 | 25 | -0.0703 [-0.357, +0.215] | -1.05 |
| 1 | 45,200,000 | 11,377 | 113,584 | 2.6309 | 18 | +0.0098 [-0.270, +0.277] | -0.78 |
| 2 | 45,400,000 | 11,319 | 124,903 | 2.6259 | 18 | +0.0117 [-0.283, +0.281] | -0.94 |
| 3 | 45,600,000 | 11,312 | 136,215 | 2.6291 | 18 | **+0.1484** [-0.135, +0.438] | -1.00 |
| 4 | 45,800,000 | 11,388 | 147,603 | 2.6211 | 23 | +0.0625 [-0.229, +0.340] | -1.87 |
| 5 | 46,000,000 | 11,289 | 158,892 | 2.6157 | 19 | -0.0352 [-0.322, +0.248] | -1.44 |

### Control

| Round | Regen seed | Hands | Corpus rows | Val CE | Epochs | Gate delta [95% CI] | pt/hand |
|---|---|---|---|---|---|---|---|
| 0 (W3) | — | — | 102,207 | 2.6456 | 15 | -0.5664 [-0.865, -0.279] | -1.61 |
| 1 | 45,300,000 | 11,398 | 113,605 | 2.6391 | 14 | +0.0332 [-0.252, +0.309] | +0.10 |
| 2 | 45,500,000 | 11,345 | 124,950 | 2.6408 | 16 | +0.0352 [-0.256, +0.322] | -1.39 |
| 3 | 45,700,000 | 11,382 | 136,332 | 2.6348 | 15 | -0.0508 [-0.338, +0.221] | -1.13 |
| 4 | 45,900,000 | 11,335 | 147,667 | 2.6312 | 16 | -0.2500 [-0.533, +0.023] | -1.63 |
| 5 | 46,100,000 | 11,329 | 158,996 | 2.6277 | 14 | -0.1113 [-0.387, +0.168] | -1.91 |

No looped-round gate CI excludes zero (round-0 control's deficit was the only
significant result, and one round of on-policy self-play erased it). Treatment's
gate delta exceeds control's in every looped round; the treatment−control gap on
the shared 7M block peaks at r3 (+0.199) and r4 (+0.3125). Both arms bleed points
per hand vs the incumbent in nearly every gate while staying near mark parity —
they lose the point battle but hold the marks-to-seven race the pricing head
optimizes. (One exception: round-1 control was point-positive, +0.10/hand.)

BEST = round-3 treatment, `otis/models/otis_v0_treatment_r3.pt` — the largest
treatment mark-delta vs incumbent across the loop.

## Confirmation (BEST = treatment r3, three fresh 512-game blocks)

| Confirmation | Team B | Seed | Delta [95% CI] | CI excl. 0 | A wins | pt/hand |
|---|---|---|---|---|---|---|
| vs incumbent (fresh stage-0 block) | `champion/margin_net_r8.pt` | 9,000,000 | -0.0332 [-0.342, +0.262] | no | 248/512 | -1.28 |
| vs round-0 treatment | `otis_v0_treatment.pt` | 8,700,000 | +0.1348 [-0.180, +0.414] | no | 272/512 | +0.10 |
| vs best-round control (r3) | `otis_v0_control_r3.pt` | 8,800,000 | +0.1016 [-0.191, +0.391] | no | 273/512 | -0.02 |

The r3 +0.1484 vs incumbent does not reproduce on the independent 9M block
(-0.0332). The looped gates reused the same 7M seed block every round (by design,
for cross-round comparability), so the fresh-block confirmation is the decisive
read: the r3 spike is consistent with paired-marks noise at 512 games
(half-width ≈ ±0.28). BEST does not clear the two-block promotion standard
(CI-excludes-zero on a reserved block).

## P5 read — "the loop moves marks" (data only; grading in W7)

Registered band: round-0 → converged improves the otis bidder **≥ +0.2 marks/game
paired** = PASS; **< +0.1** = falsifier (flat loop).

- Direct paired measurement (BEST vs its own round-0 checkpoint, fresh 8.7M block):
  **+0.1348 [-0.180, +0.414]**. The point estimate lands *between* the bands —
  above the +0.1 falsifier line, below the +0.2 PASS line — and the CI spans both.
- On the incumbent yardstick: round-0 treatment gate -0.0703 → BEST confirmed
  -0.0332 on the fresh block, a movement of ~+0.04 — below the falsifier line on
  that (unregistered) reading.
- Context: the loop *did* move control massively once (round-0 -0.5664 → round-1
  +0.0332, a significant deficit erased), then both arms plateaued at
  incumbent-parity; treatment's own trajectory wobbles inside the ±0.28 noise band.

## P4 second-half read — best treatment vs incumbent on 7M AND 9M (data only)

- 7M block (looped gate, r3): **+0.1484 [-0.135, +0.438]** — CI spans zero.
- 9M block (fresh confirmation): **-0.0332 [-0.342, +0.262]** — CI spans zero.
- The two blocks disagree in sign; neither excludes zero. Supporting head-to-head:
  treatment r3 beats control r3 by +0.1016 [-0.191, +0.391] on the fresh 8.8M
  block (CI spans zero) — inside P4's registered tie band (|Δ| < 0.3, CI overlaps 0).

## Verification appendix (fable review, 2026-07-15)

Everything below re-checked from primary artifacts, not from round reports:

- **Numbers**: all 13 gate deltas (10 looped + 3 confirmation) recomputed from
  `per_game.csv` (`marks_a − marks_b` mean) — exact match to `summary.json` and
  the round JSONs; a-wins match; independent 10k-resample bootstrap CIs agree
  with the reported CIs to < 0.05. Round-0 baselines match `w3_gates.md` (gates
  C/D, same 7M block). W3 val CEs (2.6407/2.6456) match `w3_training.md`.
- **Bit-exactness**: beyond each round's recorded 100-row check
  (`strict_load_ok=true, max_abs_diff=0.0` in every `r*_train.json`), an
  independent 20-row spot check with a fresh seed on both r5 checkpoints:
  strict load through `champion.value_bidder.load_margin_net`, `torch.equal`
  bit-exact, max_abs_diff 0.0, treatment flag correct per arm.
- **Cumulative corpora**: parquet row counts equal recorded hands (fates = 5×
  hands) for all 10 arm-rounds; cumulative N grows monotonically and each
  increment equals exactly that round's parquet hands over the shared 102,207-row
  W2 base. `build_cumulative_split` (otis/loop_data.py) injects only data;
  `otis/train.py` builds a fresh `OtisNet` under `torch.manual_seed(42)` with
  fixed W3 hyperparameters and the untouched W2 val — retrain-from-scratch
  confirmed in code, seed 42 recorded in every train JSON.
- **Seeds**: self-play logs confirm each regen seed used exactly once
  (45.2M–46.1M, +200k cadence, treatment/control disjoint); every looped gate
  cfg records base-seed 7,000,000 (per spec); confirmation seeds 9.0M / 8.7M /
  8.8M are disjoint from all training and gate blocks.
- **Referees**: `crosscheck_ok == n_hands` in every round train JSON (the W2
  parser stack asserts P1 identity, recorded-points cross-check, and ledger
  identity on every hand and raises on mismatch).
- **Tests**: `pytest otis/ -q` → 61 passed, 0 skipped.
- Zero numeric discrepancies found between the round reports and the artifacts.
  One narrative wrinkle: the round-2 report's "same pattern seen in round 1"
  (both arms bleeding points/hand) is wrong for round-1 control, which was
  point-positive (+0.10/hand); the underlying data files are correct.

Artifacts: `scratch/otis-night/loop/round_{1..5}.json`, `confirm.json`,
`round_summary.md`, per-gate dirs `r{n}_{arm}_gate/` and `confirm_*/`;
checkpoints `otis/models/otis_v0_{arm}_r{1..5}.pt`.
