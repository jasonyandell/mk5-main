# champion/evidence — load-bearing artifacts rescued from gitignored scratch

These files back wiki conclusions that previously lived **only** in gitignored
`scratch/` (one `scratch`-clear or one re-run from being lost). Copied here so the
record survives. Raw working copies were under `scratch/champion-run/run26cal/` and
`scratch/jud_demo/` (2026-06-14).

## run26_selfplay/ — the #26 self-play fixed-point result

The 4-round calibrated (`pmake_scale=0.70`) run behind
[[w42-champion-selfplay-fixed-point]]. Per round:

- `kl.txt` — symmetric belief-KL(Bₙ,Bₙ₊₁) + per-seat belief-acc (renamed from the
  run's `kl.log`, which `*.log` gitignore would have dropped). The convergence
  signal: KL plateaus **0.072–0.080** from round 1, and the A-vs-B acc gap collapses
  Δ −0.043 → ~0. (Note: the "~0.045 nats/slot seed floor" cited in the wiki has **no
  surviving measurement artifact** — it was a hardcoded literal in the run script. The
  plateau + acc-gap collapse are the real, self-sufficient convergence evidence.)
- `ab.json` — the bidder-quality arena A/B vs `net:wp` (80 games). The belief bidder
  loses every round: mean mark margin −2.23 / −2.20 / −2.59 / −2.01, CI excludes zero.
  Raw (uncalibrated) margins were ~−3.4; calibration cuts the loss ~34% (not "half").

## jud_demo/ — the candlewax "un-melt" first picture

The session sketch behind [[jud]]'s "first picture". `manifest.json` ranks scanned
positions; the headline `pos_b1000_i1_tp12` un-melts world ESS 128 → 10.5 and a
contract p_make 0.20 → 0.61. The honest counter-example `pos_b1000_i2_tp12` sharpens
*toward a loss* (p_make 0.23 → 0.07) — kept deliberately: belief sharpens toward the
worlds it believes, which is not always optimism. `unmelt_*.png` are the figures.

Caveat (see the adversarial review, 2026-06-14): the headline position is the rank-0
maximizer of a selection function that rewards the displayed effect, and the demo
measures motion toward the *belief posterior*, not toward solved ground truth. It is
an illustration of "belief un-melts the blob," not yet a proof of "toward truth."

## See also

- `champion/optimism_meter.py` + `champion/optimism_gap.json` — the oracle-vs-realized
  make-rate gap, computed from existing data (replaces the unprovenanced "0.58/0.83").
- `arena/results/` — the small 4-game belief-vs-`net:wp` smoke (now committed).
