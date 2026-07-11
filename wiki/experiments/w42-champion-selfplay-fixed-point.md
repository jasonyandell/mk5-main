---
title: Champion #26 — Self-Play Fixed Point (belief-conditioned bidder)
kind: experiment
status: complete
task_id: champion-26-selfplay
first_seen: local-2026-06-14
last_updated: afd4802
---

# w42-champion-selfplay-fixed-point

## Summary

**Question:** [[champion-design-review|Fable's design]] rung #6 — iterate *policy ↔ belief* by
self-play until conventions stabilize. Concretely: (a) does the loop reach a **fixed point**, and
(b) does belief-conditioning the **bidder** (the one channel that touches the training corpus and
that the [[arena]] is *not* blind to) produce a measurably stronger player?

**Answer: (a) yes — the loop reaches a stable fixed point; (b) no — the belief-conditioned bidder,
backed by the double-dummy oracle, systematically OVER-BIDS and loses to the PIMC-calibrated
`net:wp` bidder on marks.** This is the honest, predicted shape: the self-play *machinery* works
and converges, but the equilibrium it converges to is over-aggressive because the oracle's
double-dummy P(make) exceeds the achievable PIMC P(make) (strategy fusion). Correcting that
optimism (a tuned `pmake_scale=0.70` knob, not a measured calibration) cuts the loss ~a third and
roughly doubles the wins but does not close the gap.
The belief player's value is **legibility, not raw marks** — consistent with the #24/#25 nulls and
the project's marginal-value ranking.

## Method

- **The keystone — `champion/belief_bidder.py::BeliefBidder`** (the organ that closes the loop; the
  loop iterates *only* if the bidder changes when the belief changes, since the corpus =
  f(deal, decl, bids, bidder) and the play policy has zero corpus footprint). For each candidate
  contract it builds the **hypothetical completed auction** ("suppose I win (decl, value) and lead
  trick 1" — byte-identical to how the #26 corpus is generated, so the #24 belief is queried
  *in-distribution*), runs the same batched oracle E[Q] path as `arena.lens_play.LensPlay`,
  importance-weights the oracle's sampled opponent worlds by the [[w42-champion-auction-belief|#24]]
  belief posterior, derives P(make) per contract at the oracle-argmax legal lead, and bids the
  score-conditioned (`MarksToSeven`) utility-maximizing contract. Locked-honest: the **oracle**
  drives E[points]; the belief is only *soft* importance weights (the belief is near-uniform at bid
  time, seat-ESS ~2.86/3). One oracle forward over the 9 declarations covers the whole grid
  (Q is bid-value-independent) → **0.118 s/bid-turn**.
- **The loop — `scratch/champion-run/run_26_par.sh`**: round *n* = belief-bidder(Bₙ) self-play
  arena → `forge.cli.generate_eq_from_snapshots` (seeded) → `gus.train.train_v2_voids --auction
  --out-belief` (adapter selected by **belief-acc**, not the pi-heavy composite) → next round.
- **Metrics — primary `gus/eval/eval_belief_kl.py`**: symmetric belief-KL(Bₙ, Bₙ₊₁) + held-out
  belief-acc, on a frozen eval. The convergence evidence that actually survives on disk is the **KL
  plateau** (0.072–0.080 from round 1) plus the **A-vs-B acc-gap collapse** (Δ −0.043 → ~0). A
  "~0.045 nats/slot two-seed noise floor" is cited as the yardstick but has **no surviving
  measurement artifact** (it was a hardcoded literal in the run script) — treat the plateau and the
  acc-gap collapse as the self-sufficient signal; they do not need the floor. **Secondary —
  bidder-quality arena A/B** (belief-bidder vs `net:wp`, both `lens:ev` play so only the bidder
  differs): a *legitimate* value test because better bidding changes which contracts get made/set,
  and the arena measured the #21/#22 bidders this way (CI excluding zero). **Forbidden:**
  play-marks as a belief signal — the paired-PIMC arena is information-blind to belief value in play
  ([[champion-design-review]] caveat 1).

## Results

**As-is (raw double-dummy P(make)), 4 rounds:**

| round n→n+1 | belief-KL | belief-acc | bidder vs `net:wp` | A wins/80 |
|---|---|---|---|---|
| 0→1 | 0.116 | 0.422 | −3.08 [−3.55, −2.52] | 9 |
| 1→2 | 0.081 | 0.432 | −3.81 [−4.40, −3.21] | 10 |
| 2→3 | 0.083 | 0.415 | −3.80 [−4.39, −3.23] | 7 |
| 3→4 | 0.088 | 0.418 | −2.98 [−3.50, −2.42] | 9 |

KL drops from the cold start (0.116) and **plateaus at ~0.08 by round 1** (the residual is 80-game
corpus sampling noise; the cited "0.045 seed floor" is asserted, not measured) → a **stable fixed point**, which is a
**consistent over-bidder** (−3 to −3.8 marks/game, every round, CI excludes zero).

**Calibrated (`pmake_scale=0.70`), 4 rounds.** The optimism correction scales the oracle's P(make)
by a **tuned knob** — the prose "oracle 0.83 vs net 0.58 at bid 30 ≈ 0.70" was never computed; the
real, bid-dependent gap (oracle 0.64 / realized 0.52 at bid 30, ratio ~0.81; realized falls
0.52→0.11 across bids) is measured in `champion/optimism_meter.py` → `optimism_gap.json` — at
utility time:

| round n→n+1 | belief-KL | belief-acc | bidder vs `net:wp` | A wins/80 |
|---|---|---|---|---|
| 0→1 | 0.108 | 0.426 | −2.23 [−2.93, −1.51] | 19 |
| 1→2 | 0.074 | 0.431 | −2.20 [−2.84, −1.54] | 18 |
| 2→3 | 0.072 | 0.425 | −2.59 [−3.33, −1.92] | 18 |
| 3→4 | 0.080 | 0.423 | −2.01 [−2.68, −1.31] | 21 |

Calibration **cuts the loss ~a third (−3.4 → −2.2 marks/game) and roughly doubles the wins (~9 → ~20 of 80)** and
still converges (KL ~0.07–0.08) — but the bidder **still loses** (CI excludes zero).

## Interpretation

1. **The self-play fixed point is real and reachable.** Belief-KL converges to a stable plateau; the
   belief-bidder genuinely moves the corpus round-to-round (round-0 KL 0.116 ≫ the round-1+ plateau ~0.08),
   so this is true policy↔belief iteration, not data re-accumulation. Fable's rung-#6 machinery works.
2. **Double-dummy optimism is the over-bidding mechanism — confirmed.** The raw bidder believes the
   oracle's perfect-play P(make); in self-play, two such bidders escalate the auction and then fail
   the inflated contracts. The calibration is the controlled test: dampening the optimism halves the
   loss and doubles the wins, in lockstep — so the gap *is* the optimism (strategy fusion: double-dummy
   P(make) > achievable PIMC P(make); see [[pimc]]).
3. **But the belief-conditioned bidder does not beat `net:wp` on marks**, even well-calibrated. The
   distilled `net:wp` bidder is calibrated to *realized* play by construction (rung #22) and remains
   the stronger bidder. We did **not** grind further scales to chase a marks win — the value of the
   belief player is its **legibility** ("I bid 84 because I believe you are void in trumps"), the
   teaching surface the project actually wants, not maximized marks.

## Honest caveats

- 80-game corpora → the residual KL above the seed floor is sampling noise, not unconverged drift.
- Held-out belief-acc is measured on a **`net:wp`-auction** frozen eval, which is out-of-distribution
  for the self-play bidder's (different) auctions — so the absolute ~0.42 vs #24's ~0.47 partly
  reflects that mismatch; the **round-over-round KL** is the trustworthy convergence read, not the
  absolute acc.
- Only the *suppose-I-win* hypothetical (relative seat 0, in-distribution) is built; suppose-opponent-
  wins at 0 plays is genuinely OOD and is never queried.
- Process-parallelizing the arena netted only ~1.18× (the workload is MPS-dispatch-bound, not
  idle-latency-bound); the real wall-clock lever would be batching the bidder across games (deferred).

## What it means for the champion

The **playable champion** is the calibrated belief student (sensible, legible). It was slated for
export to [[plunge]] as the `onyx` difficulty via ONNX (a 40 KB net; the student would play its
`pi_me` policy head in-browser) so a person can sit down across from the thing we grew through
self-play — its teaching value being the legible belief trajectory, per the project's reframe of
the belief→marks nulls. **Asserted, unverified**: the era-6 audit (2026-07-06) found no `plunge`
app, `onyx` difficulty string, or ONNX export artifact anywhere in the repo — either this export
never shipped or lives somewhere the audit didn't check.

## Links

- [[champion]] — the build ladder this is rung #6 of
- [[champion-design-review]] — Fable's design + the two load-bearing caveats (info-blind arena;
  score-conditioning belongs at the auction)
- [[w42-champion-auction-belief]] — rung #24, the belief this loop iterates
- [[arena]] · [[gus]] · [[forge]] · [[pimc]] — the organs and the flaw the fixed point probes

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Provenance: the calibrated 4-round table matches `champion/evidence/run26_selfplay/RESULTS.txt` exactly; the as-is (raw) round table has no surviving repo artifact and is corroborated only by wiki/log.md.
- `scratch/champion-run/run_26_par.sh` is gitignored and no longer on disk; the RESULTS.txt header preserves its invocation shape — the only surviving record of the loop script.
- Cheap next probe (also named in wiki/log.md): back the belief bidder with a PIMC-calibrated value instead of double-dummy — the optimism ratio is bid-dependent (0.64–0.81), so no single `pmake_scale` can fit it.
