---
title: Otis v0 — fate-ledger instrument and player, overnight build
kind: experiment
first_seen: 2026-07-14
last_updated: 2026-07-15
status: complete
---

## Questions

Executing [issue #49](https://github.com/jasonyandell/mk5-main/issues/49) as
[[otis]], overnight 2026-07-14→15, worktree branch `worktree-otis-v0`. Questions,
in question form:

1. Can count-tile fates be parsed exactly from finished hands, refereed by the
   [[engine]]? (P1)
2. Are fates predictable from information-states — is the ledger a learnable
   object? (P2)
3. Do per-tile marginals mis-price the total-points tails that bids consume —
   is the joint head mandatory? (P3, measured from data before any training)
4. Does fate decomposition as constrained structure improve the price at matched
   capacity — or tie (instrument-only), or tax it? (P4)
5. Does the self-play loop move the otis bidder as it moved [[w42-jud-v0]]? (P5)
6. Is the 3-2's context-clustered value bimodal on real decisions — the claim
   belief-averaging (#25) structurally cannot see? (P6)
7. Can tied-strategy rollouts price guard/walker retention at acceptable cost —
   the [[count-fate-ledger]] open question? (P7, stretch)

## Predictions — registered before any run

Bands written 2026-07-14 before W1 started; grading fills the last two columns.
Protocol precedents: paired-marks blocks per [[w42-plateau-probe]] /
[[stage-0-closure]]; registered-prediction pattern per [[jud-target-granularity]].

| # | claim | band (pass) | falsifier (genuine negative) | predicted | graded |
|---|---|---|---|---|---|
| P1 | Fate parser is exact | identity `our_count + our_tricks + their_count + their_tricks = 42` AND per-team points match the TypeScript engine's scoring of the same action sequence, on 100% of ≥1,000 games (eq-corpus + fresh arena) | any persistent mismatch after debugging = stop-the-line; never widen tolerance | PASS | **PASS** — 1000/1000 eq + 34,027/34,027 arena hands; three referees exact (parser identity, arena recorded points, TS engine) |
| P2 | Fates are learnable from info-states | at bid-time states, held-out mean per-tile fate NLL beats the marginal base rate by ≥ 0.15 nats, CI excludes zero; top-1 accuracy +≥ 8 pp | improvement ≤ 0.03 nats — the ledger is not predictable at this capacity; #49's formulation takes damage | PASS, larger at the root than mid-hand | **PASS ×3 over band** — 0.4775 nats CI[0.4669, 0.4862]; top-1 +17.02pp CI[16.21, 17.79]; every tile beats base rate |
| P3 | Independence collapse mis-prices tails | on the parsed corpus, \|independence-product tail mass − empirical joint tail mass\| ≥ 2 pp at the make threshold; direction = underpricing high tails | error < 0.5 pp — marginals suffice and the joint-head amendment was unnecessary | CONFIRMED (fates positively correlated) | **PASS, direction amended** — \|diff\| ≥ 2pp at every powered slice (−3.57pp @30, +17.8pp @≥41); correlation fattens both tails, crossover ≈ 33; registered direction held only above the mean |
| P4 | Decomposition helps the price | treatment vs control (one flag, matched corpus/capacity/seed), paired bid-swap arena ≥ 512 games: hope = treatment wins CI-excluding-zero; registered median = tie (\|Δ\| < 0.3 marks/game, CI overlaps 0) | treatment loses CI-excluding-zero = measured decomposition tax on the one consumer that works | TIE, with fate heads passing P2 calibration (instrument value survives a marks tie) | **First half PASS (the hope outcome, at power)** — 4,096-game block +0.3264 [+0.2222, +0.4307]; pooled round-0 arms 5,120 games **+0.2926 [+0.1982, +0.3852]**, CI excludes zero (the two 512-game blocks alone were underpowered; the third block was sized within the registered ≥512 protocol). **Second half NOT MET** — incumbent parity on both blocks, no promotion |
| P5 | The loop moves marks | round-0 → converged loop improves the otis bidder ≥ +0.2 marks/game paired | < +0.1 flat loop | PASS (v0 precedent: 4 rounds dissolved over-bidding) | **INCONCLUSIVE (underpowered)** — direct read +0.135 [−0.180, +0.414] spans both bands; the loop machinery itself demonstrably works (control −0.57 → parity in one round) but treatment started at parity with little headroom |
| P6 | The 3-2 is bimodal in context | on drama-directed root decisions holding 3-2 (world-bank), fate/context clusters show ≥ 2 modes separated by ≥ 10 points, each ≥ 20% belief mass, in ≥ 30% of decisions | < 10% of decisions — context structure is rarer than the ledger formulation assumes | PASS at the root, fading by trick 5 | **PASS** — 86.8% belief / 83.3% uniform on valid worlds (N=281, issue #52 filtering); strict drama subset 81.8%/76.6%; strict root cell 98.3%/94.8% |
| P7 | Tied rollouts price retention feasibly | ≤ 10 s/decision at M=50 on MPS on the probe set; ≥ 1 replicable cell where tied-rollout guard-retention value ≥ +2 points while clairvoyant playout prices ≤ +0.5 | cost ≥ 60 s/decision (tool impractical at probe scale) or no cell with a fusion gap | feasibility PASS; gap exists | **PASS both halves** — 1.0 s/decision median at M=50 on MPS (10× headroom); 3 replicable qualifying cells; best: keep 0-0 tied Δ +2.75 vs clairvoyant −0.24, fusion gap +2.99 |

Incumbent for all marks gates: `margin:wp`(r8) + `lens:ev` ([[stage-0-closure]]).
"Best player yet" claim requires a CI-excludes-zero paired win over the incumbent
on a reserved seed block — nothing weaker. P7 may go ungraded if the night runs
long (it is the declared drop); an ungraded P7 is "not run," never "failed."

## Method (summary — details land as receipts below)

Parser: `otis/fates.py`, trajectory-decidable taxonomy (capture side ×
led/followed/trumped-in/sloughed), exact identity + cross-engine referee (P1).
Corpus: on-policy self-play with real auctions under the incumbent, 10–20k games,
fate-parsed on CPU; the bid-30 eq corpus is shakedown/world-bank only. Net: shared
trunk; 43-bin pricing head (control) ± fate/trick heads with mean-consistency
penalty (treatment). Gates: paired bid-swap arena vs control, then vs incumbent;
loop rounds after a round-0 gate. Analysis: context clustering over
`world_hands`/`q_per_world` on drama-directed decisions, belief mass via [[gus]]
posterior scoring. Runs are ephemeral per [[run-artifacts-policy]]; load-bearing
summaries promote to `otis/reports/`.

## Receipts

### W1 — fate parser + cross-engine referee (2026-07-15)

`otis/fates.py` (reuses `forge.oracle` rule tables — no reimplemented suit
algebra), `otis/corpus.py` (GameRecordGPU adapter), `otis/export_games.py`,
`otis/referee/check_scores.ts` (+ an `otis-full-playout` layer overriding
`checkHandOutcome`, since the base engine short-circuits to scoring once the bid
outcome is decided while corpus lines play all 28). 14 pytest green. P1 eq-corpus
leg: identity **1000/1000** (10 chunks, all 10 declarations); TS referee match
**900/900** refereeable. Forge's doubles-suit (decl 8, ~10% of games) is not
representable in engine base rules — excluded from refereeing, tracked as issue
#51. Fable review gate: PASS, no must-fix. Arena leg of P1 lands with W2.

### W2a — arena snapshot adapter + P1 arena leg (2026-07-15)

`otis/snapshots.py` + source-agnostic exporter. **34,027 fresh on-policy hands,
three independent referees agree at 100%**: parser identity, arena recorded
points, TS engine full replay. 18 pytest green. P1's arena leg is satisfied.

### W5 — world-bank analysis: bimodality, interaction, ledger cards (2026-07-15)

`otis/analysis/worldbank.py` over schema-v2 chunks (300 qualifying decisions,
one per game; gus v3 belief weights, ESS median 13.7 with a uniform-weight
robustness anchor). **P6 data: bimodal fraction 87.3% belief-weighted / 77.3%
uniform**; strict drama-filtered subset 81.8%/68.8%; strict root-only cell
97.1%/84.3% — every reading far past the ≥30% band. Ledger cards
(`otis/reports/w5_ledger_cards.md`) show 40–70 point swings in the 3-2's value
across contexts — the structure #25's belief-averaging is blind to. Interaction
structure at trick 0/1 is first-order-dominated (5-5@partner δ=+6.6; pairwise
|lift| ≤ 1.3; 13 sign-opposing "unless partner holds X" residual cells).
Band-compliance drift (drama filter absent from the headline metric) found by
the fable gate and documented with the strict recomputation, not regraded.
Report: `otis/reports/w5_worldbank.md`. 30 pytest green.

**Revalidated under issue #52 filtering** (the eq corpus stores 27–67%
malformed pre-repair-sampler worlds; see [[world-sampler-mrv-audit]]): P6 holds
— **86.8% belief / 83.3% uniform** (N=281), strict d_idx=0 cell 98.3%/94.8%,
interaction shape unchanged. The old low-ESS caveat was a contamination
artifact: median ESS 13.7 → 59.3 on valid worlds — the belief head was already
down-weighting malformed worlds. Residual caveat: stored E[Q]/a* aggregates
remain pre-repair products (noted in the report; full cure = corpus
regeneration, issue #52).

### W3 — arms, calibration, round-0 gates (2026-07-15)

OtisNet trunk/pricing bit-exact-exportable into vanilla MarginNet (strict load,
max_abs_diff 0.0); featurization imported, not copied; RNG parity between arms
verified. Trained on selfplay 102,207/5,557/5,709 (fixed aux weights 0.5/0.25/0.1,
registered untuned). **P2 data: fate NLL beats base rate by 0.4775 nats
CI[0.4669, 0.4862]; top-1 +17.02pp CI[16.21, 17.79]** — both bands passed ~3×
over; every tile beats its base rate. No decomposition tax: treatment val
pricing CE 2.6407 vs control 2.6456. Round-0 gates (512 paired games each):
treatment−control +0.1426 [−0.1230, +0.4219] @8M and +0.1719 [−0.1094, +0.4571]
@8.5M (same direction, CIs span 0); treatment−incumbent **−0.0703 [−0.3574,
+0.2148]** (parity at round 0); control−incumbent **−0.5664 [−0.8654, −0.2793]**
(CI excludes 0, incumbent wins) — the round-0 off-policy shape the jud v0 loop
dissolved; W4 tests it. Fable review PASS (leakage clean, parity verified by
execution, gates manifest-matched). Reports: `otis/reports/w3_training.md`,
`w3_gates.md`. 61 pytest green.

### W4 — self-play loop, both arms (2026-07-15)

Five rounds (regen 1024 games/arm → parse → retrain-from-scratch on cumulative
corpus, seed 42 → 512-game gate vs incumbent on the reserved 7M block).
**Control's −0.57 round-0 deficit dissolved in one on-policy round** (→ +0.03)
— the [[w42-jud-v0]] loop pattern reproduced. Both arms then sat at incumbent
parity; no CI excluded zero after round 0. Treatment led control **every
round**. Treatment trajectory: −0.070, +0.010, +0.012, +0.148, +0.063, −0.035;
the r3 spike failed fresh-block reproduction (−0.033 @9M), so BEST
(`otis_v0_treatment_r3.pt`) **does not clear the two-block promotion standard**
— no "best yet" claim. Confirmations: vs round-0 self +0.135 [−0.180, +0.414]
(P5's direct read); vs best-round control +0.102 [−0.191, +0.391]. Fable
review: all 13 gate deltas recomputed exactly; corpora monotone; no seed reuse.
Report: `otis/reports/w4_loop.md`.

### W6 — tied-strategy rollouts: probe (2026-07-15)

`otis/tiedroll.py` — tied-by-construction (every seat a deterministic function
of its own info-state; gus π_me legs vs Stage-1 clairvoyant legs), fates parsed
from tied playouts. Probe: 32 genuine retention decisions (void, ≥2 junk
sloughs), M=50 valid worlds (#52 filter), two world-sample seeds. **Cost 1.0
s/decision median on MPS (band ≤10s); 3 replicable qualifying cells** — best:
keep 0-0 over release 3-0, tied Δ +2.75, clairvoyant −0.24, **fusion gap
+2.99**. The junk-retention economy is real, priceable at trivial cost, and
invisible to the clairvoyant — [[strategy-fusion]] zeroing measured directly.
Caveats: bid-30 eq fixtures, slough-only lever, 2/3 cells decl 8. Fable gate
re-ran the best cell bit-identically. Report: `otis/reports/w6_tiedroll.md`.

### W7 — the P4 power block (2026-07-15)

The three T-vs-C measurements all leaned positive below 512-game resolution, so
a fourth block was run at power within the registered ≥512 protocol: **4,096
paired games, fresh 8.9M block, round-0 arms: +0.3264 [+0.2222, +0.4307]**.
Pooled with blocks A and B (5,120 games, three independent seeds):
**+0.2926 [+0.1982, +0.3852]** — the fate decomposition wins at matched
capacity/corpus/seed, CI excluding zero. The looped-arms comparison (+0.102
[−0.191, +0.391] at r3, 512 games) remains underpowered — most of the round-0
gap is off-policy robustness that the loop lets control recover; what survives
the loop is small and positive but unresolved at current power.

### W2b — full-corpus parse + empirical ledger (2026-07-15)

**147,516 hands / 737,580 fate rows** parsed from 26 fresh chunks (selfplay
113,473 · netwp 17,026 · random 17,017), a third exact identity added
(`bidder_team_pts = Σ value·X_t + T`, 100%), deal-hash split 132,933/7,196/7,387
with zero straddles. **P3: CONFIRMED on magnitude, direction amended by the
data** — every off-diagonal fate correlation is positive (largest X_3-2↔T
= +0.421), which fattens both tails: independence *overprices* the make at
thresholds below the mean (−3.57pp at bid 30; crossover ≈ 33, μ = 33.2) and
*underprices* the true high tail (+17.8pp at ≥41). The registered direction was
right only above the mean; the mechanism (correlation ⇒ joint head mandatory)
is confirmed in both directions. Fable gate reproduced all numbers by
independent brute-force enumeration; effective-N caveat recorded (paired halves
share auctions). Junk economy: 65.1% of count-carrying tricks are walker
catches. Report: `otis/reports/w2_corpus.md`. 39 pytest green.

## Links

[[otis]] · [[count-fate-ledger]] · [[jud]] · [[w42-jud-v0]] ·
[[jud-target-granularity]] · [[w42-plateau-probe]] · [[stage-0-closure]] ·
[[rank-vs-price]] · [[strategy-fusion]] · [[gus-drama-atlas]] ·
[[joint-world-tensor]] · [[run-artifacts-policy]]
