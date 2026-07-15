---
title: Otis v0 — fate-ledger instrument and player, overnight build
kind: experiment
first_seen: 2026-07-14
last_updated: 2026-07-14
status: active
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
| P1 | Fate parser is exact | identity `our_count + our_tricks + their_count + their_tricks = 42` AND per-team points match the TypeScript engine's scoring of the same action sequence, on 100% of ≥1,000 games (eq-corpus + fresh arena) | any persistent mismatch after debugging = stop-the-line; never widen tolerance | PASS | — |
| P2 | Fates are learnable from info-states | at bid-time states, held-out mean per-tile fate NLL beats the marginal base rate by ≥ 0.15 nats, CI excludes zero; top-1 accuracy +≥ 8 pp | improvement ≤ 0.03 nats — the ledger is not predictable at this capacity; #49's formulation takes damage | PASS, larger at the root than mid-hand | — |
| P3 | Independence collapse mis-prices tails | on the parsed corpus, \|independence-product tail mass − empirical joint tail mass\| ≥ 2 pp at the make threshold; direction = underpricing high tails | error < 0.5 pp — marginals suffice and the joint-head amendment was unnecessary | CONFIRMED (fates positively correlated) | — |
| P4 | Decomposition helps the price | treatment vs control (one flag, matched corpus/capacity/seed), paired bid-swap arena ≥ 512 games: hope = treatment wins CI-excluding-zero; registered median = tie (\|Δ\| < 0.3 marks/game, CI overlaps 0) | treatment loses CI-excluding-zero = measured decomposition tax on the one consumer that works | TIE, with fate heads passing P2 calibration (instrument value survives a marks tie) | — |
| P5 | The loop moves marks | round-0 → converged loop improves the otis bidder ≥ +0.2 marks/game paired | < +0.1 flat loop | PASS (v0 precedent: 4 rounds dissolved over-bidding) | — |
| P6 | The 3-2 is bimodal in context | on drama-directed root decisions holding 3-2 (world-bank), fate/context clusters show ≥ 2 modes separated by ≥ 10 points, each ≥ 20% belief mass, in ≥ 30% of decisions | < 10% of decisions — context structure is rarer than the ledger formulation assumes | PASS at the root, fading by trick 5 | — |
| P7 | Tied rollouts price retention feasibly | ≤ 10 s/decision at M=50 on MPS on the probe set; ≥ 1 replicable cell where tied-rollout guard-retention value ≥ +2 points while clairvoyant playout prices ≤ +0.5 | cost ≥ 60 s/decision (tool impractical at probe scale) or no cell with a fusion gap | feasibility PASS; gap exists | — |

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
#50. Fable review gate: PASS, no must-fix. Arena leg of P1 lands with W2.

## Links

[[otis]] · [[count-fate-ledger]] · [[jud]] · [[w42-jud-v0]] ·
[[jud-target-granularity]] · [[w42-plateau-probe]] · [[stage-0-closure]] ·
[[rank-vs-price]] · [[strategy-fusion]] · [[gus-drama-atlas]] ·
[[joint-world-tensor]] · [[run-artifacts-policy]]
