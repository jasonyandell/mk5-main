---
title: Gus (sibling project — neural player + belief + value)
kind: entity
first_seen: 2026-04-23
last_updated: 2026-07-13
status: complete
---

## What it is

Gus is a third sibling project alongside [[lem]] and [[burl]]: a multi-task transformer
that provides neural policy, belief, and value heads for [[texas-42]]. It **skips the
reasoning channel** — instead of rationalization or tool orchestration, it trains directly
against [[forge]]'s E[Q] oracle as a variance-free reward signal
([[student-distillation]]). Grounded in PG-beats-CFR (ICLR 2025), LAMIR (Oct 2025),
Bridge-AI BMCS, and PRR-TM teammate modeling. (commit messages @ 42a7535, 31e10ef)

The line ran April 2026 and is **concluded**: the player was built and measured to its
ceiling, the look-ahead question was closed by exhaustion, and the belief head moved
forward as the posterior engine of the [[jud]] frontier. The full
walkthrough — every experiment, in order, with findings — is [[gus-line]].

## Architecture

Five heads share one transformer state encoder (gus/BUILD_PLAN.md @ 31e10ef):

| Head | Output | Target |
|---|---|---|
| `belief_head` | P(dom ∈ seat) | True deal |
| `V_head` | scalar E[Q] | Oracle mean over sampled worlds |
| `π_me_head` | action softmax | Oracle argmax E[Q] |
| `world_encoder + Q_head` | per-world Q | Per-world oracle Q from the [[joint-world-tensor]] |
| `π_opp_head` | opponent action softmax | Oracle argmax per opponent seat ([[pi-opp-head]]) |

A world-conditioned Q_head caches `Q_m` per sampled world so belief updates re-weight
the cache instead of re-querying the oracle — the [[lamir1]] premise. Training data:
seeds 0–99 train, 900000–900099 eval (the LEM/Burl convention); corpora are
[[joint-world-tensor]] dumps, streamed via [[lazy-iterable-dataset]] at 10k-game scale.

## Headline findings

- **Best fast player**: `v3_consistency_10000g` — 0.551 Q-pt regret, 76.07% bot-match
  on 560 held-out decisions; first sub-1.0 regret; the [[consistency-regularizer]]
  scales better than plain distillation ([[gus-v3-consistency-full-run]],
  [[gus-scaling-ladder]]). Still the best-known single adapter, cited in [[jud]]'s
  asset map.
- **Data dominates capacity**; explicit void features are marginal — the transformer
  infers voids attentionally ([[gus-v2-voids-1000g]], [[dense-q-supervision]]).
- **Belief top-1 is at the Bayes ceiling** (39.184%) — an information limit, not an
  architecture gap ([[belief-bayes-ceiling]]); calibration improves but doesn't
  propagate downstream ([[belief-propagation-gap]], [[belief-co-train]]).
- **Look-ahead over distilled heads fails**: all 8 LAMIR-1 modes lose to direct π_me;
  scalar distillation noise flips argmax at decision boundaries ([[lamir1-ceiling]]).
  The pivot taken (option 4, self-play without CFR+) became [[w42-jud-v1]] and [[jud]].
- **Routing beats ensembling**: a belief-sampled Q-mean second opinion on ~6% of
  decisions reaches ~0.42-0.43 regret with zero new blunders ([[gus-qmean-router]]);
  oracle-fallback routing reaches 0.49 ([[gus-router-pilot]], [[blunder-detector]]).
- **q-bootstrap-belief**: worlds sampled from the belief head beat oracle-corpus worlds
  for Q aggregation (0.655 vs 0.685) — the standing evidence for wiring belief into
  world sampling ([[gus-belief-co-train]]).
- **The craft of 42 is front-loaded**: 26.2% of decisions are fog-of-war drama and 62%
  of drama is the opening lead ([[gus-drama-atlas]], [[past-belief-future-direction]]).
- **The student is legible**: counterfactual V sensitivity matches oracle E[Q] deltas
  within 0.5 Q-pts ([[gus-probe]]).

## Where the value went

**[[jud]]** (2026-06): the jud direction makes Gus's belief head the posterior
engine of the unified player — conditioned on auction + play history (belief v2, rung
#24 measured win: +2.59pp, [[w42-champion-auction-belief]]) and wired into oracle world
sampling. Belief-weighted *play* sampling measured dead (rung #25); belief's value routes
to bidding/defense via self-play ([[belief-conditioned-self-play]]). `gus/bidding/`
(2026-04) uses Gus as the simulating player for contract evaluation and is the substrate
for the champion's auction work.

**[[burl]]**: the production [[belief-trajectory]] tool serves the `v3_consistency_10000g`
belief adapter (per-domino posterior, shift-since-last, V, CLS attention @ d858781) —
Burl consumed a Gus checkpoint before Gus's own development was documented here.

**[[book-strategy-player]]** (designed 2026-05-03; build pending): Gus as input encoder —
belief tensor + V_head + world_encoder feeding a strategy-selector model. Feeding
decision-time models is what Gus was originally designed for; the consumer turned out
not to be the LLM.

**[[w42]]**: the [[gus-strategy-tags-probe]] side branch (book-derived tags cut tiny-model
regret 2.012 → 1.181) was promoted to the top-level w42 workstream the next day.

## Routes

- [[gus-line]] — the trail: full arc, all 22 experiments with hooks.
- Key leaves: [[gus-scaling-ladder]] (the ladder), [[gus-v3-consistency-full-run]] (best
  adapter), [[lamir1-ceiling]] (the pivot record), [[gus-belief-co-train]] (belief
  endgame), [[gus-qmean-router]] (last routing word), [[gus-drama-atlas]] (what lies
  past belief).
- Concepts: [[student-distillation]] · [[dense-q-supervision]] · [[regret-eval]] ·
  [[v-pi-decoupling]] · [[consistency-regularizer]] · [[qmae-plateau]] ·
  [[belief-bayes-ceiling]] · [[joint-world-tensor]] · [[gen-fleet]] (the fleet that
  never launched).
