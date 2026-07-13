---
title: Research Lane Selection (2026-07)
kind: decision
first_seen: 2026-07-13
last_updated: 2026-07-13
status: active
---

The selection step [[partnership-research-gates]] anticipated ("a later
research decision selects one causal question from the evidence ledger",
build-ladder step 3) — taken 2026-07-13. Source: an external deep-research
synthesis over the wiki record, reviewed against
[[partnership-wall-research]] and adopted; its literature grounding is filed
at [[search-literature-transfer]]. Selected lanes run per the
[[research-night]] playbook.

This decision selects **experiments and their order**, not an architecture.
Promotion still gates through [[partnership-research-gates]] unchanged.

## The frontier, restated

The research question is not "find a better generic imperfect-information
algorithm." It is: turn [[forge]]'s exact per-world counterfactuals into one
legal, decentralized, information-set-consistent continuation policy — while
public actions change what partners and opponents believe and therefore do.
What E[Q] cannot price is **continuation strategy fusion**, not a lack of
tactical planning ([[champion-design-review]], [[strategy-fusion]]).

## Prerequisite — Stage 0 measurement closure

The repaired-sampler CUDA benchmark, the historical exposure scan, and the
two-block P0/C0 reproduction close before any lane result is graded
([[partnership-wall-research]] Stage 0, [[world-sampler-mrv-audit]]). The
expected lane deltas are small enough that silent world-distribution bias
could produce or erase them.

## Selected lanes

| lane | question | why selected | first gate |
|---|---|---|---|
| **A — auction decoder** ([[auction-decoder]]) | Do auction likelihoods, decoded with role/order/score and a policy-type mixture, improve held-out inference — and then a realized-outcome consumer? | The only surface where learning has already won marks; +2.59pp auction-conditioned belief; the book's bid-semantics corpus; direct literature support ([[search-literature-transfer]]). | Held-out NLL/calibration + true-world rank beats a hand-independent bid model; consumers stay realized-outcome-priced. |
| **B — target granularity × capacity** ([[jud]] v2 ladder) | Does per-move supervision beat hand-level supervision at fixed capacity? | The cleanest attack on the diagnosed play wall; v1 left target and capacity entangled; [[lamir1-ceiling]] supplies the mechanism prior. Two signals kept distinct by consumer: dense E[Q] ranking auxiliary vs policy-conditioned realized continuation value. **First gate graded 2026-07-13 ([[jud-target-granularity]]): the parent-side dense auxiliary is a marks null at fixed capacity/corpus — ranking gain without marks gain does not pass.** Residuals: child-state continuation targets, corpus volume, capacity×target interaction. | Held-out move-ranking or marks gain over hand-level targets at fixed capacity; calibration alone is not passage. |
| **C — convention factorial, before any convention search** ([[convention-aware-blueprint-search]]) | Does one installed book convention produce a sender × reader marks interaction under fixed vs shuffled partners? | Measures the channel before building machinery to optimize it. If a literal installed convention cannot produce the interaction, a tree will not create the evidence. | A sender-by-partner-reader interaction that survives opponent readers, on one compact convention family. |
| **D — information-honest blueprint search** ([[belief-weighted-jud-mcts]], [[convention-aware-blueprint-search]]) | Does single-agent search over a fixed blueprint beat the blueprint — legally? | The main architectural frontier, eligible only after A supplies calibrated likelihoods or C demonstrates a convention. Semantics 1 only at first; J3 before a calibrated likelihood instrument collapses into J2. | Paired marks over the blueprint with information-honesty verified (no partner-node max backups). |

A and B ran in parallel as the primary lanes; C follows; D builds on their
outputs. Any lane's deselection criteria are recorded on its page.

**Reprioritized 2026-07-13 (post-execution, adopted at PR review):** with
Lane B's first gate graded null in both target forms
([[jud-target-granularity]]) and Lane A's instrument validated cleanly
([[auction-decoder-v0]]), the order becomes **A first** (enriched-bid corpus
→ book fixtures → likelihood consumers), **C second** (the convention
factorial and its information-reactive harness), **B parked** (its residuals
— capacity×target interaction, on-policy loop data, opponents-in-rollout —
reopen only if A produces a likelihood worth putting inside a search, or the
itch strikes), D unchanged (eligible after A supplies calibrated likelihoods
or C demonstrates a convention). The play-leaf ground measured cold; the
information ground measured warm.

## Held back, with reasons

- **Whole-game CFR or ReBeL** — guarantees are two-player zero-sum; the team
  game needs the coordinator/prescription representation first, and even then
  only bounded microgames ([[search-literature-transfer]]).
- **A centralized "team player"** — sees both partner hands; illegal. A legal
  coordinator sees common information and emits prescriptions.
- **Public-history-only ISMCTS** — over-merges; complete-world MCTS
  under-merges. Node identity must be actor-relative
  ([[belief-weighted-jud-mcts]]).
- **Root-belief determinized MCTS sold as an imperfect-information solution**
  — J2 may add tactical value; it is reported as a search improvement only.
- **Double-dummy bidding labels** — rejected by the project's own fixed-point
  experiment ([[w42-champion-selfplay-fixed-point]]); feature or bound, never
  target.
- **A generic symbolic plan library** — Forge Q already prices ordinary
  within-world plans; only cross-world consistency, partner-visible intent,
  concealment, and belief-dependent continuation remain plausible
  ([[champion-design-review]]).
- **Another undirected AlphaZero/self-play run** — the ledger already shows
  this shape learned competent play without crossing E[Q]
  ([[alphazero-under-imperfect-information]], [[zeb]]).
- **An LLM as move selector** — LLM value sits in rules-as-tools,
  explanation, hypothesis generation, and microscope work ([[burl]]); gated
  separately in [[partnership-research-gates]].
- **A fixed risk transform over the Q PDF** — EV already beat p_make, CVaR,
  and robust quantiles ([[w42-lens-v1-utility-head-to-head]]); the surviving
  question is contextual shape use, which remains a cataloged direction.

## What would reopen this selection

A Stage 0 failure (C0 does not reproduce on the repaired sampler) suspends
lane grading and reopens the measurement question. Both primary lanes failing
their first gates returns selection to the
[[partnership-wall-research]] Stage 2 ledger, which preserves the competing
explanations for exactly that purpose.

## Links

[[partnership-wall-research]] [[partnership-research-gates]] [[the-wall]]
[[auction-decoder]] [[search-literature-transfer]] [[jud]]
[[belief-weighted-jud-mcts]] [[convention-aware-blueprint-search]]
[[world-sampler-mrv-audit]]
