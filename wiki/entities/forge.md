---
title: Forge
kind: entity
first_seen: a8bccfa
last_updated: local-2026-07-06
status: active
---

## What it is

Forge is the project's central shared ML infrastructure — the oracle, the E[Q] framework,
and the training/analysis substrate no single project owns. It was originally built for
[[lem]] and [[burl]]; its current consumers are [[gus]], [[w42]], jud, and [[champion]]
(`forge/oracle/`, `forge/eq/` — see `wiki/entities/jud.md` lines 19, 31, 321), with LEM and
Burl now dormant. (lem/narrate/OVERVIEW.md @ a8bccfa; burl/OVERVIEW.md @ 8d26e0d)

## Origin

Named and structured on 2025-12-30 ("Crystal Forge: Lightning-first ML pipeline," `2559818`):
`forge/oracle/` (the GPU tablebase solver, moved in from a three-generation solver lineage
built two days earlier — see [[the-oracle]]), `forge/ml/` (LightningModule + DataModule),
`forge/cli/`. Naming moment: "the crystal forge sounds so badass. I can say 'yeah over in the
forge' and I feel like a cool dude." (conv bec8b3d4). See [[breakthrough-and-oracle]] for the
full era this was built inside — the algebra that made the solver trustworthy
([[suit-algebra]]) and the theoretical result ([[strategy-fusion]]) that shaped how its
output could later be consumed for bidding.

## Three distinct components (vocabulary enforced at 8d26e0d)

Earlier LEM-era docs informally called all three "the oracle." The Burl OVERVIEW enforces
the distinction:

| Term | What it is | Path |
|---|---|---|
| **Perfect-information solver** | Takes a fully-specified deal + declaration, enumerates reachable states, solves by backward induction. Deterministic, exact game value under full information. Neither LEM nor Burl calls this directly. | `forge/oracle/` |
| **E[Q] framework** | Uses the solver as a subroutine. For an imperfect-info position, enumerates plausible hidden-hand realizations, evaluates each via the solver, averages weighted by prior. Produces E[Q] given what is actually seen. Used for K1 grading. | `forge/eq/` |
| **E[Q] bot** | The `argmax(E[Q])` player. Picks the legal move with highest expected value. The comparison target in [[k1-grading]]. | Throughout game + narration code |

(burl/OVERVIEW.md @ 8d26e0d)

## Also in Forge: Zeb

[[zeb]] (`forge/zeb/`) is a 3.3M-parameter learned belief model (72% top-1 accuracy on
opponent-hand prediction, trained via self-play + oracle distillation). It is Burl-first
infrastructure but lives in Forge as shared infrastructure. LEM did not call Zeb directly.
(burl/OVERVIEW.md @ 8d26e0d)

Zeb was subsequently parked: E[Q] n=10's outcome PDF replaced it as the project's belief
primitive. See [[zeb-parked-eq-primitive]].

## Components consumed by LEM at this frontier

| Component | Path | Purpose for LEM |
|---|---|---|
| E[Q] greedy policy pipeline | `forge.eq.generate.pipeline.generate_eq_games_gpu` | Plays full games at N=10, produces 28 decision records per game |
| Q-value checkpoint | `forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt` | Required by the pipeline for [[expected-q-value]] scoring |
| Hand dealing from seed | `forge/eq/generate/deals.py` | Deterministic starting state from an integer seed |
| Bidder | `forge/bidding/` | Selects bid value and declaration (described as "broken but reasonable") |
| Suit and trick logic | `forge/oracle/tables.py` | `can_follow`, `led_suit_for_lead_domino`, `trick_rank` — used by [[narration]] to emit lines like "you were void in fives" |

(lem/narrate/OVERVIEW.md @ a8bccfa)

## Role in the LEM pipeline

The [[narration]] generator calls `generate_eq_games_gpu` to play a full game and receive
the sequence of decisions. The Q-value checkpoint embedded in that pipeline provides the
[[expected-q-value]] scores that drive both the game-play policy (E[Q] greedy, N=10) and
the [[k1-grading]] step in Stage 1. (lem/narrate/OVERVIEW.md @ a8bccfa)

## Relationship to LEM

Forge is not part of LEM. LEM imports Forge's public interface at the paths listed above
and does not modify it. The game generation policy (N=10 E[Q] greedy) was verified during
an earlier Forge experiment (zeb) as not materially worse than N=100 at a fraction of the
cost. (lem/OVERVIEW.md @ a8bccfa)

## Standalone analytics workstream

[[forge-analysis]] (`forge/analysis/`) is a sustained statistical-analysis workstream
over the oracle's ~300M-state output, distinct from any of the modeling siblings.
It produces a publication-shaped report; its load-bearing epistemic frame
([[oracle-vs-human-play]]) flags that all findings describe perfect-information
minimax structure, not human play. Highlights include the [[risk-return-inverse]]
correlation (r=−0.38) and the [[q0-positional-bias]] sub-investigation that
characterized a slot-0 anomaly in forge's trained Q-value model.
