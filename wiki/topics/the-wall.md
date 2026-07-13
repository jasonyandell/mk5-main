---
title: The Wall
kind: topic
first_seen: afd4802
last_updated: b28fb55a
status: active
---

The project's central question, stated precisely. Every era since January 2026 is an attempt on it; every experiment page in this wiki grades against it.

## The wall

[[expected-q-value|E[Q]]] n=10 — expected Q over ten sampled worlds consistent
with public history, evaluated per move — is the undefeated champion at pure
play as of `afd4802` (2026-07-06). It has beaten every learned challenger:
[[zeb]], [[burl]], [[gus]], and [[jud]]'s play half. It carries no behavioral
action likelihood, partner convention, or information-set plan across turns;
within each sampled exact world, however, Forge Q already prices play through
the end of the hand. It computes p(make) and EV and stops. Jason's diagnosis:
"like a Stephen Hawking voice modulator — you can understand it; it is obvious
within seconds it isn't like a person." The distributions it emits are
[[candlewax]]: gorgeous, honest, and mute on what to want.

The founding condition (late January 2026, Jason): "maybe the most amazing thing I've ever built and I dunno what to do with it, so let's friggin try stuff." The wall was sighted at least six times before it was named — the MCCFR kill note ("'boring and competent' isn't worth the squeeze"), the 2-2 lead filed as an aggregation bug, the January 10 scope cut ("I don't care about signaling. **yet**"), the [[argmax-q-ceiling|74% argmax ceiling]] celebrated as validation, the 4am candlewax naming, and March's "distilling that run would just be distilling the heuristic I used to select ≥18." See [[the-wall-biography]] for the full arc.

## The goal

Jason's statement (2026-07-06), the project's specification:

> Something that can take the E[Q] data and **reason** with it and **do better** and has a **plan/strategy/SOMETHING that actually succeeds**. That is a goal.

Every clause is falsifiable: "does better" is measurable against the champion; "has a plan" is checkable in the traces; "reasons with it" is the [[jud]] direction — belief-weighted where E[Q] is monochrome.

**Explicitly rejected as a goal:** "a 42 player that plays like a person" — "WAY too poorly specified" (Jason). The voice-modulator complaint is a symptom report, not a spec. Person-ness remains the felt criterion; the goal above is its specifiable projection.

## Goals vs instruments — do not promote

The archaeology's most common distortion is ideas promoted to goals. The standing hierarchy:

- **The goal**: reason with E[Q], do better, with a plan that succeeds. Nothing else.
- **Instruments**: narration/interpretability (a *verification surface* — get a talker, catch it saying actually-correct things, bootstrap on those; the hope that later turned out to be named STaR. "A player that sounds right is just a means to an end" — Jason, 2026-07-06); belief modeling (the posterior engine feeding decisions, per [[gus]] and champion rung #24); the book ([[w42]], a graded source of candidate plans); self-play (an idea probed by [[zeb]] — "can learning on its own break the wall?" — answered no at that mechanism).
- **Side benefits**: a player that is fun to sit across from; a player that can teach. Welcome, never load-bearing.

Sessions that lack this page invent their own goals (documented instances: the 2026-06-14 belief-legibility session; era-5's "better narrator" reframe). The corrective: state each experiment's question in question form, subjunctive preserved, and grade it against the goal above.

## Current coordinates (frontier)

From [[w42-jud-v1]] (`afd4802`): **the wall has a crack and coordinates.** Bidding validates — jud v0's `margin:wp`(head_8) is the first learned component ever to beat the hand-tuned champion on marks (+0.38 [+0.09,+0.67], +0.42 [+0.12,+0.72]). Play is mechanism-limited at the leaf: `judsearch` recovers two-thirds of the play gap oracle-free (−3.44 → −1.16) and stops; neither more worlds nor a better-calibrated head closes the rest. Verdict sentence: *a 470k MLP on hand-level Monte-Carlo labels cannot out-rank E[Q] n=10's per-move oracle.* The named-but-unbuilt continuation is jud v2: bigger leaf, per-move E[Q]-distill targets with the consumer declared, opponents-in-rollout.

The full ruled-in/ruled-out record across all mechanisms: [[consumption-ledger]]. The untried inventory: [[ideated-not-built]] and [[the-wall-biography]] §4.

## Registered directions

No successor architecture is selected; each direction below is a surviving
explanation, and none is the default build. The shared
measurement spine — repaired world sampler, canonical decision records,
two-block C0 reproduction — serves all of them equally
([[partnership-wall-research]] Stage 0). [[research-lane-selection]]
(2026-07-13) selects the next *experiments* — the [[auction-decoder]] and jud
v2's target-granularity ladder first, the convention factorial after — without
promoting any architecture past [[partnership-research-gates]].

- **jud v2** — the named-but-unbuilt continuation from [[w42-jud-v1]]: bigger
  leaf, per-move E[Q]-distill targets with the consumer declared,
  opponents-in-rollout. [[lamir1-ceiling]] supplies the per-move-target prior
  and marks CFR+ over distilled values as the one sanctioned look-ahead path
  never walked.
- **Partnership/coordination** — the newest instrumented family.
  [[partnership-wall-research]] gives it the same measurement spine and
  evidence discipline as the other directions without presuming it causes the
  wall or selecting a next experiment. Its target, [[partnership-value]], is a
  marks interaction — a mutually legible fixed pair must gain more than the
  same policies with partners shuffled. Natural policy legibility is broader
  than sparse intentional signaling; its aggregate value remains untested.
- **Contextual distribution consumer** — the mode/signal/hedge/gamble
  meta-strategy set of [[past-belief-future-direction]]; its training data
  already sits in the oracle's per-world tensor.

The plan correction is load-bearing across all of them. Ordinary within-world
plans are not missing from Forge Q merely because `lens:ev` chooses again next
turn. The plausible residual is narrower: action-derived inference, role/order
semantics, partner-visible intent, information-set consistency, contextual
distribution use, and auction/match-score value. A candidate clears the wall
only by beating `margin:wp(head_8) + lens:ev` in paired marks for a
demonstrated strategic reason.
