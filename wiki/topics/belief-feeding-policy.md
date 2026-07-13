---
title: Belief Feeding Policy
kind: topic
first_seen: 2026-07-06
last_updated: 2026-07-13
status: complete
---

## The question, asked as a throwaway on the way out the door

[[zeb]]'s belief head — added `93b7037` on 2026-02-08, predicting which seat holds each
domino — ran alongside its policy head for the rest of the era, connected only through a
shared training loss. It never fed a single move choice. The era's own last recorded note,
3:45am on the way out (2026-02-18), asks the question directly rather than proposing an
answer:

> "so how do we get beliefs feeding policy? right now we kind of have 2 things side by
> side only connected via loss essentially" (2026-02-18T03:45,
> [[sources/claude/era4-zeb-era|conversation digest]])

This sat unaddressed through the [[full-teacher-eq-experiment]] closeout the same week —
that experiment fed [[expected-q-value|E[Q]]] into the policy head directly, not belief
into policy — and through the rest of era 4. It is, in the era-4 retrospective's own
reading, the era's most valuable output, filed as an aside rather than pursued.

## Where the answer eventually comes from

The question is answered months later, by a different project, not by extending Zeb:
[[gus]]'s belief head and [[belief-trajectory]] give [[burl]] a belief signal that
actually participates in decisions, rather than sitting beside a separate policy
mechanism connected only by loss. Zeb itself is parked as a belief-only tool
([[zeb-parked-eq-primitive]], 2026-04-18) before being superseded by
`v3_consistency_10000g` ([[zeb]], 2026-04-23) — belief and policy stay decoupled inside
Zeb; the coupling happens architecturally elsewhere.

Even there, the coupling is imperfect. [[belief-propagation-gap]] (a later [[gus]]
finding, `137a8e7`) shows that a measurably better-calibrated belief head does not
automatically propagate to better downstream plays (Q, [[pimc]], the
[[blunder-detector]]) — "better beliefs don't produce better plays" on its own. Making
belief *available* to a policy (Zeb's gap) and making a policy actually *use* an improved
belief (the propagation gap) turn out to be two separate unsolved problems, not one.
[[belief-co-train]] is the concrete later attempt to close the second gap by training
belief, world-encoder, and Q head jointly rather than in sequence.

## Why this matters more than the "capacity ceiling" framing

[[full-teacher-eq-experiment]] named its result a capacity ceiling. The evidence from
Zeb's own self-play scaling (six-times the params bought about five points vs random)
argues against that framing — it looks like an information/architecture ceiling instead.
The decoupled belief head is a concrete, checkable instance of that architecture gap: a
real signal existed, was accurate enough to publish, and simply never reached the
decision it should have informed. "Make the model bigger" would not have fixed this;
wiring an existing signal into the decision path might have.

See [[zeb]] · [[gus]] · [[belief-trajectory]] · [[belief-propagation-gap]] ·
[[belief-co-train]] · [[sources/claude/era4-zeb-era|conversation digest]].
