---
title: "The Argmax-Q Ceiling (~74%)"
kind: topic
first_seen: 2026-01-17
last_updated: 2026-07-11
status: complete
---

## Overview

There is a hard, provable ceiling on how accurately an argmax-over-[[expected-q-value]] player
can match oracle-labeled ground truth: approximately 74%. It is not a training deficiency — it
is a fact about the game's tie structure. Established during [[eq-genesis]] (era 3), 2026-01-12,
and documented as a repo artifact the same week (`d9402cf`, "random tiebreaker ceiling
analysis"; `docs/random-tiebreaker-ceiling.md @ 233b7dc5`).

## The measurement

A 10M-sample Monte Carlo tie analysis found that only 55.7% of game states have a single
uniquely-best action (22.73% are 2-way ties, with the remainder split further). An argmax-Q
player that breaks ties with a coin flip therefore caps out at approximately **73.96%** accuracy
against oracle ground truth. The first Stage-2 training run landed at 74.2% — matching theory
almost exactly (`val/accuracy` 66.6% → 74.2%, alongside `val/q_gap` 2.02 → 0.356 and `val/q_mae`
7.73 → 3.48, 2026-01-11/12).

Jason's own gloss on the result (2026-01-12T03:56:42):

> "I think it's not that I should panic at 74, it's that anything OTHER than 74 is wrong."

The scan's far tail: exactly one state in the 10M sample has all seven actions tied —
sample 7,851,428, sixes trump, V = −36. P2, dealt no trumps at all, must play third into
a trick already won by P1's 6-5 (only the 6-6 beats it, and its holder P0 has already
committed); the game tree from that point is fully deterministic and every path loses by
36. Zero-agency states exist, but at one in ten million.
(docs/random-tiebreaker-ceiling.md @ 233b7dc5)

## Why breaking it requires a plan, not a better model

The ceiling exists because an argmax player breaks ties with a coin. The only way to do better
than ~74% is to break ties with a *reason* — a read on what the opponent likely holds, a signal,
a plan for the rest of the hand. That is precisely the consumption question [[strategy-fusion]]
named and deferred in the same week ("I also don't care about signaling. yet."). The 74% number
and [[eq-genesis|the wall]] are the same fact seen from two angles: hitting 74.2% felt like
arrival, but it is a precise measurement of how far argmax can go before it needs a personality.

Not to be conflated with [[zeb]]'s ~74% *vs-random win-rate* plateau (Feb 2026) — a different
quantity on a different (and suspect — [[vs-random-eval-is-suspect]]) metric. The numeric
coincidence is just that; Jason joked about it in-window ("it's always 74 with this game").

## Links

[[eq-genesis]] [[strategy-fusion]] [[expected-q-value]] [[q0-positional-bias]]
