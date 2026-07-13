---
title: K1 Grading — Beat the Bot It Replaced
kind: topic
first_seen: 2026-04-09
last_updated: 2026-04-11
status: superseded
---

## Overview

K1 is the success criterion for a [[star]] trace in [[lem]] Stage 1. A trace is accepted if and only if `E[Q][gemma] >= E[Q][bot]` — that is, the model's chosen action must be at least as good in expectation as the action the replaced bot would have taken (lem/OVERVIEW.md @ a8bccfa).

## Relief-pitcher framing

K1 is grounded in the "relief-pitcher" metaphor: [[gemma-4-e2b]] takes over mid-game from a bot and must do at least as well in expectation. The bot's [[expected-q-value]] provides the acceptance threshold. This framing gives a natural baseline and a realistic starting-state distribution for training games (lem/OVERVIEW.md @ a8bccfa).

## Equivalence

Because the bot is E[Q]-greedy (argmax over N=10 rollouts), K1 reduces to: Gemma picked an action whose E[Q] ties or beats the bot's argmax. In practice, K1 passes when Gemma picks an argmax-tied action. There is no partial credit — any legal action at or above the bot's E[Q] threshold passes (lem/gemma_star/star_harness.py @ 7538016).

## Self-sharpening property

As Gemma improves, its E[Q] rises, which raises the effective threshold for future K1 grades. The baseline self-adjusts without any manual tuning — better model outputs become harder to beat, keeping the training signal live (lem/OVERVIEW.md @ a8bccfa).

## First measurements

Base-model Gemma 4 E2B (no adapter) passes K1 on 60% of 10 trick-6 decisions, measured via the local runner (llama.cpp, CPU). Breakdown: 60% pass, 30% fail (legal but suboptimal), 10% illegal (hand tracking error), 0% parse fail. The surprisingly high base rate implies many trick-6 decisions have near-unanimous-argmax correct answers — the E[Q] signal is concentrated. See [[base-model-k1-baseline]] (f578bfa).

## Failure path

Traces that fail K1 are not discarded. They are routed to [[r1-rationalization]], which reveals the bot's action and asks the model to produce a trace that arrives at that action. At this frontier, illegal parses are also rationalized rather than discarded (lem/gemma_star/star_harness.py @ 7538016).

## Pass-rate progression across 15 STaR iterations

With Stage 0 adapter as starting point, K1 pass rate across 15 iterations: 30% → 34% → 33% → 36% → 35% → **42%** → 36% → **42%** → 38% → 36% → 39% → 41% → 40% → 39% → 38%. Best recorded: 42% at iterations 5 and 7. Plateau band: 38–41% in iters 10–14. The base-model K1 baseline (no adapter, 60%) is measured on a different distribution — the base model is tested via llama.cpp on a small sample, whereas STaR iterations run on the Stage 0 adapter against a filtered subset of the training pool (lem/OVERVIEW.md @ 908773a). See [[star-10-iterations]].

## Ceiling hypothesis (superseded — see revised framing below)

After 15 STaR iterations, pass rate plateaued at 38–41% and did not improve further. The
frontier's explanation at the time: "The plateau at ~40% likely reflects the ceiling of K1
grading without fact-verification. The model may be learning wrong game-facts that happen
to produce correct plays ~40% of the time but can't go further because the reasoning is
polluted." (lem/OVERVIEW.md @ 908773a)

K1 rewards matching the argmax play, but a model can match the argmax by accident or by
reasoning from incorrect game-state that coincidentally produces the right answer. Without
a mechanism to verify that the reasoning chain is factually grounded, the improvement
signal caps at whatever fraction of decisions the model can get right despite polluted
reasoning. This motivated [[scratchpad-validation]] as a proposed remediation — but the
premise was never tested, because a cheaper lever (better Stage 0 curriculum) broke the
plateau first. See below.

## Revised ceiling framing

The ~40% K1-without-fact-verification reading (ingest 10, 908773a) was premature. v3
pushed through it to 48% (ingest 13, 8c1bb14) — the [[kerry-curriculum]] and
[[trump-drilling]] curriculum rounds raised the STaR ceiling without any
fact-verification mechanism at all. The revised hypothesis: K1's apparent ceiling
depends on the rules-comprehension floor Stage 0 provides — better Stage 0 → higher
STaR plateau. [[scratchpad-validation]] may still matter eventually but is not proven
necessary; the ingest-10 "structural ceiling" framing was curriculum-bounded, not
K1-structural. See [[stage-0-progression-star]] and [[learned-by-playing]] (8c1bb14).

## Stricter grading attempted and shelved

For a 14-minute window on 2026-04-11 (commits 380f3fa → 78ba940), K1 was superseded by a `valid_pass` grade that required engine-verified scratchpad facts in addition to E[Q] dominance. The first iteration produced 64.5% `invalid` traces and only 5 trainable examples — too strict for a model that had never seen the scratchpad format. Reverted to simple K1 in 78ba940. See [[scratchpad-validation]].

## Links

[[star]] [[expected-q-value]] [[r1-rationalization]] [[lem]] [[gemma-4-e2b]] [[base-model-k1-baseline]] [[star-harness]] [[scratchpad-validation]]
