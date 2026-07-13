---
title: Single-Fact Enumeration (vs Long-Enumeration Failure)
kind: topic
first_seen: 2026-04-17
last_updated: 2026-04-17
status: active
---

## Overview

A structural finding from Stage 0 v9 and the 14B capacity experiment: long-enumeration answer formats produce 0% accuracy, while single-fact questions on the same underlying knowledge produce near-100% accuracy. The model has the knowledge; the output format is the problem (b857299, 0c7392f).

## The finding

`visibility_audit` in [[game-context-qa]] asks the model to enumerate all visible and unseen dominoes — a long list. It scored 0% on both the 1.7B ([[v9-adapter]]) and 14B ([[qwen3-14b]]) models. `highest_unseen_in_suit`, a single-fact question requiring the same underlying knowledge ("which unseen domino would win if played in this suit?"), scored 100% on the same 1.7B model (b857299).

The cross-model confirmation on 14B is decisive: this is not a capacity problem. The 14B model hit 0% on `visibility_audit` with the same answer format. "Long-enumeration answers fail autoregressive truncation." (b857299) Under autoregressive decoding, the model drifts or truncates before completing a long enumeration; the partial output fails the grader.

## Generalizable recommendation

For tasks that could be formulated as "enumerate everything," decompose into many single-fact questions whose answers collectively reconstruct the enumeration. Each individual question is answerable reliably; the set of answers gives the same coverage (b857299).

This is the design principle behind the single-fact supporting categories (`highest_unseen_in_suit`, `is_trump`) that reliably hit high accuracy alongside compound tasks (`beaters_in_unseen`, `partner_response`) that are harder (b857299).

## Implications for scratchpad validation

[[scratchpad-validation]] requires the model to enumerate HAND, VOIDS, COUNTS in a structured block. The long-enumeration finding suggests these fields should be decomposed into single-fact checks if scratchpad validation is re-enabled — or verified one claim at a time rather than as a full enumeration block (0c7392f).

## Links

[[game-context-qa]] [[rules-adapter]] [[v9-adapter]] [[qwen3-14b]] [[experiments/qwen-14b-capacity]] [[scratchpad-validation]]
