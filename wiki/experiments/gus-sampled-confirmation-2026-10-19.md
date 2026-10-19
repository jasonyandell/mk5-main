---
title: "Gus: independently confirming the sampled policy"
kind: experiment
first_seen: 2026-10-19
last_updated: 2026-10-19
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and rationale

Does the secondary sampled-policy gain in [[gus-unanchored-outcomes-2026-10-19]]
repeat on independent deals and action streams? [[gus-tiny-gpt]] learns an action
distribution, whereas greedy decoding always takes its highest probability.
Those are different policies. This measurement precedes another training
change; it cannot replace the earlier comparison's failed primary endpoint.
[Stochastic policies and expected return](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html),
[PPO exploration and exploitation](https://spinningup.openai.com/en/latest/algorithms/ppo.html#exploration-vs-exploitation).

## Frozen comparison

All four checkpoints are fixed before new outcomes: initial parent, previously
selected unanchored round24, anchored round16, rehearsal round8. Each gets
512 new physical deals, four independently seeded sampled complete hands and
one greedy hand per deal:10,240 trajectories in total. No training, checkpoint
reselection, temperature sweep or automatic promotion. Same185,652-parameter
architecture, original own-seven/public-history inputs, tile head and legal
mask; fixed G0 partner, pinned Zeb opponents and bid30/pip task. The splat head
is not evaluated as a calibrated current-policy forecast.

The512 roots exclude604,195 prior physical deals, including every reserved root
and scheduled tournament deal of the preceding study. All512 become future
training exclusions. Different draws change the learner action-stream namespace;
models share the same root/draw streams. Nonlearner behavior is recomputed after
changed histories. Deterministic policy budgets stay fixed.

## Endpoints and uncertainty

The sole primary replication endpoint is unanchored sampled success minus
parent sampled success: at least1 percentage point and paired95% lower bound
above zero. Average the four draws within each root, then bootstrap512 physical
roots20,000 times, seed91943. Repeated hands are not independent physical deals.

A stronger deployment-evidence claim additionally requires unanchored sampled
success at least2 points above the greedy parent and1 point above sampled
rehearsal, each lower bound above zero. Neither criterion automatically replaces
the deployed model. Anchored controls, all greedy policies and decoding contrasts
are reported descriptively. No outcome-dependent stopping or confirmation
reselection. Scope remains these weights, training seed, fixed field and bid30
hand success, not general full-match or auction strength.

## Resumability, verification and budget

CPU float32, one example, one Torch thread; existing lazy immutable records.
Model tensors, source files, native binary, runtime, scripts and registration
hashes must match. Each unique state gets at most24 cooperative45-second
segments, each capped at90 seconds, with at most2 live evaluators. Every
completed unit survives interruption. Stop file `STOP` prevents continuation;
identity errors, replay errors and exhausted budgets preserve partial evidence
and halt. No silent budget growth.

Independently replay all hands and reconstruct learner inputs, legal support and
rewards. Audit paired frozen-seat prefixes until first learner divergence. Four
reverse-order byte rechecks per checkpoint give16 in total. Audit caps are180
seconds per state, at most2 concurrent; summary cap90 seconds. Preflight used
only two already-consumed development draws, checked distinct seeds and exact
reverse-order replay before fresh roots were generated.

## Public registration and live state

[Verified preregistration](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/7c2230d5b4a7a9336f327c1e717360f253a5df18/lem/gus-sampled-confirmation-2026-10-19/registration) precedes evaluation:16 files, including all
four checkpoints, excluded deals, roots, scripts and checksums. Registration
SHA256:`e65fd2893a800a9ad589b1bf541f63163e13bc18889a61d6eea389a5f450cbbb`.
Source:`cc5e021a`; local recovery:`scratch/lem/gus-sampled-confirm-v1`.
`run.py` owns the bounded evaluator/audit/summary controller. Inspect its process,
`controller.json`, segment receipts and `CURRENT.md` before any resume. No
results are reported in this registration.

The preceding study's [completion addendum](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/1e02a48b65bf66c96734741690e6a7903de17b77/lem/gus-unanchored-outcomes-2026-10-19)
is verified; canonical delivery is resolved. Jason authorized continuing.
The [[gus-book-silas-tools]] refresh still has no new Lem207/Book121 replies;
SilasPR166 remains open. Existing replay and lazy-record instruments suffice;
the transformer causal bridge remains separate. No additional request sent.

## Operational recovery before readout

The first segment writes found missing `progress/` directories. Completed lazy
records were retained. The controller was stopped cooperatively; its failure,
logs and segment receipts remain. Recovery created `progress/` and `audit/`
directories and resumed the same immutable protocol. No scripts, seeds, weights,
endpoints or record bytes changed, and no scores were inspected. Failed attempts
still consume the registered segment budget. This was filesystem preparation,
not a changed experiment or an added evaluation budget.
