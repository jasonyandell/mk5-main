---
title: Narration
kind: topic
first_seen: a8bccfa
last_updated: 7f1994e
status: active
---

## Overview

Narration converts a played [[texas-42]] game into second-person prose from one player's seat. The output is a game record as a human-readable text that a language model can reason over (lem/narrate/OVERVIEW.md @ a8bccfa).

## Consumers

1. **Humans** — eyeballing whether the voice is natural and self-contained.
2. **[[rules-adapter]] parsing checks** — Stage 0 uses held-out narrations to verify the adapter can parse game text.
3. **[[star]] prompts** — Stage 1 truncates narrations at the narrator's decision point to form the STaR prompt input (lem/narrate/OVERVIEW.md @ a8bccfa).

## Voice rules

- **Second-person** for the narrator ("You play the 6-6").
- **Third-person** for the other three players ("Player 1 follows with the 5-2").
- Partner is called "your partner" on first reference.
- Facts only — no strategy commentary in the baseline version.
- Points are counted at the end of every trick.
- Running score is kept as input; count tags are present in full narrations.
- **Forced-play annotations** are emitted when a player has only one legal move. These prevent [[star]] from hallucinating strategy on non-choices (lem/narrate/render.py @ a8bccfa).

## Truncation for STaR

The renderer supports a `stop_at_decision` parameter. When set, it truncates the narration before the narrator's specified turn and appends a "What do you play?" prompt with the narrator's remaining hand and score-so-far. This is how Stage 1 STaR prompts are built (lem/narrate/render.py @ a8bccfa).

## Batch mode

`lem/narrate/batch.py` processes N seeds × 10 declarations × 4 perspectives in a single pass. For each narrator turn at trick 6 it applies two filters before writing an example:

1. `|legal| >= 2` — must be a real choice, not a forced play.
2. E[Q] gap between best and second-best action >= threshold (default 1.0 point) — removes near-coinflip decisions where there is no learnable signal.

Each output JSONL record includes the full narration prompt (rules primer prepended), `legal_actions`, `bot_action`, `bot_eq`, `best_action`, `best_eq`, `eq_gap`, and `all_eq`. This grading data is embedded at generation time so the [[star]] harness can grade traces without re-running the oracle (lem/narrate/batch.py @ b99c64d).

Throughput: 200 seeds → 3148 trick-6 training examples in 8.5 min on a 3050 Ti (lem/narrate/batch.py @ b99c64d).

Eval seeds (900000–909999) are protected by default; a `--allow-eval-seeds` flag is required to generate from that range (lem/narrate/batch.py @ b99c64d).

## Datasets produced

At this frontier, datasets are local files — not yet uploaded to HuggingFace (b99c64d):

| File | Seeds | Examples |
|---|---|---|
| `lem/data/narrations_train.jsonl` | 0–199 | 3148 |
| `lem/data/narrations_eval.jsonl` | 900000–900049 | 812 |

The eval seed range is a deliberate holdout. See [[decisions/eval-seed-holdout]].

## Post-trick public state block (v3 format)

As of 7f1994e, narrations include a structured state block after every completed trick. The block contains:

- All dominoes played so far, as `N/28`
- Count-domino status: which team took each count domino, or that it is still live
- Narrator's remaining hand

Rationale: "A 2B model shouldn't reconstruct game state from prose any more than a human should memorize 28 dominoes. In real 42, this information is public and visible at the table." (7f1994e) The state block surfaces information that was already in the narration in distributed prose form but required active bookkeeping to reconstruct. Cost: ~60 tokens per trick, ~300 extra tokens per prompt — cheap relative to the reduction in bookkeeping burden on the model.

This format is narration v3. V1 narrations (without state block) are no longer used by default for Stage 1 going forward. See [[decisions/public-state-block]] for the design rationale.

## Implementation

`lem/narrate/render.py` contains `render_narration()`. It consumes a `GameRecordGPU` from [[forge]]'s E[Q] generation pipeline, using `forge.oracle.tables` for suit/trick logic and `forge.oracle.declarations` for trump classification (lem/narrate/render.py @ a8bccfa).

## Open questions

- Whether to restate running count and void information each trick or only at the decision point. **Resolved 7f1994e**: a structured state block is emitted after every trick.
- Whether the setup block should include full declaration rules or rely on [[rules-adapter]] weights. (?) (lem/narrate/OVERVIEW.md @ a8bccfa)

## Links

[[lem]] [[star]] [[rules-adapter]] [[forge]] [[expected-q-value]] [[texas-42]] [[decisions/eval-seed-holdout]] [[decisions/public-state-block]]
