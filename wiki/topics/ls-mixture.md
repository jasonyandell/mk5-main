---
title: LS-Mixture (verbosity blender)
kind: topic
first_seen: 2026-04-19
last_updated: 2026-07-11
status: retired
---

## Overview

LS-Mixture is a corpus construction technique that blends concise and verbose traces in a [[star]] training corpus. Disambiguation: "LS" here is the arxiv-2505.03469 **long/short** (verbosity) blend — the sense of the source design doc (burl/experiments/corpus_blend_design.md, deleted; @ 233b7dc5). Wiki glosses that expand the abbreviation as *legal-but-suboptimal* trace mixing ([[commit-discipline-collapse]], the index) point to this same staged workstream — no suboptimal-trace mixing was ever designed; the legal-but-suboptimal concept lives in [[r1-rationalization]] grading, a different mechanism. For [[burl]], it addresses the problem that trimming the rules primer (iter-1) produced traces with 3× larger `<|channel>thought` bodies — useful at inference time but potentially harmful as training signal if the model learns to always ramble (3414507).

## Method

The blender shortens long `<|channel>thought` blocks in existing traces while preserving every `tool_call`/`tool_response` envelope and the terminal `commit_play`. The blend ratio is controlled by `target_short_ratio` (default 0.33 — the short cohort is one-third of the corpus). The 0.33 value is nudged above the LS-Mixture paper's reported 25-50% band because Burl's dominant failure mode is verbosity, not under-reasoning (3414507).

Preview corpus from iter-0 and iter-1 rows: 118 total (79 long + 39 short). Short cohort mean: 767 chars; long cohort mean: 5,078 chars — an 85% reduction on shortened rows, beating the paper's reported 47.6% (3414507).

## Key finding from training

Gemma 4 E2B's `chat_template.jinja` strips `<|channel>thought` blocks before tokenization. SFTTrainer calls `apply_chat_template` per row, so the trainer never sees the thought prose. This means the verbosity difference between long and short traces is invisible at training time. The blend's actual benefit is: (1) 118 vs 79 rows for commit-discipline coverage, and (2) data-augmentation regularization (same tool-call chain, two framings) (eebcae5).

This also explains why iter-0 and iter-1 still ramble at inference: rambly thoughts are base-model reflex, not trained-in behavior (eebcae5).

Never used to train a shipped adapter — [[commit-discipline-collapse]] itself still calls it "staged" as of its own later writing. The [[burl]] line went dormant (2026-05-07) before this preview corpus was ever promoted to a training run.

## Links

[[burl]] [[star]] [[r1-rationalization]]
