---
title: Preserve Thoughts at SFT
kind: topic
first_seen: 2026-04-19
last_updated: 2026-04-25
status: active
---

## Overview

Preserve-thoughts is an SFT training variant for [[burl]] that bypasses `apply_chat_template`'s `strip_thinking()` filter so assistant-turn `<|channel>thought...<channel|>` regions reach the loss computation. The default (`preserve_thoughts=False`) strips thought blocks before tokenization, meaning Gemma's winning reasoning chains never produce a gradient (20f4fa2).

## Reframe (edf86e9)

The iter-4-thoughts byte-identical A/B result (see "Result: null" below) was very likely a **truncation artifact**, not LoRA capacity saturation. TRL's `SFTConfig` defaults to `max_seq_length=1024` when the parameter is unset. The preserve-thoughts corpus has median 2,054 tokens and max 4,210 tokens per row — exactly the thought-bearing rows the recipe is meant to train on were being silently truncated to 1,024 tokens. Both paths had the same latent default: the Modal recipe (`star.py`) and the local MLX recipe (`star_mlx.py`) (edf86e9).

Fix: explicit `max_seq_length=4096` in `SFTConfig`. Local path was fixed first in 6fea6ab; Modal path fixed in edf86e9.

**Consequence:** no Burl adapter trained before iter-5 was trained on complete thought-to-tool-call traces. The thought tokens were present in the `formatting_func` output but were being clipped before they reached the loss (edf86e9).

This is a parallel to LEM's [[decisions/sft-completion-only-loss]] finding: TRL defaults silently trap gradient. The pattern — "we think we're training on X, but a default is preventing it" — has now appeared twice in this project (edf86e9).

**Follow-up:** iter-5 with `max_seq_length=4096` explicit shows preserve_thoughts moves bot-match +3.3pp at N=26 ([[iter5-e1-rank-sweep]]); the [[burl-star-run3]] data point at N=560 (run-3c vs run-3b) confirms the directional signal at scale and is the definitive result — see "Result: confirmed" below. [[experiments/iter4-null-preserve-thoughts]] updated accordingly.

## Implementation

A `formatting_func` path in `burl/train/star_iter0.py` is gated behind `preserve_thoughts: bool = False`. When enabled, assistant-turn content is appended verbatim — thought prose and all — instead of going through `apply_chat_template`. User-turn rendering stays canonical. `GEMMA4_TURN_TERMINATOR` is appended after assistant content to maintain format validity (20f4fa2).

## Empirical scale

On a real iter-3-rules corpus row: `apply_chat_template` path produces 1,714 tokens (thoughts stripped); `formatting_func` path produces 2,246 tokens (thoughts preserved) — +532 thought tokens per row. Across ~130 rows × 3 epochs: ~200K additional thought tokens that should reach the loss once `max_seq_length=4096` is set (20f4fa2, edf86e9).

## Result: null (suspect — see Reframe above)

A/B comparison against the stripped baseline (iter-4-thoughts vs iter-3-rules, identical recipe otherwise) produced **byte-identical adapter weights**. At the time this was interpreted as LoRA capacity saturation. The truncation discovery (edf86e9) makes this result uninformative about whether preserve-thoughts works — the thought tokens never reached the loss in either branch (dbadb5f, edf86e9).

## Result: confirmed (run-3c, 2026-04-25)

[[burl-star-run3]] is the cleanest preserve-thoughts vs no-preserve-thoughts A/B in the project to date — same 1062-row strict-pool corpus from [[burl-2000-harvest]], same rank-8 lr-3e-5 1-epoch recipe, same `--max-seq-length 2048`, same early-stop, same held-out 560 eval. The only delta is the `--preserve-thoughts` flag.

| Metric | Run-3b (OFF) | Run-3c (ON, n=560 final) |
|---|---|---|
| Best val loss | 0.238 | 0.268 (~13% higher) |
| Wall-clock (training) | 32 min | 45 min |
| Eval n | 113 (killed mid-run) | 560 (full) |
| **Thought-block emission at inference** | **0 / 113 = 0%** | **537 / 560 = 95.9%** |
| Bot-match | 60.2% | 66.6% |
| Legal | 100% | 100% |
| Mean \|Δ\| | 2.89 | 2.111 |
| Mean signed Δ | not measured | −1.984 |

Within run-3c, the thought-vs-no-thought split tightened from the early-sample to the full-sample read — at n=304 the no-thought subset looked ~8pp worse on bot-match, but at n=560 the gap is ~1.5pp:

| Subset | n | Bot-match | Mean \|Δ\| | Mean signed Δ |
|---|---|---|---|---|
| Thinking | 537 | 66.7% | 2.110 | −1.985 |
| No-thought | 23 | 65.2% | 2.130 | −1.965 |

This is a phase change *on the thought-emission axis*, not a knob — without `--preserve-thoughts` a Burl LoRA trains reasoning *out* of the model. The [[iter5-e1-rank-sweep]] +3.3pp signal at N=26 was the directional read; at N=560 it's a 95-pp swing on thought-block presence with bot-match holding or improving. **Default it ON for any [[burl]] STaR or [[r1-rationalization]] run.**

A note of caution surfaced by the n=560 eval: the *thought-emission axis* is conclusively answered, but the *play-quality axis* needed regret-based evaluation to read clearly. The post-hoc rescore lands the play-quality verdict (see below).

### Play-quality axis (paired n=130 result, 2026-04-26)

The 2026-04-26 STaR-shaped post-hoc rescore (`scratch/belief_trajectory_rollout/star/STAR_EVAL_REPORT_2026-04-26.md`) controls for decision difficulty by comparing run-3b and run-3c on the *same 130 indices* run-3b managed to evaluate before being killed:

| Metric                  | run-3b @ 130 | run-3c @ paired 130 |
|-------------------------|-------------:|--------------------:|
| k1_pass_rate (Δ ≥ 0)    |   61.5%      |  64.6%              |
| match_oracle            |   56.9%      |  57.7%              |
| mean_signed_delta       |   −2.92      |  **−2.16**          |
| **mean_oracle_regret**  |   **3.02**   |  **2.255 (−25%)**   |
| forced_commit_rate      |   32.3%      |  **32.3%** (identical) |

**This is the cleanest preserve-thoughts-helps-play-quality result the project has to date.** Per-decision: run-3c beats run-3b on regret on 28/130 decisions, loses on 23, same on 79 — most decisions are unchanged, but the moved subset moves the right way. The identical 32.3% forced-commit rate independently confirms preserve-thoughts is *not* the driver of [[commit-discipline-collapse]] — the FC inflation is a separate, adapter-family-level issue that affects both variants equally.

So the headline becomes: **`--preserve-thoughts` is a phase change on thought-block emission AND a real per-decision improvement on play quality, controlling for FC inflation.** The `--preserve-thoughts` decision is not a tradeoff between "more thinking" and "less play quality" — it is a strict win on both axes once the eval is regret-based ([[regret-eval]] §"Ported to Burl evaluation").

## Links

[[burl]] [[burl-star-run3]] [[iter3-rules-adapter]] [[iter5-e1-rank-sweep]] [[experiments/iter4-null-preserve-thoughts]] [[decisions/sft-completion-only-loss]] [[decisions/sft-max-seq-length]] [[regret-eval]] [[commit-discipline-collapse]]
