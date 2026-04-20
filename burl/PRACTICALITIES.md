# Burl — Practicalities learned en route

The original plan in [`OVERVIEW.md`](OVERVIEW.md) is the plan. This file is the running log of **practicalities discovered while executing it** — assumptions that were too clean, surprises that changed how we ship, and the adaptations that absorbed them without changing the vision.

Each entry follows the same shape: **what the plan assumed**, **what we observed**, **how we adapted**, **where the evidence lives**. New practicalities append at the end; nothing gets deleted (even superseded entries stay as history, with a note).

---

## 1. Gemma 4 emits its native tool-call format zero-shot — don't fight it

**Plan assumed**: our hand-rolled `<think>` / `<tool>` / `<commit>` XML tags were a universal protocol Gemma would learn from warmup.

**Observed**: Move 3's first 3/3 rollouts produced `turns=[]` with retry-exhaustion. Ergonomics probes (`burl/GEMMA_4_ERGONOMICS.md`) showed Gemma's native thinking channel ate the full 512-token budget before the XML protocol ever started. Counter-probes then confirmed: given the correct chat template and `skip_special_tokens=False`, base Gemma emits `<|tool_call>` natively with no fine-tuning.

**Adapted**: switched to native-format harness (`burl/harness/tool_loop_native.py`, `burl/modal/gemma_serve_native.py`). Added `commit_play(domino_id)` as a tool — because Gemma wanted to emit every answer as a tool call anyway, we named one for it. Legal-move compliance is still a software invariant; commit just *rides* the tool channel instead of fighting it.

**Evidence**: `burl/GEMMA_4_ERGONOMICS.md`, `SPIKE_REPORT.md` R3 section.

---

## 2. The rules primer is dual-use — rules *and* commit discipline

**Plan assumed**: once the model "knew" 42, the LEM rules primer could be trimmed or dropped entirely.

**Observed**: iter-3-v2 (no primer) hit 20% retry-exhaustion despite producing better per-commit quality on the 80% that did commit. Conversely, iter-0 (full primer baked into SFT) fossilized Gemma's overconfidence — it stopped asking tools and bulldozed into the commit. The primer was doing two separate jobs: teaching the rules *and* teaching the commit-after-you-reason discipline. Dropping it lost the second; baking it in over-corrected.

**Adapted**: rules-as-tools preamble (`burl/tools/rules.py` + 645-byte preamble). Compact shape carries the discipline scaffolding; callable rules tools (`trick_winner_if`, `what_beats_what`, `contract_progress`) carry the rule lookups. iter-3-rules (90% bot-match, 100% first-legal, 0 retry-exhausted) is the receipt.

**Evidence**: `burl/experiments/iter3_v2_eval_writeup.md`, `burl/experiments/iter3_rules_eval_writeup.md`, `burl/experiments/rules_as_tools_design.md`.

---

## 3. The model invents its own reasoning idiom — grade by outcome, not tool-mix conformity

**Plan assumed**: Burl reasons by querying `eq_outcome_distribution`, probing with `conditional_outcome`, and synthesizing the shapes.

**Observed**: the first two winning adapters both *route around* that menu.
- **iter-1** reasoned structurally via `trump_declared` (9/9 decisions) + `is_trump` (7/9) + `is_legal` (16). Called `eq_outcome_distribution` 1/9 times. 88.9% bot-match, mean E[Q] delta −0.16.
- **iter-3-rules** reinforced `trick_winner_if` from 1.56 per-decision base → 1.70 post-SFT. 90% bot-match, 100% first-legal.

Two genuinely different idioms, both beat base. Neither uses the tool surface the way the architecture diagram shows.

**Adapted**: "go with the grain" was promoted from design principle to operating rule. K1 grading stays on committed E[Q] delta — never on *which* tools were called. Two points on a Pareto frontier (iter-1 for eq-delta, iter-3-rules for robustness) is a feature, not a failure to converge. The tool surface is a *menu*, and the model's entitled to pick a different meal than we expected — so long as the meal is good.

**Evidence**: `burl/experiments/iter3_rules_eval_writeup.md`, `SPIKE_REPORT.md` iter-1@r7 table.

---

## 4. `conditional_outcome` is still zero-shot invisible — environment-shape problem, not training

**Plan assumed**: the architecturally load-bearing counterfactual probe would fire naturally when `eq_outcome_distribution` returned a multimodal PDF. Model sees two modes, asks `conditional_outcome` to disambiguate.

**Observed**: **0 calls across 145+ decisions.** Haiku, Opus 4.7, every Burl adapter iter-0 through iter-4. Not a training-data gap — the ceiling models skip it too. Working hypothesis: the raw 85-bin PDF hides bimodality behind a histogram. Models read `stdev=20.5` but don't gestalt "two modes 41 Q apart, worth probing."

**Adapted**: two layers of return-shape redesign — both *still the same tool*, just rendered so the model can act on them.

- **Candlewax** (commit `1efb9c5`) extends `eq_outcome_distribution`'s return with `distribution_shape`, `modes`, `gap_between_modes`, and `suggested_counterfactuals`. The tool now *invites* the probe by naming specific useful assumptions with outcome-directional rationales (*"collapses the left tail — rules out the disaster swing"*).
- **Spike drivers** (commit `b0952a2`) answer the same question in Gemma's native vocabulary. For each mode, report the (seat, domino) assignments over-represented in that spike: *"if partner has 5, you win; if partner has 14, you lose."* The motivation is grain-recon — Gemma articulates decisions in bid-satisfaction vocabulary, not raw-PDF shape. Same information, language the model already uses.
- **`what_would_change_my_mind(play)`** (commit `7321952`, ITER4_PLAN §2 Candidate C) is a meta-tool that ranks probe-worthy assumptions *before* the model has to read any PDF. Shows up earlier in the tool menu than `eq_outcome_distribution` and may catch models that would otherwise never ask for the distribution at all.

**Status**: T11 smoke (no LLM, 10 plays × 5 decisions) produces legible, action-shaped hints. Team-lead's mockup spike showed base Gemma zero-shot reaches for `conditional_outcome` and quotes the rationale string verbatim when handed a candlewax-shaped response. T12 live eval under `--enable-rules-tools` blocked upstream — Gemma never asked for the distribution at all, so the redesigned return shape never entered the conversation. Trimmed-primer re-run queued as the disambiguator.

**Evidence**: `burl/experiments/iter5_e2_candlewax_eval_writeup.md`, `burl/tools/eq_distribution.py`, `burl/tools/meta_tools.py`, `burl/tools/test_eq_distribution_spike_drivers.py`.

---

## 5. Truncation silently steals training signal — always pin `max_seq_length`

**Plan assumed**: TRL's `SFTConfig` defaults were safe for our trace lengths.

**Observed**: iter-4-thoughts (`preserve_thoughts=True` on iter-3-rules' corpus) produced **byte-for-byte identical output** to iter-3-rules across 42/42 turns. Initial interpretations split between (a) LoRA-capacity ceiling and (b) Gemma's thinking reflex being pre-trained beyond SFT's reach. Audit during T12 staging then uncovered the real cause: `SFTConfig` was missing `max_seq_length`, silently truncating at TRL's default (~2048). Exactly the thought-bearing rows were cut. The preserve-thoughts recipe never saw the thoughts it was supposed to learn from.

**Adapted**: pinned `max_seq_length=4096` in `burl/train/star.py` (commit `edf86e9`). iter-5 E1's rank-16 adapter on the corrected path then diverged cleanly from base (D0 went from retry-exhaust → commit-match). The recipe was right; the training config was lying.

**Meta-lesson**: for any training-stack surprise, audit *every* `apply_chat_template` consumer and *every* dataloader for a silent length cap before interpreting the result as a scientific null.

**Evidence**: commit `edf86e9`, `burl/experiments/iter5_e1_capacity_eval_writeup.md` § "Why rank-64 and rank-128 collapse", `burl/experiments/iter5_e2_candlewax_eval_writeup.md` § "Parallel finding — training-path audit".

---

## 6. LoRA-capacity sweet spot at rank 16 on MLX-LM — more rank ≠ better

**Plan assumed** (ITER4_PLAN §1): hypothesis (a) capacity-bound → scale rank → unlock signal.

**Observed**: with `max_seq_length` fixed, the dose-response curve above rank 16 is **monotone bad**.

| rank | bot-match | n_completed | n_retry_exhausted | notes |
|---:|---:|---:|---:|---|
| base | 66.7% | 9/10 | 1 | — |
| 16 | **70.0%** | 10/10 | 0 | only adapter that beats base |
| 64 | 55.6% | 9/10 | 1 | malformed tool JSON, regresses below base |
| 128 | 0% | 0/10 | 10 | total policy collapse, 16k-char token-salad |

Root cause: **MLX-LM lacks gradient clipping.** At rank 64+ the LoRA subspace is large enough that the unclipped update on a 26-row corpus pushes adapter weights outside Gemma's effective manifold.

**Adapted**: **next lever is corpus size, not rank.** ITER4_PLAN §3's "iter-5-hybrid = rank + N=100" recipe is disconfirmed; corpus-size sweep at rank 16 replaces it. Gradient clipping in `burl/train/star_mlx.py` is a prerequisite for any future rank > 16 experiment.

**Evidence**: `burl/experiments/iter5_e1_capacity_eval_writeup.md`, `burl/train/star_mlx.py` (no-clipping note in-file).

---

## 7. Batch generation lifted the corpus-scale ceiling — 43 → 1334 tok/s on M5 Max

**Plan assumed**: corpus ≥ 100 rows → Modal for time reasons. Local M5 Max was "good for quick experiments, bad for scale."

**Observed**: `mlx_lm.batch_generate` on real Burl prompts (mean 2378 tokens, iter-3-rules shape):

| batch | aggregate tok/s | vs single-stream (83 tok/s) | peak mem |
|---:|---:|---:|---:|
| 1 | 93 | 1.1× | 10.1 GB |
| **64** (recommended) | **1206** | **14.5×** | **13.8 GB** |
| 128 (peak) | 1334 | 16.1× | 15.5 GB |
| 256 | 1309 | 15.8× | 19.2 GB (plateau) |

N=500 rollouts drop from hours-of-wall-time on the single-stream path to **~3.5 minutes**. Memory headroom is enormous (15 GB of 48 GB on batch=128).

**Adapted**: corpus-scale stops being a capacity decision. "What if the corpus is 10-20× larger?" — the exact lever Practicality 6 identified as load-bearing — is now free. Next: productize into `burl/modal/gemma_local_batched.py` + `burl/eval/run_move4_star_rollout_batched.py`, then layer `prompt_caches` reuse (untested, expected multiplier on top).

**Evidence**: commit `ed3cfc3`, `burl/experiments/batch_throughput_bench.md`, `burl/eval/bench_batch_throughput.py`.

---

## 8. M5 Max is an iteration multiplier, not a Modal replacement

**Plan assumed**: Modal L4 for rollouts, Modal B200 for training, everything cloud.

**Observed**: Apple Silicon unified memory + MLX-LM runs rank-64 LoRA in ~30 min at $0. iter-5 E1's four-way capacity sweep (base + 3 ranks × N=10) cost **$0.00 cloud spend**. Rollout at batch=64 is cheaper than L4 vLLM for corpus generation. *But*: Modal B200 still wins for final iter-N sweeps — reproducibility across runs, parallel adapter comparison, the `huggingface_hub` publish flow, team-lead access patterns.

**Adapted**: split path, parameterized by design.
- **Local (M5 Max)**: disambiguation experiments — rank sweeps, candlewax smoke, batch-throughput sweeps, prompt-shape A/Bs. Burn 10+ runs in a day at $0.
- **Modal B200**: canonical iter-N training runs that ship to HuggingFace. Reproducible, publishable, comparable.
- **Modal L4**: base-model rollout serving when scale outruns the M5 Max OR when a team-lead doesn't have local MLX.

Training recipe stays parameterized: `burl/train/star.py` (Modal/TRL/Unsloth) and `burl/train/star_mlx.py` (local/MLX-LM) are siblings consuming the same corpus schema.

**Evidence**: `burl/train/star_mlx.py`, `burl/modal/gemma_local.py`, `burl/modal/gemma_serve_native.py`, `burl/experiments/iter5_e1_capacity_eval_writeup.md` § Budget.

---

## Pattern across practicalities

Three meta-shapes keep recurring:

**Go with the grain** (P1, P3, P4). Every time we tried to impose a shape — XML tags, a specific tool-use path, a raw-PDF return — the model's own instincts won, and the right move was to redesign the environment around what it naturally emits. Candlewax and spike_drivers are the same tool rendered in the model's own language; `commit_play` is the model's "I want to tool-call my way out" turned into a sanctioned tool.

**The tool *surface* is the lever, not the training data** (P2, P4, P7 partial). When the model doesn't reach for a tool zero-shot, the cheap fix is usually the tool's *menu position, return shape, or vocabulary* — not more authored examples in the SFT corpus. Authored demonstrations are a weight-hammer when environment-shape is a scalpel.

**Audit before interpreting a null** (P5). Byte-identical output looked like a scientific finding. It was a missing kwarg. The meta-lesson applies anywhere: before concluding "the model can't do X," verify the model *saw* X at the expected fidelity.

---

## Adding to this log

When the next practicality lands, append it as section 9 with the same four-part shape. Link the evidence (commit hashes + writeup paths). Don't edit the earlier entries — if one gets superseded, add a note at its end pointing at the newer entry. This file is the running receipts; the plan it pays back stays in [`OVERVIEW.md`](OVERVIEW.md).
