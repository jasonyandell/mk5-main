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

## 9. Hard-gated HATEOAS works, but the harness was silently dropping every tool response for two days

**Plan assumed** (Practicality 4): the right lever to un-freeze `conditional_outcome` was the **return shape** of `eq_outcome_distribution` — candlewax hints, spike_drivers, `what_would_change_my_mind`. The tool *menu* itself was treated as stable; only the data inside responses would be redesigned.

### First reading (premature, 2026-04-20 morning)

`burl/wax_museum/` hard-gated the tool surface: `initial=[explore_game]` → `after_explore=[+probe_*, +ask_rule]` → `after_probe=[+commit_play]`. Base Gemma 4 E2B N=3 local on M5 Max in 65 s hit **probe rate 3/3** (vs 0/145 historical across Haiku, Opus 4.7, and every Burl adapter). First-glance celebration: menu composition is a new lever. Logged this entry with wrinkles (hallucinated numbers, duplicate calls, thinking-channel leak) noted as "not blockers."

### Second reading (four A/Bs later)

Expanded to N=5 and ran three successive tool-output rendering A/Bs to fix the hallucinated-values wrinkle:

- **JSON** (dense nested `{"result":{"summary":{"mean":-3.85,...},"spikes":[...]}}`): 0 faithful quotes, 6+ hallucinated.
- **Prose tables** (indented `mean = -3.85 Q` with numbered modes): 0 faithful quotes, 13+ hallucinated.
- **ASCII chart + if/then + pivot synthesis** (bar chart on Q axis, `IF partner holds 0(0-0): Q → +24 (25% of worlds, wins by 28)`, one-line pivot): 0 faithful quotes, 5 hallucinated. Zero mentions of "bimodal"/"pivot"/"spike" in the model's thoughts. Model kept writing *"I cannot see the results yet"* with the ASCII chart sitting visible above it.

At that point the pattern was unmistakable: **the three renderings were equivalently invisible to the model.** Verdict drafted: "2B Gemma 4 E2B doesn't attend to prior tool output; environment-shape has hit a ceiling." Pivot recommended.

### Third reading (the actual bug)

User asked me to read a single turn carefully. Ran `tokenizer.apply_chat_template()` on a minimal `[system, user, assistant w/ tool_call, tool_response]` fixture and inspected the output. Found that Gemma 4's chat template contains:

```jinja
{%- for message in loop_messages -%}
    {%- if message['role'] != 'tool' -%}
        ... render block ...
    {%- endif -%}
{%- endfor -%}
```

**Every `role="tool"` message in the harness was being silently dropped by the chat template, for every rollout we've ever run in `burl/harness/tool_loop_native.py`.** The model had been hallucinating tool responses because the responses literally never arrived in the prompt. Three A/B runs of "re-render the payload" all produced the same behavior because the payloads were invisible — 0 vs 6 vs 13 vs 5 hallucinated numeric quotes is **within-noise for three invisible rendering strategies.**

The correct Gemma 4 shape (from the template's `format_tool_response_block` macro + the `message.get('tool_responses')` loop) is: tool responses live on the **assistant** message as a sibling field to `tool_calls`:

```python
{
    "role": "assistant",
    "content": thought_text,
    "tool_calls": [{"type": "function", "function": {"name": "...", "arguments": {...}}}],
    "tool_responses": [{"name": "...", "response": "<prose string>"}],
}
```

Renders as `<|tool_call>call:NAME{args}<tool_call|><|tool_response>response:NAME{value:<|"|>prose<|"|>}<tool_response|>` — the native shape Gemma was post-trained on.

### After the fix

N=1 sanity check, same d0 seed (blanks trump, hand {13(4-3), 21(6-0)}, defense):

- Turn 4 thought quoted the tool response verbatim: *"**Playing 13 (4-3)**: expected outcome (**-20.9**), with p_make=**0.08**. **Playing 21**: Best case (if partner holds 0-0): **Q → +24**. Worst case (if left opp holds 0-0): **Q → -23**. **The pivot is the status of the 0-0 domino.**"*
- Every bolded value matches the actual tool response. Pivot synthesis got quoted verbatim.
- 1/1 bot-match, 13.5 s wall.

### Adapted

- Fixed `burl/wax_museum/harness.py` to build one assistant message per turn with structured `tool_calls` + `tool_responses` carrying the prose output. `commit_play`, gate rejections, and illegal-play retries all flow through the same channel.
- Fixed `burl/wax_museum/run_pilot.py::PilotLogger.live_input` to render the structured fields so `live.log` matches what the chat template actually renders.
- `burl/harness/tool_loop_native.py` still has the same bug for every other Burl experiment. Every adapter — iter-0, iter-1, iter-3-rules, every corpus the STaR loops trained on — was generated with tool responses invisible to the model. **The adapters' bot-match numbers are real, but the learned behavior was "commit a play that matches the bot based on the prompt framing alone," not "commit a play based on the distribution the tool returned."**

### Meta-lesson

This is Practicality 5 at a higher altitude: *audit before interpreting a null.* Three consecutive A/Bs (JSON → prose → ASCII) looked like a ceiling ("2B model can't reason about distributions"). The ceiling was harness plumbing silently dropping the payload. **Before concluding "the model can't do X," verify the model saw X at the expected fidelity** — not just "the bytes we sent" but "the bytes the chat template emitted after processing our messages."

Concrete rule: when a new experiment depends on a chat template we haven't personally stepped through the Jinja of, render one fixture and grep for the expected content in the serialized output. If the content isn't there, the experiment isn't measuring what we think it's measuring.

### Evidence

- `burl/wax_museum/logs/n5_live/` — JSON rendering A (0 faithful, 6+ hallucinated).
- `burl/wax_museum/logs/n5_prose/` — prose tables A (0 faithful, 13+ hallucinated).
- `burl/wax_museum/logs/n5_visual/` — ASCII + if/then + pivot A (0 faithful, 5 hallucinated).
- `burl/wax_museum/logs/n1_native2/` — after the fix; d0 turn-4 thought in `thoughts/d0_t4.md` quotes real tool values.
- Chat template diagnosis: `tokenizer.apply_chat_template` on `mlx-community/gemma-4-e2b-it-bf16`; Jinja macros `format_tool_response_block` and the `message.get('tool_responses')` loop in the template source.

---

## 10. Chat-template shapes are model-specific — Qwen wants `role="tool"`, Gemma wants `assistant.tool_responses`

**Plan assumed** (implicit in P9's fix): the native-tool-response rendering we landed for Gemma 4 would port to any other model we tried. One harness, swap the backend.

**Observed** (2026-04-20): porting `burl/wax_museum/` to Qwen3.6-35B-A3B-4bit on M5 Max via mlx-lm needed both a new parser and a different message shape. Qwen's chat template reads tool responses from a separate `role="tool"` message (OpenAI-style `<tool_response>…</tool_response>` block), while Gemma drops `role="tool"` silently and reads from `assistant.tool_responses`. An empirical rendering test (`apply_chat_template` on the minimal fixture, grep for expected content) caught this in under a minute — same diagnostic that unearthed the P9 bug.

Two entirely different tool-call output syntaxes too:

- **Gemma 4**: `<|tool_call>call:NAME{args}<tool_call|>`
- **Qwen3.6**: `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>` (nested XML), with Hermes JSON as a fallback.

**Adapted**: the wax_museum harness (`burl/wax_museum/harness.py::run_decision_waxed`) now takes two plugin kwargs:

- `parse_completion: Callable[[str], tuple[thought, tool_specs, commit]]` — defaults to the Gemma parser; pass `burl.wax_museum.qwen_parser.parse_qwen_completion` for Qwen.
- `tool_response_style: Literal["gemma_native", "role_tool"]` — selects whether responses live on the assistant turn (Gemma) or as separate messages (Qwen/OpenAI-style).

Both variants exit through the same `WaxResult` shape so analysis code is model-agnostic. `burl/wax_museum/run_pilot.py` has `--model qwen` routing.

**Meta-lesson**: backend-pluggable does NOT mean chat-template-compatible. Any new model backend requires the three-step audit: (1) render a fixture and grep for the tool response; (2) check the tool-call output syntax the base model emits zero-shot; (3) verify `role` handling. Cheap and catches invisible bugs.

### Early Qwen signal (spot-read of one turn 1, N=5 run interrupted)

Base Qwen3.6-35B-A3B-4bit on the same d0 (defense, hand {13(4-3), 21(6-0)}, blanks trump, 3/4 position) produced a 6k+ token `<think>` block that:

- derived the game state via hypothesis→test→correct ("Wait, that doesn't make sense… let me re-read…" ×~15 beats)
- correctly applied 42 rules unprompted: led suit = 6, 21(6-0) is both trump AND led suit, must follow suit
- correctly ranked trumps under blanks-trump (6-0 = highest)
- noted 27(6-6) is NOT trump (a rule Gemma never got right across 5 N=5 runs)
- did the strategic math: "Team 1 has 20 count; winning this trick → 21; Team 0 at 0; bid 30; they're set"
- arrived at the answer then emitted the tool call with: *"So the answer is to play 21(6-0). **But let me use the explore_game tool to confirm, as per the protocol.**"*

Qualitatively different from Gemma: Gemma reasons *toward* the tool; Qwen reasons *from first principles* and treats the tool as a confirmation ritual. For a "must follow suit" decision where there's only one legal play, **Qwen's approach is strictly correct** where Gemma's 5/5 bot-match was semi-lucky.

Open question deferred to the next Qwen run: does the tool surface still do work on decisions where the rulebook alone doesn't force a play (e.g. d2 0(0-0) vs 23(6-2) with a bimodal distribution)? The N=5 run was stopped after turn 1 of d0 because the information density of that single turn was already decisive — trace length is not a cost as long as the thinking is coherent, and this thinking was coherent.

**Evidence**: `burl/wax_museum/logs/n5_qwen/live.log` lines 1–261 (turn 1 only; run killed mid-d0). Rendering audit: `tokenizer.apply_chat_template` test against Qwen3.6's template (template snippet at `_im_start|>user / <tool_response> / </tool_response> / <|im_end|>`).

---

## Adding to this log

When the next practicality lands, append it as section 11 with the same four-part shape. Link the evidence (commit hashes + writeup paths). Don't edit the earlier entries — if one gets superseded, add a note at its end pointing at the newer entry. This file is the running receipts; the plan it pays back stays in [`OVERVIEW.md`](OVERVIEW.md).
