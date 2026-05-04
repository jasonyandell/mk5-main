---
title: Burl — Tool-using Texas 42 agent
kind: entity
first_seen: 8d26e0d
last_updated: 063fcac
status: active
---

## What it is

Burl is an agentic tool-using [[texas-42]] player on [[gemma-4-e2b]]. It is a sibling
project to [[lem]], not a successor. The name is a common Texas name from the 1930s; it
is not an acronym. (burl/OVERVIEW.md @ 8d26e0d)

## The premise

LEM teaches the game in model weights — flashcards, rationalizations, structured templates.
It reached 86% comprehension on [[qwen3-1.7b]] but plateaued at 55% bot-match on
open-ended play. Open-ended play selection is a different skill than fact recall.

Burl takes a different shape entirely:

- The [[engine]] is the authority on rules and visible state. Burl asks it.
- [[zeb]] is the authority on beliefs about hidden state. Burl asks it.
- Burl's job is **reasoning with what the tools return** and **committing to a play**.

> "Small models are much better at asking the right questions and synthesizing tool
> responses than at memorizing many facts." (burl/OVERVIEW.md @ 8d26e0d)

## LEM vs Burl

| | LEM | Burl |
|---|---|---|
| Base model | [[qwen3-1.7b]] | [[gemma-4-e2b]] |
| Training data | Flashcards: `(state, Q) → A` | Trajectories: tool calls + play |
| Objective | Pattern-match comprehension | Tool-mediated play |
| Eval | Comprehension accuracy, rationalization | Legal rate, retry count, bot-match |
| Product slot | "Explain this position" | "Play against AI" |
| Directory | `lem/` | `burl/` |

Shared infrastructure (owned by neither): [[forge]] (engine, E[Q] framework, solver, [[zeb]], visualizer).

## Why Gemma 4 E2B for Burl (not Qwen)

Gemma 4 E2B was retired from LEM due to comprehension ceiling and training throughput. For
Burl those reasons do not apply:
- Comprehension matters less — tools provide facts; Burl does not need to memorize them.
- Training volume is smaller (trajectories, not tens of thousands of flashcards).
- Gemma 4 was designed for agentic function-calling; native tool-use and thinking mode
  are first-class; agentic benchmarks favor Gemma 4 at the 2B scale.

(burl/OVERVIEW.md @ 8d26e0d)

## Tool surface

**Allowed** (epistemic — facts about current state):

| Tool | Returns |
|---|---|
| `is_legal(dom)` | Legality + reason |
| `is_trump(dom)` | Trump under current declaration |
| `unseen()` | Dominoes not in hand and not yet played |
| `void_audit(player, suit)` | Has this player been proven void? |
| `trump_declared()` | e.g. "blanks", "fives", "doubles" |
| `get_belief(player, dom)` | Zeb's `[P_L, P_P, P_R]` for opponent seat |
| `posterior(event)` | Zeb-or-engine marginal probability |
| `conditional_outcome(play, assumption)` | E[outcome] conditional on assumption, weighted by P(assumption) |

**Forbidden** (evaluative — distillation short-circuits): `get_eq(dom)`, `best_move()`,
`simulate_plan(actions)`. Tools answer "what IS the state?", never "what SHOULD you do?".

(burl/OVERVIEW.md @ 8d26e0d)

## First experimental moves

1. **Move 1** — Expose [[zeb]] as a callable tool (`burl/tools/zeb.py`). Pure software, prereq for everything.
2. **Move 2** — Calibration eval of Zeb's belief head. Brier score, reliability diagram.
3. **Move 3** — Prototype: Gemma 4 + tool harness, zero training. Measure what the base model does.
4. **Move 4** — Training corpus design based on Move 3 failure modes.
5. **Move 5** — First SFT run + STaR iteration.

(burl/OVERVIEW.md @ 8d26e0d)

## Design principles (carried from LEM)

- Decision-quality, not outcome-quality. Grade on E[Q], not realized Q.
- 100%-legal by construction via engine retry loop — not a trained behavior.
- Reasoning = exploring the distribution. Tool-call pattern expresses this.
- Two distillation dodges: probabilistic articulation + stylized play.

## Eval targets

Legal rate, retry count, bot-match. Different from LEM's comprehension accuracy +
rationalization quality.

## Moves 1-4 shipped (2026-04-18/19, commits d9baf3b–3781dce)

### Move 1 — Tool-call harness (d9baf3b)

Four load-bearing pieces landed:
- `tools/eq_distribution.py` — E[Q] N=10 outcome PDF as Burl's belief primitive.
  Counterfactual shift validated on seed 900013 (Δmean +15, p_make 0.6→1.0). 290ms/play.
- `tools/engine.py` — `is_legal`, `is_trump`, `unseen`, `void_audit`, `trump_declared` as
  thin wrappers over forge tables and voids. Duck-typed state.
- `tools/zeb.py` — shipped but **parked** after calibration eval (see below and [[zeb]]).
- `harness/tool_loop.py`, `retry.py`, `trace.py` — ReAct think/act/observe loop with
  XML-tag parser (`<think>`, `<tool>`, `<commit>`), illegal-retry with traces, stable JSON
  schema for STaR training corpora.

Key pivot: Zeb parked; E[Q] N=10 outcome PDF replaces it as belief primitive. See
[[experiments/zeb-calibration-eval]] and [[decisions/zeb-parked-eq-primitive]].
(commit message @ d9baf3b)

### Move 3 — Base Gemma, XML path (4b3ba3d)

10 held-out decisions, base Gemma 4 E2B, zero fine-tuning, XML harness. Results:

| Metric | Result |
|---|---|
| Legal rate | 100% |
| Retries | 0 |
| Bot-match | 60% |
| K1 (P(E[Q] ≥ bot)) | 70% |
| Cost | $0.09 |

**Premise survives.** But tool-use breadth limited: only `is_legal` called;
`eq_outcome_distribution` and `conditional_outcome` never reached. Model also hallucinates
a fake `play` tool 80% of the time (harmless — commit still lands, but flags prompt
ambiguity). Finding: `enable_thinking` is a no-op on Gemma 4's Jinja template; `max_tokens`
budget is what actually controls thinking-channel consumption. See [[experiments/burl-move3-base]].
(commit message @ 4b3ba3d)

### Move 4 R3 spike — Native tool-use format (3781dce)

Migrated from XML harness to Gemma's post-trained native tool-use format. Results vs Move 3:

| Metric | Move 3 XML | M4 R3 native |
|---|---|---|
| Completed | 10/10 | 9/10 |
| Legal rate | 100% | 100% |
| Bot-match | 60% | **88.9% (+28.9pp)** |
| K1 | 70% | **88.9% (+18.9pp)** |
| Mean E[Q] delta | −4.65 | −1.92 |
| `eq_outcome_distribution` calls | 0 | 15 |
| `trump_declared` calls | 0 | 9 |
| Hallucinated tools | 8 | 0 |

Cost: $0.16 spike. Tool-use breadth is finally real. See [[experiments/burl-move4-native-spike]]
and [[decisions/native-tool-use-format]].

Two blockers fixed:
1. XML `<commit>INT</commit>` is off Gemma's trained path. Fix: added `commit_play(domino_id)`
   as a native tool in the schema — bend the harness to the model's grain.
2. `max_retries=3` insufficient for native path (one tool-call per turn). Production config
   defaults to 7.

**Design principle added**: "Go with the model's grain; catch it doing right." STaR trains
the model's own best behavior back into itself. (commit message @ 3781dce)

## Phase 1-4 iter-0 pipeline (2026-04-19, commits b8116b5–789e14d)

### Phase 1 — Primer + 42-aware framing (b8116b5)

Added LEM's 1549-word rules primer + a "42-aware context" block (partner seat, opponent
seats, team role offense/defense, bid contract, tricks completed, score so far). Prompt grew
to ~11 KB / 2.7K tokens. New `game_summary(game_state)` tool added as structured view —
NOT yet wired into the native tool registry.

Results on 10 held-out decisions:
- Bot-match: 88.9% → **70%** (regressed — primer makes Gemma overconfident, suppresses
  distribution tool calls: 15 → 2)
- 42 vocabulary in traces: 0 → **5-11 mentions/trace** (partner, team, offense/defense,
  count, bid)

This is the intended trade: vocabulary rises for STaR substrate quality at the cost of
short-term bot-match. See [[experiments/burl-phase1-primer]] and [[decisions/primer-tradeoff]].
(commit message @ b8116b5)

### Phase 2 — STaR corpus build (fd6032b)

Rolled Layer-1 endpoint on N=50 held-out decisions (seeds 900010+, balanced 5/declaration).
K1-filtered, rationalized 23 legal losses via hinted re-prompt. Corpus: 50 entries,
27 K1 wins + 23 rationalizations, 794 KB. See [[experiments/burl-phase2-starcorpus]].

Phase 2 stats:
- K1: 54% (27/50)
- Mean E[Q] delta: −5.07
- Retries/illegal: 0
- Tool histogram: `is_legal` 72, `is_trump` 8, `eq_outcome_distribution` 8 on 50 decisions
  (vs 15 `eq_outcome_distribution` calls on just 10 decisions at spike v2 — primer suppression)
- 23/23 rationalizations converged first pass — hint-and-format pattern, not genuine reasoning
- Cost: $0.91 (primer caused 3.3× warm time, 3.7× tokens vs spike v2)

(commit message @ fd6032b)

### Phase 3 — LoRA train (0168210)

[[burl-iter0-adapter]] trained on B200. Loss 53 → 4.5, token accuracy 3.5% → 29.5%, 3 min
wall, $0.60. (commit message @ 0168210)

### Phase 4 — iter-0 eval (789e14d)

Iter-0 on 10-decision held-out: **60% bot-match — regressed 10pp from Layer 1 and 29pp
from spike v2.** 100% legal. Tool distribution unchanged from Layer 1 (adapter internalized
Layer-1 Gemma's eq-shy behavior). See [[experiments/burl-iter0-eval]].

**Key finding**: STaR iter-0 trained the adapter to reproduce Layer 1 Gemma, including its
pathologies. Root cause: the 50 K1 wins were already eq-shy (harvested from Layer-1 base).
Primer is too long (~40K chars/decision). Rationalizations teach format, not judgment. One
catastrophic tail decision accounts for most of the mean regression. See [[decisions/primer-tradeoff]].

**vLLM-LoRA blocker**: vLLM 0.19 rejects `Gemma4ForConditionalGeneration` for LoRA. Fix:
`hf_overrides` forcing architecture to `Gemma4ForCausalLM` at load. (Different blocker from
LEM's 2c2b851 attempt, which hit multimodal weight layout issues.) (commit message @ 789e14d)

**Next**: iter-1 — drop the primer, keep the 42-aware framing block (vocabulary at low
cost), re-harvest corpus on lighter prompt.

### iter-1 — Trimmed primer + SFT, mixed result (09b841e)

Dropping the primer entirely broke commit discipline immediately (0% wins, 50% retry-exhausted
on 6 decisions — killed). Trimmed primer (~500 words vs 1549) recovered commit discipline:
N=30 rollout at 43% K1, 0 retries, healthier tool diversity (`is_legal` 44, `trump_declared` 4).

[[burl-iter1-adapter]] trained on the 30-trace corpus. Eval on 10 held-out: **5/10
retry-exhausted, 80% bot-match + mean E[Q] delta -0.76 on the 5 completed** (within 1pp of
spike v2 on the completed subset). Cost: $1.37, session total $3.22.

**Finding**: trimmed primer amplified "reason deeply" at cost of "emit commit_play." The
trimmed rules text removed a load-bearing commit-discipline scaffold. See
[[experiments/burl-iter1-mixed]] and [[decisions/primer-tradeoff]].

**iter-2 options**: (a) harvest from spike v2 shape (no primer, no framing), trim max_turns;
(b) keep trimmed primer + add explicit "MUST emit commit_play" line; (c) N=50 corpus,
max_retries=7 at eval. (commit message @ 09b841e)

### iter-2 prep — four parallel infrastructure workstreams (2026-04-19, commits f164796–b5d05de)

No new adapter trained in this cluster. Four infra workstreams landed in parallel:

**(a) EQ-gate STaR rejection sampling** (f164796, 761587c): replaces the Phase-B
"reveal answer, rationalize" pattern with a non-leaking gate. On legal-but-suboptimal
commits, the gate feeds back a nudge and only keeps traces where Gemma self-corrects to
the bot play. Gate outcomes: `converged_first_try`, `self_corrected`, `forced_flip`,
`stubborn`, `exhausted`. Only `self_corrected` traces go into the SFT corpus (tagged
`source=eq_gate_self_correct`). CLI flags: `--gate-variant {minimal,tool_nudge,social}`,
`--max-gate-retries N`, `--eq-epsilon F`. (commit messages @ f164796, 761587c)

**(b) LS-Mixture corpus verbosity blender** (3414507): shortens `<|channel>thought` blocks
from long traces while preserving tool-call/tool-response envelopes and terminal
`commit_play`. Blends at `target_short_ratio=0.33`. Preview corpus: 118 rows (79 long + 39
short); short mean 767 chars vs long mean 5078 (−85%). **Key discovery from training
launcher** (eebcae5): Gemma 4 E2B's `chat_template.jinja` strips `<|channel>thought` blocks
before tokenization — SFTTrainer never sees thought prose. The verbosity blend's benefit is
therefore (1) coverage (118 vs 79 rows for commit discipline) and (2) data-augmentation
regularization, not thought-content training. Inference-time rambly thoughts are base-model
reflex, not trained behavior. (commit messages @ 3414507, eebcae5)

**(c) Rules-as-tools scaffold** (b3a27e2, 80704f0): four tools (`count_dominoes_remaining`,
`trick_winner_if`, `what_beats_what`, `contract_progress`) that replace the 1549-word primer
with on-demand engine-authoritative answers. Wired behind `enable_rules_tools` flag (default
False). When enabled, a ~645-byte preamble renders in place of the trimmed primer; framing
block is unchanged. iter-2's verbosity-blend experiment keeps the flag False (single
variable); iter-3 flips it. (commit messages @ b3a27e2, 80704f0)

**(d) [[haiku-4-5]] reference-trace spike** (1f13f92, b5d05de): Anthropic's Haiku 4.5 via
the Agent SDK runs against the same tool surface. N=30 run: 29/30 complete, 72.4% bot-match,
$0.78. Key findings: `conditional_outcome` never used zero-shot (must be synthesized by
STaR); 4 big-gap misses all skipped `eq_outcome_distribution`; Haiku uses 7 tools/decision
vs Gemma iter-1's 3. These are reference-ceiling traces for distillation, not training data.
(commit messages @ 1f13f92, b5d05de)

### iter-3 prep — prompt-shape matrix + async speed (2026-04-19/20, commits c698091–c2aa3a7)

No new adapter trained. Three infra additions:

**Three-mode `enable_primer` flag** (c698091, 65c749c, faefca7): codifies the prompt-shape
space as a controlled matrix for iter-3 A/B experiments. See [[decisions/primer-tradeoff]].

| Mode | Flags | Prompt shape |
|---|---|---|
| Default | `enable_primer=True` | Trimmed primer + 42-framing (iter-1 shape) |
| Rules-as-tools | `enable_rules_tools=True` | Rules-as-tools preamble + 42-framing |
| No primer | `enable_primer=False` | 42-framing only, no primer (spike v2 shape, where `eq_outcome_distribution` was called 15× on base Gemma) |

`enable_primer=False ∧ enable_rules_tools=True` raises `ValueError` — rules-as-tools IS a
primer; the combination is incoherent. The flag is threaded through rollout, EQ-gate, and
`compose_sft_record` so training and eval prompts cannot silently diverge.
See [[topics/rules-as-tools]].

Two new launchers added: `star_iter3_rules.py` (adapter `burl-iter3-rules`, `enable_rules_tools=True`)
and `star_iter3_v2.py` (adapter `burl-iter3-v2`, `enable_primer=False`, no blender).
(commit messages @ c698091, abb1b3d, 65c749c, faefca7)

**Async concurrency in STaR rollout** (c2aa3a7): `asyncio.to_thread` + semaphore-gated
batching for both Phase A (initial rollouts) and Phase B (EQ-gate chains). New
`--concurrency N` flag, default=1 (backwards-compat: output bit-identical at 1). Local
benchmark: N=8 @ concurrency=4 → 3.92× speedup. Expected real-world 2.5-3.5× on L4.
(commit message @ c2aa3a7)

### iter-3 winner + iter-4 null + arena (2026-04-19, commits 35c75ff–dbadb5f)

**[[iter3-rules-adapter]] is the winner — 90% bot-match, 0 retry-exhausted, 100%
first-legal.** Trained with `enable_rules_tools=True`, `enable_primer=off`. `trick_winner_if`
usage INCREASED after SFT, validating [[topics/rules-as-tools]]: the adapter learns to call
rules-tools more aggressively, not less. Surpasses spike v2 (88.9%) by 1.1pp with 0
exhaustions vs 1. See [[experiments/iter3-comparison]]. (commit message @ dbadb5f)

**iter-4 preserve_thoughts — null result** (20f4fa2): bypassing `strip_thinking()` at SFT
adds +532 thought tokens/row (~200K tokens over 130 rows × 3 epochs) to the gradient.
Byte-identical A/B result vs baseline. The winning reasoning shape was already present in
the tool-call/tool-response envelopes the template preserves; thought prose adds no signal
at this LoRA rank. See [[experiments/iter4-null-preserve-thoughts]]. (commit message @ 20f4fa2)

**[[selfplay-arena]] launched** (35c75ff, 39aafaf): 4-model full-game orchestrator in
`burl/arena/`. First run Haiku seed 900010: bidder team 0-42 shutout, $0.84, 7m22s.
Opus vs Haiku same seed: Opus salvages 7-35 vs Haiku's 0-42, $5.58. Tool-use efficiency:
Opus trusts the system prompt (1× `trump_declared` vs Haiku's 24×) and leads with
`eq_outcome_distribution` probes. See [[experiments/opus-vs-haiku-arena]].

**Opus parallel-tool-use lock** (2830be0): `asyncio.Lock` around
`_handle_sdk_mcp_request` in `haiku_spike/agent.py` — Opus 4.7 emits multiple
`tool_use` blocks per message, racing the shared MCP tool cache and wedging the channel.
Naive monkey-patch fix; HTTP transport is an alternative.

**`conditional_outcome` structural finding** (4th observation): across 145+ decisions —
single-decision Haiku, single-decision Opus, full-game Haiku, full-game Opus — no model
has ever called `conditional_outcome` zero-shot. See
[[topics/conditional-outcome-structural-nonuse]]. If Burl is to use this tool, STaR must
synthesize explicit demos. (commit messages @ 35c75ff, 39aafaf, dbadb5f)

**Session spend**: ~$20.50 of $40 authorized. (commit message @ dbadb5f)

### MLX-LM local pipeline + SFT truncation finding (2026-04-19, commits 6fea6ab–edf86e9)

**[[mlx-lm]] local path** (6fea6ab): full corpus-generation + training + eval loop now runs
on M-series Mac without Modal. Two new files:
- `burl/modal/gemma_local.py` — in-process [[gemma-4-e2b]] via MLX-LM with `NativeModelCallable`
  interface; adapter swap via constructor arg. Model: `mlx-community/gemma-4-e2b-it-bf16`.
- `burl/train/star_mlx.py` — port of `star.py` to `mlx_lm.tuner`; `preserve_thoughts`
  bypass intact. Peak memory: ~10 GB at rank 4 with `grad_checkpoint`, ~26 GB at rank 16
  with `max_seq_length=4096`. At least 1.86× wall speedup vs Modal.

`--local` / `--model-source local` flags added to eval runners. Weight mmap shares across
up to 5 workers on 48 GB unified memory. (commit message @ 6fea6ab)

**SFT truncation finding** (edf86e9): TRL's `SFTConfig` defaults `max_seq_length=1024`.
Burl's `preserve_thoughts` corpus has median 2054 tokens and max 4210 tokens/row — both
modal `star.py` and local `star_mlx.py` were silently truncating thought regions. Fix:
`max_seq_length=4096` applied to both paths.

**This reframes ingest B7's iter-4 null result**: the byte-identical A/B was almost
certainly truncation, not LoRA capacity saturation. No prior Burl adapter was trained on
complete thought-to-tool-call traces. Parallel to LEM's [[decisions/sft-completion-only-loss]]
discovery — TRL defaults are traps. See [[topics/preserve-thoughts]] and
[[experiments/iter4-null-preserve-thoughts]]. (commit message @ edf86e9)

### iter-5 + candlewax + MLX batch + multimodal spike (2026-04-19/20, commits 1efb9c5–0545342)

**iter-5 E1 — rank sweep with truncation fixed** (ceca203): first real `preserve_thoughts`
adapters trained on complete thought-to-tool-call traces.

| Adapter | Bot-match | E[Q] delta | Note |
|---|---|---|---|
| Base Gemma | 66.7% | −3.15 | — |
| rank-16 | **70.0%** | **−2.83** | First real preserve_thoughts adapter |
| rank-64 | 55.6% | −5.59 | 50% empty-tool-rollout — degraded |
| rank-128 | 0% | — | 10/10 retry-exhausted; 16K chars/decision |

Dose-response is catastrophic collapse above rank-16 on a 26-row corpus. MLX-LM lacks
gradient clipping; LR × rank × small-corpus interaction drives instability. Verified on 147
rows — rank-64 still diverges at LR peak. Next lever: N=100+ rollouts at rank-16, not more
rank. See [[experiments/iter5-e1-rank-sweep]]. (commit message @ ceca203)

**iter-5 E2 — candlewax-aware tool surface** (1efb9c5, 7321952, b0952a2): three new fields
added to `eq_outcome_distribution` to make bimodality legible at the tool surface:
`distribution_shape`, `modes`, `gap_between_modes`, `suggested_counterfactuals`,
`spike_drivers` (empirical mode-catalyst dominoes per spike), and `enumerate="auto"` (exact
at pool ≤ 12, ~7ms on M5 Max). New `what_would_change_my_mind` meta-tool surfaces top-K
world-assumptions that most shift E[Q] — called once zero-shot in a live smoke.

Candlewax E2 result: **null in terms of changing model behavior.** E3 rollout (N=500, 98
`eq_outcome_distribution` calls, 74 non-unimodal, 53 mixed-mode): still 0
`conditional_outcome` calls. Three traces show Gemma considering it in thought prose then
declining in favor of breadth-first alternative-play evaluation. Environment-shape lever is
validated at its firing site; the blocker is Gemma's policy preference. See
[[experiments/iter5-e2-candlewax-null]]. (commit messages @ 1efb9c5, ceca203)

**MLX batch_generate ceiling** (ed3cfc3, 6a97d55): 43 → 1334 tok/s at batch=128 (16×
aggregate). 90% of peak at batch=64. Memory plateau 15 GB on 48 GB host. N=500 rollouts
in ~3.5 min. Operationalized in `GemmaLocalNativeBatched` / `run_move4_star_rollout_batched.py`:
2.3× wall on N=16. See [[experiments/batch-throughput-bench]]. (commit messages @ ed3cfc3, 6a97d55)

**[[candlewax-spike]] (0545342)**: multimodal PDF rendering + engine fact-checker + MLX
LoRA STaR end-to-end on Qwen 3.6-35B-A3B via mlx-vlm. Image-as-alignment: Haiku d000
13→21 flipped by the candlewax image. v7 adapter +15% bot-match on 33 held-out examples.
**Pivots away from LLM-as-reasoner**: reasoning-coherence verification is the bottleneck
and needs a multi-week verifier subproject. See [[topics/reasoning-coherence-verification]].
(commit message @ 0545342)

**PRACTICALITIES.md split** (aeafe22): 8 entries logged (native tool-call format, dual-use
primer, model-invented idioms, `conditional_outcome` zero-shot invisibility, `max_seq_length`
truncation, rank-16 LoRA sweet spot, 43→1334 tok/s batch ceiling, M5-Max-as-multiplier).
OVERVIEW Pareto frontier table: iter-3-rules 90% robustness vs iter-1 −0.16 eq-delta.
(commit message @ aeafe22)

**Incidental finding** (7321952): `WorldSamplerMRV` marginal distribution is biased vs
uniform enumeration by ~6.8 Q points at trick 6. Enumeration is ground truth; sampling is
a biased estimator. Affects all historical Burl eval numbers and forge/eq training data
quality. Not fixed; filed for follow-up.

### wax_museum + chat-template bug + belief_trajectory — Burl end-of-replay (2026-04-20/23, commits 54f7776–1bf1885)

**THE CONFOUND: Gemma 4 chat-template silently drops `role="tool"` messages** (54f7776).

`tool_loop_native.py` had been packing tool responses into `role="tool"` messages. Gemma 4's
Jinja template wraps the entire render loop in `{%- if message['role'] != 'tool' -%}` —
every response silently discarded. **Every Burl rollout from B2 through B9 had tool outputs
invisible to the model.**

Findings that must be re-interpreted:
- `conditional_outcome = 0/145` ([[topics/conditional-outcome-structural-nonuse]]): trivially
  explained — model never saw tool responses, not a structural policy preference.
- "Environment-shape ceiling" from three A/B runs (JSON → prose → ASCII): confounded.
- Every adapter's measured bot-match (iter-0 60%, iter-1 80%\*, iter-3-rules 90%): confounded.
  Note: [[iter3-rules-adapter]] achieved 90% **without any visible tool responses** — remarkable,
  but we don't know what a correct-tool-responses run would have yielded.

**Fix**: tool responses on `assistant.tool_responses=[{name, response}]` for Gemma (native);
`role="tool"` for Qwen (OpenAI-style). Harness takes `parse_completion` + `tool_response_style`
plugins. **Meta-lesson**: audit the *rendered prompt*, not the messages dict, before
concluding "the model can't do X."

**Post-fix N=5 held-out**: base Gemma 4 E2B **5/5 bot-match**, faithful numeric quoting,
pivot quoted verbatim. See [[experiments/chat-template-fix-validation]].
(commit message @ 54f7776)

**[[wax-museum]] subproject** (54f7776, 1bf1885): hard-gated HATEOAS harness
(`explore_game → probe_* → commit_play`). Forces model to reason about outcome distributions
before committing. Three extension hooks: `system_prompt_transform`, `preload_tool_calls`
(model wakes with [[belief-trajectory]] already in context), `menu_override`. Used for
Phase 1 six-variant belief-trajectory sweep. (commit messages @ 54f7776, 1bf1885)

**[[belief-trajectory]] tool** (d858781): exposes [[gus]]'s `v3_consistency_10000g` belief
adapter as per-domino posterior + shift-since-last + V + CLS attention. Dual output (dict +
LLM-legible STRONG/MEDIUM/WEAK prose). Bridged via `_gus_adapter.py`. Wired into
[[wax-museum]] as a free side-call in every gate state. Replaces [[zeb]] as Burl's production
belief source. (commit message @ d858781)

**[[gus]] first appearance in Burl**: Gus is the third sibling project (neural policy +
belief + value). Its belief head now powers [[belief-trajectory]]. Gus's own replay trail
begins in a later session. (commit message @ d858781)

### Phase A guards + 2000-decision harvest (2026-04-24/25, commit 063fcac)

The next-step that 1bf1885 set up: harvest a STaR-quality corpus at 2000 decisions on the winning `D_required_first` variant. Two prerequisites landed:

**Phase A guards (063fcac)**: turn-budget extension on commit reject + forced-commit fallback on turn-cap, both wired into [[wax-museum]]'s `run_decision_waxed`. Pre-Phase-A blunder rerun had 3/29 decisions committing illegally; post-Phase-A and on the full 2000-decision harvest: zero illegal commits. See [[wax-museum]] "Phase A guards" section.

**Batched 2000-decision harvest (v2, finished 2026-04-25 13:15)**: 5h 46m wall, batch=6, `max_tokens=2048` (see [[max-tokens-2048-floor]]), zero quarantine fires across 333 batches. Output at `scratch/belief_trajectory_rollout/harvest_batched_20260425_072910/` (gitignored). See [[burl-2000-harvest]].

The harvest replaced an earlier v1 run that SIGKILLed at 1398/2000 and was contaminated by a turn-1 truncation bug at `max_tokens=1024`: 11.4% of v1 decisions had `belief_called_turns` not starting at turn 1 because the model had exhausted its budget mid-thinking-block on turn 1 and never reached [[belief-trajectory]]. v2 is at 0.0%. Per-decision audit on the overlap between v1 and the sequential 560 found 43% of decisions had moved between buckets — bucket-parity at the distribution level had been a false pass. **Methodological lesson: distribution-level parity gates can hide systematic per-decision regressions when the regression has multiple compensating directions.**

Bucket distribution on v2 (n=2000):

| Class | n | % |
|---|---:|---:|
| Strict pool (Burl matches oracle) | 1062 | 53.1 |
| → non-trivial gold (excl. ALL_AGREE_CORRECT) | 202 | 10.1 |
| BURL_BREAKS_CONSENSUS (sharpest loss / [[r1-rationalization]] target) | 299 | 14.9 |
| All other loss buckets | 420 | 21.0 |
| FORCED_COMMIT (guard fired) | 219 | 10.9 |
| ILLEGAL | 0 | 0.0 |

Strict pool is 3.6× the sequential pilot's 294 rows; non-trivial gold is 3.9× the 52 rows that the prior 71-row STaR run had collapsed on. [[star]] run-3 is the planned next step on this corpus, with rank=8 + lr=3e-5 + val-loss + early-stopping per the conservative-hyperparam lesson from the prior collapse.

A side artifact: `scratch/belief_trajectory_rollout/harvest_batched.py` now hosts the per-wave OOM-resilience layer (quarantine ledger + SIGKILL sentinel) — see [[batched-harvest-resilience]]. Dead code on the success path; existence is the entire point.

### burl-chat workbench + post-commit Q&A research direction (2026-04-30)

A standalone interactive workbench under `burl/chat/` lets the user load any wax_museum decision as a conversation prefix and chat with Burl about it. See [[burl-chat]] for the architecture (FastAPI + in-process [[mlx-lm]] + Svelte 5 + typed-segment rendering) and [[burl-chat-spike]] for the first sessions.

First-session findings opened a new research direction, [[post-commit-q-and-a]]: talking with Burl after a hand the way a teammate would. Closest published precedent is chapters 2-8 of Roberson's *Winning 42*, which structurally is a worked-example dialogue corpus and which the project owns as canonical text via the user's family heritage.

Two findings from this spike worth carrying forward:

- **[[chat-mode-primer]]** is load-bearing. A synthetic "Yeah, I committed N. The decision is done — ask me anything" assistant turn injected after `commit_play` flips base Gemma from play-decision mode into chat mode. Without it, even the base model produces "I am Burl, my next action is to call commit_play" responses.
- **[[play-adapter-lock-in]]** is real and complete. e1-rank16 ([[experiments/iter5-e1-rank-sweep]]) cannot be talked out of `commit_play` even with primer + explicit "do not output a tool call" + a non-tool question. Base Gemma + same prefix engages cleanly. Implication: any post-commit Q&A adapter must be co-trained or trained from base, not stacked on a play adapter.

A side product of the first session: base Gemma critiqued the existing tool surface unprompted, suggesting structured summaries before raw histograms, explicit strategic labels, and a "why" framing that reads as the [[topics/at-risk-points]] frame Roberson uses. First product feedback from Burl on its own tools — usable backlog item.

## Candidate role for [[book-strategy-player]] (2026-05-03)

The book-strategy-player architecture (designed 2026-05-03 in W42 book validation) gives
Burl a use case it was unusually well-suited for but had not yet found:
**strategy-selector model (Model A) over a structured action space.**

The setup: encoded book strategies have human-readable names ("singleton_lead_to_void",
"throwaway_ladder", "trump_pulling"). The action space at decision time is "which named
strategy should fire?" rather than "which of 7 dominoes should I play?" — small (~15-30
strategies), structured, and naturally narrative. An LLM can reason about strategy
*rationales* in chain-of-thought, justify the pick, and commit. STaR-style training has
clean ground truth: we record per-decision (game_state, applicable_strategies, chosen,
hand_outcome) and Burl can learn from labeled traces or from self-rationalization on the
ones where it disagrees with the hand-crafted priority arbitration.

Why this is better than raw-action policy for Burl:
- 15-30 strategy choices vs 7-domino × 28-trick raw action space → much smaller decision
  surface
- Each strategy has a one-paragraph rationale → natural fit for LLM narrative reasoning
- Failure modes are interpretable ("Burl picked throwaway_ladder when trump_pulling would
  have been right") rather than opaque ("Burl picked the 5-2 instead of the 4-1")
- Recorded counterfactual (Lens(ev) action) gives free training signal for "when does
  Burl's strategy pick beat the EV-greedy fallback?"

Speculative until the book-strategy framework lands. See [[book-strategy-player]] for
architecture; this is referenced as Model A in that page's recording → training pipeline
section.

## Open questions at this frontier

- Can Gemma 4 E2B tool-use reliably at 2B scale? (Move 3 answers cheaply.)
- Is Zeb's 72% belief accuracy useful to Burl's reasoning? (Move 2 + 3.)
- Does tool-mediated reasoning transfer to tool-less reasoning (ablation)?
- Does `conditional_outcome` leak evaluative signal through the back door?
- Is 3M-parameter Zeb strong enough, or does Burl need more capacity?
- Does v2's residual −2.4pp gap on `BURL_BREAKS_CONSENSUS` (vs sequential 560) reflect real batched-mode policy drift or sampling noise? A paired McNemar test on the 560 overlap would settle it.
