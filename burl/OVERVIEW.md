# Burl — Tool-using Texas 42 agent

Can a small model play 42 through **tool orchestration** — asking the engine for facts, asking the E[Q] framework for outcome distributions, reasoning between calls — instead of memorizing the game in its weights?

The name is a common Texas name from the 1930s. It's not an acronym. Sibling to LEM, not successor.

## The premise

LEM teaches a model the game in its *weights* — flashcards, rationalizations, structured templates. It gets to 86% comprehension accuracy on Qwen 3 1.7B and produces lucid "explain this position" traces. It's a good teacher/companion artifact.

But LEM also has a ceiling problem: 55% bot-match on open-ended play, and the v10-maskfix experiment showed capacity and gradient allocation don't fix it. Open-ended play selection is a different skill than fact recall, and it's not clear more SFT signal of the same shape would move the number.

Burl takes a different shape entirely:

- The engine is the authority on rules and visible state. Burl asks it.
- The E[Q] framework is the authority on outcome-under-uncertainty. Burl asks it for **distributions**, not point estimates.
- Burl's job is **reasoning with what the tools return** and **committing to a play**. The engine validates; if illegal, Burl gets another turn.

The bet is that small models are much better at *asking the right questions* and *synthesizing tool responses* than at memorizing many facts. If true, the tool-call harness lets a 2B-class model play competently without the comprehension curriculum LEM spent months on — because tools replace memorization.

## Not a successor — a sibling

LEM and Burl solve different problems and ship as separate artifacts:

|                | LEM                                    | Burl                                   |
|----------------|----------------------------------------|----------------------------------------|
| Base model     | Qwen 3 1.7B                            | Gemma 4 E2B (agentic-strong)           |
| Training data  | Flashcards: `(state, Q) → A`           | Trajectories: tool calls + play        |
| Objective      | Pattern-match comprehension            | Tool-mediated play                     |
| Eval           | Comprehension accuracy, rationalization| Legal rate, retry count, bot-match     |
| Product slot   | "Explain this position" mode           | "Play against AI" mode                 |
| Lineage        | Fresh from Qwen 3                      | Fresh from Gemma 4 (no LEM lineage)    |
| Directory      | `lem/`                                 | `burl/` (this project)                 |

Shared infrastructure (owned by neither, used by both): `forge/` (engine, E[Q] framework, solver, visualizer).

## Why Gemma 4 E2B for the base

We previously pivoted Stage 0 off Gemma 4 to Qwen 3 1.7B because Gemma hit 60% on comprehension while Qwen hit 100%, and Gemma's per-layer embeddings + KV-sharing layers tank training throughput on B200.

For Burl those reasons don't apply in the same way:

- Comprehension matters less — tools provide facts, Burl doesn't need to memorize them.
- Training volume is smaller — tool-using trajectories are thousands, not tens of thousands of flashcards.
- Gemma 4 was *designed* for agentic function-calling; native tool-use training is baked in; thinking mode is actively used during tool loops.
- Agentic benchmarks favor Gemma 4 at the 2B scale.

The architecture issues Gemma had for LEM's training recipe become lower priority when we're training on smaller trajectory corpora and relying on inference-time behavior.

Warmup confirmed (2026-04-18): base Gemma 4 E2B emits our `<tool>{"name":...}</tool>` XML zero-shot from a simple system prompt. The harness parser accepts it without fine-tuning. Training budget goes to judgment, not syntax.

## Vocabulary (matters — be precise)

| Term | What it is | Where it lives |
|---|---|---|
| **Perfect-information solver** | Takes a fully-specified deal + declaration, enumerates reachable states, solves by backward induction / minimax. Deterministic. Produces the exact game value under full information. | `forge/oracle/` |
| **E[Q] framework** | Uses the solver (or its learned Q-value distillate) as a subroutine. For an imperfect-info position (your hand + visible plays), samples plausible hidden-hand realizations, evaluates each, averages. Produces the distribution of outcomes given what you actually see. | `forge/eq/` |
| **E[Q] distribution tool** | The per-play outcome distribution exposed to Burl — samples `N` consistent worlds (respecting inferred voids), evaluates via the Stage-1 Q-value oracle, returns the 85-bin PDF over Q ∈ [-42, +42]. Default `N=10` (Zeb evals: N=10 head-to-head ≈ N=100 ≈ N=1000). | `burl/tools/eq_distribution.py` |
| **E[Q] bot** | The `argmax(E[Q])` player. Picks the legal move with highest expected value. Comparison target in K1 grading. | Throughout game + narration code |
| **Zeb** | Learned belief model (3.3M-param transformer, trained via self-play + oracle distillation). Outputs `P(opponent ∈ {L, partner, R})` per domino. **Parked for Burl** after calibration eval found hidden-only top-1 ≈ 39% (advertised 72% was inflated by averaging over already-played dominoes — trivial play-history lookup). `burl/tools/zeb.py` kept as history; not wired into the tool loop. | `forge/zeb/` |
| **Engine** | The TypeScript game engine — rules, state transitions, legality, visible information. Burl talks to it via thin Python wrappers over `forge/oracle/tables` + `forge/eq/voids`. | `src/core/`, `burl/tools/engine.py` |

Do **not** conflate the solver with the E[Q] framework. The solver sees everything and is deterministic. The E[Q] framework runs the solver over a distribution of hidden states and averages. Burl never calls the solver directly; Burl's decisions are graded against the E[Q] framework.

## Architecture

```
               ┌──────────────────────────────────────────┐
               │             BURL (the agent)             │
               │      Base: Gemma 4 E2B                   │
               │      Role: reasoning + orchestration     │
               └───┬────────┬────────┬────────┬───────────┘
                   │        │        │        │
                   │  tool calls (all channels; native <|tool_call|>)
                   ▼        ▼        ▼        ▼
   ┌──────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
   │  ENGINE  │ │ E[Q] DIST.   │ │    RULES     │ │ COMMIT CHAN. │
   │ (facts)  │ │  (outcomes)  │ │  (lookups)   │ │  (terminate) │
   │          │ │              │ │              │ │              │
   │ is_legal │ │ eq_outcome_  │ │ trick_       │ │ commit_play( │
   │ is_trump │ │  distribution│ │  winner_if   │ │  domino_id)  │
   │ unseen   │ │   + candle   │ │ what_        │ │  → engine    │
   │ void_    │ │     wax      │ │  beats_what  │ │    validates │
   │  audit   │ │   + spike    │ │ contract_    │ │    + retries │
   │ trump_   │ │     drivers  │ │  progress    │ │    on illegal│
   │  declared│ │ conditional_ │ │              │ │              │
   │ game_    │ │  outcome     │ │  (see Prac.2)│ │  (see Prac.1)│
   │  summary │ │ what_would_  │ │              │ │              │
   │          │ │  change_my_  │ │              │ │              │
   │          │ │   mind       │ │              │ │              │
   └──────────┘ └──────────────┘ └──────────────┘ └──────────────┘
        │              │                │                │
        └──────────────┴──────┬─────────┴────────────────┘
                              │
                              ▼
                ┌────────────────────────────────────────┐
                │  Burl reasons with:                    │
                │   • engine facts (hard truths)         │
                │   • E[Q] PDFs + shape hints + drivers  │
                │   • rules lookups (trick outcomes,     │
                │     beats-what, contract progress)     │
                │   • counterfactual PDFs (what-ifs)     │
                │  → emits commit_play(...) to terminate │
                │    (engine validates inside the tool;  │
                │     illegal → retry with rejection in  │
                │     context)                           │
                └────────────────────────────────────────┘

   ──────────────────────  inference-time boundary  ──────────────────────

                ┌────────────────────────────────────┐
                │      E[Q] FRAMEWORK (grading)      │
                │  - Ground truth for K1 grading     │
                │  - Same machinery the tool wraps,  │
                │    plus the bot baseline play for  │
                │    comparison                      │
                └────────────────────────────────────┘

                ┌────────────────────────────────────┐
                │   PERFECT-INFO SOLVER (dev-time)   │
                │  - Seeds the E[Q] framework        │
                │  - Neither Burl nor LEM touch it   │
                └────────────────────────────────────┘
```

## Tool surface: allow and forbid

**Allow** (epistemic — facts about current state, distributions over outcomes, rule lookups, and the commit channel):

*Engine facts* (`burl/tools/engine.py`):

| Tool | Signature | Returns |
|---|---|---|
| `is_legal` | `(dom) → (bool, str)` | Legality + reason if not |
| `is_trump` | `(dom) → bool` | Trump under current declaration |
| `unseen` | `() → set[dom]` | Dominoes not in your hand and not yet played |
| `void_audit` | `(player, suit) → bool` | Has this player been proven void in this suit? |
| `trump_declared` | `() → str` | e.g. "blanks", "fives", "doubles" |
| `game_summary` | `() → dict` | Pre-parsed hand + trick + contract + score (written, not yet wired into the registry) |

*Outcome distributions* (`burl/tools/eq_distribution.py`, `burl/tools/meta_tools.py`):

| Tool | Signature | Returns |
|---|---|---|
| `eq_outcome_distribution` | `(play, n_samples=10, enumerate="auto")` | 85-bin PDF + mean/stdev/p_make/percentiles + **candlewax hints** (`distribution_shape`, `modes`, `gap_between_modes`, `suggested_counterfactuals`) + **spike_drivers** (catalyst dominoes per mode, in bid-satisfaction vocabulary). `enumerate="auto"` switches to exact enumeration when the unseen-world pool is small (≤12). See Practicality 4. |
| `conditional_outcome` | `(play, assumption, n_samples=10) → OutcomeDistribution` | Same PDF but restricted to sampled worlds satisfying `assumption` (`{"player": seat, "holds": dom}`, `{"player": seat, "void_in_suit": suit}`, or callable predicate). **Currently zero-shot invisible** (0/145+ decisions across every model). Practicality 4 tracks the three environment-shape fixes in flight. |
| `what_would_change_my_mind` | `(play, n_samples=10) → list[assumption+rationale]` | Ranks unseen-world assumptions by how much each shifts the play's E[Q]. Appears in the tool menu *before* a PDF is ever requested — catches models that would otherwise never ask for the distribution. |

*Rules lookups* (`burl/tools/rules.py` — the compact substitute for the LEM primer; see Practicality 2):

| Tool | Signature | Returns |
|---|---|---|
| `trick_winner_if` | `(dom) → seat` | Who wins the current trick if I play `dom`? (winner — 1.56→1.70 per decision after iter-3-rules SFT) |
| `what_beats_what` | `(dom_a, dom_b) → dom` | Which of two dominoes beats the other under current trump |
| `contract_progress` | `() → dict` | Bid vs points taken so far; remaining-to-make |
| `count_dominoes_remaining` | `() → dict` | Per-suit dominoes left in play (dead — 0 calls across 145+ decisions; deletion queued) |

*Commit channel* (see Practicality 1):

| Tool | Signature | Behavior |
|---|---|---|
| `commit_play` | `(domino_id: int) → None` | Terminates the turn. Engine validates inside the tool; illegal → `RetryExhausted` exception with rejection prompt fed back. Rides the native `<\|tool_call>` channel because Gemma wants to tool-call its way out of turns anyway. |

**Forbid** (evaluative — these are distillation short-circuits):

| Tool | Why forbidden |
|---|---|
| `get_eq(dom) → float` | Just emits the oracle's scalar mean; model learns `argmax`, not distribution-shape reasoning |
| `best_move()` | Absolute distillation |
| `simulate_plan(actions)` | Brute-force search wrapper; short-circuits the reasoning |

The rule: tools answer "**what IS the state?**" or "**what does the outcome distribution look like?**" or "**what are the rules here?**", never "**what SHOULD you do?**". Rules lookups are epistemic facts about 42, not evaluations of a position. Under that discipline, Burl has to actually reason with the facts and distribution shapes to pick a move — the whole point.

## How it composes at runtime (one play)

1. Burl receives a game state + "your move, explain yourself."
2. Burl's base weights (Gemma 4) know how to ask structured questions but nothing about 42. So it asks.
3. Tool calls: `trump_declared()` → "blanks". `unseen()` → `{4-4, 5-5, 3-0, ...}`. `void_audit(partner, sixes)` → False.
4. Burl queries the E[Q] distribution tool across legal plays: `eq_outcome_distribution(6-2)` → `distribution_shape: "bimodal"`, μ=+3.4, p_make=0.70, `suggested_counterfactuals: [{"player": "right_opp", "holds": 5-5, "rationale": "collapses the left tail — rules out the disaster swing"}]`, `spike_drivers: [{mode: -12, "right_opp holds 5-5, freq=0.80, lift=1.8"}]`. `eq_outcome_distribution(5-2)` → `distribution_shape: "unimodal"`, μ=+2.0, p_make=1.00 but narrower. Burl takes the suggested counterfactual: `conditional_outcome(5-2, {"player": right_opp, "holds": 5-5})` → μ shifts from +2.0 to -8.0, p_make collapses to 0.35.
5. Burl reasons with the shapes: *"5-2 is safer unconditionally but the 5-5 branch torches it. 6-2 gives up some mean for partial immunity to that branch."*
6. Burl commits via tool call: `commit_play(6-2)`. Engine validates inside the tool → legal → trace complete.

If the committed play had been illegal, the `commit_play` tool raises, the harness feeds the rejection back as context, and Burl gets another turn. Legal-move compliance is a **software invariant**, not a trained behavior. (See Practicality 1 for why commit rides the tool channel instead of a free-form `<commit>` tag.)

## How it composes at training time (STaR iteration)

1. Burl rolls out plays against the E[Q] bot on held-out decisions (gap ≥ 1.0 filter, same as LEM's decision set).
2. The E[Q] framework (grading-side only) computes the E[Q] of Burl's chosen play and the bot's.
3. K1 filter: keep traces where Burl's play has `E[Q] ≥ E[Q]_bot` (or regret < ε).
4. Kept traces — with their tool-call histories intact — become the next SFT corpus.
5. Repeat.

The tool-call histories are part of the training signal. Burl doesn't just learn "which play wins," it learns "which sequence of tool-asks, reasoning steps, and commitment looked like this."

## Infrastructure: Modal hosting + LoRA serving

Move 3 stands up `burl/modal/gemma_serve.py` — a vLLM endpoint on Modal L4 with the class shape chosen so later moves extend rather than refactor:

- `@app.cls` + `@modal.enter` loads vLLM once per container, not per call.
- `@modal.method generate(prompt, max_tokens, stop)` is the remote entry point.
- `@modal.concurrent(max_inputs=4)` lets vLLM's continuous batching serve parallel Burl rollouts on a single L4.
- PLE-safe vLLM config: `bfloat16`, `limit_mm_per_prompt={"image":0,"video":0,"audio":0}`, `max_model_len=8192`.

For Move 4+ (STaR iterations producing LoRA adapters), the extension is near-zero surgery:

- Base model loads with `enable_lora=True, max_loras=4, max_lora_rank=64` at `@modal.enter`.
- `generate()` gains `adapter_name: str | None = None`. First call for a new name: `snapshot_download` the adapter from HuggingFace (matches LEM's `jasonyandell/gemma-4-e2b-texas42-*` pattern — zero new infra), wrap as a `vllm.lora.request.LoRARequest`, cache in `self._adapters`. Subsequent calls reuse.
- A/B between STaR iterations is two `.remote()` calls with different `adapter_name`s — vLLM routes each to the right adapter and serves up to 4 concurrently on the same GPU.
- Adapter cache lives on a Modal Volume; persistent across container restarts, no repeated downloads.

No custom adapter registry, no homemade hot-swap. All leverage of vLLM's built-in `LoRARequest` API. Training remains LEM's Unsloth-on-Modal recipe — Burl just consumes the adapters it produces.

## Design principles

- **Decision-quality, not outcome-quality.** Grade Burl on E[Q], not realized Q. We're training a better bettor, not a luckier player.
- **100%-legal by construction.** Engine retry loop, not a trained behavior. Rule-knowledge becomes irrelevant to Burl's legality guarantees.
- **Reasoning = exploring the distribution.** Burl's tool-call pattern is the expression of this — the E[Q] distribution tool gives distribution shape; engine gives facts; Burl weighs scenarios via counterfactual distributions.
- **Two distillation dodges.** Probabilistic articulation (tool-mediated via the E[Q] distribution tool) and stylized play (future work, t42-8aep) are the non-distillation axes. Composable.
- **Visualizer is the canonical teacher artifact.** `forge/analysis/results/web/eq_pdf_discs.html`, `eq_surface_3d.html`, `eq_game_journey.html`. Shared with LEM. Burl's rationalizations should, when visualized, trace regions of the discs/surface a strong human would.
- **Go with the model's grain; catch it doing right.** Small models have their own instincts — Gemma reaches for a `play` verb even when our menu doesn't define one, emits thoughts in markdown when asked to reason, and keeps calling `is_legal` before every commit. Rather than fight those tendencies, bend the harness around them. If `play` is what Gemma wants, `play` is what we give it. STaR trains the *model's own best behavior* back into itself — their words, their corrections, their self-checks. We're not imposing a shape; we're amplifying one we noticed.

## Current state — 2026-04-19

Pareto frontier of two shipped adapters (both on HuggingFace, both at `--max-retries 7`, both N=10 held-out):

| adapter | bot-match | first-legal | retry-exhausted | mean E[Q] Δ | idiom |
|---|---:|---:|---:|---:|---|
| `jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules` | **90%** | 100% | 0 | −1.78 | rules-as-tools + `trick_winner_if` heavy |
| `jasonyandell/gemma-4-e2b-texas42-burl-iter1` | 88.9% | 90% | 0 | **−0.16** | trimmed-primer + trump-structural |

iter-3-rules is the robustness winner; iter-1 is the per-commit-quality winner. They're genuinely different idioms, not two versions of the same thing. Whether a single adapter can land both is the iter-5 question.

Two structural open items carry forward into iter-5:

- **`conditional_outcome` has been called zero times across 145+ decisions** covering every model tested (Haiku, Opus 4.7, every Burl adapter iter-0 through iter-4). Candlewax + spike_drivers + `what_would_change_my_mind` are the three environment-shape levers queued; mockup spike confirmed the downstream path works. See Practicality 4.
- **LoRA-capacity sweet spot at rank 16** on MLX-LM. Scaling rank further monotonically regresses (rank-64 55.6%, rank-128 collapses at 0%). Next lever is corpus size, not rank. See Practicality 6.

**What we've learned executing the plan — in depth**: [`PRACTICALITIES.md`](PRACTICALITIES.md) is the running receipts log. Read it alongside the "Current state" table above for the full picture of which assumptions survived contact with Gemma 4 and which got replaced.

**Full scientific log**: [`SPIKE_REPORT.md`](../SPIKE_REPORT.md). **Forward experiment plan**: [`ITER4_PLAN.md`](ITER4_PLAN.md).

## Experimental moves

Ordered by most-learning-per-dollar. Moves 1-3 are done; Moves 4-5 are where iter-1 through iter-5 live. Detailed findings live in [`PRACTICALITIES.md`](PRACTICALITIES.md) and the per-iteration writeups under `burl/experiments/`.

### Move 1 — [parked] Zeb as a callable belief tool

Built, calibration-evaluated, parked. Hidden-only top-1 ≈ 39% (the advertised 72% was inflated by averaging over already-played dominoes — trivial play-history lookup). `burl/tools/zeb.py` kept as history; `burl/eval/belief_calibration.py` kept for the methodology.

### Move 1′ — [done] E[Q] distribution as Burl's belief primitive

`burl/tools/eq_distribution.py` wraps `forge/eq/` sampling + Stage-1 oracle + outcome bucketing. N=10 default. Returns the 85-bin PDF plus summary stats — preserves the multimodal "melted candlewax" shape that makes counterfactual reasoning load-bearing. The return shape has since been extended with candlewax hints (shape/modes/gap/suggested counterfactuals) and spike_drivers (catalyst dominoes per mode), because of Practicality 4. This replaces Move 1 as Burl's outcome/belief primitive.

### Move 2 — [done] Engine facts + harness scaffold

- `burl/tools/engine.py` — engine tools, thin wrappers over `forge/oracle/tables` + `forge/eq/voids`.
- `burl/harness/{tool_loop,retry,trace}.py` — XML-tag parser + illegal-retry loop + stable JSON trace schema.
- `burl/harness/tool_loop_native.py` — native `<|tool_call|>` path, added when Practicality 1 landed.

### Move 3 — [done] Base Gemma 4 E2B end-to-end on held-out decisions

- `burl/modal/gemma_serve.py` + `gemma_serve_native.py` — Modal L4 endpoints (XML + native paths).
- `burl/modal/gemma_local.py` + `gemma_local_batched.py` — local MLX-LM paths (Practicality 7-8).
- `burl/eval/decision_dataset.py` — 50 held-out decisions (seeds ≥ 900000, trick 6, |legal|≥2, E[Q] gap ≥ 1.0), balanced across declarations.
- `burl/eval/run_move4_spike.py` + `run_move4_star_rollout.py` — base-adapter eval + STaR rollout with EQ-gate + async concurrency.

**What Move 3 measured** (and still does): legal rate (100% by construction), first-legal rate (raw-competence signal), bot-match, mean E[Q] delta vs bot, P(E[Q] ≥ bot) (K1 rate), retry distribution, tool-use histogram, empty-tool rollouts.

### Move 4 — [ongoing] Training corpus design via STaR rollouts

Five corpora iterated through so far (iter-0 through iter-4-thoughts, plus iter-5 E1 rank sweep). The dominant corpus shape that won iter-3-rules is:

- **Rollouts against the E[Q] bot** on held-out decisions (gap ≥ 1.0 filter).
- **K1 filter**: keep traces where Burl's committed play has E[Q] ≥ bot (or regret < ε). Tool-call histories kept intact — they're part of the training signal.
- **Rules-as-tools preamble** (Practicality 2) provides commit-discipline scaffolding without flashcard-ing rules into the weights.
- **Preserve-thoughts recipe** reopened by the `max_seq_length=4096` fix (Practicality 5); iter-5 E1 rank-16 is the first adapter that actually earned a thought-gradient signal.

iter-5+ focus: corpus *size*, not rank (Practicality 6). Batched rollout harness (Practicality 7) makes N=500 cheap enough to try.

### Move 5 — [ongoing] STaR iteration loop

SFT via `burl/train/star.py` (Modal/TRL/Unsloth) or `burl/train/star_mlx.py` (local MLX-LM). LoRA adapter served from either `gemma_serve_native.py` (Modal vLLM) or `gemma_local.py` (local MLX). Eval: legal rate, first-legal rate, bot-match, retry count, mean E[Q] delta.

Five iterations shipped so far (iter-0 through iter-5 E1 rank sweep, plus iter-4-thoughts). `jasonyandell/gemma-4-e2b-texas42-burl-iter3-rules` and `-iter1` are the current Pareto frontier. Next iteration (iter-5 E3) targets closing the gap between the two idioms via a corpus-size bump at rank 16 with gradient clipping in place — details in [`ITER4_PLAN.md`](ITER4_PLAN.md).

## What we've learned executing the plan

Eight practicalities (and counting) have been absorbed without changing the vision. The full log with evidence pointers is in [`PRACTICALITIES.md`](PRACTICALITIES.md); quick-reference:

1. Gemma 4 emits its native tool-call format zero-shot — `commit_play` is a tool now.
2. The rules primer is dual-use — rules-as-tools is the compact substitute.
3. The model invents its own reasoning idiom — grade by outcome, not tool-mix conformity.
4. `conditional_outcome` is still zero-shot invisible — candlewax + spike_drivers + `what_would_change_my_mind` are the environment-shape fixes in flight.
5. Truncation silently steals training signal — always pin `max_seq_length`.
6. LoRA-capacity sweet spot at rank 16 on MLX-LM — next lever is corpus size.
7. Batch generation lifted the corpus-scale ceiling — 43 → 1334 tok/s on M5 Max.
8. M5 Max is an iteration multiplier, not a Modal replacement.

## Open questions

Questions that are *still* open (answered ones have migrated into [`PRACTICALITIES.md`](PRACTICALITIES.md)):

- **Can a single adapter land both iter-1's eq-delta and iter-3-rules's robustness?** Currently two points on a Pareto frontier. iter-5 E3 (rank-16 × corpus-size bump × gradient clipping) is the natural next try. See [`ITER4_PLAN.md`](ITER4_PLAN.md).
- **Does the candlewax + spike_drivers + `what_would_change_my_mind` stack clear the `conditional_outcome = 0` gap end-to-end in a live rollout?** Mockup spike + T11 smoke say yes for isolated exposure. Trimmed-primer re-run of T12 is the pending disambiguator.
- **Does tool-mediated reasoning transfer to tool-less reasoning?** If we ablate tools at inference, does Burl reason from what it previously retrieved, or collapse? Matters for product scenarios where tool latency is prohibitive. Untested.
- **Does `conditional_outcome` leak into E[Q] argmax through the back door?** A Burl that always asks `conditional_outcome(play, {})` (trivially satisfied assumption) is essentially calling `get_eq`. Watch for this pattern post-candlewax-ship; if it appears, require non-trivial `assumption` structure at the tool layer.
- **Is N=10 E[Q] samples enough for Burl's decisions?** Zeb evals showed N=10 ≈ N=100 for picking a play. Shape-classification stability (bimodal-vs-unimodal) sits at ~83% at N=10, ~92% at N=100 — see `iter5_e2_candlewax_eval_writeup.md`. Whether the variance matters for Burl's counterfactual reasoning is still untested on a live adapter.
- **STaR on Burl traces vs STaR on LEM traces.** Does the tool-call history make the training signal cleaner or just longer? Three iterations in, the answer looks like "cleaner" (iter-3-rules reinforces specific tools; iter-1 discovers its own structural reasoning). But we haven't run the matched comparison.

## Directory plan (current)

```
burl/
├── OVERVIEW.md                    ← this file (vision + architecture)
├── PRACTICALITIES.md              ← ✓ running log of lessons learned executing the plan
├── ITER4_PLAN.md                  ← ✓ forward experiment plan (iter-5+)
├── GEMMA_4_ERGONOMICS.md          ← ✓ Gemma 4 native-format probes (Practicality 1 origin)
├── tools/                         ← tool harness implementations
│   ├── engine.py                  ← ✓ is_legal, is_trump, unseen, void_audit, trump_declared, game_summary
│   ├── eq_distribution.py         ← ✓ eq_outcome_distribution (+ candlewax hints + spike_drivers)
│   │                                  + conditional_outcome, enumerate="auto"
│   ├── meta_tools.py              ← ✓ what_would_change_my_mind — assumptions ranked by E[Q] swing
│   ├── rules.py                   ← ✓ trick_winner_if, what_beats_what, contract_progress,
│   │                                  count_dominoes_remaining (Practicality 2)
│   └── zeb.py                     ← ✓ parked; kept as history (calibration eval was decisive)
├── harness/                       ← agent runtime
│   ├── tool_loop.py               ← ✓ ReAct-style think/act/observe loop (XML path, legacy)
│   ├── tool_loop_native.py        ← ✓ Gemma native <|tool_call|> loop (Practicality 1)
│   ├── retry.py                   ← ✓ illegal-play retry logic
│   ├── trace.py                   ← ✓ trace structure for training + debugging
│   ├── agent_runner.py            ← ✓ ties remote/local model + tools into run_decision()
│   └── eq_gate.py                 ← ✓ rejection-sampling gate for STaR Phase B
├── modal/                         ← Modal + local hosting
│   ├── gemma_serve.py             ← ✓ Gemma 4 E2B vLLM endpoint on L4 (XML path)
│   ├── gemma_serve_native.py      ← ✓ vLLM endpoint with native tool-call format
│   ├── gemma_local.py             ← ✓ MLX-LM single-stream on M5 Max (43 tok/s baseline)
│   └── gemma_local_batched.py     ← ⚙ batched MLX-LM path (per Practicality 7, in progress)
├── train/                         ← training recipes
│   ├── star.py                    ← ✓ Modal/TRL/Unsloth SFT — canonical iter-N runs
│   ├── star_mlx.py                ← ✓ local MLX-LM SFT — disambiguation experiments
│   ├── star_iter{2,3_rules,3_v2,4_thoughts}.py  ← ✓ per-iteration launchers
│   └── test_{formatting_func,iter2_loader,star_mlx}.py ← ✓ training-invariant pytests
├── eval/                          ← evaluation + scoring
│   ├── belief_calibration.py     ← ✓ parked; produced the decisive Zeb finding
│   ├── decision_dataset.py        ← ✓ held-out decisions + bot baselines
│   ├── run_move3.py               ← ✓ Move 3 orchestrator + grader (XML-era)
│   ├── run_move4_spike.py         ← ✓ native-format adapter eval
│   ├── run_move4_star_rollout.py  ← ✓ STaR rollout with EQ-gate + async concurrency
│   ├── bench_batch_throughput.py  ← ✓ MLX-LM batch_generate sweep (Practicality 7)
│   └── results/                   ← adapter evals + traces (per-experiment subdirs)
├── selfplay/                      ← 4-model full-game arena
│   └── arena.py                   ← ✓ model-agnostic seat runner; --tag for artifacts
├── haiku_spike/                   ← reference-model spikes (Haiku, Opus via Agent SDK)
│   └── agent.py                   ← ✓ MCP lock fix for parallel tool use
├── experiments/                   ← durable writeups (one markdown per experiment)
│   ├── README.md                  ← ✓ session index
│   ├── iter{2,3_rules,3_v2,4_thoughts,5_e1_capacity,5_e2_candlewax}_eval_writeup.md
│   ├── {corpus_blend,eq_gate,iter4_thoughts,rules_as_tools}_design.md
│   ├── {batch_throughput,concurrency}_bench.md
│   ├── {haiku_spike,haiku_full,opus_spike,arena}_notes.md
│   └── launch_iter2.md
├── corpus/                        ← training corpora (committed — small)
├── adapters/                      ← local LoRA checkpoints (gitignored — regeneratable)
└── data/                          ← evaluation artifacts (gitignored)
```

## Related beads

- `t42-14h4` (P1) — Burl's founding umbrella. Remains open; Moves 1-3 slot under it.
- `t42-8aep` (P3) — Stylized teachers. Long-term research direction for characterful Burl variants.
- `t42-3oj7` (P4) — Tool-augmented LLM judge. General tool-use capability, may connect to Burl's trace-grading.
- `t42-0dg6` (P2) — Probabilistic reasoning in LEM. Separate track from Burl; result will inform whether in-weights calibration complements Burl's tool-mediated approach.

Closed-as-landed (see `git log` for detail): `t42-56gu` (Zeb wrapper), `t42-oyq9` (engine + harness), `t42-jljb` (belief calibration), `t42-2ap0` (eq_distribution), `t42-ozju` (Gemma warmup).

## Not goals

- Bidding. Burl plays from a dealt state forward. Bidder is out of scope and lives elsewhere.
- General-purpose tool use. Burl's tool menu is 42-specific.
- Self-play against Burl. First iterations go against the E[Q] bot. Self-play comes later, if at all.
- Human-interface UX. Burl ships its reasoning traces as structured JSON; any UI (game integration, teaching display) is downstream work.

## Relationship to LEM's OVERVIEW

This is the forward-looking plan. `lem/OVERVIEW.md` is the history of the project that produced LEM, the comprehension adapter, and — relevant here — the vocabulary, curriculum principles, and E[Q] framework we're building on. Read both. LEM's OVERVIEW explains *why* we know what we know; this one explains *what we're doing next*.
