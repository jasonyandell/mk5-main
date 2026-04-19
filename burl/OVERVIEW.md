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
               └────────┬──────────────┬──────────────────┘
                        │              │
                        │   tool       │   tool
                        │   calls      │   calls
                        ▼              ▼
        ┌──────────────────────┐   ┌───────────────────────────────┐
        │      ENGINE          │   │   E[Q] DISTRIBUTION TOOL      │
        │  (rules + visible)   │   │   Wraps Stage-1 oracle        │
        │                      │   │                               │
        │   Epistemic facts:   │   │   Outcome distributions:      │
        │   • is_legal         │   │   • eq_outcome_distribution   │
        │   • is_trump         │   │       (play, n_samples=10)    │
        │   • unseen           │   │     → 85-bin PDF + mean/stdev │
        │   • void_audit       │   │                               │
        │   • trump_declared   │   │   • conditional_outcome       │
        │                      │   │       (play, assumption, n=10)│
        │                      │   │     → PDF sliced by assume    │
        └──────────────────────┘   └───────────────────────────────┘
                        │                       │
                        └───────────┬───────────┘
                                    │
                                    ▼
                ┌────────────────────────────────────────┐
                │  Burl reasons with:                    │
                │   • engine facts (hard truths)         │
                │   • E[Q] PDFs (shape of outcomes)      │
                │   • counterfactual PDFs (what-ifs)     │
                │  → commits to a play (engine validates)│
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

**Allow** (epistemic — facts about current state, and distributions over outcomes):

| Tool | Signature | Returns |
|---|---|---|
| `is_legal` | `(dom) → (bool, str)` | Legality + reason if not |
| `is_trump` | `(dom) → bool` | Trump under current declaration |
| `unseen` | `() → set[dom]` | Dominoes not in your hand and not yet played |
| `void_audit` | `(player, suit) → bool` | Has this player been proven void in this suit? |
| `trump_declared` | `() → str` | e.g. "blanks", "fives", "doubles" |
| `eq_outcome_distribution` | `(play, n_samples=10) → OutcomeDistribution` | 85-bin PDF over Q plus mean, stdev, p_make |
| `conditional_outcome` | `(play, assumption, n_samples=10) → OutcomeDistribution` | Same PDF but restricted to sampled worlds satisfying `assumption` (structured: `{"player": seat, "holds": dom}` or `{"player": seat, "void_in_suit": suit}`; or a callable predicate) |

**Forbid** (evaluative — these are distillation short-circuits):

| Tool | Why forbidden |
|---|---|
| `get_eq(dom) → float` | Just emits the oracle's scalar mean; model learns `argmax`, not distribution-shape reasoning |
| `best_move()` | Absolute distillation |
| `simulate_plan(actions)` | Brute-force search wrapper; short-circuits the reasoning |

The rule: tools answer "**what IS the state?**" or "**what does the outcome distribution look like?**", never "**what SHOULD you do?**". Under that discipline, Burl has to actually reason with the facts and distribution shapes to pick a move — the whole point.

## How it composes at runtime (one play)

1. Burl receives a game state + "your move, explain yourself."
2. Burl's base weights (Gemma 4) know how to ask structured questions but nothing about 42. So it asks.
3. Tool calls: `trump_declared()` → "blanks". `unseen()` → `{4-4, 5-5, 3-0, ...}`. `void_audit(partner, sixes)` → False.
4. Burl queries the E[Q] distribution tool across legal plays: `eq_outcome_distribution(6-2)` → bimodal PDF, μ=+3.4, p_make=0.70. `eq_outcome_distribution(5-2)` → μ=+2.0, p_make=1.00 but narrower. Then tests a counterfactual: `conditional_outcome(5-2, {"player": right_opp, "holds": 5-5})` → μ shifts from +2.0 to -8.0, p_make collapses to 0.35.
5. Burl reasons with the shapes: *"5-2 is safer unconditionally but the 5-5 branch torches it. 6-2 gives up some mean for partial immunity to that branch."*
6. Burl commits: `play(6-2)`. Engine validates → legal → trace complete.

If the committed play had been illegal, the engine rejects and Burl gets another turn with the rejection in context. Legal-move compliance is a **software invariant**, not a trained behavior.

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

## Experimental moves

Ordered by most-learning-per-dollar. Status current as of 2026-04-18.

### Move 1 — [parked] Zeb as a callable belief tool

Built, calibration-evaluated, then parked. Found hidden-only top-1 ≈ 39% (vs advertised 72%, inflated by averaging over already-played dominoes — trivial play-history lookup). Brier 0.224, ECE 0.067 — moderately calibrated but too weak to anchor Burl's reasoning. `burl/tools/zeb.py` exists; not wired into the tool loop. `burl/eval/belief_calibration.py` kept for the methodology.

### Move 1′ — [done] E[Q] distribution as Burl's belief primitive

`burl/tools/eq_distribution.py` wraps `forge/eq/` sampling + Stage-1 oracle + outcome bucketing. N=10 default. Returns the 85-bin PDF plus summary stats — preserves the multimodal "melted candlewax" shape that makes counterfactual reasoning load-bearing. Counterfactual validated on seed 900013: `conditional_outcome(play=3, {"player": right_opp, "holds": 7})` shifted mean by +15 points and p_make 0.60 → 1.00. 290 ms/play at N=10 on a 3050 Ti. This replaces Move 1 as Burl's outcome/belief primitive.

### Move 2 — [done] Engine facts + harness scaffold

- `burl/tools/engine.py` — 5 engine tools, thin wrappers over `forge/oracle/tables` + `forge/eq/voids`. Duck-typed game state.
- `burl/harness/{tool_loop,retry,trace}.py` — XML-tag parser (`<think>`, `<tool>`, `<commit>`), illegal-retry loop with `RetryExhausted` on cap, stable JSON trace schema for STaR corpora.

### Move 3 — [in progress] Base Gemma 4 E2B end-to-end on held-out decisions

- `burl/modal/gemma_serve.py` — vLLM Gemma 4 E2B on Modal L4 (LoRA-ready class shape).
- `burl/harness/agent_runner.py` — ties remote Gemma + local tools into `run_decision(state) → BurlTrace`.
- `burl/eval/decision_dataset.py` — 50 held-out decisions (seeds ≥ 900000, trick 6, |legal|≥2, E[Q] gap ≥ 1.0), balanced across declarations.
- `burl/eval/run_move3.py` — orchestrates rollouts, produces the grading table.

**What Move 3 measures**: legal rate (should be 100% by construction), first-legal rate (honest raw-competence signal), bot-match, mean E[Q] delta vs bot, P(E[Q] ≥ bot) (K1 rate), retry distribution, tool-use histogram, empty-tool rollouts (a failure mode).

### Move 4 — Training corpus design

Based on Move 3's failure modes. Likely a mix of:

- Positive exemplars (human- or LLM-authored traces that correctly use each tool class)
- Negative exemplars (traces that skip tool calls and hallucinate, so Burl learns to prefer tool-grounded reasoning)
- Retry-recovery traces (illegal first attempt → engine rejection → successful second attempt)

Size: unknown until Move 3 tells us what the gap looks like.

### Move 5 — First training run + iteration

SFT on the corpus from Move 4 via LEM's Unsloth-on-Modal recipe, producing a LoRA adapter served from the same endpoint (see Infrastructure above). Eval: legal rate, bot-match, retry count, reasoning structure.

Once bot-match stabilizes, start STaR iterations: roll out on fresh decisions, filter with K1, retrain, repeat.

## What Moves 1-2 taught us

- **Lossy proxies for rich primitives are worse than the rich primitive.** A 3-way categorical belief per domino cannot substitute for the E[Q] distribution over outcomes — the distributions are irreducibly multimodal and `conditional_outcome`'s counterfactual power comes from the full PDF, not from marginal seat-ownership probabilities.
- **Published accuracy numbers can be inflated by the aggregation mask.** Zeb's 72% top-1 was real but averaged over dominoes where the answer is trivial (already played, readable from history). The operationally-relevant subset for Burl (hidden) is ~39%. Always check the subset that matches the consumer's actual query.
- **Base Gemma 4 E2B emits our XML tool-call format zero-shot.** No syntax-correction data needed in training. Budget goes to judgment (which tool when, synthesis of results), not format.
- **3050 Ti cannot host Gemma 4 E2B at useful speed.** PLE architecture + ~5B real params means bf16 OOMs on 4 GB. Only viable local path is Q4 GGUF via llama.cpp CPU (~7 tok/s) — dev smoke-testing only. Modal L4 is the cheapest practical inference target.

## Open questions

- **Can Gemma 4 E2B actually tool-use *judiciously* at 2B scale?** Zero-shot syntax is free; the real question is whether it picks the right tools, avoids redundant calls, and synthesizes results into reasoning that's load-bearing for the final play. Move 3 answers this directly.
- **Is N=10 E[Q] samples enough for Burl's decisions?** Zeb evals showed N=10 ≈ N=100 *for picking a play*. Whether it's enough for Burl to reason with counterfactual shifts is a separate question. Measure variance of `conditional_outcome` results across re-draws during Move 3.
- **Does tool-mediated reasoning transfer to tool-less reasoning?** If we ablate tools at inference, does Burl reason from what it previously retrieved, or collapse? Matters for product scenarios where tool latency is prohibitive. Measured via ablation eval in Move 3+.
- **Does `conditional_outcome` leak into E[Q] argmax through the back door?** A Burl that always asks `conditional_outcome(play, {})` (trivially satisfied assumption) is essentially calling `get_eq`. Watch for this pattern in Move 3 traces; if it appears, require non-trivial `assumption` structure at the tool layer.
- **STaR on Burl traces vs STaR on LEM traces.** Does the tool-call history make the training signal cleaner or just longer? Unknown until we run iterations.

## Directory plan (current + planned)

```
burl/
├── OVERVIEW.md                    ← this file
├── tools/                         ← tool harness implementations
│   ├── engine.py                  ← ✓ is_legal, is_trump, unseen, void_audit, trump_declared
│   ├── eq_distribution.py         ← ✓ eq_outcome_distribution + conditional_outcome
│   └── zeb.py                     ← ✓ parked; kept as history (calibration eval was decisive)
├── harness/                       ← agent runtime
│   ├── tool_loop.py               ← ✓ ReAct-style think/act/observe loop
│   ├── retry.py                   ← ✓ illegal-play retry logic
│   ├── trace.py                   ← ✓ trace structure for training + debugging
│   └── agent_runner.py            ← ✓ ties remote model + tools into run_decision()
├── modal/                         ← Modal hosting
│   └── gemma_serve.py             ← ✓ Gemma 4 E2B vLLM endpoint (LoRA-ready class shape)
├── train/                         ← training recipes [Move 5]
│   ├── sft.py
│   └── star.py
├── eval/                          ← evaluation + scoring
│   ├── belief_calibration.py     ← ✓ parked; produced the decisive Zeb finding
│   ├── decision_dataset.py        ← ✓ held-out decisions + bot baselines
│   └── run_move3.py               ← ✓ Move 3 orchestrator + grader
└── data/                          ← training/eval corpora (gitignored)
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
