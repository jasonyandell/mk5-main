# Burl — Tool-using Texas 42 agent

Can a small model play 42 through **tool orchestration** — asking the engine for facts, asking Zeb for beliefs, reasoning between calls — instead of memorizing the game in its weights?

The name is a common Texas name from the 1930s. It's not an acronym. Sibling to LEM, not successor.

## The premise

LEM teaches a model the game in its *weights* — flashcards, rationalizations, structured templates. It gets to 86% comprehension accuracy on Qwen 3 1.7B and produces lucid "explain this position" traces. It's a good teacher/companion artifact.

But LEM also has a ceiling problem: 55% bot-match on open-ended play, and the v10-maskfix experiment showed capacity and gradient allocation don't fix it. Open-ended play selection is a different skill than fact recall, and it's not clear more SFT signal of the same shape would move the number.

Burl takes a different shape entirely:

- The engine is the authority on rules and visible state. Burl asks it.
- Zeb is the authority on beliefs about hidden state. Burl asks it.
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

Shared infrastructure (owned by neither, used by both): `forge/` (engine, E[Q] framework, solver, Zeb, visualizer).

## Why Gemma 4 E2B for the base

We previously pivoted Stage 0 off Gemma 4 to Qwen 3 1.7B because Gemma hit 60% on comprehension while Qwen hit 100%, and Gemma's per-layer embeddings + KV-sharing layers tank training throughput on B200.

For Burl those reasons don't apply in the same way:

- Comprehension matters less — tools provide facts, Burl doesn't need to memorize them.
- Training volume is smaller — tool-using trajectories are thousands, not tens of thousands of flashcards.
- Gemma 4 was *designed* for agentic function-calling; native tool-use training is baked in; thinking mode is actively used during tool loops.
- Agentic benchmarks favor Gemma 4 at the 2B scale.

The architecture issues Gemma had for LEM's training recipe become lower priority when we're training on smaller trajectory corpora and relying on inference-time behavior.

Provisional; revisit if prototype says otherwise.

## Vocabulary (matters — be precise)

| Term | What it is | Where it lives |
|---|---|---|
| **Perfect-information solver** | Takes a fully-specified deal + declaration, enumerates reachable states, solves by backward induction / minimax. Deterministic. Produces the exact game value under full information. | `forge/oracle/` |
| **E[Q] framework** | Uses the solver as a subroutine. For an imperfect-info position (your hand + visible plays), enumerates plausible hidden-hand realizations, evaluates each via the solver, averages weighted by prior. Produces expected value given what you actually see. | `forge/eq/` |
| **E[Q] bot** | The `argmax(E[Q])` player. Picks the legal move with highest expected value. This is the comparison target in K1 grading. | Throughout game + narration code |
| **Zeb** | Learned belief model (3.3M-param transformer, trained via self-play + oracle distillation). Predicts `P(opponent ∈ {L, partner, R} | visible state)` for each of 28 dominoes. 72% top-1 accuracy. | `forge/zeb/` |
| **Engine** | The TypeScript game engine — rules, state transitions, legality, visible information. | `src/core/` |

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
        ┌──────────────────────┐   ┌───────────────────────┐
        │      ENGINE          │   │         ZEB           │
        │  (rules + visible)   │   │   Belief model        │
        │                      │   │                       │
        │   Epistemic facts:   │   │   Learned beliefs:    │
        │   • is_legal         │   │   • get_belief(       │
        │   • is_trump         │   │       player, dom)    │
        │   • unseen           │   │     → [P_L, P_P, P_R] │
        │   • void_audit       │   │                       │
        │   • trump_declared   │   │   (tool-wrapped from  │
        │                      │   │    forge/zeb, ~500    │
        │                      │   │    lines of wrapper)  │
        └──────────────────────┘   └───────────────────────┘
                        │              │
                        └──────┬───────┘
                               │
                               │ Composed via higher-order tool:
                               ▼
                ┌────────────────────────────────────┐
                │  conditional_outcome(play, assume) │
                │  1. Zeb → P(assume | visible)      │
                │  2. E[Q] framework → outcome       │
                │     conditional on `assume`        │
                │  3. Return the slice               │
                └────────────────────────────────────┘

   ──────────────────────  inference-time boundary  ──────────────────────

                ┌────────────────────────────────────┐
                │      E[Q] FRAMEWORK (training)     │
                │  - Ground truth for K1 grading     │
                │  - Not a Burl-callable tool        │
                └────────────────────────────────────┘

                ┌────────────────────────────────────┐
                │   PERFECT-INFO SOLVER (dev-time)   │
                │  - Seeds the E[Q] framework        │
                │  - Neither Burl nor LEM touch it   │
                └────────────────────────────────────┘
```

## Tool surface: allow and forbid

**Allow** (epistemic — facts about the current state):

| Tool | Signature | Returns |
|---|---|---|
| `is_legal` | `(dom) → (bool, str)` | Legality + reason if not |
| `is_trump` | `(dom) → bool` | Trump under current declaration |
| `unseen` | `() → set[dom]` | Dominoes not in your hand and not yet played |
| `void_audit` | `(player, suit) → bool` | Has this player been proven void in this suit? |
| `trump_declared` | `() → str` | e.g. "blanks", "fives", "doubles" |
| `get_belief` | `(player, dom) → [P_L, P_P, P_R]` | Zeb's belief over opponent seat |
| `posterior` | `(event) → float` | Zeb-or-engine marginal probability |
| `conditional_outcome` | `(play, assumption) → float` | E[outcome] conditional on `assumption`, weighted by P(assumption) |

**Forbid** (evaluative — these are distillation short-circuits):

| Tool | Why forbidden |
|---|---|
| `get_eq(dom)` | Just emits the oracle; model learns `argmax`, not reasoning |
| `best_move()` | Absolute distillation |
| `simulate_plan(actions)` | Brute-force search wrapper; short-circuits the reasoning |

The rule: tools answer "**what IS the state?**", never "**what SHOULD you do?**". Under that discipline, Burl has to actually reason with the facts to pick a move — the whole point.

## How it composes at runtime (one play)

1. Burl receives a game state + "your move, explain yourself."
2. Burl's base weights (Gemma 4) know how to ask structured questions but nothing about 42. So it asks.
3. Tool calls: `trump_declared()` → "blanks". `unseen()` → `{4-4, 5-5, 3-0, ...}`. `void_audit(partner, sixes)` → False.
4. Burl reasons about what matters. It asks Zeb: `get_belief(right_opponent, 5-5)` → `[0.15, 0.20, 0.65]` — right opponent probably holds the 5-5.
5. Burl incorporates: *"If right opponent has 5-5 (65%), leading fives is bad because they trump it."*
6. Burl commits: `play(6-2)`. Engine validates → legal → trace complete.

If the committed play had been illegal, the engine rejects and Burl gets another turn with the rejection in context. Legal-move compliance is a **software invariant**, not a trained behavior.

## How it composes at training time (STaR iteration)

1. Burl rolls out plays against the E[Q] bot on held-out decisions (gap ≥ 1.0 filter, same as LEM's decision set).
2. The E[Q] framework (training-side only) computes the E[Q] of Burl's chosen play and the bot's.
3. K1 filter: keep traces where Burl's play has `E[Q] ≥ E[Q]_bot` (or regret < ε).
4. Kept traces — with their tool-call histories intact — become the next SFT corpus.
5. Repeat.

The tool-call histories are part of the training signal. Burl doesn't just learn "which play wins," it learns "which sequence of tool-asks, reasoning steps, and commitment looked like this."

## Design principles (carried from LEM's vision)

- **Decision-quality, not outcome-quality.** Grade Burl on E[Q], not realized Q. We're training a better bettor, not a luckier player.
- **100%-legal by construction.** Engine retry loop, not a trained behavior. Stage-0 rule-knowledge becomes irrelevant to Burl's legality guarantees.
- **Reasoning = exploring the distribution.** Burl's tool-call pattern is the expression of this — Zeb gives distribution shape; engine gives facts; Burl weighs scenarios.
- **Two distillation dodges.** Probabilistic articulation (now tool-mediated via Zeb) and stylized play (future work, t42-8aep) are the non-distillation axes. Composable.
- **Visualizer is the canonical teacher artifact.** `forge/analysis/results/web/eq_pdf_discs.html`, `eq_surface_3d.html`, `eq_game_journey.html`. Shared with LEM. Burl's rationalizations should, when visualized, trace regions of the discs/surface a strong human would.

## First experimental moves

Ordered by most-learning-per-dollar. Assumes LEM's probability-category and path-A work proceed in parallel on the LEM side.

### Move 1 — Expose Zeb as a callable tool (~500 lines, 2-4 hours)

Wrap `forge/zeb/` as `burl/tools/zeb.py` with a stateless `get_belief(game_state, player, domino) → [P_L, P_P, P_R]` function. Handles checkpoint loading, batching, error cases, caching.

Prereq for everything downstream. Pure software task, no ML.

### Move 2 — Calibration eval of Zeb's belief head

Before Burl composes Zeb into reasoning, we need to know: when Zeb says "P=0.72," is it actually right 72% of the time? Brier score, reliability diagram, per-domino accuracy breakdown. If calibration is poor, we either retrain Zeb with a bigger belief-loss weight, or Burl treats Zeb's outputs as ordinal rather than numeric.

Cheap. Uses existing Zeb checkpoint + held-out games.

### Move 3 — Prototype Burl: Gemma 4 + tool harness, zero training

Minimal tool harness (8 tools above) + base Gemma 4 E2B + 20-50 decisions end-to-end. **No training yet.** Just see what the base model does when handed these tools. Does it:

- Make legal plays (via retry)?
- Use the tools at all, or ignore them?
- Call `get_belief` on meaningful dominoes or random ones?
- Produce reasoning traces that reference tool outputs correctly?

This is the cheapest way to find out whether the Gemma 4 base is agentic enough for our purposes before investing in training data.

### Move 4 — Training corpus design

Based on Move 3's failure modes. Likely a mix of:

- Positive exemplars (human- or LLM-authored traces that correctly use each tool class)
- Negative exemplars (traces that skip tool calls and hallucinate, so Burl learns to prefer tool-grounded reasoning)
- Retry-recovery traces (illegal first attempt → engine rejection → successful second attempt)

Size: unknown until Move 3 tells us what the gap looks like.

### Move 5 — First training run + iteration

SFT on the corpus from Move 4. Eval: legal rate (should be 100% by construction), bot-match, retry count, reasoning structure.

Once bot-match stabilizes, start STaR iterations: roll out on fresh decisions, filter with K1, retrain, repeat.

## Open questions

Honest list of things we don't know yet. Each gets answered empirically by one of the moves above or a quick eval.

- **Can Gemma 4 E2B actually tool-use reliably at 2B scale?** Historical baseline for small-model tool use is mixed. Move 3 answers this cheaply.
- **Is Zeb's 72% belief accuracy enough to be useful in Burl's reasoning?** 72% beats uniform (33%) hugely, but whether it's *useful* depends on how fine-grained Burl's decisions hinge on belief differences. Move 2 + Move 3 answer.
- **Does tool-mediated reasoning transfer to tool-less reasoning?** If we ablate tools at inference, does Burl reason well from what it previously retrieved, or does it collapse? Matters for product scenarios where tool latency is prohibitive. Measured via ablation eval.
- **Does the `conditional_outcome` composition actually give Burl the right reasoning primitive**, or does it leak too much (becomes E[Q] argmax through the back door)? Design-review question; the rule "no evaluative tools" should protect against leakage but worth testing on contrived positions.
- **Is a 3M-parameter Zeb strong enough**, or does Burl need a 10×-capacity belief module eventually? Easy to retrain if needed; currently no evidence we need it.
- **STaR on Burl traces vs STaR on LEM traces** — does the tool-call history make the training signal cleaner or just longer? Unknown until we run iterations.

## Directory plan (what will live here as we build)

```
burl/
├── OVERVIEW.md              ← this file
├── tools/                   ← tool harness implementations
│   ├── engine.py            ← is_legal, is_trump, unseen, void_audit, trump_declared
│   ├── zeb.py               ← get_belief wrapper around forge/zeb/
│   └── conditional.py       ← conditional_outcome (composes Zeb + E[Q])
├── harness/                 ← agent runtime
│   ├── tool_loop.py         ← ReAct-style think/act/observe loop
│   ├── retry.py             ← illegal-play retry logic
│   └── trace.py             ← trace structure for training + debugging
├── train/                   ← training recipes
│   ├── sft.py
│   └── star.py
├── eval/                    ← evaluation + scoring
│   ├── legal_rate.py
│   ├── bot_match.py
│   └── reasoning_structure.py
└── data/                    ← training/eval corpora (gitignored)
```

Not everything here day one — each file appears when the corresponding move lands.

## Related beads

- `t42-14h4` (P1) — Burl Stage 1: ReAct-style play. This is effectively Burl's founding bead. Will be renamed/updated to reference `burl/` directly.
- `t42-8aep` (P3) — Stylized teachers. Long-term research direction for characterful Burl variants.
- `t42-3oj7` (P4) — Tool-augmented LLM judge. General tool-use capability, may connect to Burl's trace-grading.
- `t42-0dg6` (P2) — Probabilistic reasoning in LEM. Separate track from Burl (in-weights calibration); result will inform whether Burl's Zeb-tool approach is strictly better than in-weights calibration or if both have a place.

## Not goals

- Bidding. Burl plays from a dealt state forward. Bidder is out of scope and lives elsewhere.
- General-purpose tool use. Burl's tool menu is 42-specific.
- Self-play against Burl. First iterations go against the E[Q] bot. Self-play comes later, if at all.
- Human-interface UX. Burl ships its reasoning traces as structured JSON; any UI (game integration, teaching display) is downstream work.

## Relationship to LEM's OVERVIEW

This is the forward-looking plan. `lem/OVERVIEW.md` is the history of the project that produced LEM, the comprehension adapter, and — relevant here — the vocabulary, curriculum principles, and E[Q] framework we're building on. Read both. LEM's OVERVIEW explains *why* we know what we know; this one explains *what we're doing next*.
