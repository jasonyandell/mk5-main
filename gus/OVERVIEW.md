# Gus — Neural policy + belief + value for Texas 42

> **Status — 2026-04-20: pivoted to joint-world distillation.**
> Tire-kick validated the per-world `(world_layout, Q_per_world)` tensor
> as a distillation target (`forge/eq/generate/` now saves this natively
> on `--save-joint-worlds`, tested end-to-end on MPS in ~10s/game at
> SEM<0.5 adaptive). Move 0 is now a five-head LAMIR-ready student. See
> [`BUILD_PLAN.md`](BUILD_PLAN.md) for the concrete architecture +
> training plan currently underway.

Can a small neural network — policy + belief + value, multi-task trunk, trained
against the E[Q] oracle — play 42 competently, after LEM and Burl both stalled
at the same reasoning-verification ceiling?

The name is a common East Texas name from the 1930s. It's not an acronym.
Sibling to LEM and Burl, not a successor.

## The premise

LEM taught rules to a small LLM via comprehension SFT. It hit 86% on
comprehension but plateaued at 55% bot-match on open-ended play.

Burl handed the LLM tool access and let it reason through the oracle. Iter-3-rules
reached 90% bot-match but STaR iteration converged on noise — V8 plateaued,
V9 regressed, because the filter couldn't distinguish "coherent reasoning,
unlucky play" from "lucky bot-match on garbage reasoning."

Both stalled at the same place: **you can't train a model to reason well when
you can't grade its reasoning.** LLM-as-reasoner was the hill. We took it as far
as the garage project's budget for "verifying reasoning traces" could reach.

Gus skips the reasoning channel entirely. Trains a neural policy on
`(state, belief) → action`. Grades by play outcome, which the E[Q] oracle
gives variance-free. When the model plays well, LEM attaches *downstream*
as a commentary head — narrator of a player that already works, not the
other way around.

## Why this, why now

The field moved while we were in the reasoning-trace hole:

- **Policy gradient beat CFR in 2025.** [Reevaluating PG Methods for IIGs
  (ICLR 2025)](https://arxiv.org/abs/2502.08938) ran 7000+ training runs
  and found plain PPO matches or beats CFR-based DRL on exploitability.
  [A NeurIPS 2025 companion paper](https://www.mit.edu/~gfarina/2025/neurips25_policy_gradient/neurips25_policy_gradient.pdf)
  proves global convergence for PG in extensive-form. The CFR machinery
  the Bridge literature scared us off with? Not required.
- **[LAMIR (Oct 2025)](https://arxiv.org/abs/2510.05048)** is the first IIG
  algorithm to learn a clustered-information-set abstraction *without
  domain knowledge* and use it for continual resolving. Beats model-free
  PG on Goofspiel-15. 42 is structurally close.
- **Modern Bridge AI is policy + value + belief + Belief Monte Carlo Search.**
  [IEEE JAS 2024](https://www.ieee-jas.net/article/doi/10.1109/JAS.2024.124488)
  and [Thousand (MDPI 2025)](https://www.mdpi.com/2076-3417/15/4/2121) both ship
  this exact shape. Partnership trick-taking is a solved recipe — we just haven't
  run it on 42 yet.
- **[PRR-TM (2025)](https://link.springer.com/article/10.1007/s13042-025-02607-y)**
  introduces explicit teammate modeling for adversarial team games — a module
  that predicts partner's action given partner's observations. Honest model
  of the signaling humans use in 42.

## Not a successor — three siblings now

|                | LEM                             | Burl                             | Gus                                    |
|----------------|---------------------------------|----------------------------------|----------------------------------------|
| Base           | Qwen 3 1.7B                     | Gemma 4 E2B                      | Small transformer, from scratch        |
| Channel        | English reasoning               | LLM + tool calls                 | Neural policy + value + belief         |
| Training       | Comprehension SFT + STaR        | Tool-use SFT + STaR              | BC-from-oracle + PPO self-play         |
| Grading        | Comprehension accuracy; bot-match | Bot-match; retry count         | Bot-match; E[Q] delta; exploitability  |
| Explainability | Primary output                  | Primary output                   | Downstream (LEM as narrator head)      |
| Product slot   | "Explain this position"         | "Play against AI" (LLM)          | "Play against AI" (neural)             |
| Lineage        | Fresh from Qwen 3               | Fresh from Gemma 4               | Fresh architecture; reuses LEM tokens  |
| Directory      | `lem/`                          | `burl/`                          | `gus/` (this project)                  |

Shared infrastructure (owned by none, used by all): `forge/` (engine, perfect-info
solver, E[Q] framework, Zeb's architecture, narration tokenizer, decision-dataset
machinery).

## The 42 cheat codes Gus leans on

Most researchers don't have these. We do:

1. **Oracle as reward signal.** E[Q] labels every state variance-free. We don't
   have to rely on noisy realized outcomes for learning signal — we have an
   oracle that gives expected value directly. This is a massive, underrated
   advantage.
2. **Supervised belief training.** At training time we know every hidden hand.
   Train the belief head by cross-entropy to the true posterior, then deploy
   at inference where hands are hidden. Zeb stalled at 39% top-1 partly because
   it was distilled from self-play — real supervised signal should exceed it.
3. **Forge infrastructure.** Engine, solver, E[Q] framework, Zeb architecture,
   LEM tokenizer, decision dataset, held-out seeds, `forge/analysis/results/web/`
   visualizers. Half the project is already built.

## Vocabulary (matters — be precise)

Reuses LEM and Burl's vocabulary (solver, E[Q] framework, E[Q] bot, Zeb, engine).
Adds:

| Term | What it is |
|---|---|
| **Shared trunk** | Transformer encoder over `(public_history, your_hand, belief_features)` tokens |
| **Policy head** | π(a \| state, belief) — softmax over legal dominoes |
| **Value head** | V(state, belief) — scalar, trained to match oracle E[Q] |
| **Belief head** | P(each unseen domino ∈ {L, partner, R}) — supervised against truth at train time |
| **Warmstart (BC)** | Behavior-cloning from `(state, argmax_E[Q])` pairs before any RL |
| **Self-play fine-tune** | PPO with partner weight-sharing (same net runs in all 4 seats) |
| **PIMC leaf search** | Optional look-ahead via the E[Q] framework when policy entropy is high |
| **Exploitability** | Best-response value gap — game-theoretic quality metric (from the DRL-PG literature) |

## Architecture

```
                     ┌──────────────────────────────────────────┐
                     │           GUS (the player)               │
                     │    Shared transformer trunk              │
                     │    Input: public history + your hand     │
                     └──┬─────────────┬──────────────┬──────────┘
                        │             │              │
                        ▼             ▼              ▼
                 ┌────────────┐ ┌───────────┐ ┌─────────────┐
                 │ POLICY     │ │ VALUE     │ │ BELIEF      │
                 │ π(a│s,b)   │ │ V(s,b)    │ │ P(hand│obs) │
                 │ → softmax  │ │ → scalar  │ │ → cat/player│
                 │   over     │ │   (E[Q])  │ │             │
                 │   legals   │ │           │ │             │
                 └─────┬──────┘ └─────┬─────┘ └──────┬──────┘
                       │              │              │
   Training            │              │              │
   signal:       ┌─────┴──────┐ ┌─────┴─────┐ ┌──────┴──────┐
                 │ BC: argmax │ │ Oracle    │ │ True hand   │
                 │ of E[Q]    │ │ E[Q] label│ │ posterior   │
                 │ then PPO   │ │ (variance-│ │ (known at   │
                 │ self-play  │ │  free)    │ │  train time)│
                 └────────────┘ └───────────┘ └─────────────┘

   ─────────────────────  inference-time boundary  ──────────────────────

                 ┌────────────────────────────────────────┐
                 │  PIMC leaf search (optional)           │
                 │  When policy entropy > threshold,      │
                 │  sample hidden worlds, solve via       │
                 │  E[Q] framework, re-rank policy's      │
                 │  top-k. Modern Bridge-AI recipe.       │
                 └────────────────────────────────────────┘

                 ┌────────────────────────────────────────┐
                 │  LEM as commentary head (later)        │
                 │  Input: (state, Gus action, attention) │
                 │  Output: English narration             │
                 │  Trained AFTER the player works.       │
                 └────────────────────────────────────────┘
```

## Design principles

- **Neural as player, LLM as narrator.** Reasoning quality gets baked into
  commentary *after* we have a player. This reverses the LEM/Burl order and
  side-steps the reasoning-verification problem that stalled both.
- **Oracle as training signal.** E[Q] labels give variance-free rewards.
  Don't waste that advantage on realized-outcome noise.
- **Belief supervised, not distilled.** We know the hands at training time.
  Train to truth; deploy to uncertainty.
- **Partnership via weight-sharing, not signaling.** Same policy net in all
  four seats. Partners can't signal at play time, but correlation lives in
  weights — the ex-ante correlation trick from the team-zero-sum literature.
- **Bridge-AI recipe, not poker.** PIMC-at-leaves + neural policy is what
  actually ships in modern trick-taking AIs (NooK, Wbridge5, JACK, BMCS).
  42 is closer to Bridge than to poker; borrow from the right precedent.
- **100% legal by construction.** Engine validates every commit. Same
  invariant as Burl.
- **Grade by E[Q] delta and exploitability.** Same north star as LEM and
  Burl for E[Q]. Add exploitability (from the 2025 DRL-PG literature) as
  a game-theoretic quality measure that realized win-rate can't capture.

## Experimental moves

Ordered by most-learning-per-dollar. Each move is also a potential "stop here
and ship if it's good enough" checkpoint.

### Move 0 — [in flight] Joint-world distillation → five-head LAMIR-ready student

**Pivoted 2026-04-20 after tire-kicking the joint-world tensor.** The original
"BC the E[Q] bot" plan threw away too much structure; the per-world
`(world_layout, Q_per_world)` tensor from the oracle carries catalyst
signals that post-hoc tools (`spike_drivers`, `what_would_change_my_mind`)
were trying to reconstruct. Preserving it natively makes the student
LAMIR-ready by construction — the Q head can re-weight cached per-world
values during look-ahead without any oracle calls.

Five heads on a shared state encoder:

1. **`belief_head`**: state → P(dom ∈ seat). Supervised against truth.
2. **`V_head`**: state → E[Q]. Supervised against oracle mean.
3. **`π_me_head`**: state → action softmax. Supervised against oracle argmax.
4. **`world_encoder + Q_head`**: (state, world) → Q per world. **LAMIR-critical.**
5. **`π_opp_head`** (v2): (state, opp_seat) → opp action. Deferred — needs
   opponent-view oracle queries in the corpus.

Corpus: 100 games, adaptive sampling to SEM<0.5, generated on MPS in ~17 min.
See [`BUILD_PLAN.md`](BUILD_PLAN.md) for concrete architecture, loss shape,
data schema, training stages (v0 belief-only → v1 full → v2 LAMIR), and
success bars.

### Move 1 — [todo] PPO self-play fine-tune

- Partner weight-sharing (same net in all 4 seats).
- Reward = E[Q] delta vs bot baseline, not realized Q.
- Measure: exploitability (per the ICLR 2025 DRL-PG methodology),
  bot-match, E[Q] delta.

### Move 2 — [todo] PIMC leaf search at high-entropy positions

- Fast policy for the ~90% of decisions where policy entropy is low.
- Invoke the existing E[Q] framework for look-ahead when policy is uncertain.
- Hybrid exactly like modern Bridge AIs. Preserves inference speed while
  catching the hard cases.

### Move 3 — [speculative] LAMIR-style information-set abstraction

If Moves 0-2 plateau, try [LAMIR](https://arxiv.org/abs/2510.05048)'s
clustered-information-set learning for continual resolving. October 2025
paper; reimplementing on 42 could be a real contribution.

### Move 4 — [speculative] PRR-TM-style teammate modeling

Explicit partner-action prediction head. The honest model of 42's signaling
(leads as information). Most likely to matter when bidding is added.

### Move 5 — [speculative] LEM as commentary head

Once the player works: SFT LEM on `(state, Gus's action, Gus's attention
patterns) → English narration`. LEM becomes the narrator of a player that
already plays well. Closes the reasoning-quality gap from the other direction
— instead of training reasoning to drive action, train narration to describe
action.

## Directory plan (intended)

```
gus/
├── OVERVIEW.md             ← this file (vision + architecture)
├── model/                  ← transformer trunk + three heads
│   ├── trunk.py
│   ├── heads.py            ← policy, value, belief
│   └── tokenize.py         ← reuses/extends LEM's tokenizer
├── data/                   ← E[Q]-labeled state/action/belief tuples
│   └── build_corpus.py
├── train/                  ← warmstart + PPO + curriculum
│   ├── warmstart_bc.py     ← Move 0
│   ├── selfplay_ppo.py     ← Move 1
│   └── leaf_search.py      ← Move 2
├── eval/                   ← exploitability + bot-match + belief calibration
│   ├── bot_match.py
│   ├── exploitability.py
│   └── belief_calibration.py
└── experiments/            ← per-spike writeups (one markdown each)
```

## Open questions

- **Does BC-only hit 90% bot-match**, or plateau lower than Burl's iter-3-rules
  (90%)? First real data point in Move 0.
- **Does supervised belief training exceed Zeb's 39% top-1?** We're training
  to truth instead of distilling from self-play. The answer tells us whether
  Zeb was architecturally bottlenecked or signal-bottlenecked.
- **Is partner weight-sharing enough to recover team signaling**, or do we
  need explicit teammate modeling (PRR-TM style)?
- **Is PPO stable with E[Q] as the reward?** The DRL-PG literature uses
  realized outcomes; nobody's pumped oracle-labeled rewards through PPO
  at scale. Might just work; might need variance-reduction tricks.
- **Does LEM-as-narrator produce coherent commentary on Gus's policy?**
  This is the reasoning-quality question from the other direction. If LEM
  can narrate a policy that plays well, we've got both halves of the
  original "reasoning + play" vision — we just built them in the opposite
  order.
- **Can Gus fit in under 10M params?** Zeb was 3.3M. Belief + policy + value
  shouldn't need more capacity than Zeb needed for belief alone. Worth
  finding out before over-parameterizing.

## Related beads

- `burl/` — LLM-as-player. Stalled at reasoning-verification. Infrastructure
  and lessons carry forward (especially `burl/tools/engine.py`,
  `forge/eq/`, and the decision dataset).
- `lem/` — comprehension + rationalization. Stalled at open-ended play
  ceiling. Reusable as Gus's commentary head in Move 5.
- `t42-14h4` — Burl's founding umbrella bead. Gus carries the "play-mode
  AI" goal forward.

## Not goals

- **Bidding.** Same as Burl. Play from dealt-and-declared state forward.
- **Matching the perfect-info solver.** Impossible by construction.
- **CFR machinery.** 2025 evidence says PPO is competitive; skip the plumbing.
- **Reasoning traces upstream of the policy.** That's what LEM/Burl tried
  and what Gus deliberately avoids. Reasoning comes downstream, as
  commentary on a working player.
- **Human-interface UX.** Gus ships actions and (eventually) narrations
  as structured JSON. UI work is downstream.

## Relationship to LEM and Burl

LEM explained. Burl played via LLM tool-use. Gus plays via neural policy.
If Gus works, LEM gets a second life as Gus's narrator, and the original
"reason about the distribution, then commit to a play" vision is reconstructed
with each component playing the role it's actually good at.

If Gus doesn't work, we've learned something real about whether small-model
parameter budgets + oracle supervision are sufficient for partnership
imperfect-info games. Fun either way.
