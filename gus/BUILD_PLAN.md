# Gus — Joint-world distillation build plan

> **Status: active build — 2026-04-20.** This doc says what we're about to do.
> Appends to `OVERVIEW.md`'s Move 0. Receipts log as it happens.

## What we're building

The five-head LAMIR-ready student that distills the oracle's per-world
`(world_layout, Q_per_world)` tensor into a small neural net.

Five heads sharing one state encoder:

1. **`belief_head`**: `state → P(dom ∈ seat)`. Supervised against truth
   (the real hand in the original deal).
2. **`V_head`**: `state → scalar E[Q]`. Supervised against oracle mean
   over sampled worlds.
3. **`π_me_head`**: `state → action softmax`. Supervised against oracle's
   `argmax E[Q]` (the existing E[Q] bot's choice).
4. **`world_encoder + Q_head`**: `(state, world_layout) → Q per action`.
   Supervised against per-world oracle Q. **This is the LAMIR-critical head**
   — it supports belief re-weighting during continual-resolving look-ahead
   without re-querying the oracle.
5. **`π_opp_head`**: `(state, opp_seat) → opp action softmax`. Supervised
   against oracle's argmax from the opponent's seat in each sampled world.
   *Deferred to v2* — requires a corpus re-gen with opponent-view oracle
   queries saved.

## Why this shape

### Why world-conditioned (Q_head) instead of marginal (V_head only)

At LAMIR's look-ahead search leaves, belief over hidden hands updates as
opponent plays are observed. A marginal student gives `E[Q | current_belief]`
and can't re-condition. A world-conditioned student caches `Q_m` for
`m ∈ [1..M]` worlds at each leaf; belief update is then a cheap
`Σ_m w_m(new_belief) · Q_m` re-weight over the cache. **Look-ahead
through a tree with belief updates becomes a cached-and-reweight operation
instead of a re-query operation.** That's what makes LAMIR cheap.

### Why five heads on a shared encoder

The five outputs all require the same understanding of the state (whose
turn, what's been played, trump, my hand, score). One encoder trains five
times over, regularized by all supervisions at once. Parameter-efficient
and cross-task regularized. Total target: ~3-5M params, Zeb-class.

## Data generation

Adaptive sampling to SEM < 0.5 (validated today):

```bash
python -u -m forge.eq.generate \
    --start-seed 0 --n-games 100 \
    --adaptive \
    --min-samples 100 --max-samples 50000 \
    --sem-threshold 0.5 --batch-size 200 \
    --save-joint-worlds \
    -o gus/data/corpus_100.pt
```

**Expected**: ~10s/game on MPS, so ~17 min for 100 games. ~600 MB file.

Seeds `0-99` for training. Held-out seeds `900000-900099` for eval
(same convention as LEM and Burl).

### What's in the corpus per decision

```python
DecisionRecordGPU(
    player,              # 0-3
    e_q,                 # [7] marginal oracle E[Q] per action → V_head target
    action_taken,        # int → π_me_head target
    legal_mask,          # [7] boolean
    e_q_var, e_q_pdf,    # [7] / [7, 85] — existing oracle outputs
    world_hands,         # [M, 3, 7] opponent hands per world (NEW)
    q_per_world,         # [M, 7] oracle Q per world (NEW)   → Q_head target
    ...
)

GameRecordGPU(
    decisions=[...],     # 28 per game
    hands=[...],         # full initial deal (all 4 hands) → belief_head truth target
    decl_id,
)
```

Every field the student needs is here. No second pass through the oracle.

## Student architecture (v1 target)

```
Input: flat state features [F ≈ 200-400 dims]
       ├── current_player (4-dim one-hot)
       ├── declared_suit (10-dim one-hot)
       ├── my_hand (28-dim multi-hot)
       ├── played_mask (28-dim multi-hot)
       ├── current_trick_plays (variable, flatten up to 3 plays × 28)
       ├── trick_history_summary (per-trick: winner + dominoes played)
       └── scores (2-dim team points + bid)

             │
             ▼
       state_encoder
       (MLP: [F → 256 → 256 → z])    z dim = 256
             │
    ┌────────┼────────┬────────┬────────┐
    ▼        ▼        ▼        ▼        ▼
belief     V_head   π_me    world_enc   (π_opp, v2)
[84 out]   [1 out]  [7 out]  [28×4→64]
                               │
                               ▼
                            fuse(z, w_emb)
                               │
                               ▼
                            Q_head [7]
```

**Parameter budget** (rough):
- state_encoder: ~150K params (F=300, 2×256 hidden)
- belief_head: 256 × 84 = ~22K
- V_head: 256 × 1 = 256
- π_me_head: 256 × 7 = ~2K
- world_encoder: 112 × 64 = ~7K
- Q_head: (256+64) × 256 × 7 = ~580K

Total: ~760K params. **Leaves budget for a transformer encoder later.**

## Losses

```
L = α · L_belief
  + β · L_V
  + γ · L_π_me
  + δ · L_Q
```

- **L_belief**: cross-entropy per (unseen domino, seat) slot.
  Mask out known-played and in-my-hand dominoes.
- **L_V**: MSE of V_head(state) vs oracle E[Q] for action_taken.
- **L_π_me**: cross-entropy of π_me_head softmax vs oracle's argmax action.
  Label smoothing ε=0.1 to handle near-ties.
- **L_Q**: MSE of Q_head(state, world_m) vs q_per_world[m] per sampled world.
  Averaged over M worlds per decision.

**Weighting (v1 starting point)**: `α=1.0, β=0.5, γ=0.5, δ=1.0`.
Belief and Q are the core distillation targets; V and π_me are
consistency/shaping signals. Tune after first run.

## Training staging

### v0 — Belief-only sanity check (today)

- State encoder + belief_head only.
- `α=1.0`, all others = 0.
- Train to truth on the 100-game corpus (seeds 0-99).
- Evaluate on held-out (seeds 900000-900099): **belief top-1 accuracy per
  unseen domino**. Baseline: chance ≈ 33% with 3 seats.
- **Success bar**: top-1 > 50%. (Zeb's hidden-only was 39%, so anything
  above that beats the prior belief artifact we have.)
- **Fail bar**: top-1 < 35%. Means state representation is underbaked or
  training signal is too sparse — debug before building more.

### v1 — Full 4-head student (this week)

- Add V_head, π_me_head, world_encoder, Q_head.
- Joint loss with all four weighted.
- Evaluate:
  - Belief top-1 on held-out (should be ≥ v0's value)
  - V_head MAE vs oracle E[Q] (bar: < 2 Q-points)
  - π_me bot-match rate (bar: > 80% — aspirational, Burl's iter3-rules was 90%)
  - Q_head per-world MAE (bar: < 3 Q-points)

### v2 — Add π_opp + LAMIR look-ahead (future)

- Requires corpus re-gen with opponent-view Q cached per sampled world
  (3× oracle calls per decision).
- Opponent policy head trained supervised on that signal.
- Then: implement the LAMIR continual-resolving inference loop using all
  five heads. Compare bot-match with and without look-ahead.

## Repo layout

```
gus/
├── OVERVIEW.md                 ← vision
├── BUILD_PLAN.md               ← this doc
├── data/
│   ├── corpus_100.pt           ← 100-game training corpus (gitignored — regen-able)
│   └── corpus_eval.pt          ← held-out eval corpus
├── model/
│   ├── __init__.py
│   ├── features.py             ← DecisionRecordGPU → flat state features
│   ├── student.py              ← encoder + heads Module
│   └── losses.py               ← four weighted losses
├── data/
│   └── dataset.py              ← PyTorch Dataset over .pt corpus
├── train/
│   └── train_v0_belief.py      ← v0 belief-only trainer (today)
│   └── train_v1_full.py        ← v1 full 4-head trainer (later)
└── eval/
    └── eval_belief.py          ← belief top-1 + calibration
```

`data/` and model checkpoints are gitignored. Code and eval scripts committed.

## What success looks like

**v0 outcome** (today): belief top-1 on held-out > 50%. Proves the joint
tensor carries a learnable belief signal.

**v1 outcome** (this week): π_me bot-match > 80% AND Q_head per-world MAE
< 3 Q-points. Proves we have a working world-conditioned value function —
a LAMIR-ready student that also happens to play 42.

**v2 outcome** (later): LAMIR look-ahead + π_opp improves bot-match over
v1. Proves continual resolving on the abstracted latent is worth the
engineering cost on 42.

## Not yet

- Transformer state encoder (MLP first, upgrade if v1 plateaus)
- Self-play / PPO (still needed downstream; distillation first)
- LEM as narrator head (needs a working player to narrate)
- Multi-game batch of M > 1 concurrent corpora (single-game-adaptive is fine for now)

## Related

- `forge/eq/generate/` — oracle + joint-world infrastructure (this session)
- `scratch/joint_worlds_tire_kick.py` — tire-kick eyeball script
- `burl/tools/eq_distribution.py` — downstream consumers that could use a
  student E[Q] approximation once this ships
