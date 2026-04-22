# Gus — Morning Status 4 (2026-04-22)

## TL;DR

Completed a full LAMIR-1 implementation pass: bug-hunted the eval harness,
distilled a π_opp head, and ran the complete ladder of look-ahead modes.
**Depth-1 look-ahead does not beat direct π_me on the 560-decision held-out
set, even with a trained opp model.** The architectural bottleneck is that the
distilled Q_head was trained on initial-deal world assignments and cannot
reliably evaluate mid-rollout states where played dominoes are absent from
those assignments. The direct π_me baseline (regret=0.551) is the current
production ceiling.

---

## Full inference ladder (560-decision held-out, mean regret, lower = better)

| mode | regret | bot-match | notes |
|---|---|---|---|
| direct π_me (baseline) | **0.551** | 76.1% | legal-masked argmax |
| q-bootstrap | 0.685 | 73.0% | depth-1 Q_head, no rollout |
| lamir1-qleaf (pre-Bug6) | 2.006 | 64.3% | rollout + Q_head leaf, stale assignment |
| lamir1-piopp (pre-Bug6) | 2.268 | 62.1% | π_opp rollout + Q_head leaf |
| lamir1-qleaf (Bug6 fix) | 2.156 | 63.2% | Bug 6 applied: zeroed played dominoes |
| lamir1-piopp (Bug6 fix) | 2.350 | 62.5% | Bug 6 applied |

All rollout modes are worse than direct π_me. q-bootstrap (no full rollout,
world-conditioned Q at depth 1) is the best look-ahead variant at 0.685 but
still trails baseline.

---

## Bugs found and fixed

### Bug 5 — Token/world domino mismatch (commit `e4e6862`)

Opp tokens in the look-ahead rollout were built from the real dealt hands,
not the world's hypothetical opp hands. Each world must have its own token
sequence built from `world_hands[m]`. Fix: `_world_game_hands(real_hands,
world_hands_m, P)` substitutes world hands for each absolute opp player;
applied before every `_build_tokens_voids` call.

### Sign-flip fix — V_head team frame (commit included in Bug 5 fix session)

V_head output is in the leaf player's team frame. When the leaf player is
on the opponent team relative to P, V_head must be negated. Fix: compare
`(leaf_cp % 2) == (P % 2)` — if False, flip sign. This improved pos=2
from regret=1.692 → 0.868 in isolation.

### Bug 6 — Stale world_assignment at Q_head leaf (commit `2c380a6`)

After 1-3 opp rollout plays, the world_assignment tensor passed to the
leaf Q_head still reflected the pre-rollout hand layout. Dominoes played
out of opp hands remained marked as present. Fix: track `played_dominos[m]`
per world during the rollout loop; zero those rows in `world_assign_leaf`
before the Q_head call.

**Result: Bug 6 fix did not help.** qleaf regret went from 2.006 → 2.156.
The mechanically correct fix exposed a distribution-shift problem: the
Q_head was trained exclusively on initial-deal world assignments (full 21
opp dominos) and has not been trained on post-play partial assignments
(18 or fewer opp dominos). The OOD input hurts rather than helps.

---

## Why look-ahead doesn't beat direct π_me with our current heads

Three independent reasons, ordered by severity:

**1. Q_head is only trained on initial-deal world assignments.**
At training time, `world_assignment` reflects the full opp hands at each
decision point. In a depth-1 rollout, the leaf Q_head sees a partially
depleted assignment (played dominos absent). This is out-of-distribution
and the Q_head produces unreliable leaf values — worse than not looking
ahead at all. Bug 6 fix is mechanically correct but exposes this gap.

**2. Strategy fusion kills PIMC even with a good leaf.**
Averaging world-conditioned best actions over worlds gives the best action
for each world individually — but those are different actions. The argmax
of an average Q is not the action committed best action across worlds.
`argmax_world` (single representative world) partially mitigates this but
the Q_head noise dominates.

**3. V_head is architecturally world-blind.**
V_head takes only `state_emb` (no `world_encoder` input). All M world
samples return identical V_head values (std=0.000 confirmed empirically).
V_head cannot differentiate between world-hypotheses, making it useless
as a look-ahead leaf evaluator.

---

## Schema v2 infrastructure: complete

Generated 1000-game v2 corpus (seeds 0-99 × decls 0-9, fixed 200 samples,
~770 MB total in 10 chunks). Schema v2 adds:
- `oracle_softmax_per_seat [4, 7]` — per-seat oracle policy target
- `legal_mask_per_seat [4, 7]`
- `voids_per_seat [4, 3, 8]`

Corpus: `gus/data/corpus_v2_train_*_d0-9.pt` + `gus/data/corpus_v2_eval.pt`

---

## π_opp head: trained, 68.6% oracle accuracy

Trained `PiOppHead(d_model, seat_embed_dim=8)` — a 3-way seat embedding
concatenated with `state_emb` and projected to 7 logits. Frozen trunk,
only 1,879 head parameters trained.

- 20 epochs, 1000-game v2 corpus
- Best eval acc: **68.57%** (epoch 13)
- Loss converged at ~0.625
- Adapter: `gus/adapters/v3_10k_piopp.pt`

At 68.6% oracle accuracy, π_opp is substantially better than rotated π_me
(~55% oracle match). Despite this, lamir1-piopp is slightly *worse* than
lamir1-qleaf — confirming the leaf evaluator, not the opp model, is the
bottleneck.

---

## Next-steps recommendations

### (a) Retrain Q_head with post-rollout world assignment distribution

The cleanest fix: augment training to include world assignments where some
dominos have been "played out" (zeroed). Concretely: for each decision at
`d_idx`, also generate training items at `d_idx` with 1-3 randomly selected
opp dominos removed from the world assignment. This teaches Q_head to
evaluate mid-game states with partial world info. This is what the LAMIR
paper implicitly assumes — the Q function is evaluated at *any* game state
including mid-trick.

### (b) End-to-end LAMIR training instead of distillation

The paper's Q_head is trained jointly with the rollout policy in an RL-style
loop (PIMC + CFR+ solver). Our distilled Q_head is trained on single-shot
decisions, not on the (world, action, rollout) tuples that a look-ahead
policy actually encounters. True LAMIR training would use the rolled-out
states as Q_head training targets.

### (c) Multi-valued-states leaf (paper §4.3)

The LAMIR paper evaluates the leaf with a V function trained on multi-valued
states — the full joint (state, world) distribution. Our V_head is world-
blind by architecture (no world_encoder input). Adding world conditioning
to V_head training is a smaller change than (b) and might be the right
next experiment.

### (d) Accept depth-1 ceiling and focus on direct π_me improvements

q-bootstrap at 0.685 is only 24% above the direct baseline. The direct
baseline at 0.551 is already at 76% oracle match. The remaining 24% error
gap is likely: (i) irreducible (near-ties where any legal action is fine),
(ii) cases where look-ahead genuinely helps but requires better leaf values,
and (iii) cases requiring multi-ply look-ahead (> 1 trick). The simpler path
to improvement may be more training data and a larger direct π_me head rather
than look-ahead.

---

## Current best adapter

`gus/adapters/v3_consistency_10000g.pt` — direct π_me argmax, regret=0.551,
bot-match 76.1% on 560-decision held-out set.

The look-ahead harness is ready to pick up again once one of the above
architectural fixes is in place.
