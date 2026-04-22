# Gus — Morning Status 4 (2026-04-22)

## TL;DR

We tried to beat the 0.551 direct π_me baseline via 1-ply look-ahead in 8
variants across two sessions. None beat it. q-bootstrap at 0.679 is the best
look-ahead result (+25% regret vs direct). The bottleneck is intrinsic scalar
value-head noise: distilled V/Q heads trained on marginal oracle E[Q] don't
preserve the action-ordering signal that look-ahead requires. The architectural
fix (T×T multi-valued-states matrix from Kubíček & Lisý 2025) is months of
work. Direct π_me at 0.551 is the current production ceiling.

---

## Final inference ladder (560-decision held-out, lower = better)

| mode | regret | bot-match | notes |
|---|---:|---:|---|
| direct π_me (baseline) | **0.551** | 76.07% | π_me argmax — winner |
| q-bootstrap | 0.679 | 72.50% | depth-1 Q_head, world-conditioned |
| v-bootstrap | 1.645 | 66.96% | depth-1 V_head, world-blind |
| lamir1-qleaf | 2.006 | 64.29% | full rollout + Q_head leaf |
| lamir1 | 2.094 | 60.36% | full rollout + V_head leaf |
| lamir1-piopp | 2.268 | 62.10% | trained π_opp rollout + Q_head leaf |
| lamir1-piopp + Fix 6 | 2.350 | 62.50% | assignment-update made it worse |

Fix 6 (zeroing played dominoes from world_assign at leaf) made every rollout
mode worse — the training convention keeps the original assignment intact as
the played_mask advances, so our "correction" broke a training invariant.

---

## What worked

**Paper read and divergence-mapped.** `scratch/lamir_paper_notes.md` has a
precise accounting of what Kubíček & Lisý 2025 build vs what we built.
We're not building LAMIR — we built depth-1 determinization rollout with
distilled leaf evaluators.

**Bug-hunting: 5 bugs probed, 2 confirmed and fixed.**
- *Bug 5 (token/world mismatch, commit `e4e6862`)*: opp tokens were built
  from the real dealt hands, not the world's hypothetical opp hands. Each
  world now gets its own token sequence via `_world_game_hands()`. Clean fix.
- *Sign-flip fix*: V_head output is in leaf player's team frame; must negate
  when leaf player is on opp team relative to P. Real bug for pos=2 (−0.8
  Q-pt improvement in isolation).

**Schema v2 pipeline: end-to-end working.** Types → generator → loader →
eval corpus → train corpus. 1000-game diverse-seed corpus in ~13 minutes
(10 seeds × 10 decls × 10 chunks). `oracle_softmax_per_seat [4,7]`,
`legal_mask_per_seat`, `voids_per_seat` all plumbed through.

**π_opp head trained: 68.57% oracle accuracy.** `PiOppHead` (1,879 params,
3-way seat embedding) trained on 1000-game v2 corpus in 20 epochs.
`gus/adapters/v3_10k_piopp.pt`. Real signal — substantially better than
rotated π_me (~55%).

---

## What didn't work + why

**Fix 6 (stale world_assignment update).** Regret ticked up across the board.
The training convention preserves the original world_assignment as the game
advances — played_mask encodes what's been played, not the assignment tensor.
Our zeroing of played-domino rows broke a training invariant.

**π_opp rollout.** Even at 68.6% oracle match, lamir1-piopp is *worse* than
lamir1-qleaf. Opp quality is not the bottleneck. The leaf evaluator is.

**Full rollouts.** Every rollout variant is worse than depth-1. Compounding
errors across 1-3 opp steps overwhelm any leaf signal. The look-ahead
*hurts* for all leaf evaluators we have.

---

## The architectural conclusion

Kubíček & Lisý explicitly warned: "the value function from RNaD cannot be
used as a value function for look-ahead reasoning." Our V_head is oracle-
distilled (not RNaD), but the same failure mode applies: scalar-prediction
noise of the distilled value is enough to flip argmax at decision boundaries,
while π_me trained on argmax directly preserves ordering.

The paper's answer is the T×T **multi-valued-states matrix value function** —
a |T|×|T| table of expected values under pairs of strategy transformations.
Building it requires retraining the entire Gus stack jointly with CFR+.
Months of work, not hours.

**V_head is architecturally world-blind** (confirmed empirically: std=0.000
across 200 world samples for the same decision). V_head takes only
`state_emb` with no world_encoder input. Cannot serve as a look-ahead leaf.

**q-bootstrap (depth-1, no full rollout)** at 0.679 is the best look-ahead
result because: it uses the initial-deal world assignment (no depletion
issue), avoids strategy-fusion by not averaging across worlds, and the Q_head
world-conditional signal is real — just not strong enough.

---

## Next directions for the user to consider

**1. Accept depth-1 ceiling, ship q-bootstrap as an alternative inference
mode.** +25% regret vs direct but world-conditioned. Could serve as a second
opinion in a router (use q-bootstrap when π_me entropy is high). Small
engineering lift.

**2. Train a look-ahead-compatible V-head.** Train V on expected Q under
sampled opp play rather than marginal oracle E[Q]. This is closer to the
paper's recipe without requiring CFR+. Medium effort.

**3. Implement LAMIR paper faithfully.** Multi-valued states + CFR+ solver.
Large effort, research-grade. Only justified if competitive play vs. expert
humans is the explicit goal.

**4. Bridge-AI / BMCS recipe.** The OVERVIEW's Move 1 (PPO self-play) would
move V/Q heads to a better equilibrium shape. The 68.6% π_opp head and
world-conditioned Q_head are the raw materials for a Bridge-AI-style player
rather than a LAMIR clone. Gus doesn't have to be LAMIR.

---

## Path (a) post-session experiment: Q_head depletion augmentation (attempted, didn't help)

**Hypothesis**: Q_head is OOD at rollout leaves because it was trained only on
initial-deal world assignments (all 21 opp dominos present). After 1-3 opp
plays, the played_mask advances but the assignment tensor has fewer assigned
dominos. Fine-tuning with random partial depletion (zero k random assigned rows,
k ~ Uniform(1,3), p=0.5 per item) should teach Q_head to be robust.

**Training**: `train_q_head_augmented.py`, 15 epochs, lr=5e-5, frozen trunk
(encoder + belief + v_head + π_me), trainable: q_head + world_encoder
(160,007 params). Best eval q_mae improved 8.277 → 8.169 (epoch 10).
Adapter saved as `gus/adapters/q_head_aug.pt`.

**Eval**: lamir1-qleaf + Bug 6 + augmented adapter

| mode | regret | bot-match |
|---|---:|---:|
| lamir1-qleaf + Bug6 (baseline) | 2.156 | 63.2% |
| lamir1-qleaf + Bug6 + aug Q_head | **2.216** | 63.8% |

**Result**: Slightly worse. The small q_mae improvement (0.1 pts) did not
translate to better rollout decisions — the 0.06 regret regression is noise,
not signal either way. Path (a) closed.

**Why it didn't work**: OOD augmentation fixes a measurement artifact
(q_mae on depleted inputs) but not the fundamental ordering problem: scalar
noise from distillation is large relative to the action-value gap at decision
boundaries. The leaf evaluator would need to be trained end-to-end in the
rollout context (path b) to fix this. Augmenting the input distribution alone
is insufficient.

---

## Artifacts

| file | description |
|---|---|
| `gus/adapters/v3_consistency_10000g.pt` | Best adapter — direct π_me, regret=0.551 |
| `gus/adapters/v3_10k_piopp.pt` | π_opp head, 68.57% oracle acc |
| `gus/data/corpus_v2_train_*_d0-9.pt` | 1000-game v2 train corpus (10 chunks) |
| `gus/data/corpus_v2_eval.pt` | 560-decision v2 eval corpus |
| `gus/eval/lamir1.py` | Full inference harness, all 6 modes |
| `gus/train/train_pi_opp.py` | π_opp head training script |
| `scratch/lamir_paper_notes.md` | Paper divergence map |
| `scratch/lamir1_*.json` | Per-mode eval results |
| `gus/adapters/q_head_aug.pt` | Augmented Q_head fine-tune (path a, didn't help) |
| `gus/train/train_q_head_augmented.py` | Depletion-augmentation fine-tune script |
| `gus/eval/belief_ceiling.py` | Bayes-optimal belief top-1 diagnostic |

---

## Addendum — Belief is at the Bayes ceiling (2026-04-22)

Ran `gus/eval/belief_ceiling.py` on `corpus_eval_20.pt` to compute the
theoretical best top-1 achievable from the oracle's own sampled worlds:

```
Bayes-optimal belief top-1:  39.184%
Gus v3 belief head (§6):     ~38-39%
```

**Gus is at ceiling.** The "belief is hard" feeling was correct, but the
reason is that **the signal isn't there** — not that the architecture or
capacity is wrong. Zeb's 39% wasn't a plateau, it was the information limit.

Per-decision-idx breakdown confirms it: d_idx 0-5 sits at ~33% (pure prior;
no play info yet), d_idx 18+ rises to 50-75% (deterministic voids + played
domino exclusions), d_idx 26 hits 100%. Early-hand belief is fundamentally
unsharpen-able without more observations.

**Reorients the belief roadmap**: top-1 accuracy is a dead lever. The real
unfinished work is posterior *shape* — §15 already showed distribution-target
training closes 47% of the calibration gap (KL 0.078 → 0.062). The open loop
is co-training belief + world_encoder + Q_head **jointly** so that
better-calibrated belief actually propagates to better look-ahead value
estimates. That's a concrete overnight-scale experiment, not a mystery.

See `gus/PRACTICALITIES.md §21` for the full receipt.

---

## Addendum 2 — Joint co-train run: falsified hypothesis + accidental win

Ran the §21 proposed experiment (`gus/train/train_belief_q_joint.py`).
Results landed two findings:

**Hypothesis falsified**: joint co-training lowered belief KL 20%
(0.0840 → 0.0672) as designed, but downstream q-bootstrap regret got
slightly worse (0.685 → 0.718). Q_head was already sitting at a sweet
spot for the original belief's output distribution; moving the belief
head moved Q_head off it. Calibration improvement doesn't propagate
additively in a distillation pipeline where downstream heads were
trained against the old belief's shape.

**Accidental win**: to run the A/B properly, had to build a new inference
mode `q-bootstrap-belief` that samples worlds from the belief head
instead of reading them from the oracle's saved corpus. On the
ORIGINAL adapter (no co-train), this gives **regret 0.655** — the
closest any look-ahead variant has come to the 0.551 direct baseline
(gap 0.104, ~19%).

Likely mechanism: oracle adaptive sampling can overconcentrate on a
few worlds at near-consensus decisions; belief softmax sampling is
smoother. Reuses `gus/model/sample_worlds.py` (the file symmetry-checker
wrote off-plan during the original overnight — it turned out to be
exactly what this follow-up experiment needed).

See PRACTICALITIES §21 for full numbers and the mechanism discussion.
