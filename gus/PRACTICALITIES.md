# Gus — Practicalities

> Running receipts log. Appends what we actually learned while building the
> LAMIR-ready distillation pipeline. Written in the spirit of
> `burl/PRACTICALITIES.md`.

Read alongside `OVERVIEW.md` (vision), `BUILD_PLAN.md` (architecture + plan),
and the three `MORNING*_STATUS.md` snapshots (point-in-time results).

---

## 1. Regret is vs the oracle teacher, not the theoretical optimum

`regret = oracle_best_eq - eq[student_action]` — the distance between the
student's pick and the oracle's best legal action, measured in the oracle's
own E[Q] vector. Two implications:

- **A perfect student has regret 0**, which is in principle reachable — we're
  distilling, not playing a new game.
- **The oracle itself is noisy.** Stage-1 Q-network MAE is ~0.94, plus SEM=0.5
  from the per-decision adaptive sampling. The practical floor for student
  regret is ~0.5-1 Q-pt; below that we're chasing teacher noise.

So 1.39 mean regret at 3000g = ~2× the teacher-noise floor. Room to improve,
but don't expect zero.

**Do not** chase regret against an unknown optimum — that's a different project
(RL, self-play, game-theoretic solving).

## 2. The Q-MAE ≈ 10 number is not what it looks like

Q-head MAE ~10 Q-pts on a ±42 scale *sounds* bad. It isn't — it's measuring
per-world Q predictions against a target whose intrinsic variance is σ ≈ 20
(decision 0 empirically). Predicting per-world outcomes of a hidden-info game
from partial observations has an irreducible component that no amount of model
capacity can predict away.

Contrast:
- **Perfect-info oracle Q-MAE: 0.94.** Target is deterministic given full deal.
- **Imperfect-info Q-MAE: ~10.** Target is stochastic given visible state, and
  the randomness is what we're modeling through the belief head.

**V-MAE (~5 for the 3000g big) is the right "how good is the student's
positional understanding" number.** It predicts the marginal (averaged-over-
worlds) E[Q], which is a much smoother target. 5 Q-pt V-MAE on a 42 scale
is usable but has headroom.

**π_me regret (1.39 Q-pt) is the right "how well does the student play"
number.** Everything else is infrastructure.

## 3. Single-step PIMC underperforms direct π_me — strategy fusion

PIMC-via-Q_head (sample K worlds from belief, compute Q_head per world,
average, argmax) loses to the π_me head by 3-4 percentage points of bot-match
regardless of K (tested K=50, K=200, K=500). All saturate at ~62%.

Why: **strategy fusion.** PIMC averages over per-world "best actions" as if
the policy could condition on the world. Real policies have to pick one
action without knowing which world is true. Averaging per-world Qs gives a
score that no single committed policy can realize.

π_me is trained directly on the oracle's argmax(marginal E[Q]), which already
internalizes the anti-strategy-fusion correction. PIMC at inference can't add
it back — it literally undoes what π_me learned.

**Rule of thumb**: if you have a policy trained on marginalized-E[Q]
supervision, don't re-marginalize at inference. Just use the policy.

**Multi-step LAMIR is different** because each level of the search tree
commits to an action — no strategy fusion. But that requires a π_opp head
(not yet trained) and engine plumbing (deferred).

## 4. Data scaling dominates capacity at these corpus sizes

The clean result of the session:

| corpus | 1.2M params | 3.4M params | 7.4M params |
|---|---|---|---|
| 100g | 2.48 | — | — |
| 1000g | 2.16 | 2.10 | — |
| 2000g | — | 1.60 | 1.65 (XL flat) |
| 3000g | — | **1.39** | 1.54 (XL regressed) |

Reading:
- Doubling data reduced regret every time (100g → 1000g: −0.32; 1000g → 2000g:
  −0.56; 2000g → 3000g: −0.21).
- Going from 1.2M to 3.4M helped marginally at 1000g (−0.06 regret).
- Going from 3.4M to 7.4M actively hurt regret at 2000g and 3000g, despite
  composite score improving.

The 7.4M XL on 3000g gave the highest composite score (0.219 vs 0.208) but
WORSE regret (1.54 vs 1.39). The heavier regularization (dropout 0.15, WD
0.03) improved the auxiliary head MAEs that the composite rewards but
*blunted* the policy head.

**Lesson**: composite metrics can disagree with the metric that matters.
Regret > composite > bot-match > individual head accuracy.

## 5. The regret distribution is bimodal — 73% perfect, 6% blunder

For the 3000g_big adapter on 560 held-out decisions:

| regret bucket | count | % |
|---|---|---|
| optimal / dead-tie (<0.1 Q-pt) | 410 | 73.2% |
| near-tie (0.1-0.5) | 23 | 4.1% |
| minor slip (0.5-1) | 13 | 2.3% |
| modest miss (1-2) | 23 | 4.1% |
| real miss (2-4) | 30 | 5.4% |
| big miss (4-8) | 27 | 4.8% |
| **BLUNDER (8+)** | **34** | **6.1%** |

Percentiles: p50=**0** (median is perfect!), p90=4.4, p95=9.1, p99=20.1, max=29.1.

**1.39 mean regret is entirely driven by the 6% blunder tail.** If we could
halve the blunder rate without hurting the perfect rate, mean regret drops to
~0.9.

This is consistent with 42's strategic structure: most decisions are trivial
or near-tied, a few are make-or-break, and the student occasionally walks off
the cliff. Future work on blunder reduction should probably target these
specific decisions (identify via regret, augment training with additional
examples, or use LAMIR look-ahead as a safety net on high-entropy positions).

## 6. Explicit void features barely help — transformer infers voids from plays

Adding engine-computed void features (24-dim: 3 opponents × 8 suits, fed to
CLS position) moved belief top-1 from 37.2% → 38.6% at 1000g and left π_me
flat. The transformer was already learning void inference attentionally from
play tokens.

**Corollary**: explicit features are worth it when they encode information
the network CANNOT derive from inputs. If the derivation is straightforward
and the architecture is attention-based, the explicit version is cosmetic.

## 7. GPU contention on MPS is real but linear-ish

Running 10k-game gen concurrently with XL training on one M5 Max:
- Gen chunk time: 7 min (alone) → ~9 min (contended)
- XL epoch time: 22s (alone) → ~65s (contended)

Both cost ~30% wall overhead. Not free, but parallel scheduling still beats
sequential when total compute > 1 Apple unit. **Run them simultaneously when
wall-time pressure is low**; sequentialize when a single run needs to finish
fast.

## 8. Data-to-oracle training-set comparison

The Stage-1 Q oracle was trained on 11.24M tokenized states (1,124 seeds × 10
declarations, ~215 GB).

Gus's current corpus (3,800 games):
- **360M (state, world) training examples** — ~32× more in raw count
- **106K distinct info-states** — 100× FEWER than the oracle
- **3,800 seed/decl combinations** — 3× fewer than the oracle

We get more data per state (3,400 sampled worlds per decision) but much less
state diversity. At 10k games: 280K distinct states — still way below the
oracle's 11M.

**Unseen lever**: generate 10 declarations per seed instead of 1 (the oracle's
recipe). Would 10× state diversity at the same gen cost. Worth trying if
further data scaling plateaus.

## 9. The joint-world tensor is the pipeline's keystone

Saving `(world_hands, q_per_world)` per decision was a substantive
infrastructure lift (edits to `forge/eq/generate/{types,actions,pipeline,
adaptive,cli}.py`). Without it, the full 4-head student wouldn't be
possible — the Q_head's supervision signal lives entirely in that tensor.

**Storage is manageable**: ~6 MB per game at SEM<0.5, so 10k games ≈ 60 GB.
Gitignored, regeneratable.

**Opt-in**: `--save-joint-worlds` flag keeps the tensor off normal gen runs.

## 10. Timeline calibration

On M5 Max (MPS, solo run):
- 100-game adaptive gen at SEM<0.5: ~7 min
- 1000-game training (3.4M params, 60 epochs): ~10 min
- 3000-game training (3.4M params, 60 epochs): ~25 min
- 3000-game training (7.4M params, 80 epochs): ~70 min alone, ~90 min contended

The gen generator is the long pole for scaling experiments. Budget 7-10 min
per 100-game chunk of corpus expansion.

## Open questions (that came up, didn't block)

- **Does a better teacher exist?** Stage-1 oracle Q-MAE is 0.94. A higher-
  accuracy oracle (more solver samples, bigger Q network) would move the
  student's teacher-noise floor down.
- **Does distillation with the raw PDF help?** We distill on per-world Q
  scalars, not the 85-bin outcome PDF. The PDF is richer; might help Q-head
  calibration if targeted.
- **Are the 6% blunder decisions identifiable at inference?** If the student
  could flag them (high belief entropy? high Q variance across sampled
  worlds?) we could fall back to the oracle or LAMIR just for those.
- **Is there a fundamental mid-game hardness floor?** Decisions 4, 8, 11, 12,
  16 always have high regret. Is that the oracle's own noise concentrating
  there, or real strategic ambiguity no student can resolve?
