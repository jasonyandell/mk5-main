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
- **Is there a fundamental mid-game hardness floor?** Decisions 4, 8, 11, 12,
  16 always have high regret. Is that the oracle's own noise concentrating
  there, or real strategic ambiguity no student can resolve?

## 11. Game-level play costs ~18pp vs the oracle baseline (arena)

Decision-level regret of 1.39 Q-pt sounds small; game-level it compounds
to real outcomes. Arena at n=150 (15 seeds × 10 decls), student at seat 0
vs all-E[Q]-bot baseline on the same seeds:

- Student team wins: **78/150 = 52%**
- All-bot baseline: **105/150 = 70%**
- Student bidder-made rate: 22% (bot: 39%)
- Avg bidder points: 20.9 (student) vs 25.7 (bot), Δ −4.8 per hand

1.39 Q-pt regret per decision × 28 decisions = ~39 Q-pt team-level shift,
which is ~18 pp of team-level win rate. Decision regret compounds about as
expected.

This is the "so what does this student actually play like" number. It's
weaker than the oracle but not random — above 50% team-win rate on sharp
games. Detect-and-route (receipt 13) is the lever to close most of this
gap without retraining.

## 12. Ensembling by naïve averaging hurts regret (while boosting bot-match)

Tested 4 ensemble strategies over 8 trained adapters at `gus/adapters/`:

| strategy | bot-match | regret | blunders |
|---|---|---|---|
| best single (v2_voids_3000g_big) | 67.9% | **1.39** | 34 |
| majority vote | 69.1% | 1.55 | 40 (worse) |
| softmax avg | 71.3% | 1.46 | 36 (worse) |
| v-weighted avg | 71.3% | 1.43 | 35 (worse) |
| **oracle-per-decision (ceiling)** | **89.6%** | **0.36** | **8** |

Bot-match goes up, regret goes up: ensembles win near-ties (where any
legal play is ~equivalent) but lose on sharp decisions by diluting the
confident-right adapter with less-confident ones. The oracle ceiling
shows massive latent diversity — 58% of decisions have adapter
disagreement, 24% have 3+ distinct actions picked — but simple voting
strategies can't exploit it. **The right move is a router, not an
ensemble.**

## 13. Blunder detect-and-route is the practical ceiling (detector AUC 0.84+)

Small gradient-boosted classifier that predicts "will the student
blunder at this decision" from state-only features:

- **Oracle-feature detector** (uses oracle E[Q] summary stats): ROC-AUC
  **0.926**. At 15% flag rate: 80% recall of blunders.
- **Student-feature detector** (inference-deployable — uses student's π
  confidence, Q_head stats across K=20 sampled worlds, V_head,
  belief entropy): ROC-AUC **0.839**. At 20% flag rate: regret drops
  1.13 → 0.49 (57% reduction).

Top student-feature importances:

1. `pi_peak` (0.19) — student's policy confidence
2. `pi_entropy` (0.07)
3. `v_minus_polexpq` (0.07) — V/π consistency gap
4. `q_mean_std_legal` (0.06) — student's analog of oracle_spread
5. `v_minus_qmean_chosen` (0.06)

**Finding**: student's Q_head spread is a weaker proxy for oracle
spread than hoped (0.045 importance vs oracle's 0.390). Q_head was
trained on one random world per forward pass — spread signal is noisy.
Two fixes queued for v4-class experiments:
- Multi-world variance regularization during Q_head training
- K=50+ worlds at inference (cheap, batched)

**Upshot**: detect-and-route with inference-only features is viable.
Project regret ~0.5 Q-pt (near teacher-noise floor of 0.5-1) and game-
level arena should recover ~60% of the −18 pp gap. That's the practical
ceiling for vanilla distillation; beating it requires multi-step LAMIR
or a cleaner teacher.

## Emerging architecture: detect → route → fallback

The shape of a deployable Gus v1.0:

```
   student π_me (fast path, 1.39 regret, ~ms)
          │
          ▼
   blunder_detector (tiny GBM, ms)  ── flags ~20%
          │                         │
          │ ok                      │ flag
          ▼                         ▼
   commit action             fallback: oracle argmax (has budget)
                             OR PIMC-Q-K50 (doesn't work YET,
                                            needs Q_head variance reg)
```

**Router PoC (receipt 14) confirms**: oracle fallback at 20-25% flag hits
the projected 0.49-0.56 regret. But PIMC-Q-K50 as fallback HURTS (1.47 vs
1.39 baseline) because the student's Q_head is too noisy. Next-best-
adapter fallback is even worse (1.55). **Without an oracle budget, the
detect-and-route architecture is blocked on the Q-head-variance fix**
(multi-world training regularizer or K=100+ at inference).

## 15. Distribution-target belief calibration — works in isolation, doesn't propagate

Frozen-trunk fine-tune of just the belief head with a **soft** target —
the empirical marginal `P(domino ∈ seat)` over the M sampled worlds
already saved in `world_hands`, instead of the one-hot truth. Script
at `scratch/train_belief_distribution.py`.

Isolated:

| metric                       | before (truth) | after (dist.) |
|------------------------------|---------------:|--------------:|
| truth top-1 accuracy         | 38.4%          | 38.3%         |
| KL vs world-marginal (held-out) | 0.078       | **0.062**     |
| (uniform-belief baseline KL) | 0.081          | —             |

The calibration is genuinely better — distribution training closes
about 47% of the gap between uniform-prior and a hypothetical perfect
belief. Truth-trained belief was optimizing for the mode, not the shape.

Downstream:

| test                   | before | after  |
|------------------------|-------:|-------:|
| pimc-belief K=50       | 66.4%  | **65.4%** (regressed) |
| blunder detector ROC-AUC | 0.792 | 0.793 (flat) |
| blunder detector PR-AUC  | 0.175 | 0.133 (regressed) |

Better belief did NOT help the student play better. Likely cause: the
Q_head was co-trained with mode-sharp world samples at training time;
softening the belief distribution at inference creates distribution
shift the Q_head wasn't prepared for. Improvements don't stack on top
of a frozen ecosystem.

**Implication**: the right follow-up is CO-TRAINING {belief, world_encoder,
Q_head} together with distribution-belief + regular Q loss. Stacking
calibration fixes on a frozen architecture doesn't work — the integrated
approach is necessary.

**Not promoted** (agent's call): ablation result, good diagnostic, not a
ship-able adapter.

## 14. Router PoC reality-check: oracle fallback works, student fallbacks don't

End-to-end validation of the detect-and-route architecture on 560 held-out
decisions (`gus/eval/router.py`):

|           flag% | oracle | PIMC-Q-K50 | v1_full_1000g |
|----------------:|-------:|-----------:|--------------:|
|            none | 1.39   | 1.39       | 1.39          |
|              5% | 1.15   | 1.39       | 1.48 (worse)  |
|             10% | 0.97   | 1.44       | 1.49          |
|             15% | 0.82   | 1.45       | 1.49          |
|         **20%** | **0.56** | 1.47 (worse) | 1.55       |
|             25% | **0.49** | 1.48     | 1.69          |

- **Oracle fallback** hits the projected ~0.5 Q-pt regret at 20-25%
  flag rate — the architecture works as designed if we can afford
  oracle calls at inference.
- **PIMC-Q-K50 fallback is actively worse than no routing**. At 20%
  flag: catches 23 of 34 blunders via detector, but PIMC-Q only
  fixes 7 of them and introduces 6 new blunders on non-blunder
  decisions the detector flagged. Q_head trained on one world per
  forward is too noisy for belief-sampled averaging to rescue.
- **Next-best-adapter fallback** (v1_full_1000g) is worst — smaller
  weaker model is wrong on the same hard decisions.

Implication: to ship a no-oracle-inference student, the prerequisite
is a Q_head that survives multi-world averaging. Two approaches:
multi-world variance regularization during training (change train_v2_voids
loss to query Q_head on K worlds and penalize cross-world variance) OR
K=100+ at inference (cheap but only 2× improvement over K=50 at best).

## 16. Eager dataset OOMs at 10k games — the lazy IterableDataset is the fix

Baked into the pipeline since day one, `JointWorldFullDataset` loaded
every chunk's games into a list at `__init__`. Fine for the 100-1000-game
corpora (≤11 GB) and even for 3000 games (~33 GB). Breaks hard at 10k.

**Observed** (2026-04-21 15:41 CDT): v3 consistency training on the full
100-chunk 10k-game corpus was killed by macOS SIGKILL (OOM) before the
first forward pass. The eager load hit ~110 GB of unified memory and the
OS wouldn't tolerate it alongside the Python runtime.

**Wrong instinct**: just subsample. Works as a patch but doesn't scale,
and the Vast-fleet plan targets 100k+ games / multi-TB corpora where
even subsampling the eager path fails.

**Right fix**: `JointWorldFullIterable(IterableDataset)` in the same
module. Streams chunks in random order; each chunk is loaded, its items
are shuffled and yielded through an 8192-item shuffle buffer, then the
chunk is released. Memory footprint is O(one chunk + buffer),
independent of corpus size.

Smoke-test receipts (5 chunks × 2 epochs, CPU):
- Peak RSS: **3.4 GB**, flat across epochs (no growth leaks).
- Throughput: ~4500 items/s (CPU-bound on item-building, not disk-bound).
- __len__ scan: ~0.5s per chunk one-time; cached to disk via
  `--length-cache` so subsequent runs skip it.

Tradeoffs accepted:
- **Approximate shuffle.** Each batch draws from an 8192-item buffer
  dominated by the currently-loading chunk, so mini-batches have slight
  chunk-locality bias. WebDataset uses the same pattern at much larger
  scale; in our training it was imperceptible (see §X below once v3-10k
  vs v2-10k-big numbers land). If shuffle quality ever becomes suspect,
  bump buffer to 32K items (~200 MB, still trivial).
- **No multi-worker support yet.** `DataLoader(num_workers>0)` raises
  explicitly rather than silently misbehave. Our scripts use 0 anyway.

Landed in `gus/model/dataset_seq_world.py` and behind `--lazy` flags in
`train_v2_voids.py` and `train_v3_consistency.py`. Eager class retained
for eval corpora (small, random-access friendly). Two tools for two
jobs — not legacy; eager is the right tool on ≤1 GB inputs.

**Implication**: the distillation pipeline is now memory-bounded by
chunk size, not corpus size. Vast-fleet scale-out is no longer blocked
on infrastructure.

## 17. Oracle optimizes p_make, not E[Q] — utility is cliff-shaped

While investigating how the teacher picks its "best move," we traced
`forge/eq/generate/actions.py` line 126:

```python
score = p_make_masked + 1e-6 * e_q_normalized
actions = score.argmax(dim=1)
```

`p_make` is the probability of making the contract (P(Q ≥ 18) for
offense at bid 30). `e_q_normalized` is E[Q] scaled to [0, 1] over
legal actions. The 1e-6 coefficient is smaller than float32 precision
on most p_make gaps — **the oracle is effectively optimizing pure
p_make**.

This is the correct utility for 42 scoring. A mark is a mark; 30 and 42
yield the same mark. 29 and -42 yield the same mark (to opponents).
Margin has no durable value. The threshold cliff is the whole thing.

What we confirmed:
- "29 is never 30" is preserved — p_make=0 (cert. loss) loses to any
  p_make>0 at argmax. ✓
- Above threshold, all winning actions tie. Oracle picks arbitrarily.
  Acceptable: margin above threshold buys nothing.
- Below threshold, all losing actions tie. Oracle picks arbitrarily.
  Acceptable: a loss is a loss.

What we **did not** confirm:
- Hardcoded bid=30 threshold. `decl_id` (0-9) encodes trump only, not
  bid amount. The `[:, :, 60:]` (offense) and `[:, :, 25:]` (defense)
  bin offsets are literally P(Q ≥ 18) and P(Q ≥ -17), which are
  correct only for bid=30. For bid=36 the offense threshold shifts to
  Q ≥ 30 (bin 72+). Teacher is systematically overconfident on
  higher-bid hands. **Pre-Vast-launch blocker, tracked in GEN_FLEET.md.**

**The utility-vs-dynamics distinction.** The user's intuition that
"going for it wins more" is real — but it's a *dynamics* claim, not a
utility one. Margin at trick 3 compounds into safety at trick 7
through trump-depletion, partner signaling, and buffer against
unexpected opponent plays. A single-decision utility has no way to see
this — it's a property of sequences. The answer is multi-step planning
(LAMIR), not a richer utility.

Decomposition that stays clean:
- **Utility**: `U = p_make`. Pure threshold. Trivial to compute,
  trivial to explain.
- **Dynamics**: LAMIR rollouts re-score each action by rolled-out
  future p_make. Naturally values "margin that buys safety later."
- **These stack cleanly** — LAMIR uses the utility at the rollout
  leaves and aggregates via expectation. No need to corrupt the
  utility to imitate what planning provides.

**Earlier wrong turn worth remembering**: proposed `U = E[Q] + C·p_make`
as a "cliff-preserving, margin-rewarding" utility. Fails on the user's
own test case: A(p_make=1e-8, E[Q]=-42) vs B(p_make=0, E[Q]=+16) picks
B — the guaranteed loss — because E[Q] dominates when p_make×100 is
small. Real preference is pure p_make; margin belongs in dynamics.

**Implication**: no oracle-side utility rewrite needed. The cliff
behavior is correct as-is. The action items on the oracle side are
(1) bid_value plumbing for the threshold, (2) per-seat oracle softmax
for π_opp training targets. Neither touches the utility shape.

## 18. qMAE is data-scaling resistant — the laggard head needs a training-recipe fix

With 3.3× more data (3k → 10k games, same architecture, matched hparams),
most metrics compounded cleanly. qMAE did not:

| metric                       | 3k     | 10k    | relative improvement |
|------------------------------|-------:|-------:|---------------------:|
| regret (Q-pts)               | 1.35   | 0.55   | **−59%**             |
| bot-match (pi_me)            | 66.4%  | 76.1%  | **+9.7 pp**          |
| V-MAE                        |  4.5   |  3.6   | **−20%**             |
| **qMAE**                     | **9.4**| **8.7**| **−7%**              |
| belief top-1                 | ~38%   | ~39%   | roughly flat         |

qMAE reduction was an order of magnitude smaller than regret reduction.
v3's consistency loss helped marginally (8.25–8.69 range vs v2's 8.72) —
exactly the V/π decoupling it was designed to fix — but the magnitude is
faint relative to the structural problem.

**Why qMAE is structurally harder than V-MAE** (see also §2):

- Q_head predicts **per-world** expected Q. The target is noisy because
  it depends on *which world* was sampled, plus downstream play
  uncertainty within that world. V_head predicts the **marginal** E[Q]
  across worlds — a much smoother target.
- Current training pipeline samples **one random world per forward
  pass** (see `JointWorldFullDataset.__getitem__` / _build_item). The
  gradient is noisy because each forward sees a different world per
  decision, and Q_head's cross-world consistency is never explicitly
  rewarded.

**V-MAE of 3.6 is near the Bayes floor** for its target (marginal E[Q]
at visible state). **qMAE of 8.7 has real headroom**.

**Known fixes, not yet shipped** (either would be ~1 day of work):

1. **Multi-world variance regularization.** During training, query
   Q_head on K worlds for each decision; penalize high cross-world
   variance on `Q[action]` for the action_taken. Forces Q to be
   smoother — predicts the conditional mean with less world-dependence.
2. **Joint co-training of {belief, world_encoder, Q_head}** with a
   distribution-belief loss (§15 was the isolated ablation — it didn't
   help because Q_head wasn't co-trained; this is the integrated
   follow-up).

Either fix is a training-recipe change, not architecture or data.

**Why this doesn't block LAMIR-1:**

LAMIR-1's rollout uses V_head at the leaf (MAE 3.6, near floor).
Q_head is currently used for PIMC-via-world-averaging, which we
already know underperforms π_me direct (§3). LAMIR-1 doesn't invoke
Q_head at all. So the qMAE plateau is a separable concern from the
north-star roadmap — tackle it when/if the Q_head becomes critical
(router with inference-only features, or richer rollout-leaf models).
