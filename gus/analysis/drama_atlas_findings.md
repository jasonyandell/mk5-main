# §22 Drama Atlas — Findings

**Generated:** 2026-04-22  
**Corpus:** 10k train games (280,000 decisions) + 20 eval games (560 decisions)  
**Method:** Pure analytics over existing `q_per_world [M, 7]` tensors — no new model training  
**Time:** 703 seconds to compute (belief head inference on CPU + oracle tensor aggregation)

---

## The Reframe

Gus's belief head is at the Bayes ceiling: 39.2% top-1 accuracy matches the theoretical maximum achievable from oracle-sampled worlds (§21). Belief is solved. The remaining 42-craft is not about *knowing better* — it is about *acting well under unresolvable uncertainty*.

This document reports the first systematic measurement of where that uncertainty lives in the game.

---

## Three Quantities

Each of the 280,560 corpus decisions was scored on three axes, all computable from the existing `q_per_world [M, 7]` tensors without any new oracle calls:

1. **Outcome variance** — `q_per_world[:, action_taken].std()` over M ≈ 4000 worlds. Mean: **13.1 Q-pts**, median: **14.4**, p99: **29.1**.

2. **Action fragility** — count of distinct oracle-best actions across M worlds. Distribution: 1 (oracle agrees completely): **40.3%** of decisions; 2: **21.3%**; 3-7 (genuine fog-of-war): **38.4%**. Mean fragility: **2.59**.

3. **Belief sharpness** — `1 − H(P_belief) / log(3)` averaged over unseen dominoes. Mean: **0.064**, median: **0.020**. Belief stays near-uniform (sharpness ≈ 0) through d_idx 20, then spikes only in the final 2-3 tricks when played dominoes give near-deterministic inference.

---

## Quadrant Analysis

Decisions classified using thresholds: foggy = belief_sharpness < 0.020 (train median); fragile = action_fragility ≥ 3.

| Quadrant | Description | Count | Fraction |
|---|---|---|---|
| **easy-robust** | Oracle agrees (fragility < 3), Gus has signal | 106,155 | **37.8%** |
| **foggy-forced** | Oracle agrees, but Gus is guessing | 66,727 | **23.8%** |
| **easy-high-stakes** | Oracle disagrees, Gus has signal | 34,121 | **12.2%** |
| **drama** | Oracle disagrees AND Gus is in the dark | 73,557 | **26.2%** |

Drama fraction: **26.2%** — more than one in four corpus decisions is a genuine fog-of-war call.

**Late game (d_idx ≥ 20): drama fraction = 0.0%.** By tricks 5-7, played dominoes collapse the belief space deterministically. The game's craft is entirely front-loaded.

---

## Where Does Drama Concentrate?

**Finding: the opening lead is the heart of 42.**

| Trick position | Drama fraction | Mean fragility |
|---|---|---|
| **Lead (1st play)** | **64.9%** | **3.97** |
| 2nd play | 14.6% | 2.18 |
| 3rd play | 12.7% | 2.17 |
| 4th play | 12.7% | 2.04 |

**62% of all drama decisions are lead decisions.** The player choosing what to play first — before any trick information is revealed — faces the maximum oracle disagreement and minimum belief signal.

This quantitatively confirms the human expert intuition: "the lead tells partner your hand." The first play is also the most informationally expensive decision. The oracle itself would choose from 3-7 different dominoes depending on which hidden world is true. The 4th player, by contrast, has seen three dominoes played this trick and faces a near-forced choice.

**Game phase:**
- Early (d_idx 0-11): 39.4% drama
- Mid (d_idx 12-19): 32.6% drama
- Late (d_idx 20-27): 0.0% drama

---

## Gus vs Oracle Mode

The central §22 question: is Gus a mode-player, or does it learn something subtler?

| Quadrant | Gus-mode agreement |
|---|---|
| easy-robust | **86.5%** |
| foggy-forced | **85.0%** |
| easy-high-stakes | 46.5% |
| **drama** | **52.6%** |
| **Overall** | **72.4%** |

**Gus is primarily a mode-player but degrades on drama decisions.** On easy decisions, 87% agreement with the oracle's marginal-mode. On drama decisions, only 53% — close to coin-flip.

**This is NOT evidence of learned meta-strategy.** The 35-point drop in agreement is explained by ambiguous training signal: when oracle worlds are split across 5-7 different best actions (as in the top drama decisions), the BCE loss from those training examples points in multiple directions. The policy head compresses this into a single output, and when the distribution is nearly flat, the compressed output drifts.

Concretely: the #1 drama decision has oracle worlds split 33/24/16/13/11% across five actions. The BCE label is the marginal argmax (33%), but the true distribution is nearly uniform. Gus outputs action_0 (the mode, matching 33% worlds) — correct by the training criterion, but only marginally better than random among the top-4 options.

---

## Top Drama Decision: A Concrete Example

**Seed 900016, d_idx=0, player=0, decl=6:**
- Belief sharpness: 0.007 (near pure prior — first play of the game, no information)
- Action fragility: 7 (ALL 7 legal actions are oracle-best in some world)
- Outcome variance: 27.9 Q-pts (the taken action's outcome swings ±28 points by world)
- Oracle world distribution: action_0 in 33%, action_1 in 24%, action_6 in 16%, action_3 in 13%, action_2 in 11%, action_4 in 3%, action_5 in 0%
- Gus chose: action_0 (the mode)
- Action taken in game: action_6
- e_q_taken = -10.76, e_q_max = -6.43 (the game's play was suboptimal by 4.3 Q-pts)

This is the prototypical "what you do past belief" moment: you're leading trick 1, you have perfect knowledge of your own hand but no information about opponents, and the oracle says "it depends on which world you're in" with no clear winner. Human players use signals, partner conventions, and suit hierarchy heuristics here. Gus uses the marginal mode. Neither is obviously right.

---

## The Research Implication

The drama atlas surfaces exactly where meta-strategy choice matters. These 73,557 decisions (26% of the corpus) are the ones where the oracle itself has no consensus answer — the right play genuinely depends on unobservable hidden information.

The next step from this analysis:
1. Label the drama decisions by the shape of the oracle world distribution (mode-dominant vs. flat vs. bimodal).
2. For the flat/bimodal cases, compute what a hedge player would choose (minimax over worlds) and what a signal player would choose (action whose identity conveys the most information to partner).
3. Use these as multi-target training labels for a richer student that outputs a distribution over meta-strategies, not just a mode.

No LAMIR/CFR+ required — just the oracle per-world data we already have, relabeled with multiple meta-strategy targets.

---

## Surprising Finding

The most unexpected result: **the opening lead (d_idx=0) is always maximum fragility (7 distinct oracle choices) across all 10k games.** The very first decision of every game is, by oracle measure, the hardest decision: all legal actions are plausible in some world, and no belief signal is available. Human 42 teaching confirms this — "your opening lead is your most important communication to partner" — but we now have quantitative backing from 10k games: the oracle disagrees with itself maximally at exactly this moment.

This also explains why the first trick lead is where signaling systems are most valuable: it is the single decision where the oracle's mode has the smallest world-support (often 30-35%), meaning any systematic non-mode play carries maximal information value.

---

## Artifacts

| File | Contents |
|---|---|
| `gus/analysis/drama_atlas.parquet` | 280,560 rows, 22 columns (split, seed, drama quantities, Gus outputs, quadrant, drama_score) |
| `gus/analysis/drama_atlas.ipynb` | Full analysis notebook |
| `gus/analysis/build_drama_atlas.py` | Compute script (703s on CPU) |
| `gus/analysis/figures/drama_hero_scatter.png` | 3D scatter hero figure (eval, 560 decisions) |
| `gus/analysis/figures/drama_quadrant_by_didx.png` | Quadrant × game timeline |
| `gus/analysis/figures/drama_trick_position.png` | Drama by trick position (lead dominates) |
| `gus/analysis/figures/drama_trajectory.png` | Belief sharpness + fragility over game |
| `gus/analysis/figures/drama_gus_vs_mode.png` | Gus vs oracle mode by quadrant |
| `gus/analysis/figures/drama_score_distribution.png` | Drama score histogram |
| `gus/analysis/tables/drama_summary_by_split.csv` | Mean/median/p99 per split |
| `gus/analysis/tables/quadrant_summary.csv` | Per-quadrant statistics |
| `gus/analysis/tables/quadrant_deep_analysis.csv` | Extended quadrant analysis |
| `gus/analysis/tables/top20_drama_eval.csv` | Top 20 drama decisions (eval, full context) |
| `gus/analysis/tables/top20_drama_train.csv` | Top 20 drama decisions (train) |
| `scratch/drama_atlas/events.jsonl` | Build log |
| `scratch/drama_atlas/thoughts/` | Design decisions and findings notes |
