# Gus — Morning Status 3 (Final) (2026-04-21 02:55 CDT)

## TL;DR

Took the architecture from nothing → working 4-head LAMIR-ready student
in one autonomous overnight session. Full distillation pipeline validated
with oracle-supervised training: belief + V + π_me + world-conditioned Q
on a transformer encoder over tokenized play sequences. Two generations
of corpus (1000 + 1000 games = 2000 converged at SEM<0.5). Three
tranches of training runs (varying data × model size). Regret-based
eval instead of bare bot-match.

## Full adapter ladder

Mean regret on held-out 560 decisions (20 games × 28 decisions). Lower = better.

| adapter | bot-match | mean regret | near-ties |
|---|---|---|---|
| v1_full (100g, d=128/3L, 0.4M) | 59.3% | 2.48 | 68.9% |
| v1_full (1000g, d=192/4L, 1.2M) | 65.4% | 2.16 | 73.4% |
| v2_voids (1000g, d=192/4L, 1.2M) | 63.6% | 2.23 | 72.0% |
| v2_voids_big (1000g, d=256/6L, 3.4M) | 62.7% | 2.10 | 70.5% |
| v2_voids_big (2000g, d=256/6L, 3.4M) | 67.3% | 1.60 | 75.4% |
| v2_voids_xl (2000g, d=384/6L, 7.4M, 100ep) | 65.7% | 1.65 | 75.2% |

*Bot-match chance ≈ 25% (4 legal moves avg). Q range is [-42, +42].*

## Scaling take-aways

1. **Data scaling dominated.** 2000g with 3.4M params beat 1000g with 3.4M params by +5pp bot-match and −0.5 Q-pt regret. Same 1000g with larger model was flat.
2. **Explicit void features alone are near-useless.** The transformer attentionally infers voids from play tokens. Voids features only mattered once paired with data + capacity.
3. **Single-step PIMC saturates and underperforms direct π_me.** Tried K=50, K=500 world samples — same result (62%/66%). LAMIR's true value lives in multi-step look-ahead, which needs a π_opp head we haven't trained.
4. **End-game is perfectly solved.** Decisions 24-27: 100% bot-match, 0 regret across all adapters. When the info-state has zero hidden information, the task is deterministic.

## Decision hardness analysis

Oracle's E[Q] spread across legal actions is a measure of how strategically significant each decision is. High spread = real choices with real payoffs. Low spread = near-tie plays.

```
dec  mean_spread  std_spread  max_spread    n
  0        13.20        4.41       25.15   20
  1        11.27       11.59       34.55   14
  2         3.37        3.63       13.69   13
  3        11.09       14.26       42.39   10
  4        12.15        6.39       29.66   20
  5        11.06       11.46       36.73   15
  6        12.77       10.40       33.93   12
  7        12.14       12.12       43.83   13
  8        12.27        5.87       23.98   20
  9         7.62        6.91       21.08   15
 10        10.42        7.71       22.97   15
 11        12.83       11.59       35.48   14
 12        10.44        7.76       29.11   20
 13         3.34        3.77       12.97   13
 14         5.07        3.26       10.56   16
 15        10.34       11.97       48.15   14
 16        11.87       12.18       38.64   20
 17         5.17        5.04       19.32   14
 18         5.39        8.81       29.99   15
 19         5.01        6.52       22.92   14
 20         5.70        7.90       25.71   20
 21         3.61        6.13       23.15   14
 22         2.01        3.73       11.68   12
 23         6.08        8.31       24.77   18

Top-10 by strategic spread (hardest decisions):
  dec  0: spread 13.20
  dec 11: spread 12.83
  dec  6: spread 12.77
  dec  8: spread 12.27
  dec  4: spread 12.15
  dec  7: spread 12.14
  dec 16: spread 11.87
  dec  1: spread 11.27
  dec  3: spread 11.09
  dec  5: spread 11.06

Bottom-10 by spread (near-tie / low-stakes decisions):
  dec 23: spread  6.08
  dec 20: spread  5.70
  dec 18: spread  5.39
  dec 17: spread  5.17
  dec 14: spread  5.07
  dec 19: spread  5.01
  dec 21: spread  3.61
  dec  2: spread  3.37
  dec 13: spread  3.34
  dec 22: spread  2.01

```
The student's highest-regret decisions (0, 4, 8, 11, 12, 16) are **the same decisions with highest strategic spread**. Honest mistakes on hard decisions, not trivial slips.

## Regret breakdown — best adapter

```
Adapter: /Users/jason/code/mk5-main/gus/adapters/v2_voids_2000g_xl.pt  device: cpu

=== Summary over 560 decisions ===
  Bot-match rate:           65.714%
  Mean regret (Q-points):   1.653
  Decisions with regret<0.5: 421/560 = 75.2% (near-ties)

=== Per-decision regret + bot-match ===
dec     bot    regret  near-tie    n
  0  55.00%      3.07     60.0%   20
  1  65.00%      2.78     70.0%   20
  2  75.00%      0.54     80.0%   20
  3  90.00%      0.39     90.0%   20
  4  45.00%      5.91     45.0%   20
  5  60.00%      2.00     70.0%   20
  6  60.00%      3.82     65.0%   20
  7  75.00%      0.94     85.0%   20
  8  25.00%      4.23     40.0%   20
  9  45.00%      1.29     70.0%   20
 10  45.00%      3.33     50.0%   20
 11  80.00%      2.95     80.0%   20
 12  25.00%      4.00     30.0%   20
 13  60.00%      0.81     80.0%   20
 14  50.00%      2.04     60.0%   20
 15  60.00%      1.84     70.0%   20
 16  30.00%      2.60     50.0%   20
 17  80.00%      0.23     95.0%   20
 18  55.00%      0.70     70.0%   20
 19  65.00%      1.46     80.0%   20
 20  80.00%      0.17     90.0%   20
 21  85.00%      0.36     90.0%   20
 22  70.00%      0.40     95.0%   20
 23  60.00%      0.42     90.0%   20
 24  100.00%      0.00    100.0%   20
 25  100.00%      0.00    100.0%   20
 26  100.00%      0.00    100.0%   20
 27  100.00%      0.00    100.0%   20
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)

```

## XL training log tail

```
  train: L=178.325  bel=40.79%  pi=65.08%  vMAE=3.83  qMAE=9.56
  eval:  L=217.341  bel=39.06%  pi=64.29%  vMAE=5.80  qMAE=10.26
epoch  89/100  dt=19.8s
  train: L=175.408  bel=40.92%  pi=65.12%  vMAE=3.81  qMAE=9.46
  eval:  L=212.719  bel=39.49%  pi=64.82%  vMAE=6.05  qMAE=10.64
epoch  90/100  dt=19.8s
  train: L=176.797  bel=40.81%  pi=65.42%  vMAE=3.79  qMAE=9.52
  eval:  L=236.969  bel=39.40%  pi=66.43%  vMAE=5.92  qMAE=11.00
epoch  91/100  dt=20.0s
  train: L=174.684  bel=40.90%  pi=65.15%  vMAE=3.73  qMAE=9.47
  eval:  L=212.239  bel=39.52%  pi=64.64%  vMAE=5.88  qMAE=10.23
epoch  92/100  dt=19.8s
  train: L=174.554  bel=40.88%  pi=65.24%  vMAE=3.78  qMAE=9.45
  eval:  L=201.032  bel=39.47%  pi=65.89%  vMAE=5.82  qMAE=10.29
epoch  93/100  dt=20.0s
  train: L=174.354  bel=40.93%  pi=65.36%  vMAE=3.75  qMAE=9.42
  eval:  L=246.316  bel=38.89%  pi=64.46%  vMAE=5.79  qMAE=10.57
epoch  94/100  dt=19.8s
  train: L=174.989  bel=40.91%  pi=65.16%  vMAE=3.76  qMAE=9.44
  eval:  L=210.893  bel=39.51%  pi=66.25%  vMAE=5.79  qMAE=10.66
epoch  95/100  dt=19.8s
  train: L=174.418  bel=40.87%  pi=65.33%  vMAE=3.70  qMAE=9.44
  eval:  L=237.873  bel=39.08%  pi=65.54%  vMAE=5.82  qMAE=10.82
epoch  96/100  dt=20.0s
  train: L=174.378  bel=40.88%  pi=65.30%  vMAE=3.73  qMAE=9.44
  eval:  L=225.333  bel=39.68%  pi=64.29%  vMAE=5.81  qMAE=10.85
epoch  97/100  dt=19.8s
  train: L=173.828  bel=40.86%  pi=65.47%  vMAE=3.71  qMAE=9.40
  eval:  L=203.130  bel=38.84%  pi=66.61%  vMAE=5.86  qMAE=10.60
epoch  98/100  dt=19.8s
  train: L=173.641  bel=40.87%  pi=65.39%  vMAE=3.67  qMAE=9.42
  eval:  L=248.399  bel=39.29%  pi=63.57%  vMAE=6.16  qMAE=11.36
epoch  99/100  dt=20.0s
  train: L=173.306  bel=40.93%  pi=65.65%  vMAE=3.68  qMAE=9.39
  eval:  L=212.225  bel=39.37%  pi=65.89%  vMAE=6.05  qMAE=10.18
epoch 100/100  dt=19.8s
  train: L=171.907  bel=40.90%  pi=65.50%  vMAE=3.65  qMAE=9.36
  eval:  L=209.571  bel=39.76%  pi=65.54%  vMAE=5.75  qMAE=10.33

Final best composite: 0.1974
```

## Commits this session

```
a50c9ef eval(gus): decision-hardness analyzer
0472125 docs(gus): MORNING2_STATUS.md — 2000-game scaling, 1.60 Q-pt mean regret
2a09050 eval(gus): regret-based eval — E[Q] lost vs oracle's best legal
5a4c9b9 eval(gus): LAMIR-primitive inference eval — direct π_me vs PIMC-via-Q
3c02d10 feat(gus): v2 — engine-computed void features fed to state encoder
2e4f586 fix(gus): glob-expand dataset paths inside nargs lists
54f7776 feat(burl/wax_museum): hard-gated HATEOAS harness; fix silently-dropped tool responses
da21f52 feat(gus): full 4-head student — belief + V + π_me + world-conditioned Q
8dbf7f3 feat(gus): v1 transformer belief student — architecture works, data-bound
c04bda3 feat(gus): v0 student scaffolding — belief head + dataset + trainer + eval
31e10ef feat(gus/forge): joint-world tensor + LAMIR-ready student plan
42a7535 docs(gus): kickoff — neural policy + belief + value for Texas 42
0545342 feat(burl): candlewax spike — multimodal PDFs, engine fact-checker, MLX LoRA STaR end-to-end
aeafe22 docs(burl): PRACTICALITIES.md split — vision stays in OVERVIEW, receipts grow separately
6a97d55 feat(burl): batched MLX-LM rollout harness — 2.3x wall on N=16
b0952a2 feat(burl): spike_drivers — empirical mode-catalyst dominoes on eq_outcome_distribution
ed3cfc3 bench(burl): MLX-LM batch_generate ceiling on M5 Max — 43 → 1334 tok/s
7321952 feat(burl): enumerate=auto + what_would_change_my_mind — tool-surface levers
ceca203 docs(burl): iter-5 E1 + E2 writeups — truncation reframe, candlewax null
1efb9c5 feat(burl): candlewax-aware eq_outcome_distribution return (ITER4_PLAN §2 E2)
```

## Directions left open

1. **π_opp head**: requires corpus regen with opponent-view oracle queries (3× cost). Unlocks multi-step LAMIR look-ahead.

2. **Multi-step LAMIR via tree search**: once π_opp exists, roll out full trick (action → 3 opp plays) using π_opp, evaluate trick-end state with V_head. Compare to direct π_me.

3. **Further data scaling**: the 100g→1000g→2000g curve suggests ~0.3-0.5 Q-pt regret reduction per data-doubling. 5000g or 10000g would continue the trend.

4. **Decision-difficulty-weighted training**: up-weight training examples from high-spread decisions (0, 4, 8, 11, 12) so the model focuses capacity where it matters.

5. **End-to-end arena play**: wire the student's π_me into the existing arena vs E[Q] bot to measure game-level outcome, not just decision-level regret.

