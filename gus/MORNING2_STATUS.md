# Gus — Morning Status 2 (2026-04-21 02:18 CDT)

## TL;DR — overnight-2 run

Built on the previous night's results. Major additions:
- Engine-computed void features (`gus/model/voids.py`) + new student v2
- Regret-based eval (`gus/eval/eval_regret.py`) — measures E[Q]-points lost by the student's choice vs oracle's best legal action. More meaningful than bot-match because many decisions have near-tie alternatives.
- LAMIR-primitive inference comparison: π_me vs PIMC-via-Q_head (`gus/eval/eval_pimc.py`) — single-step PIMC underperforms direct π_me; multi-step look-ahead is the real LAMIR lever but needs a π_opp head.
- (When available) larger model trained on 2000-game corpus.

## Adapters and regret

Mean regret on held-out 560 decisions (20 games × 28 decisions). Lower = better.

**Key metric**: Mean regret (Q-points lost per decision). Q range is [-42, +42].

| adapter | bot-match | mean regret | near-ties (<0.5pt) |
|---|---|---|---|
| v1_full (100g, d=128/3L) | 59.3% | 2.48 | 68.9% |
| v1_full (1000g, d=192/4L) | 65.4% | 2.16 | 73.4% |
| v2_voids (1000g, d=192/4L) | 63.6% | 2.23 | 72.0% |
| v2_voids_big (1000g, d=256/6L) | 62.7% | 2.10 | 70.5% |
| v2_voids_big (2000g, d=256/6L) | 67.3% | 1.60 | 75.4% |

## Regret breakdown — v1_full (100g, d=128/3L)

```
Adapter: /Users/jason/code/mk5-main/gus/adapters/v1_full_100g.pt  device: cpu

=== Summary over 560 decisions ===
  Bot-match rate:           59.286%
  Mean regret (Q-points):   2.479
  Decisions with regret<0.5: 386/560 = 68.9% (near-ties)

=== Per-decision regret + bot-match ===
dec     bot    regret  near-tie    n
  0  35.00%      4.06     40.0%   20
  1  70.00%      1.99     70.0%   20
  2  75.00%      0.21     90.0%   20
  3  65.00%      0.97     70.0%   20
  4  30.00%      6.20     35.0%   20
  5  65.00%      2.66     70.0%   20
  6  55.00%      4.70     60.0%   20
  7  55.00%      3.19     60.0%   20
  8  20.00%      5.55     30.0%   20
  9  50.00%      2.93     60.0%   20
 10  45.00%      3.52     50.0%   20
 11  30.00%      5.79     45.0%   20
 12  30.00%      5.39     45.0%   20
 13  60.00%      1.63     70.0%   20
 14  55.00%      1.55     60.0%   20
 15  45.00%      5.55     50.0%   20
 16  50.00%      0.84     70.0%   20
 17  60.00%      1.40     80.0%   20
 18  40.00%      3.20     65.0%   20
 19  55.00%      1.99     70.0%   20
 20  65.00%      1.40     90.0%   20
 21  70.00%      1.69     75.0%   20
 22  80.00%      0.01    100.0%   20
 23  55.00%      2.98     75.0%   20
 24  100.00%      0.00    100.0%   20
 25  100.00%      0.00    100.0%   20
 26  100.00%      0.00    100.0%   20
 27  100.00%      0.00    100.0%   20
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)

```

## Regret breakdown — v1_full (1000g, d=192/4L)

```
Adapter: /Users/jason/code/mk5-main/gus/adapters/v1_full_1000g.pt  device: cpu

=== Summary over 560 decisions ===
  Bot-match rate:           65.357%
  Mean regret (Q-points):   2.160
  Decisions with regret<0.5: 411/560 = 73.4% (near-ties)

=== Per-decision regret + bot-match ===
dec     bot    regret  near-tie    n
  0  45.00%      5.24     50.0%   20
  1  75.00%      0.64     80.0%   20
  2  65.00%      1.34     75.0%   20
  3  90.00%      0.09     90.0%   20
  4  25.00%      8.64     30.0%   20
  5  55.00%      3.09     65.0%   20
  6  60.00%      3.83     70.0%   20
  7  80.00%      3.47     80.0%   20
  8  30.00%      4.12     35.0%   20
  9  55.00%      1.98     70.0%   20
 10  65.00%      1.44     75.0%   20
 11  75.00%      4.87     75.0%   20
 12  30.00%      3.40     30.0%   20
 13  75.00%      0.68     85.0%   20
 14  50.00%      1.43     70.0%   20
 15  55.00%      4.20     60.0%   20
 16  35.00%      3.14     55.0%   20
 17  75.00%      0.54     90.0%   20
 18  45.00%      2.37     55.0%   20
 19  55.00%      2.43     70.0%   20
 20  65.00%      1.96     75.0%   20
 21  70.00%      0.91     80.0%   20
 22  75.00%      0.40     95.0%   20
 23  80.00%      0.28     95.0%   20
 24  100.00%      0.00    100.0%   20
 25  100.00%      0.00    100.0%   20
 26  100.00%      0.00    100.0%   20
 27  100.00%      0.00    100.0%   20
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)

```

## Regret breakdown — v2_voids (1000g, d=192/4L)

```
Adapter: /Users/jason/code/mk5-main/gus/adapters/v2_voids_1000g.pt  device: cpu

=== Summary over 560 decisions ===
  Bot-match rate:           63.571%
  Mean regret (Q-points):   2.225
  Decisions with regret<0.5: 403/560 = 72.0% (near-ties)

=== Per-decision regret + bot-match ===
dec     bot    regret  near-tie    n
  0  40.00%      5.19     40.0%   20
  1  75.00%      1.62     75.0%   20
  2  60.00%      1.34     75.0%   20
  3  80.00%      1.00     80.0%   20
  4  60.00%      4.14     60.0%   20
  5  65.00%      3.78     70.0%   20
  6  80.00%      0.66     85.0%   20
  7  75.00%      1.94     80.0%   20
  8  25.00%      4.89     30.0%   20
  9  40.00%      2.18     55.0%   20
 10  45.00%      2.40     60.0%   20
 11  60.00%      4.63     60.0%   20
 12  20.00%      5.34     30.0%   20
 13  70.00%      0.70     85.0%   20
 14  70.00%      1.06     70.0%   20
 15  55.00%      2.26     65.0%   20
 16  20.00%      7.09     40.0%   20
 17  60.00%      1.07     85.0%   20
 18  50.00%      3.56     65.0%   20
 19  55.00%      1.95     75.0%   20
 20  55.00%      1.67     75.0%   20
 21  70.00%      1.71     75.0%   20
 22  75.00%      0.40     95.0%   20
 23  75.00%      1.73     85.0%   20
 24  100.00%      0.00    100.0%   20
 25  100.00%      0.00    100.0%   20
 26  100.00%      0.00    100.0%   20
 27  100.00%      0.00    100.0%   20
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)

```

## Regret breakdown — v2_voids_big (1000g, d=256/6L)

```
Adapter: /Users/jason/code/mk5-main/gus/adapters/v2_voids_1000g_big.pt  device: cpu

=== Summary over 560 decisions ===
  Bot-match rate:           62.679%
  Mean regret (Q-points):   2.103
  Decisions with regret<0.5: 395/560 = 70.5% (near-ties)

=== Per-decision regret + bot-match ===
dec     bot    regret  near-tie    n
  0  45.00%      4.63     45.0%   20
  1  70.00%      1.77     70.0%   20
  2  50.00%      1.45     65.0%   20
  3  85.00%      0.14     85.0%   20
  4  30.00%      7.53     30.0%   20
  5  60.00%      4.83     70.0%   20
  6  65.00%      4.13     70.0%   20
  7  80.00%      0.27     90.0%   20
  8  35.00%      3.68     45.0%   20
  9  40.00%      2.20     55.0%   20
 10  55.00%      2.48     65.0%   20
 11  45.00%      3.14     45.0%   20
 12  30.00%      3.65     35.0%   20
 13  75.00%      0.84     80.0%   20
 14  60.00%      1.42     70.0%   20
 15  60.00%      1.35     75.0%   20
 16  35.00%      2.42     55.0%   20
 17  75.00%      0.69     90.0%   20
 18  40.00%      2.72     50.0%   20
 19  55.00%      2.16     70.0%   20
 20  70.00%      2.22     80.0%   20
 21  60.00%      2.16     70.0%   20
 22  65.00%      1.19     85.0%   20
 23  70.00%      1.79     80.0%   20
 24  100.00%      0.00    100.0%   20
 25  100.00%      0.00    100.0%   20
 26  100.00%      0.00    100.0%   20
 27  100.00%      0.00    100.0%   20
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)

```

## Regret breakdown — v2_voids_big (2000g, d=256/6L)

```
Adapter: /Users/jason/code/mk5-main/gus/adapters/v2_voids_2000g_big.pt  device: cpu

=== Summary over 560 decisions ===
  Bot-match rate:           67.321%
  Mean regret (Q-points):   1.597
  Decisions with regret<0.5: 422/560 = 75.4% (near-ties)

=== Per-decision regret + bot-match ===
dec     bot    regret  near-tie    n
  0  55.00%      3.99     55.0%   20
  1  70.00%      1.96     70.0%   20
  2  55.00%      1.43     65.0%   20
  3  95.00%      0.35     95.0%   20
  4  50.00%      5.66     50.0%   20
  5  60.00%      2.72     70.0%   20
  6  65.00%      2.44     70.0%   20
  7  85.00%      0.41     95.0%   20
  8  20.00%      4.84     25.0%   20
  9  50.00%      1.61     75.0%   20
 10  70.00%      1.70     75.0%   20
 11  75.00%      2.21     75.0%   20
 12  30.00%      3.74     35.0%   20
 13  65.00%      0.88     75.0%   20
 14  60.00%      0.62     75.0%   20
 15  65.00%      0.83     75.0%   20
 16  45.00%      2.69     50.0%   20
 17  75.00%      0.85     90.0%   20
 18  60.00%      0.71     70.0%   20
 19  70.00%      2.07     80.0%   20
 20  60.00%      1.56     75.0%   20
 21  70.00%      0.24     85.0%   20
 22  65.00%      0.61     90.0%   20
 23  70.00%      0.60     90.0%   20
 24  100.00%      0.00    100.0%   20
 25  100.00%      0.00    100.0%   20
 26  100.00%      0.00    100.0%   20
 27  100.00%      0.00    100.0%   20
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)

```

## LAMIR-primitive inference (direct π_me vs PIMC-via-Q)

### v1_full 1000g
```
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
Adapter: gus/adapters/v1_full_1000g.pt  voids=False
Eval decisions: 560

=== Overall bot-match on 560 decisions ===
        direct: 366/560 = 65.357%
        pimc-q: 348/560 = 62.143%
   pimc-belief: 346/560 = 61.786%

=== Disagreement (direct, pimc-q, pimc-belief) → count ===
  (1, 1, 1): 290
  (0, 0, 0): 143
  (1, 0, 0): 44
  (0, 1, 1): 31
  (1, 0, 1): 17
  (1, 1, 0): 15
  (0, 1, 0): 12
  (0, 0, 1): 8

=== Per-decision bot-match ===
dec  direct  pimc-q  pimc-bel     n
  0  40.00%  40.00%    40.00%    20
  1  70.00%  75.00%    75.00%    20
  2  60.00%  60.00%    60.00%    20
  3  90.00%  95.00%    95.00%    20
  4  25.00%  25.00%    20.00%    20
  5  55.00%  55.00%    45.00%    20
  6  65.00%  75.00%    60.00%    20
  7  75.00%  75.00%    70.00%    20
  8  30.00%  30.00%    30.00%    20
  9  50.00%  45.00%    45.00%    20
 10  70.00%  40.00%    40.00%    20
 11  70.00%  65.00%    65.00%    20
 12  20.00%  25.00%    35.00%    20
 13  80.00%  55.00%    65.00%    20
 14  45.00%  55.00%    65.00%    20
 15  50.00%  70.00%    75.00%    20
 16  55.00%  25.00%    20.00%    20
 17  85.00%  65.00%    60.00%    20
 18  45.00%  45.00%    35.00%    20
 19  60.00%  60.00%    50.00%    20
 20  65.00%  65.00%    75.00%    20
 21  70.00%  60.00%    70.00%    20
 22  75.00%  70.00%    75.00%    20
 23  80.00%  65.00%    60.00%    20
 24  100.00%  100.00%   100.00%    20
 25  100.00%  100.00%   100.00%    20
 26  100.00%  100.00%   100.00%    20
 27  100.00%  100.00%   100.00%    20
```

### v2_voids_big 2000g
```
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
Adapter: gus/adapters/v2_voids_2000g_big.pt  voids=True
Eval decisions: 560

=== Overall bot-match on 560 decisions ===
        direct: 368/560 = 65.714%
        pimc-q: 350/560 = 62.500%
   pimc-belief: 349/560 = 62.321%

=== Disagreement (direct, pimc-q, pimc-belief) → count ===
  (1, 1, 1): 303
  (0, 0, 0): 147
  (1, 0, 0): 39
  (0, 1, 1): 22
  (1, 0, 1): 17
  (0, 1, 0): 16
  (1, 1, 0): 9
  (0, 0, 1): 7

=== Per-decision bot-match ===
dec  direct  pimc-q  pimc-bel     n
  0  35.00%  40.00%    35.00%    20
  1  70.00%  70.00%    80.00%    20
  2  45.00%  55.00%    50.00%    20
  3  95.00%  85.00%    95.00%    20
  4  45.00%  30.00%    35.00%    20
  5  65.00%  65.00%    60.00%    20
  6  60.00%  60.00%    55.00%    20
  7  85.00%  90.00%    90.00%    20
  8  25.00%  15.00%    25.00%    20
  9  65.00%  50.00%    55.00%    20
 10  55.00%  35.00%    40.00%    20
 11  70.00%  60.00%    70.00%    20
 12  35.00%  35.00%    30.00%    20
 13  70.00%  70.00%    65.00%    20
 14  50.00%  75.00%    65.00%    20
 15  75.00%  70.00%    70.00%    20
 16  35.00%  35.00%    30.00%    20
 17  75.00%  45.00%    50.00%    20
 18  55.00%  70.00%    60.00%    20
 19  65.00%  50.00%    60.00%    20
 20  60.00%  60.00%    55.00%    20
 21  70.00%  55.00%    50.00%    20
 22  65.00%  65.00%    65.00%    20
 23  70.00%  65.00%    55.00%    20
 24  100.00%  100.00%   100.00%    20
 25  100.00%  100.00%   100.00%    20
 26  100.00%  100.00%   100.00%    20
 27  100.00%  100.00%   100.00%    20
```

## Big-model training log tail (2000g)

```
  eval:  L=236.535  bel=40.20%  pi=65.71%  vMAE=6.08  qMAE=10.98
epoch  49/60  dt=14.7s
  train: L=186.513  bel=40.69%  pi=64.78%  vMAE=4.31  qMAE=9.78
  eval:  L=236.278  bel=38.55%  pi=63.39%  vMAE=5.78  qMAE=10.60
epoch  50/60  dt=14.7s
  train: L=188.843  bel=40.76%  pi=64.48%  vMAE=4.31  qMAE=9.84
  eval:  L=217.982  bel=38.81%  pi=65.36%  vMAE=5.82  qMAE=11.09
epoch  51/60  dt=14.9s
  train: L=186.348  bel=40.71%  pi=64.69%  vMAE=4.25  qMAE=9.76
  eval:  L=216.447  bel=37.70%  pi=65.71%  vMAE=5.85  qMAE=10.08
  -> saved best model (score=0.1769)
epoch  52/60  dt=14.7s
  train: L=186.432  bel=40.76%  pi=64.49%  vMAE=4.23  qMAE=9.78
  eval:  L=241.818  bel=37.87%  pi=65.89%  vMAE=5.98  qMAE=10.89
epoch  53/60  dt=14.7s
  train: L=185.119  bel=40.78%  pi=64.79%  vMAE=4.21  qMAE=9.74
  eval:  L=243.503  bel=37.53%  pi=65.89%  vMAE=6.08  qMAE=10.21
epoch  54/60  dt=14.9s
  train: L=185.463  bel=40.80%  pi=64.93%  vMAE=4.18  qMAE=9.75
  eval:  L=257.817  bel=38.93%  pi=65.18%  vMAE=6.13  qMAE=10.98
epoch  55/60  dt=14.6s
  train: L=185.875  bel=40.86%  pi=64.74%  vMAE=4.18  qMAE=9.78
  eval:  L=252.313  bel=38.35%  pi=64.64%  vMAE=5.80  qMAE=10.70
epoch  56/60  dt=14.8s
  train: L=184.216  bel=40.78%  pi=64.92%  vMAE=4.12  qMAE=9.72
  eval:  L=217.579  bel=38.15%  pi=65.36%  vMAE=5.71  qMAE=10.42
epoch  57/60  dt=14.6s
  train: L=185.820  bel=40.73%  pi=64.67%  vMAE=4.12  qMAE=9.76
  eval:  L=219.469  bel=38.27%  pi=65.18%  vMAE=5.94  qMAE=10.58
epoch  58/60  dt=14.8s
  train: L=181.365  bel=40.78%  pi=65.12%  vMAE=4.07  qMAE=9.67
  eval:  L=225.621  bel=39.20%  pi=65.36%  vMAE=5.61  qMAE=10.70
epoch  59/60  dt=14.6s
  train: L=181.326  bel=40.80%  pi=65.09%  vMAE=4.04  qMAE=9.65
  eval:  L=212.203  bel=37.70%  pi=65.18%  vMAE=5.72  qMAE=10.42
epoch  60/60  dt=14.8s
  train: L=179.670  bel=40.83%  pi=65.04%  vMAE=4.01  qMAE=9.60
  eval:  L=233.209  bel=38.98%  pi=65.89%  vMAE=6.07  qMAE=10.62

Final best composite: 0.1769
```

## Gen and pipeline logs

### gen_1000_1999.log
```

real	6m59.018s
user	0m21.484s
sys	0m16.685s
[Tue Apr 21 01:28:45 CDT 2026] chunk done: 1.1G
[Tue Apr 21 01:28:45 CDT 2026] generating chunk seeds 1600-1699...

real	6m42.906s
user	0m20.790s
sys	0m16.219s
[Tue Apr 21 01:35:28 CDT 2026] chunk done: 1.1G
[Tue Apr 21 01:35:28 CDT 2026] generating chunk seeds 1700-1799...

real	6m58.174s
user	0m21.338s
sys	0m16.609s
[Tue Apr 21 01:42:26 CDT 2026] chunk done: 1.1G
[Tue Apr 21 01:42:26 CDT 2026] generating chunk seeds 1800-1899...

real	6m18.087s
user	0m19.846s
sys	0m15.488s
[Tue Apr 21 01:48:44 CDT 2026] chunk done: 1.0G
[Tue Apr 21 01:48:44 CDT 2026] generating chunk seeds 1900-1999...

real	7m4.753s
user	0m21.857s
sys	0m16.962s
[Tue Apr 21 01:55:49 CDT 2026] chunk done: 1.1G
[Tue Apr 21 01:55:49 CDT 2026] seeds 1000-1999 all done
```

### overnight2_pipeline.log
```
==== [Tue Apr 21 00:48:30 CDT 2026] overnight2 pipeline start ====
[Tue Apr 21 00:48:30 CDT 2026] waiting for seeds 1000-1999 gen to finish...
[Tue Apr 21 01:56:06 CDT 2026] gen complete; all 20 chunks present
chunk count: 20

==== [Tue Apr 21 01:56:06 CDT 2026] train v2_voids_big on 2000g (d=256/6L) ====

==== [Tue Apr 21 02:11:07 CDT 2026] train v2_voids_big on 1000g (same-size, data ablation) ====

==== [Tue Apr 21 02:18:37 CDT 2026] PIMC eval — v1 1000g ====
==== [Tue Apr 21 02:18:39 CDT 2026] PIMC eval — v2 big 2000g ====

==== [Tue Apr 21 02:18:41 CDT 2026] write morning2 status ====
```

## Commits since previous status

```
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
edf86e9 fix(burl): SFTConfig max_seq_length=4096 — thought-bearing rows no longer truncate
6fea6ab feat(burl): local MLX-LM path — training + inference on Apple Silicon
f7164b4 chore(burl): gitignore burl/adapters/ — too large for git, regeneratable
dbadb5f docs(burl): session 2026-04-19 capture — iter-3 winner, iter-4 null, forward plan
39aafaf feat(burl): arena — --tag for head-to-head game artifacts
20f4fa2 feat(burl): iter-4 foundation — preserve_thoughts bypasses strip_thinking at SFT
2830be0 fix(burl): Opus parallel-tool-use lock for claude-agent-sdk in-process MCP
```

## What's NEXT (follow-ups worth picking up)

1. **π_opp head** — requires corpus re-gen with opponent-view oracle queries per sampled world (3× gen cost). Without it, LAMIR look-ahead has no simulator for opponent plays.

2. **Multi-step LAMIR** — once π_opp exists, tree-search look-ahead: roll out the full trick (action → 3 opponent plays) using π_opp, evaluate at trick-end using V_head. Compare to direct π_me.

3. **Mid-game data gap** — decisions 4, 8, 15, 16 have highest regret (~4-8 Q-pts). More games focused on these positions, or a targeted eval corpus, might unlock further improvement.

4. **Scale further** — 10k games (~12h gen, ~50GB corpus) would likely close another 1-2 Q-pts of regret based on the 100g→1000g scaling.

