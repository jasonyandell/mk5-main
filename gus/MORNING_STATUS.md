# Gus — Morning Status (2026-04-20 23:52:44 )

## TL;DR

Overnight pipeline: corpus generation (1000 games via chunking), full 4-head student training on both 100-game baseline and 1000-game corpus, per-decision evaluation on held-out.

## Best-epoch metrics (eval)

| metric | v1_full (100g) | v1_full (1000g) |
|---|---|---|
| best epoch | 26 | 38 |
| π_me bot-match | 60.2% | 66.1% |
| belief top-1 | 34.5% | 37.2% |
| V MAE | 13.61 | 7.56 |
| Q MAE | 18.15 | 12.27 |

Baselines: belief chance=33.3%, π_me chance (~4 legal moves)=25%, Q on scale [-42,+42]

## Per-decision breakdown (v1_full on 1000g corpus)

```
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
Adapter: gus/adapters/v1_full_1000g.pt
Eval corpus: gus/data/corpus_eval_20.pt
Decisions: 560

OVERALL:
  belief top-1:     36.650%
  pi_me bot-match:  65.357%
  V MAE:            7.63
  Q MAE (legal):    12.14

dec     bel      pi    vMAE    qMAE     n
  0  31.19%  40.00%    5.04   12.86    20
  1  34.50%  70.00%    5.96   19.15    20
  2  37.89%  60.00%    6.39    9.03    20
  3  35.83%  90.00%    8.25    9.97    20
  4  33.33%  25.00%    6.52   13.68    20
  5  33.53%  55.00%    7.60   10.09    20
  6  34.69%  65.00%    7.52   14.31    20
  7  37.33%  75.00%    9.48   14.22    20
  8  35.67%  30.00%    7.10   13.73    20
  9  41.07%  50.00%    8.23   13.08    20
 10  33.08%  70.00%    5.62   13.24    20
 11  36.67%  70.00%    7.45   12.41    20
 12  40.83%  20.00%    5.98   13.98    20
 13  44.09%  80.00%    5.71   12.39    20
 14  37.00%  45.00%    6.19    9.56    20
 15  31.11%  50.00%    9.18   11.21    20
 16  37.78%  55.00%    9.90   10.68    20
 17  40.00%  85.00%    7.40   11.40    20
 18  40.00%  45.00%    8.19   11.15    20
 19  39.17%  60.00%    9.81   11.82    20
 20  37.50%  65.00%    9.18    9.98    20
 21  36.00%  70.00%    7.19    7.98    20
 22  46.25%  75.00%    8.16    9.18    20
 23  40.00%  80.00%   10.46   10.94    20
 24  38.33%  100.00%    6.54    7.75    20
 25  42.50%  100.00%    6.03    6.81    20
 26  90.00%  100.00%    8.55    7.90    20
 27   0.00%  100.00%   10.01   10.18    20

```

## Per-decision breakdown (v1_full on 100g corpus baseline)

```
/Users/jason/code/mk5-main/gus/model/student.py:146: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
Adapter: gus/adapters/v1_full_100g.pt
Eval corpus: gus/data/corpus_eval_20.pt
Decisions: 560

OVERALL:
  belief top-1:     34.473%
  pi_me bot-match:  60.179%
  V MAE:            13.61
  Q MAE (legal):    18.60

dec     bel      pi    vMAE    qMAE     n
  0  37.38%  30.00%    8.33   20.53    20
  1  33.25%  70.00%   12.29   16.54    20
  2  31.32%  75.00%   16.38   24.70    20
  3  31.11%  65.00%   20.12   18.20    20
  4  31.67%  25.00%   11.24   20.83    20
  5  35.29%  65.00%   14.72   25.29    20
  6  34.69%  60.00%   15.95   21.73    20
  7  33.00%  60.00%   18.36   23.82    20
  8  31.67%  15.00%   12.25   19.42    20
  9  35.71%  50.00%   12.90   16.95    20
 10  30.77%  35.00%   13.55   22.76    20
 11  36.67%  35.00%   15.74   18.98    20
 12  30.83%  25.00%   13.42   16.15    20
 13  38.18%  65.00%   14.33   15.96    20
 14  33.50%  55.00%   12.95   22.23    20
 15  36.67%  55.00%   16.42   21.55    20
 16  33.33%  55.00%   12.29   15.67    20
 17  36.88%  70.00%   13.28   16.96    20
 18  36.43%  50.00%   14.26   13.42    20
 19  42.50%  55.00%   15.78   19.40    20
 20  33.33%  65.00%    9.40   11.11    20
 21  35.00%  70.00%   11.89   10.48    20
 22  36.25%  80.00%   14.53   13.26    20
 23  46.67%  55.00%   14.14   15.51    20
 24  48.33%  100.00%    9.87   11.35    20
 25  35.00%  100.00%   12.79   12.87    20
 26  60.00%  100.00%   11.65   10.55    20
 27   0.00%  100.00%   12.20   12.29    20

```

## Training log tail — 1000g

```
epoch  32/40  dt=4.9s
  train: L=241.237  bel=37.54%  pi=62.31%  vMAE=6.30  qMAE=11.14
  eval:  L=299.324  bel=36.36%  pi=64.29%  vMAE=8.24  qMAE=12.35
epoch  33/40  dt=5.0s
  train: L=242.856  bel=37.61%  pi=62.28%  vMAE=6.23  qMAE=11.18
  eval:  L=335.235  bel=35.92%  pi=64.64%  vMAE=7.89  qMAE=12.93
epoch  34/40  dt=4.8s
  train: L=236.398  bel=37.57%  pi=62.11%  vMAE=6.17  qMAE=10.99
  eval:  L=286.047  bel=36.65%  pi=65.36%  vMAE=7.63  qMAE=11.98
  -> saved best model (score=-0.1622)
epoch  35/40  dt=4.9s
  train: L=240.924  bel=37.69%  pi=62.38%  vMAE=6.13  qMAE=11.15
  eval:  L=289.836  bel=36.12%  pi=65.00%  vMAE=7.81  qMAE=12.19
epoch  36/40  dt=4.9s
  train: L=235.142  bel=37.77%  pi=62.35%  vMAE=6.08  qMAE=11.00
  eval:  L=289.120  bel=37.18%  pi=64.46%  vMAE=7.81  qMAE=12.35
epoch  37/40  dt=4.8s
  train: L=233.052  bel=37.83%  pi=62.35%  vMAE=6.04  qMAE=10.93
  eval:  L=272.756  bel=35.97%  pi=63.39%  vMAE=7.78  qMAE=11.85
epoch  38/40  dt=4.9s
  train: L=232.000  bel=38.05%  pi=62.34%  vMAE=6.00  qMAE=10.92
  eval:  L=279.337  bel=37.21%  pi=66.07%  vMAE=7.56  qMAE=12.27
epoch  39/40  dt=4.9s
  train: L=229.888  bel=37.98%  pi=62.51%  vMAE=5.88  qMAE=10.83
  eval:  L=274.367  bel=37.69%  pi=64.29%  vMAE=7.87  qMAE=12.49
epoch  40/40  dt=4.9s
  train: L=229.620  bel=38.03%  pi=62.54%  vMAE=5.86  qMAE=10.89
  eval:  L=298.915  bel=36.28%  pi=65.00%  vMAE=8.10  qMAE=12.50

Final best composite: -0.1622
```

## Training log tail — 100g

```
  eval:  L=731.943  bel=34.83%  pi=60.18%  vMAE=14.14  qMAE=19.85
epoch  32/40  dt=0.8s
  train: L=439.661  bel=36.19%  pi=58.07%  vMAE=8.37  qMAE=15.80
  eval:  L=812.989  bel=34.68%  pi=57.14%  vMAE=14.49  qMAE=20.68
epoch  33/40  dt=0.7s
  train: L=415.330  bel=36.13%  pi=58.07%  vMAE=8.34  qMAE=15.21
  eval:  L=736.062  bel=34.78%  pi=57.32%  vMAE=14.10  qMAE=19.08
epoch  34/40  dt=0.7s
  train: L=440.471  bel=36.41%  pi=57.89%  vMAE=8.44  qMAE=15.77
  eval:  L=733.138  bel=33.79%  pi=58.39%  vMAE=14.22  qMAE=19.52
epoch  35/40  dt=0.8s
  train: L=432.895  bel=36.93%  pi=59.14%  vMAE=8.32  qMAE=15.56
  eval:  L=736.301  bel=35.22%  pi=58.93%  vMAE=14.02  qMAE=19.90
epoch  36/40  dt=0.8s
  train: L=427.958  bel=36.65%  pi=58.82%  vMAE=8.33  qMAE=15.64
  eval:  L=711.528  bel=34.42%  pi=58.93%  vMAE=14.16  qMAE=19.00
epoch  37/40  dt=0.8s
  train: L=410.130  bel=36.83%  pi=58.50%  vMAE=8.06  qMAE=15.11
  eval:  L=748.237  bel=34.90%  pi=59.11%  vMAE=14.13  qMAE=19.67
epoch  38/40  dt=0.8s
  train: L=425.706  bel=36.64%  pi=59.43%  vMAE=8.21  qMAE=15.28
  eval:  L=699.262  bel=34.91%  pi=59.29%  vMAE=13.98  qMAE=19.10
epoch  39/40  dt=0.8s
  train: L=421.568  bel=36.54%  pi=58.50%  vMAE=7.90  qMAE=15.50
  eval:  L=673.594  bel=34.30%  pi=58.21%  vMAE=13.99  qMAE=18.96
epoch  40/40  dt=0.7s
  train: L=403.764  bel=37.04%  pi=58.71%  vMAE=7.89  qMAE=15.08
  eval:  L=694.021  bel=33.69%  pi=59.46%  vMAE=13.60  qMAE=18.98

Final best composite: -0.5032
```

## Pipeline log

```
user	0m21.313s
sys	0m15.120s
[Mon Apr 20 22:39:35 CDT 2026] chunk done: 1.0G
[Mon Apr 20 22:39:35 CDT 2026] generating chunk seeds 100-199...

real	7m34.886s
user	0m22.895s
sys	0m16.549s
[Mon Apr 20 22:47:10 CDT 2026] chunk done: 1.1G
[Mon Apr 20 22:47:10 CDT 2026] generating chunk seeds 200-299...

real	7m29.140s
user	0m22.465s
sys	0m16.544s
[Mon Apr 20 22:54:39 CDT 2026] chunk done: 1.1G
[Mon Apr 20 22:54:39 CDT 2026] generating chunk seeds 300-399...

real	7m16.398s
user	0m23.047s
sys	0m16.169s
[Mon Apr 20 23:01:55 CDT 2026] chunk done: 1.1G
[Mon Apr 20 23:01:55 CDT 2026] generating chunk seeds 400-499...

real	7m17.459s
user	0m21.985s
sys	0m15.996s
[Mon Apr 20 23:09:13 CDT 2026] chunk done: 1.1G
[Mon Apr 20 23:09:13 CDT 2026] generating chunk seeds 500-599...

real	7m1.704s
user	0m21.413s
sys	0m15.304s
[Mon Apr 20 23:16:15 CDT 2026] chunk done: 1.0G
[Mon Apr 20 23:16:15 CDT 2026] generating chunk seeds 600-699...

real	7m20.955s
user	0m22.003s
sys	0m16.146s
[Mon Apr 20 23:23:36 CDT 2026] chunk done: 1.1G
[Mon Apr 20 23:23:36 CDT 2026] generating chunk seeds 700-799...

real	7m8.697s
user	0m21.628s
sys	0m15.664s
[Mon Apr 20 23:30:44 CDT 2026] chunk done: 1.0G
[Mon Apr 20 23:30:44 CDT 2026] generating chunk seeds 800-899...

real	7m18.039s
user	0m22.055s
sys	0m16.167s
[Mon Apr 20 23:38:02 CDT 2026] chunk done: 1.1G
[Mon Apr 20 23:38:02 CDT 2026] generating chunk seeds 900-999...

real	7m35.198s
user	0m22.661s
sys	0m16.662s
[Mon Apr 20 23:45:38 CDT 2026] chunk done: 1.1G
[Mon Apr 20 23:45:38 CDT 2026] all 10 chunks complete
==== [Mon Apr 20 23:45:38 CDT 2026] train v1_full on 100g corpus (baseline re-run with 40 epochs) ====
==== [Mon Apr 20 23:46:11 CDT 2026] train v1_full on 1000g corpus (bigger model, 40 epochs) ====
```

## Commits made overnight

```
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
```

## Suggested next moves

1. **Inspect π_me per-decision accuracy** — if late-game is >90% but early-game is mixed, the bot-match ceiling is data-bound on mid-game. More games (seeds 1000+) would help.
2. **Q_head distillation quality** — if Q_MAE is < 5 Q-points, the world-conditioned value function is good enough for LAMIR leaf-eval. If >10 Q-points, needs more data / bigger model.
3. **If bot-match plateaued < 70%**, consider: (a) engine-computed void features injected into the state encoder (cheat to unblock the model), (b) bigger transformer (6 layers, d_model=256).
4. **Next architecture extension**: add π_opp_head — requires opponent-view oracle queries saved during corpus gen. ~3× corpus gen cost but unlocks LAMIR look-ahead.
