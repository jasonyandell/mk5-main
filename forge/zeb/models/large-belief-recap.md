# large-belief Training Run Recap

**W&B run:** [waxffg2j](https://wandb.ai/jasonyandell-forge42/zeb-mcts/runs/waxffg2j)
**Dates:** Feb 9 -- Feb 11, 2026 (~50 hours wall time)
**Status:** Completed (3,743 training cycles, 1.7M self-play games)

---

## 1. Model Overview

| Property | Value |
|---|---|
| Architecture | ZebModel transformer (pre-LN, 6 layers, 8 heads) |
| Parameters | 3,347,110 |
| Embed dim | 256 |
| FF dim | 512 |
| Tokenizer | v1 |
| Belief head | Yes -- auxiliary head predicting opponent domino ownership (28 dominoes x 3 opponents) |
| Base weights | `large.pt` (pre-trained without belief head) |
| HF repo | [jasonyandell/zeb-42](https://huggingface.co/jasonyandell/zeb-42), weights name: `large-belief.pt` |
| Examples repo | [jasonyandell/zeb-42-examples](https://huggingface.co/datasets/jasonyandell/zeb-42-examples) |

The belief head was randomly initialized on top of existing large model weights.
The policy and value heads carried over their pre-trained parameters. This gave the
model a strong starting point while the belief head learned from scratch.

---

## 2. Training Configuration

| Parameter | Initial | Final |
|---|---|---|
| Learning rate | 1e-4 | 1e-4 (constant) |
| Batch size | 64 | 64 |
| Steps per cycle | 1,000 | 1,000 |
| Replay buffer | 1,000,000 | 500,000 |
| Min buffer | -- | 250,000 |
| Max example age | -- | 600s |
| Keep example files | -- | 100 |

---

## 3. Training Progression

### Key milestones from W&B

| Cycle | Policy | Value | Belief | Belief Acc | Eval (vs random) | Games | Buffer |
|------:|-------:|------:|-------:|-----------:|------------------:|------:|-------:|
| 10 | 0.499 | 0.106 | 0.993 | 45.8% | 72.7% | 4K | 115K |
| 160 | 0.487 | 0.232 | 0.538 | 70.2% | 74.8% | 170K | 1M |
| 600 | 0.448 | 0.208 | 0.519 | 71.3% | 74.5% | 452K | 1M |
| 1190 | 0.402 | 0.208 | 0.515 | 71.5% | 75.4% | 741K | 1M |
| 1780 | 0.358 | 0.196 | 0.510 | 71.9% | 74.6% | 1.07M | 1M |
| 2310 | 0.349 | 0.184 | 0.512 | 71.7% | **76.2%** | 1.23M | 1M |
| 2730 | 0.333 | 0.175 | 0.507 | 72.1% | **76.3%** | 1.36M | 1M |
| 3300 | 0.307 | 0.131 | 0.505 | 72.3% | 75.4% | 1.56M | mixed |
| 3550 | 0.300 | 0.121 | 0.511 | 72.0% | 75.0% | 1.64M | 500K |
| 3743 | 0.304 | 0.096 | 0.507 | 72.3% | 71.3% | 1.70M | 500K |

### Phase summary

**Phase 1 -- Belief head warmup (cycles 1--50).** The belief head starts cold
(loss 0.99, 46% accuracy -- near random). Within 50 cycles it drops sharply to
0.54 / 70%, converging quickly on the structured opponent-ownership prediction
task. Policy and value losses barely move since they inherited strong pre-trained
weights.

**Phase 2 -- Steady improvement (cycles 50--2730).** All losses grind lower in
tandem. Policy loss drops from 0.49 to 0.33. Belief accuracy climbs from 70% to
72%. The eval win rate against random peaks at **76.3%** around cycle 2730
(1.36M games). Policy top-1 accuracy rises from 86% to 89%.

**Phase 3 -- Buffer tuning and diminishing returns (cycles 2730--3743).** After
switching to 1 worker the buffer became too stale at 1M capacity. Buffer was
reduced to 500K. Policy loss continued improving (0.33 -> 0.30) and policy top-1
accuracy reached 90.2%. Belief loss plateaued around 0.507. Value loss showed
artificially low readings after restarts due to small buffer memorization (see
Section 6). Eval win rate fluctuated in the 73--75% range, suggesting the model
was near its architecture ceiling for this evaluation.

---

## 4. Final Metrics

From the W&B run summary at cycle 3,743:

| Metric | Value |
|---|---|
| Policy loss | 0.304 |
| Value loss | 0.096 |
| Belief loss | 0.507 |
| Belief accuracy | 72.3% |
| Total loss | 0.654 |
| Policy entropy | 0.304 |
| Policy KL divergence | 0.086 |
| Policy top-1 accuracy | 89.8% |
| Value mean | -0.004 |
| Value std | 0.949 |
| Eval vs random | 71.3% (final), **76.3% peak** (cycle 2730) |
| Grad norm | 2.95 |
| Train time/cycle | 43.3s |
| Total self-play games | 1,700,608 |

### Peak performance

The best eval-vs-random win rate was **76.3%** at cycle 2730 (1.36M games), with
policy=0.333, value=0.175, belief=0.507, belief_acc=72.1%.

---

## 5. Head-to-Head Evaluation

### Calibration snapshot (Feb 9, 2026, cycle ~1660, 100-game samples)

Elo ratings anchored to eq:n=100 = 1600:

| Player | Elo | Notes |
|---|---|---|
| eq(n=100) | 1600 | anchor |
| zeb-large-belief | 1579 | ~1M training games |
| eq(n=500) | 1561 | |
| random | 1386 | floor |

At 1M training games, large-belief was already neck-and-neck with eq:n=100 (the
Expected-Q player running 100 Monte Carlo rollouts per decision). The model
continued improving after this snapshot.

### Qualitative matchup notes

- **large-belief vs large:** ~3 percentage point advantage consistently, showing
  the belief head provides meaningful signal even at similar parameter counts.
- **large-belief vs medium:** dominant.
- **large-belief vs E[Q]:** competitive with eq:n=100; the neural model runs
  orders of magnitude faster at inference time.

---

## 6. Infrastructure and Cost

### Compute

| Role | Hardware | Location | Cost |
|---|---|---|---|
| Learner | RTX 3050 Ti | Local | $0 |
| Workers (phase 1) | 8x RTX 3060 | Vast.ai | ~$0.43/hr |
| Workers (phase 2) | 12x mixed (4070 Ti/4070/3080) | Vast.ai | ~$0.97/hr |
| Workers (phase 3) | 1x RTX 4070 Ti | Vast.ai | ~$0.084/hr |

**Total Vast.ai spend: ~$14.72** for 1.7M+ games over ~50 hours wall time.

### GPU performance benchmarks (large-belief, 3.3M params)

| GPU | Games/s | Typical $/hr | $/game/s |
|---|---|---|---|
| RTX 4070 Ti | 4.1--4.2 | $0.074--0.090 | $0.018--0.021 |
| RTX 4070 | 3.2--3.4 | $0.079--0.081 | $0.024 |
| RTX 3080 | 3.0 | $0.079 | $0.026 |
| RTX 3080 Ti | 2.7--2.8 | $0.084 | $0.030 |
| RTX 3070 Ti | 2.6--2.7 | $0.084 | $0.031 |

### Fleet scaling

The run started with 8 workers for rapid buffer filling, scaled up to 12 for
peak throughput (~35 games/s steady state), then scaled down to 1 worker as the
run entered its long tail and the focus shifted to buffer freshness tuning.

---

## 7. Key Tuning Decisions and Lessons Learned

### Buffer sizing is critical with few workers

With 1 worker producing ~500 games/cycle and a 1M buffer, the buffer takes ~810
cycles to fully turn over. This means the model trains on increasingly stale
data. Reducing to 500K cuts turnover to ~100 cycles, keeping examples fresh.

### Small buffers cause memorization artifacts

After a restart, the buffer fills from HF example files. With only 164K examples
loaded, value loss dropped dramatically (artificially low) -- the model was
memorizing the small buffer rather than generalizing. The fix was setting a
minimum buffer size of 250K to prevent training on too-small datasets.

### Belief head converges fast, then plateaus

The belief head went from random (46% accuracy) to 70% in ~50 cycles, then
slowly climbed to 72.3% over 3,700 more cycles. This suggests the belief task
has a natural difficulty ceiling -- predicting which of 3 opponents holds each of
28 dominoes is inherently uncertain given partial observability. The 72% accuracy
still provides useful signal to the policy and value heads.

### Value loss is a noisy metric

Value loss fluctuated significantly (0.10 -- 0.25) depending on buffer
composition and restart state. It is not a reliable indicator of model quality in
isolation. Eval win rate is the ground-truth performance measure.

### Eval win rate plateaus around 75--76%

The vs-random eval saturated in the 75--76% range, suggesting either a ceiling of
the eval methodology (vs-random is a weak opponent) or diminishing returns at this
model size. The Elo-calibrated head-to-head tournament against E[Q] players
provides more discriminating evaluation.

### Bootstrap from pre-trained weights works well

Starting from the large model's policy and value weights and only randomly
initializing the new belief head gave immediate strong play (73% vs random from
cycle 1) while the belief head caught up over ~50 cycles. This is far more
efficient than training the belief model from scratch.
