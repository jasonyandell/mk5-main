# Crystal Forge

A ground-up machine learning pipeline that solves Texas 42 dominoes via perfect-play backward induction, trains neural networks to approximate the oracle, and extends that knowledge to imperfect-information play through Monte Carlo marginalization and AlphaZero-style self-play.

Built on weekends for the love of building — iterated 8 times and counting.

## The Big Picture

Texas 42 is a trick-taking dominoes game for 4 players in 2 teams. A "bid" declares how many points your team will capture; a "declaration" sets the trump suit. The game tree is small enough to solve exactly (~50K states per deal), but large enough to be interesting — hidden opponent hands create genuine strategic depth.

Crystal Forge attacks this in three stages:

```
Stage 1: Perfect-Information Oracle
  GPU minimax solver → exact Q-values for every state
  "If everyone could see all hands, what's the optimal play?"

Stage 2: E[Q] Marginalization
  Sample N opponent hands → query oracle in each → average
  "Given what I can see, what's the best play in expectation?"

Zeb: MCTS Self-Play
  AlphaZero-style learning with oracle leaf evaluation
  "Learn to play by playing a million games against yourself"
```

Each stage produces training data for neural networks that get progressively closer to strong imperfect-information play from a single forward pass.

## What's Here

```
forge/
├── oracle/          Stage 1: GPU minimax solver (backward induction)
├── ml/              PyTorch Lightning training core (models, data, metrics)
├── eq/              Stage 2: E[Q] imperfect-info pipeline (GPU-native)
├── zeb/             MCTS self-play with distributed Vast.ai workers
├── bidding/         Monte Carlo P(make) estimation for bid evaluation
├── cli/             Command-line interfaces for all major operations
├── flywheel/        Automated generate→tokenize→train→evaluate loop
├── models/          Pre-trained model catalog (5 checkpoints)
├── analysis/        25-module scientific analysis (98 Jupyter notebooks)
├── modal/           Modal.com cloud GPU orchestration
├── scripts/         Cloud training helpers (Lambda Labs, etc.)
├── data/            Small test datasets
├── ORIENTATION.md   Detailed operational reference (1100 lines)
└── README.md        You are here
```

## Stage 1: Perfect-Information Oracle

**Path:** [`forge/oracle/`](oracle/)

The GPU solver computes exact Q-values for every legal action at every reachable state in a Texas 42 game. States are packed into 41-bit integers encoding four 7-domino hands, trick leader, trick length, and current trick plays. Backward induction propagates values bottom-up — Team 0 maximizes, Team 1 minimizes, with rewards computed at trick completion.

Output: parquet shards with ~50K states each, one per (seed, declaration) pair. The full dataset covers 1,124 seeds across all 10 declaration types — ~215GB on disk, 11.24M tokenized training examples.

```bash
# Generate oracle shards (continuous, gap-filling)
python -m forge.cli.generate_continuous

# Single seed with Q-value inspection
python -m forge.oracle.generate --seed 0 --decl sixes --show-qvals --out /dev/null
```

**Key files:** [`solve.py`](oracle/solve.py) (backward induction), [`state.py`](oracle/state.py) (41-bit packing), [`expand.py`](oracle/expand.py) (vectorized child expansion), [`context.py`](oracle/context.py) (per-deal lookup tables)

## Model Training

**Path:** [`forge/ml/`](ml/)

A `DominoTransformer` (6 layers, 8 heads, 256-dim, 3.3M parameters) learns to predict Q-values from tokenized game states. Each state becomes a 32-token sequence with 12 features per token — pip values, trump rank, player identity, trick context. The tokenizer converts 41-bit packed states into this learnable representation.

The recommended "Q-Val Shuffle" model achieves:
- **Q-Gap 0.074** — mean regret vs oracle in points
- **99.4% zero-regret rate** — plays optimally almost always
- **0.24% blunder rate** — catastrophic mistakes (Q-gap > 10) are rare

Trained with hand-position shuffling to eliminate slot-0 positional bias, Q-value MSE loss, and a 0.5-weighted value head.

```bash
# Quick sanity check
python -m forge.cli.train --fast-dev-run --no-wandb

# Full training with W&B tracking
python -m forge.cli.train --data ../data/tokenized-full --batch-size 4096 --epochs 20 --wandb
```

**Key files:** [`module.py`](ml/module.py) (DominoTransformer + LightningModule), [`tokenize.py`](ml/tokenize.py) (state→token conversion), [`data.py`](ml/data.py) (DataModule), [`metrics.py`](ml/metrics.py) (Q-gap, blunder rate)

**Model catalog:** [`forge/models/README.md`](models/README.md) — 5 checkpoints spanning the evolution from policy cross-entropy (97%+ accuracy) to Q-value models (0.07 Q-gap, direct point prediction)

## Stage 2: E[Q] Imperfect-Information Pipeline

**Path:** [`forge/eq/`](eq/)

The E[Q] system bridges perfect information to realistic play. For each decision point, it:
1. Infers void constraints from play history
2. Samples N consistent opponent hand configurations ("worlds")
3. Queries the Stage 1 oracle for Q-values in each world
4. Averages to produce E[Q] — the expected Q-value under uncertainty

The entire pipeline runs on GPU with **12,325x speedup** over the original Python implementation. Key innovations:

- **MRV backtracking with bitmask arithmetic** — 28 dominoes fit in one int64, all constraint operations become popcount/bitwise-OR/AND
- **Adaptive convergence-based sampling** — samples until SEM < 0.1 threshold, achieving 8.4x better precision than fixed sampling
- **Bayesian posterior weighting** — reweights worlds based on consistency with observed opponent play, improving mid-game estimates

```bash
# Generate E[Q] training data (GPU, adaptive, gap-filling)
python -m forge.cli.generate_eq_continuous \
    --checkpoint forge/models/domino-qval-3.3M-shuffle-qgap0.074-qmae0.96.ckpt \
    --adaptive --posterior --posterior-window 4

# Interactive debug viewer
python -m forge.eq.viewer data/eq-games/train/seed_00000042.pt
```

**Key files:** [`generate/pipeline.py`](eq/generate/pipeline.py) (orchestrator), [`generate/sampling.py`](eq/generate/sampling.py) (world sampling), [`generate/posterior.py`](eq/generate/posterior.py) (Bayesian weighting), [`generate/adaptive.py`](eq/generate/adaptive.py) (convergence detection), [`gpu_tokenizer.py`](eq/gpu_tokenizer.py) (12,325x speedup), [`viewer.py`](eq/viewer.py) (curses-based debugger)

## Zeb: MCTS Self-Play Training

**Path:** [`forge/zeb/`](zeb/)

An AlphaZero-style system where a 557K-parameter neural network learns through self-play guided by oracle leaf evaluation. Determinized UCT with virtual loss collects batches of 32 leaves across 16 concurrent games for GPU evaluation.

Three layers of CUDA graph optimization deliver a **2.4x speedup**:
1. Depth-variant graph capture at [28, 8, 1] depths
2. Multi-step replay — K=10 simulation steps per graph replay
3. Vectorized kernels replacing Python loops (252 kernels → 12)

**Distributed architecture:** The learner runs locally on a 3050 Ti. Workers run on Vast.ai (RTX 3060s at $0.05/hr). HuggingFace Hub is the sole communication broker — no shared filesystem needed. Workers generate self-play games and upload batches every 240s; the learner polls for new examples, trains on a 200K-example GPU ring buffer, and pushes updated weights.

**Milestone: 1.5M self-play games, from first commit to 70% win rate in 5 days.** An 8-hour distributed session with 4 Vast.ai workers costs ~$2.88.

A $30K B200 datacenter GPU achieves 10.1 games/sec — only 2.8x faster than a $250 3050 Ti at 3.6 games/sec. MCTS is inherently depth-sequential; no amount of GPU width helps beyond eliminating kernel dispatch overhead. At $6.25/hr, the B200 produces ~5,800 games/$/hr. A fleet of RTX 3060s at $0.05/hr produces ~288,000 games/$/hr — **50x more cost-effective**.

```bash
# Launch 4 Vast.ai workers
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
./forge/zeb/vast/vast_up.sh 4

# Monitor fleet
./forge/zeb/vast/vast_status.sh

# Run learner locally
python -u -m forge.zeb.learner.run \
    --repo-id jasonyandell/zeb-42 \
    --examples-repo-id jasonyandell/zeb-42-examples

# Teardown
./forge/zeb/vast/vast_down.sh
```

**Key files:** [`mcts.py`](zeb/mcts.py) (UCT engine), [`gpu_mcts.py`](zeb/gpu_mcts.py) (CUDA graph optimization), [`batched_mcts.py`](zeb/batched_mcts.py) (cross-game batching), [`model.py`](zeb/model.py) (ZebModel), [`learner/run.py`](zeb/learner/run.py) (training loop), [`vast/RUNBOOK.md`](zeb/vast/RUNBOOK.md) (fleet operations)

## Bidding Evaluation

**Path:** [`forge/bidding/`](bidding/)

Monte Carlo estimation of P(make) for any hand across all 9 trump types and 13 bid thresholds (30–42). Vectorized PyTorch game simulation on GPU with Wilson score confidence intervals. Includes convergence analysis (N=500 gives ±0.04 CI), multi-GPU parallelism, and a poster generator producing publication-quality PDF heatmaps.

```bash
# Single hand evaluation
python -m forge.bidding.evaluate --hand "6-4,5-5,4-2,3-1,2-0,1-1,0-0" --samples 100

# Continuous large-scale evaluation
python -m forge.cli.bidding_continuous
```

**Key files:** [`simulator.py`](bidding/simulator.py) (vectorized game sim), [`estimator.py`](bidding/estimator.py) (P(make) + Wilson CI), [`poster.py`](bidding/poster.py) (PDF generation)

## Scientific Analysis

**Path:** [`forge/analysis/`](analysis/)

A 25-module investigation of the oracle game tree — 98 Jupyter notebooks, 200+ seeds, ~300M game states. Topics span information theory, topology, symmetry, Bayesian modeling, survival analysis, time series classification (MiniRocket), ecological diversity metrics, SHAP explainability, and strategic analysis.

### Headline findings

**The napkin formula:** `Oracle E[V] ≈ 14 + 6*(n_doubles) + 3*(trump_count)` — only two features survive multivariate regression with bootstrap CIs excluding zero. Cross-validation confirms the 2-feature model generalizes better than the full 10-feature model.

**Inverse risk-return:** Good hands are also *safer* hands (r = -0.381, p = 2.6e-8) — the opposite of financial markets.

**Game phase structure:** "Order → chaos → resolution." Opening moves have 40% best-move consistency, mid-game drops to 22%, endgame reaches 100% determinism at depth ≤ 4.

### Folk wisdom: tested and broken

Six pieces of traditional Texas 42 wisdom were tested against oracle data. **None were confirmed.** The most striking:

| Folk Wisdom | Verdict | Finding |
|---|---|---|
| Threshold cliffs at 30 and 35 | NOT CONFIRMED | Transitions are statistically average; largest cliff is at 38→39 |
| Voiding is an active strategy | NOT CONFIRMED | No support in oracle data |
| Naked lows hurt | NOT CONFIRMED | Not validated |
| Coverage protects | NOT CONFIRMED | No support |
| Voids are directional | NOT CONFIRMED | Not confirmed |
| Coverage beats raw trumps | **REFUTED (INVERTED)** | Coverage *hurts* E[V] (β = -0.288, p=0.0001). Voids enable trumping, which is more valuable. 4 trumps + voids beats 2 trumps + coverage. |

### Epistemic audit

Every module underwent a systematic audit enforcing that claims are scoped to "oracle game tree structure" rather than unqualified "Texas 42 strategy." The oracle represents perfect play with full information — not human play — and the distinction matters. All reports carry explicit scope labels, qualified language, and "Further Investigation" sections.

**Key files:** [`report/00_executive_summary.md`](analysis/report/00_executive_summary.md) (executive summary), [`notebooks/`](analysis/notebooks/) (98 notebooks), [`report/`](analysis/report/) (26 module reports)

## Automation & Infrastructure

**Flywheel** ([`forge/flywheel/`](flywheel/)): Automated iterative fine-tuning — generate oracle shards → tokenize → train → evaluate → repeat. YAML state machine with W&B logging. Each iteration fine-tunes from the previous checkpoint (2 epochs, LR=3e-5).

**Cloud compute:** Three tiers scripted — Lambda Labs for A100 training ([`scripts/`](scripts/)), Modal for B200 oracle and E[Q] generation ([`modal_app.py`](modal_app.py)), Vast.ai for distributed self-play fleet ([`zeb/vast/`](zeb/vast/)).

## Getting Started

```bash
# Install dependencies
pip install -r forge/requirements.txt

# Verify imports
python -c "from forge.ml import module, data, metrics; from forge.oracle import schema; print('OK')"

# Quick training sanity check
python -m forge.cli.train --fast-dev-run --no-wandb
```

For the full operational reference — data formats, CLI options, debugging tips, architectural invariants — see [ORIENTATION.md](ORIENTATION.md).
