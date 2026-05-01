---
title: forge-analysis (oracle game-tree analytics workstream)
kind: entity
first_seen: 5ffdf58
last_updated: 4a747f6
status: active
---

## What it is

`forge/analysis/` is a sustained workstream applying classical statistics, ML
explainability, and dimensionality reduction to the [[forge]] perfect-information
oracle's game-tree output. Distinct from the LEM/Burl/Gus modeling sibling projects:
it does not train an agent. It reads ~300M oracle states and writes a **publication-
shaped report** — figures, tables, executive summary, per-section findings.

The unit of work is a numbered notebook (`notebooks/NN_<theme>/`) that produces files
in `results/figures/` and `results/tables/`. Each gets a parallel `report/NN_<theme>.md`
write-up. Findings are summarized in `report/00_executive_summary.md`.

## Layout

| Path | What's there |
|---|---|
| `forge/analysis/notebooks/` | 21 numbered theme dirs (01_baseline → 21_survival) — Jupyter notebooks per analysis |
| `forge/analysis/report/` | Per-section markdown writeups + `00_executive_summary.md`; this is the publication scaffold |
| `forge/analysis/scripts/` | One-off `.py` scripts for memory-heavy analyses that don't fit in notebooks (E[Q] PDF export, regret distribution, slot-swap test, etc.) |
| `forge/analysis/utils/` | Shared loaders + `seed_db.py` (DuckDB query interface over parquet shards), `hand_features.py` (unified feature extraction) |
| `forge/analysis/bias/` | The [[q0-positional-bias]] investigation — 20 numbered probes into a slot-0 anomaly in forge's Q-value model |
| `forge/analysis/results/figures/`, `results/tables/`, `results/web/` | Outputs. Figures are 300dpi PNG + vector PDF. Tables are CSV. |
| `forge/analysis/CLAUDE.md` | Analyst seat: data locations, notebook config pattern, DuckDB SQL recipes, memory tips, key statistical findings reference. Read first. |

## Data

- **`/mnt/d/shards-standard/`** (~200GB) — full oracle solve, train/val/test split by `seed % 1000`.
- **`data/shards-marginalized/`** (~92GB) — 200 base seeds × 3 opponent configs each, P0's hand fixed. Enables imperfect-info E[V]/σ(V) analysis without retraining the oracle.
- **`data/flywheel-shards/`** — small local subset for quick testing.

Schema: `state` (packed int64, 41 bits), `V` (int8, Team-0 perspective, ±42), `q0`–`q6` (int8 per-action Q, −128 = illegal). [[forge]]'s `oracle/schema.py` unpacks.

## Headline epistemic frame

The report opens with a **load-bearing caveat** that every finding inherits: this is
analysis of a perfect-information minimax solver, not human gameplay. See
[[oracle-vs-human-play]] — that page exists so the caveat is one backlink away from
any claim that wants to extrapolate ("doubles are good for E[V]" is an oracle fact;
"play doubles" is an untested human-play hypothesis).

## Headline findings (oracle frame)

| # | Finding | Source |
|---|---|---|
| 1 | **Count capture explains ~92% of oracle V variance** (R²=0.99 at depth ≤ 12, 0.76 overall). | report/03_counts.md |
| 2 | **Risk and return are inversely correlated**: r(E[V], σ(V)) = −0.38 [−0.49, −0.26] at n=200 seeds. Opposite of typical financial markets. See [[risk-return-inverse]]. | report/13_statistical_rigor.md |
| 3 | **n_doubles + trump_count are the only robust E[V] predictors** (bootstrap 95% CIs exclude zero; survive 10-fold CV with R²=0.15 from 2 features vs 0.11 from 10). The "napkin formula." | report/13–14 |
| 4 | **σ(V) is fundamentally unpredictable from your hand** (CV R² < 0). The variance comes from opponent hands, not yours. | report/14b |
| 5 | **Three-phase game structure**: depth 24–28 ordered (40% best-move consistency), depth 5–23 chaotic (22%), depth 0–4 mechanical (100%). | report/15d |
| 6 | **Pareto frontier is degenerate**: only 1.5% of hands are non-dominated (all E[V]=42, σ=0). No real risk-return tradeoff exists. | report/15c |
| 7 | **Domino co-occurrence carries near-zero strategic signal** (Word2Vec/UMAP). Value comes from game context (trump, position), not hand composition. | report/16a–b |
| 8 | **Symmetry compression negligible** (1.005×). Trump + played-card history breaks pip-permutation orbits. | report/04_symmetry.md |
| 9 | **Hurst H=0.925** in oracle PV trajectories — strong persistence, far from random walk. | report/05_topology.md |

The full list is much longer; the executive summary at `forge/analysis/report/00_executive_summary.md` is the canonical index.

## Tooling

`forge/analysis/utils/seed_db.py` is the load-bearing primitive. DuckDB over parquet
with a depth UDF (`bit_count(state & 0x0FFFFFFF)`), 10× faster than Python loops for
aggregations. Notebook conversion guidance lives in `forge/analysis/CLAUDE.md`'s
"Converting Scripts to SeedDB" section.

`forge/analysis/utils/hand_features.py` consolidates feature extraction once-replicated
across `run_11*.py` scripts. The 200-seed × 20-column master table at
`results/tables/12b_unified_features.csv` is the input to most regression and SHAP work.

## Status

Mature. Last commit 2026-01-31. Active strand at the time of the LEM→Burl handoff
(8d26e0d) was the `bias/` slot-0 investigation; that strand resolved with proposed
fix `bias/20-proposed-fix-shuffle.md`.

## Related

[[forge]] — the oracle this consumes. [[zeb]] — the shared trained Q-value model whose
slot-0 anomaly drove the [[q0-positional-bias]] sub-investigation. [[gus]] — the
distillation project that picks up where this analysis leaves off, training a student
on the same oracle output.
