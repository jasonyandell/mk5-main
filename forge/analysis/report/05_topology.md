# 05: Topology Analysis

## Context

We analyzed the topological structure of the value function: how are states with the same V connected? This explains why the value head struggles (MAE 7.4) despite 97.8% move accuracy.

## Level Set Fragmentation

A "level set" is all states with the same V value. We asked: are these connected regions, or scattered fragments?

![Level Set Connectivity](../results/figures/05a_level_set_connectivity.png)

**Finding**: Level sets are **highly fragmented**. Each V value corresponds to many disconnected components, not a smooth manifold.

![Level Set Sizes](../results/figures/05a_level_set_sizes.png)

## V Transitions: Value Changes on Most Moves

![V Transitions](../results/figures/05a_v_transitions.png)

Most edges in the game tree connect states with *different* V values. The value function is discontinuous almost everywhere.

**Model relevance**: This explains why the value head (MAE 7.4) underperforms move prediction (97.8%). Moves are locally predictable; values are globally fragmented.

## Reeb Graph Structure

The Reeb graph contracts level sets to points, preserving connectivity:

![Reeb Graph](../results/figures/05b_reeb_graph.png)

**Finding**: Complex branching structure. The game tree doesn't form simple funnels—it's a web of merging and splitting value regions.

![Reeb Structure](../results/figures/05b_reeb_structure.png)

## Critical Points

Critical points are where level set topology changes (branches merge or split):

| Type | Count | Meaning |
|------|-------|---------|
| Merge | Many | Multiple paths → same outcome |
| Split | Many | One position → divergent outcomes |

**Model relevance**: The high branch count means V can change dramatically with small position changes. Value regression is fundamentally hard—the landscape is rugged.

## Why Move Prediction Works but Value Doesn't

| Task | Structure | Our Result |
|------|-----------|------------|
| Move prediction | Local (one step) | 97.8% |
| Value prediction | Global (whole game) | MAE 7.4 |

Move prediction only needs to compare Q-values of available actions—a local operation. Value prediction requires understanding the global game tree topology—much harder given the fragmentation.

**This is why Monte Carlo bidding (planned)** makes sense: instead of predicting V directly, simulate many games and average. MC handles rugged landscapes; smooth regression doesn't.

## What This Means for the Model

| Finding | Implication |
|---------|-------------|
| Fragmented level sets | Value landscape is rugged |
| V changes most moves | Discontinuous value function |
| Complex Reeb graph | No simple value decomposition |
| Local ≠ global | Move accuracy ≠ value accuracy |

**Bottom line**: The topology explains the value head's limitations. For bidding decisions that need V estimates, Monte Carlo simulation is the right approach—not regression on a fragmented landscape.

---

*Next: [06 Scaling Analysis](06_scaling.md)*
