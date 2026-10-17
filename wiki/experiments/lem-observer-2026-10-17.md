---
title: "Lem's first tiny observer: a worked hand and next-tile imitation"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
---

## Registration

Jason selected the observer-first lesson in [[lem-nanogpt-foundations]] on
2026-10-17. The immediate aim is a small model that learns something visible
from existing play records, with one inspectable hand. Population construction,
reward learning and neuron naming are later work.

Before the first training run, the predictions are:

1. A two-block, width-64, two-head causal transformer trained for 2,000 updates
   on the first 5,000 Pour A deals improves held-out next-tile log loss over its
   saved initialization and uniform guessing over 28 tiles.
2. It predicts its own plays more accurately than other seats' plays: its own
   original hand supplies information absent for the other seats.
3. It still places nonzero probability on visibly impossible tiles. Legality
   is measured on raw probabilities; no legal-action mask repairs the model.

No prediction is registered about beating the rule-based visible-support
baseline. That baseline guesses uniformly among own legal moves, or among
unplayed tiles outside the viewer's hand and the acting seat's publicly known
voids. It does not see another seat's actual hand.

## Protocol

`lem/observer/` follows the pinned nanoGPT architecture with pre-normalized
residual blocks, explicit causal attention, GELU MLPs, learned positions and
tied tile embeddings. A sequence is one viewer's original seven tiles,
declaration and fixed seat-0 bid-30 contract, then 28 public seat/tile pairs.
Seats are relative to the viewer. The loss supervises only next tiles.

Whole physical deals are hashed into 80/10/10 train/validation/test buckets
before making four viewer sequences. Checkpoints every 250 updates; the best
validation loss selects the test checkpoint. AdamW, learning rate 0.001,
batch 64, seed 42. No architecture or hyperparameter sweep. The worked hand is
the first test deal in seed order, with all four viewpoints available.

Only raw `hands`, `decl` and `plays` arrays are loaded; no values or splats.
The referee checks recordings and labels forced moves for reporting. Actual
opponent legality is never an input or a prediction filter. Each prediction
receives the recorded past, so this is teacher-forced prediction, not an
autonomously played game or a playing-strength measurement.

The inspection records native attention and MLP preactivations. This net,
these inputs (own initial hand and within-hand public history), and no concept
vocabulary yet: no clean-share or causal neuron claim is made.

## Sources

- `lem/observer/`, `lem/tests/test_observer.py`
- [[lem-pour-a-2026-10-16]]; [[clean-neurons-42]]
