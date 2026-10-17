---
title: "Lem's first tiny observer: a worked hand and next-tile imitation"
kind: experiment
first_seen: 2026-10-17
last_updated: 2026-10-17
status: complete
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

## Result

The registered run completed on local MPS: 107,392 parameters, 3,988 training
deals, 509 validation deals and 503 test deals. All four views of a deal share
its split. Step 2,000 had the best validation loss. No second run or tuning was
performed. Training code and registration: `3aa1f435`; uncertainty and inspection
verification: `5cc363a2`.

| Test measure | Trained | Untrained twin | Uniform over 28 tiles | Uniform over visible support |
|---|---:|---:|---:|---:|
| Exact next-tile accuracy | 39.56% [39.01, 40.15] | 3.34% [3.19, 3.52] | 3.57% expected | 29.32% [29.19, 29.45] expected |
| Log loss, lower is better | 1.755 [1.736, 1.773] | 3.348 [3.346, 3.350] | 3.332 | 1.749 [1.746, 1.751] |
| Probability on visibly impossible tiles | 17.44% [17.18, 17.72] | 69.41% | Varies with visible support | 0% by construction |

Intervals are percentile 95% intervals from 2,000 whole-deal bootstrap draws,
seed 42. Each draw keeps the four views and all moves of a deal together.
They condition on this trained model and this pour, not training-seed or
population uncertainty. Accuracy uses the top raw prediction; random-baseline
accuracy is its expectation, not a sampled random run.

Own-play accuracy is 61.66% [60.83, 62.52]; other-seat accuracy is 32.19%
[31.62, 32.79]. On discretionary moves it is 27.99% [27.31, 28.69], compared
with 3.25% [3.04, 3.47] for the twin and 16.78% expected for visible-uniform
guessing. Forced moves are included in the headline and separated in the
artifact. Coverage-forced declarations are also reported separately.

The model beats initialization and blind guessing, and its most likely tile
outperforms a uniform visible-support draw. Its probability distribution is
not better established than that stronger baseline: paired log-loss difference
is +0.0063 [-0.0119, +0.0240]. Own legality remains particularly unfinished:
26.32% [25.91, 26.76] probability falls outside the viewer's legal moves on
its own turns. No legality mask hides this result.

All three registered predictions held. This is the requested first observer
milestone; no playing-strength, population-learning or neuron-meaning result
is claimed. The attention and native MLP traces are available for a subsequent
[[clean-neurons-42]] investigation, with the scope above.

## Worked hand and durable evidence

The standalone lesson selects test deal **400006**, five-trump, without
searching for a favorable example. At its opening, seat 0 holds
`1-0 4-3 5-4 6-1 6-4 6-5 6-6`. The model's highest-probability tile is `5-4`
(41.2%), while the recorded player leads `6-6` (rank 5, 3.7%). The lesson lets
the reader predict, reveal, advance/backtrack, change viewer, and inspect each
attention head and each native MLP preactivation. Browser checks exercised
those controls, all 28 choices, the last-play boundary and the error log.

The [frozen evidence directory](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-research-2026-10/tree/242f07beededcd8d1750c6cdda54e72d5449cee9/lem/observer-2026-10-17)
contains initialization, every 250-step checkpoint, best/latest, exact selected
recordings and splits, data hashes, training metrics, bootstrap intervals,
the training log and `worked-hand.html`. The best checkpoint was downloaded
into a fresh directory and its SHA256 matched:
`e453ae1524f5226f74fbc9427be4743e6c7df87a2918f5382159b49276f9a5d3`.

`scripts/cpu-slot` recorded 1.734e11 CPU instructions, 13.0 CPU seconds and
16.4 wall seconds for the run including loading and evaluation. Training
itself used MPS; CPU instructions do not measure the GPU's work. These are
descriptive receipts, not a backend comparison.

## Verification

The observer tests check target alignment, initial-hand prefixes, hidden/future
noninterference, causal attention, batch isolation, legal replay, deal grouping,
evaluation RNG independence, exact CPU checkpoint continuation, rejection of
changed resume data, bootstrap aggregation and the exported inspection payload.
The full repository check was GREEN: 20 suites passed, no failures; CUDA forge
was unavailable on this machine and retired Burl is not run by the checker.
