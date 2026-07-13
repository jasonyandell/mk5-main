---
title: Q-value model slot-0 positional bias (forge/zeb investigation)
kind: topic
first_seen: 2026-01-06
last_updated: 2026-01-31
status: active
---

## The anomaly

The [[forge]] Q-value model exhibits systematic bias when reading hand position 0
versus other slots — a slot-swap test changes Q outputs in a way that should not
depend on position if the model were truly invariant under domino-set permutation.

`forge/analysis/scripts/slot_swap_test.py` and `q0_bias_check.py` quantify it; the
bias is robust and large enough to drive measurable Q errors at hand-evaluation time.

## Origin (2026-01-25..26, era 3)

The investigation was triggered mid-[[eq-genesis]] (era 3) by a mispredicted slot-0 anomaly
during E[Q] training-data work. `0feee02` (2026-01-25, "complete slot 0 positional bias
investigation") is the largest non-checkpoint commit of the era — 53 files, 8,132 insertions,
18-20 numbered analysis scripts — and the methodology is the lesson as much as the fix: "20
parallel investigations ruled out architectural causes" (no positional encoding issue, balanced
attention, identical LayerNorm stats) before the actual, mundane cause was found: `deal_from_seed()`
sorted hands by domino ID, so slot 0 only ever saw low-pip dominoes in training — a 1.74-bit KL
divergence baked into the data by data *provenance*, not architecture. The fix (`9f73c42`,
`440aeea`, shuffle hand slots at training time) moved slot-0 tie rate 0.38 → 0.17 (theoretical
target 0.14) at a small, honest cost to aggregate Q-gap (0.071 → 0.074, `domino-qval-3.3M-shuffle`
checkpoint) — a measured trade, not a clean win. See [[eq-genesis]] and
[[expected-q-value]].

## The investigation

`forge/analysis/bias/` contains 20 numbered probes (`01-attention-mask.md` through
`20-proposed-fix-shuffle.md`), each ruling in or out one mechanism for the bias.
Summary:

| # | Hypothesis | Verdict |
|---|---|---|
| 01 | Causal masking creating a "first token sees nothing" artifact | **Ruled out** — model uses bidirectional attention, no `is_causal` flag, only padding mask. |
| 02 | Tokenization layout placing structurally distinct content at slot 0 | Investigated — partial contributor only. |
| 03 | Positional encoding biasing slot 0 | Investigated — see #07. |
| 04 | Output-head bias toward slot 0 | Investigated — not the dominant cause. |
| 05 | Padding convention (right-padded vs left-padded) | Ruled out as primary. |
| 06 | Attention pattern inspection | Patterns at slot 0 differ; this is *symptom*, not cause. |
| 07 | Learned PE weights at position 0 | Anomalous — slot 0 has unique PE values. |
| 08 | "BOS token ghost" — pretrained-model artifact at position 0 | **Ruled out** — model is trained from scratch, no pretrained weights, no built-in BOS handling in `nn.TransformerEncoder`. |
| 09 | RoPE position 0 special case | Ruled out — not using RoPE. |
| 10 | Attention routing | Slot 0 is over-routed-to. |
| 11 | Context adjacency | Slot 0's neighbors are structurally different. |
| 12 | Domino ordering convention | Contributor — sort order privileges slot 0. |
| 13 | Action frequency imbalance at slot 0 | Contributor. |
| 14, 14b, 14c | Embedding analysis at slot 0 | Embedding magnitudes anomalous; `bias/14_embedding_stats.json` records the distribution. |
| 15 | Transformer edge effects | Contributor — confirmed. |
| 16 | Gradient flow during training | Slot 0 receives biased gradients. |
| 17 | Attention head specialization | Some heads specialize for slot 0. |
| 18 | Layer-wise degradation | `bias/18_layer_wise_results.json`; bias amplifies through layers. |
| 19 | LayerNorm statistics | Contributor — stats differ at slot 0. |
| 20 | **Proposed fix: input shuffle** | Output: shuffle slots before each forward pass to prevent the model from learning a slot-0-specific shortcut. Not yet trained-and-validated. |

The synthesis lives in `forge/analysis/bias/POSITIONAL-BIAS-ANALYSIS.md`.

## Why this matters

Anything in [[forge-analysis]] that compares Q-model output against oracle ground
truth at the slot level inherits this bias. Findings that aggregate over slots
(E[V], σ(V) at the hand level) are largely unaffected. Findings that compare
specific Q values per-action-position (e.g. "does the model choose the right
domino?") are biased.

The bias also affects [[zeb]]'s belief calibration — slot 0 contributes
disproportionately to the calibration gap that drove zeb's parking decision (`zeb`
calibration eval, eventual parking documented in `wiki/decisions/zeb-parked-eq-primitive.md`).

## Status

Investigation closed at `4a747f6` with a proposed fix (#20) but not validated. Any
future model retraining (e.g. [[gus]] q-head augmentation work) should incorporate
the shuffle remedy before reading further conclusions about positional Q bias.

## Links

[[forge]] [[eq-genesis]] [[expected-q-value]] [[zeb]] [[forge-analysis]] [[gus]]
