---
title: w42 Phase3 Sequence Seat Counterfactuals
kind: experiment
first_seen: local-2026-05-03
last_updated: local-2026-05-03
status: active
---

## Summary

[[w42]] bead `t42-qtwb.3` extends the phase-2 seat/position map into direct
branch-value counterfactuals at naturally occurring sequence states. The run
consumes the 75079 full legal-action rows from [[w42-tactical-claim-replication]]
and derives book-shaped labels for bidder lead plans, follow-seat obligations,
last-seat closure, partner support timing, and setter pounce windows.

The strongest results support action-local table wisdom: followers should take
control from an opponent when they can, last seat should usually close/take the
trick when both options exist, setters should pounce on count when the bidder
side is winning, and unsafe count into opponent control is heavily punished.
Partner count support is positive but small and timing-sensitive.

The bidder lead-plan result is the important nuance. A pooled called-suit lead
versus off-suit lead is negative in this corpus (`-1.935` Q across 1657 paired
states), including first-trick slices. But when the called-suit choice is the
called double versus a lower called-suit tile, the double lead is strongly
positive (`+5.720` Q across 182 pairs). This argues against a blanket "always
lead trump/called suit first" reading and for the narrower book shape: lead
commanding trump/double plans when they are the right plan, but preserve
exception buckets for early offs, count inventory, and low-trump command plays.

Claim-ledger impact: no broad central claim promotion. Local recommendations are
strong support for follow-seat control, last-seat closure, setter pounce, and
unsafe-count negative controls; partner support remains context-limited; bidder
lead sequencing remains context-limited and needs hand-shape/state-injection
follow-up.

W&B: `https://wandb.ai/jasonyandell-forge42/w42/runs/kp0otpdo`.

## Method

| field | value |
|---|---|
| bead | `t42-qtwb.3` |
| artifact directory | `w42/sequence_seat_counterfactuals/` |
| runner | `w42/sequence_seat_counterfactuals/run_sequence_seat_counterfactuals.py` |
| validation | `w42/sequence_seat_counterfactuals/validate_outputs.py` |
| input | `w42/tactical_claim_replication/all_action_rows.jsonl` |
| action rows | 75079 |
| decision states | 28000 |
| paired contrasts | 14 |
| paired slice rows | 394 |
| W&B run | `kp0otpdo` |

Rows are legal candidate actions in an already observed public sequence state.
The branch labels are offline E[Q]-style values. This is a direct
same-public-state action counterfactual, not a fully divergent multi-trick human
plan-tree simulation.

## Core Contrasts

| contrast | paired decisions | mean delta | interpretation |
|---|---:|---:|---|
| bidder called-suit lead vs off-suit lead | 1657 | -1.935 | Blanket called-suit-first is not supported in this corpus. |
| bidder called double vs lower called-suit lead | 182 | +5.720 | Boss/double trump plans are strongly supported when both are available. |
| bidder lower called-suit lead vs called double | 182 | -5.720 | Low-trump-first is not a broad rule; it needs exception gates. |
| bidder count lead vs non-count lead | 1844 | -4.877 | Count inventory matters; casual count-leading is punished. |
| follower take control from opponent vs decline | 1870 | +11.564 | Follow-seat control is one of the strongest supported slices. |
| follower count into opponent control vs non-count | 3526 | -7.322 | Unsafe count into opponent control is sharply bad. |
| partner count when bidder side controls vs other | 614 | +0.608 | Partner support count is small-positive, not automatic. |
| partner count into defense control vs other | 802 | -8.351 | Partner count into defensive control is strongly negative. |
| setter pounce count vs other | 489 | +3.846 | Setter pounce survives phase-3 relabeling. |
| setter pounce sets-now vs other | 174 | +5.930 | Immediate set pounces are stronger. |
| setter reckless count to bidder vs other | 2027 | -7.829 | Negative control is very strong. |
| closure take trick vs decline | 808 | +10.936 | Last-seat closure is strongly supported. |
| closure take count vs other | 277 | +4.666 | Taking count from closure is supported. |
| closure slough count vs other | 1803 | -6.377 | Sloughing count while declining is punished. |

## Slice Notes

Follow-seat control is robust across phases: `+16.669` Q early, `+9.232` middle,
and `+6.216` late. Last-seat closure has the same shape: `+16.007` Q early,
`+8.614` middle, and `+4.529` late.

Setter pounce count remains positive across early, middle, and late slices, and
is largest from later positions: `+2.315` Q from second seat, `+5.658` Q from
third seat, and `+6.755` Q from last seat.

Partner support is the caution case. The pooled supported-count delta is
positive (`+0.608` Q), but trick timing varies: trick 0 is positive, trick 1 and
trick 2 are negative/uncertain, trick 4 and trick 5 are stronger positive. The
book concept is real enough to keep, but only with control/timing gates.

Bidder lead sequencing needs better hand-shape gates. The pooled called-suit
lead contrast is negative in ordinary pip declarations, near zero in doubles,
and positive in doubles-suit. The double-vs-lower-called-suit contrast is
positive across pip declaration slices. That pattern supports a narrower
"commanding trump/double" detector and argues for future generated tests with
trump count, off count, reentry, and live-count inventory.

## Artifacts

| artifact | content |
|---|---|
| `summary.json` | coverage, W&B link, leakage boundary, and run metadata. |
| `label_metrics.csv` | label-level action metrics. |
| `paired_contrasts.csv` | 14 main paired counterfactual summaries. |
| `paired_contrasts_by_slice.csv` | declaration, phase, trick, role, position, count, and control slices. |
| `labeled_sequence_action_rows.csv` | compact row table with derived sequence labels. |
| `examples.json` | positive and negative human-readable examples for each contrast. |
| `manifest.json` | artifact manifest and boundary statement. |

## Validation

```bash
python -m py_compile \
  w42/sequence_seat_counterfactuals/run_sequence_seat_counterfactuals.py \
  w42/sequence_seat_counterfactuals/validate_outputs.py

python w42/sequence_seat_counterfactuals/run_sequence_seat_counterfactuals.py \
  --output-dir w42/sequence_seat_counterfactuals \
  --bootstrap-samples 2000 \
  --min-label-n 20 \
  --min-paired-n 20 \
  --min-slice-n 20 \
  --wandb-mode online \
  --wandb-group w42-sequence-seat-counterfactuals \
  --wandb-name t42-qtwb.3-sequence-seat-counterfactuals-v0

python w42/sequence_seat_counterfactuals/validate_outputs.py \
  --artifact-dir w42/sequence_seat_counterfactuals \
  --min-actions 75000 \
  --min-paired-contrasts 14 \
  --min-slice-rows 300
```

## Links

[[w42]] | [[w42-phase2-seat-position-strategy-map]] |
[[w42-tactical-claim-replication]] |
[[w42-claim-analysis-synthesis-report]] |
[[winning42-ch03-bidder-play]] |
[[winning42-ch04-partner-support]] |
[[winning42-ch05-setter-defense]]
