---
title: w42 Doubles No-Trump Legacy Mining
kind: experiment
first_seen: 2026-05-03
last_updated: 2026-07-06
status: superseded
---

**Superseded** by [[w42-phase4-doubles-notrump-regime-tests]], which
delivers the generated paired same-hand regime comparison this page's own
"Next Checks" named as its central blocker (192 same-hand regime pairs,
66.1% no-trump-preferred). The three narrower proxy-sharpening TODOs (low-double
sacrifice state gates, support-double preservation label, suit-count/walker
labels) remain genuinely open.

## Summary

[[w42]] now has a direct legacy-corpus Chapter 9 mining pass for
[[winning42-ch09-doubles-no-trump]]. It follows [[w42-claim-data-inventory]]:
the large Gus legacy corpus is not paired same-hand regime data, but it does
contain enough declaration-7 and declaration-9 play states to test within-regime
tactical proxies.

Epistemic status: powered proxy-label report. It moves Chapter 9 beyond static
odds for action-local doubles-trump/no-trump slices, but it does not validate
same-hand "no-trump over doubles-trump" declaration choice.

## Artifacts

| artifact | path |
|---|---|
| runner | `w42/doubles_no_trump_legacy_mining/run_legacy_mining.py` |
| summary | `w42/doubles_no_trump_legacy_mining/summary.json` |
| label metrics | `w42/doubles_no_trump_legacy_mining/label_metrics.csv` |
| paired contrasts | `w42/doubles_no_trump_legacy_mining/paired_contrasts.csv` |
| examples | `w42/doubles_no_trump_legacy_mining/examples.json` |
| paired examples | `w42/doubles_no_trump_legacy_mining/paired_examples.json` |
| manifest | `w42/doubles_no_trump_legacy_mining/manifest.json` |

W&B run: `https://wandb.ai/jasonyandell-forge42/w42/runs/o96omty7`.

## Data Slice

The run processes all 100 legacy `gus/data/corpus_train_chunk_*-*.pt` files.

| field | value |
|---|---:|
| source files | 100 |
| declaration slice | 7 doubles-trump and 9 no-trump |
| decision rows | 56000 |
| legal action rows | 149415 |
| labeled action rows | 25906 |
| wall time | 164.75 seconds |

The script emits compact metrics and examples only. It does not emit a full
legal-action JSONL.

## Findings

The strongest positive signal is no-trump double action value. Early no-trump
double-spend proxy actions beat same-decision non-double alternatives by about
`+5.76` Q with CI `[+5.46, +6.05]`. Late no-trump double-spend proxy actions
beat non-double alternatives by about `+3.03` Q with CI `[+2.65, +3.45]`.
Defender double-weapon actions beat same-decision alternatives by about `+12.80`
Q with CI `[+11.18, +14.46]`.

This supports the Chapter 9 idea that no-trump doubles are live control/support
assets. It does not support a simplistic "always preserve the double" rule:
in this proxy, spending the double is often the high-value action when it wins
or controls the trick.

Doubles-trump opening low-double sacrifice is not supported by the broad proxy.
Opening low-double leads are worse than non-low alternatives by about `-2.10` Q
with CI `[-2.39, -1.84]`. Against opening high-double controls in the same
decision the mean delta is about neutral (`+0.01` Q), but threshold mass is
lower by about `-0.027`. The proxy is missing the book's essential conditions:
missing higher doubles, planned loss budget, off-walker creation, and contract
survival after the sacrifice.

The `6-5` dual-suit top substrate is real but not action advice by itself.
`6-5` actions under doubles-trump trail same-decision alternatives by about
`-1.62` Q with CI `[-1.99, -1.25]`. This says the rule substrate matters, but
the label needs a state gate for when dual-suit protection is strategically
relevant.

The no-trump count-walker/capture proxy is too broad. It has high mean regret
and trails alternatives badly; the next label needs late-off ordering, suit
depletion, and support-double context instead of "non-double count action wins
now."

## Label Counts

| label | actions | decisions |
|---|---:|---:|
| `ch09_dt_opening_low_double_sacrifice_proxy` | 1036 | 720 |
| `ch09_dt_opening_high_double_control_proxy` | 481 | 432 |
| `ch09_dt_dual_suit_top_65_play` | 2282 | 2282 |
| `ch09_dt_trump_count_capture` | 1121 | 1121 |
| `ch09_nt_early_support_double_spend_proxy` | 11028 | 6424 |
| `ch09_nt_late_support_double_spend_proxy` | 5286 | 4127 |
| `ch09_nt_count_walker_or_capture_proxy` | 4672 | 3757 |
| `ch09_nt_defender_double_weapon_spend_proxy` | 697 | 697 |

## Claim-Ledger Impact

No central claim status changed. The result is strong enough to retire the
"static-only because no data" assumption for Chapter 9, but proxy labels remain
too broad for direct claim promotion.

The immediate ledger posture should be:

- static ruleset claims remain supported by [[w42-doubles-no-trump-claim-validation]];
- no-trump double weapon/control claims are now empirically promising but
  context-limited;
- low-double sacrifice and dual-suit-top action claims remain underpowered until
  the detectors add the book's state gates;
- no-trump-over-doubles-trump declaration choice still requires paired
  same-hand generation.

## Commands

```bash
python w42/doubles_no_trump_legacy_mining/run_legacy_mining.py \
  --bootstrap-samples 1000 \
  --wandb-mode online \
  --wandb-group w42-doubles-no-trump-legacy-mining \
  --wandb-name t42-0b4l.8-legacy-ch09-mining-v0

python -m py_compile w42/doubles_no_trump_legacy_mining/run_legacy_mining.py
```

## Next Checks

The next Chapter 9 pass should sharpen labels before moving statuses:

- low-double sacrifice must require first-trick bidder lead, missing higher
  doubles, planned loss budget, and a plausible off/walker payoff;
- no-trump support-double preservation needs a label for holding a live double
  when a tempting early spend is legal, not just spending doubles;
- no-trump suit-count/walker labels need public suit depletion and final-off
  ordering;
- same-hand regime comparison still needs generated paired declaration rows.

## Links

[[w42]] | [[w42-claim-data-inventory]] |
[[w42-doubles-no-trump-claim-validation]] |
[[winning42-ch09-doubles-no-trump]]
