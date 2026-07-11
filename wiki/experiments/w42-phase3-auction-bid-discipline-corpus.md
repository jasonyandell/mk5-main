---
title: w42 Phase3 Auction Bid Discipline Corpus
kind: experiment
first_seen: local-2026-05-03
last_updated: afd4802
status: complete
---

## Summary

[[w42]] bead `t42-qtwb.1` promotes the Chapter 2/12 bidding work from fixed
same-contract margin arithmetic into a generated auction-pressure corpus. It
builds full deals, estimates each seat's strongest pip-declaration contracts
with `forge.bidding`, then expands P0 decisions into opening, partner-high, and
opponent-high bid/pass/overcall contexts.

The result strengthens the operational "bid only enough" claim: across 14976
positive-margin bid-action rows, no higher-than-minimum bid improves over the
minimum winning bid under the same hand/declaration/context label. The broader
book claims remain narrower. Natural bid buckets are now measurable as empirical
max-profitable threshold buckets, but only 17.708% of contract labels land on
the chapter's natural buckets. Partner bid signal is real-looking but not a
blanket boost: P0 overcalls partner in 77 of 192 generated partner-high contexts
under this proxy.

Claim-ledger impact: no broad central ledger promotion. The local recommendation
is stronger support for the bid-only-enough operational slice, context-limited
evidence for risk-budget/natural-bucket reporting, and a sharper blocker for
partner-signal claims until a real or learned auction policy exists.

## Method

| field | value |
|---|---|
| bead | `t42-qtwb.1` |
| artifact directory | `w42/auction_bid_discipline_claim_tests/` |
| script | `w42/auction_bid_discipline_claim_tests/run_auction_bid_discipline.py` |
| validation | `w42/auction_bid_discipline_claim_tests/validate_outputs.py` |
| generated deals | 32, seeds `0..31` |
| candidate labels | 3 static-best pip declarations per seat |
| samples | 64 `forge.bidding` greedy simulations per contract |
| auction contexts | opening plus partner/left-opponent/right-opponent high bids at 30, 31, 32, 34, 35, 36 |
| W&B | `https://wandb.ai/jasonyandell-forge42/w42/runs/6cup1bat` |

The script evaluates each seat's hand as if that hand were the bidder hand in
the existing `forge.bidding` Monte Carlo simulator. For P0 it emits every legal
bid from the minimum winning bid through 42. For partner and opponents it uses
their estimated contract value as an offline pass-value comparison:

- partner high bid: pass value is partner's contract mark swing;
- opponent high bid: pass value is the negative of the opponent's contract mark
  swing;
- opening auction: pass value is `0`.

This makes the corpus auction-aware without claiming to be a fully conditioned
four-hand auction engine.

## Artifacts

| artifact | content |
|---|---|
| `contract_rows.csv` | 384 seat/declaration contract labels with static risk features and bid-threshold make labels. |
| `point_samples.csv` | raw point samples behind each contract row. |
| `auction_context_rows.csv` | 608 generated auction contexts. |
| `bid_action_rows.csv` | 16800 P0 bid-action rows with deltas versus pass and versus minimum bid. |
| `context_decision_rows.csv` | 608 pass-vs-best-overcall context decisions. |
| `bid_margin_summary.csv` | margin-level bid-only-enough contrasts. |
| `bid_margin_by_context_summary.csv` | margin contrasts split by high-bid owner and current high bid. |
| `natural_bucket_summary.csv` | max-profitable threshold buckets by seat and static risk bucket. |
| `risk_budget_summary.csv` | empirical `P(make 30)` and profitable-bid rate by static risk bucket. |
| `partner_signal_summary.csv` | partner-high overcall/pass summary. |
| `opponent_pressure_summary.csv` | opponent-high overcall/pass summary. |

## Findings

| claim slice | result | interpretation |
|---|---:|---|
| Bid above minimum improves over minimum | 0 / 14976 positive-margin rows | Strong support for the operational bid-only-enough slice. |
| Bid above minimum worsens over minimum | 12872 / 14976 positive-margin rows | Extra margin usually costs value under fixed hand/declaration labels. |
| Bid above minimum ties minimum | 2104 / 14976 positive-margin rows | Ties occur when no sampled points fall between the two thresholds. |
| Minimum bid beats pass | 337 / 1824 minimum-bid rows | Many generated contexts are still pass contexts; the corpus can separate bid amount from bid/pass decision. |
| Partner-high overcall | 77 / 192 contexts, 40.104% | Partner signal is conditional, not an automatic pass or automatic confidence boost. |
| Natural-bucket max-profitable contracts | 68 / 384 contracts, 17.708% | The book buckets appear, but most generated contracts are either not profitable at 30 or land outside those exact centers. |

Static risk buckets move in the expected direction but are not decisive. Mean
`P(make 30)` is 0.421 for `risk_le_12`, 0.387 for `risk_13_20`, and 0.331 for
`risk_gt_20`. Profitable-contract rates are 30.994%, 24.623%, and 14.286% for
the same buckets. The static risk budget is useful screening vocabulary, not a
standalone bidding policy.

Natural bucket distribution:

| max-profitable bucket | contracts | rate |
|---|---:|---:|
| no profitable bid | 280 | 72.917% |
| natural 30/31 | 41 | 10.677% |
| natural 35/36 | 25 | 6.510% |
| high 37-41 | 19 | 4.948% |
| 42 | 9 | 2.344% |
| 34 | 8 | 2.083% |
| natural 32/33 | 2 | 0.521% |

Partner-high contexts:

| current high bid | overcall contexts | total | rate |
|---:|---:|---:|---:|
| 30 | 11 | 32 | 34.375% |
| 31 | 12 | 32 | 37.500% |
| 32 | 14 | 32 | 43.750% |
| 34 | 13 | 32 | 40.625% |
| 35 | 14 | 32 | 43.750% |
| 36 | 13 | 32 | 40.625% |

Opponent-high contexts are more naturally pressure-sensitive: overcall rates
fall from 57.812% at opponent high 30 to 7.812% at opponent high 36.

## Caveats

- The corpus is generated from `forge.bidding`, not from human logs or a learned
  auction policy. It cannot measure table response to reputation or wild bids.
- Partner/opponent pass values are offline labels from each seat's bidder-hand
  Monte Carlo evaluation. They are useful comparison targets, not live features.
- The run is report-scale, not publication-scale: 32 deals and 64 samples per
  contract are enough to route claims and catch direction, but not enough for
  final confidence intervals on narrow buckets.
- Only pip declarations are used. No-trump, doubles-as-trump, 84, and score-mode
  variants stay outside this bead.
- Natural bucket evidence is descriptive. It does not prove that 32/33 are table
  mistakes; it shows they are rare as max-profitable thresholds in this generated
  slice.

## Validation

```bash
python -m py_compile \
  w42/auction_bid_discipline_claim_tests/run_auction_bid_discipline.py \
  w42/auction_bid_discipline_claim_tests/validate_outputs.py

python w42/auction_bid_discipline_claim_tests/run_auction_bid_discipline.py \
  --out-dir w42/auction_bid_discipline_claim_tests \
  --seeds 32 \
  --samples 64 \
  --decls-per-hand 3 \
  --device cpu \
  --smoke \
  --wandb-mode online \
  --wandb-group w42-auction-bid-discipline \
  --wandb-name t42-qtwb.1-auction-bid-discipline-v0

python w42/auction_bid_discipline_claim_tests/validate_outputs.py \
  --artifact-dir w42/auction_bid_discipline_claim_tests \
  --min-contract-rows 384 \
  --min-bid-action-rows 16000
```

## Links

[[w42]] | [[w42-bidding-risk-budget-claim-validation]] |
[[w42-claim-analysis-synthesis-report]] | [[winning42-ch02-bidding]] |
[[winning42-ch12-advanced-bidding-playing]] | [[winning42-ch16-statistical-odds]] |
[[forge]]

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- The W&B run (`6cup1bat`) is external and unverified; the local `summary.json` records the same URL.
- Cheap next probe: rerun with 128 deals to tighten the thin `risk_gt_20` bucket (n=14) and the natural 32/33 cell (n=2), both too small for stable rates.
- The 0/14976 bid-above-minimum result is structurally guaranteed under fixed hand/declaration labels — a higher bid can only raise the make threshold; a follow-up could note this analytically rather than empirically.
