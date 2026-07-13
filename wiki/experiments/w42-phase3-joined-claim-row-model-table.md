---
title: w42 Phase3 Joined Claim Row Model Table
kind: experiment
first_seen: 2026-05-03
last_updated: 2026-07-06
status: complete
---

## Summary

[[w42]] bead `t42-qtwb.4` normalizes the phase-2 and phase-3 claim-family
artifacts into one public-safe legal-action table and reruns the cheap row-model
probe with family ablations. It continues [[w42-claim-tag-model-probe]], but no
longer treats bidding, 84, doubles/no-trump, or hidden-threat proxies as simply
unavailable.

The joined table has 75079 legal action rows and 28000 decision states. It
includes full-coverage sequence/seat labels from
[[w42-phase3-sequence-seat-counterfactuals]], partial auction-risk joins from
[[w42-phase3-auction-bid-discipline-corpus]], partial public 84 bidder-structure
joins from [[w42-phase3-84-seed-mining-corpus]], declaration/action-local
doubles/no-trump tags, and public hidden-pressure proxies. Hidden-owner truth,
84 defender assets, and auction partner/opponent value labels are written only
as audit/eval-only context, not model features.

The main result is positive model-feature evidence: on a seed-mod split with
5600 held-out decision states, public features alone select actions at `1.3598`
mean regret and `64.464%` best-mean match. Adding all public claim families
improves this to `1.1257` mean regret and `68.107%` best-mean match. Tail regret
`>=5` falls from `9.536%` to `7.250%`.

Claim-ledger impact: no central claim promotion. This proves that the joined
claim vocabulary carries action-selection signal, especially sequence/seat
signal. It does not prove every book claim behind those tags.

W&B: `https://wandb.ai/jasonyandell-forge42/w42/runs/jvjuld7m`.

## Method

| field | value |
|---|---|
| bead | `t42-qtwb.4` |
| artifact directory | `w42/joined_claim_row_model_table/` |
| runner | `w42/joined_claim_row_model_table/run_joined_claim_row_model_table.py` |
| validation | `w42/joined_claim_row_model_table/validate_outputs.py` |
| base rows | `w42/sequence_seat_counterfactuals/labeled_sequence_action_rows.csv` |
| action rows | 75079 |
| decision states | 28000 |
| split | seed `% 5 == 4` held out |
| train rows / decisions | 59853 / 22400 |
| eval rows / decisions | 15226 / 5600 |
| auction joined rows | 7216 |
| public 84 joined rows | 5365 |
| feature labels | 128 |
| W&B run | `jvjuld7m` |

The model is intentionally small: public numeric/categorical row features plus
optional one-hot claim tags, trained with balanced logistic regression to rank
legal actions inside each decision. Oracle mean/regret are labels and metrics
only.

## Family Inventory

| family | feature labels | tag action count | status |
|---|---:|---:|---|
| sequence/seat | 50 | 494943 | available |
| bidding risk | 39 | 54062 | available, partial seed/seat/declaration join |
| 84 public bidder structure | 19 | 27671 | available, partial seed/seat/declaration join |
| doubles/no-trump | 13 | 33979 | available from declaration/action-local rows |
| hidden public proxy | 7 | 133853 | available, public pressure proxy only |
| 84 defender assets | 0 | 17240 | eval-only hidden/full-deal labels |
| hidden truth owner | 0 | 0 | eval-only hidden labels |
| auction partner/opponent values | 0 | 0 | eval-only generated pass-value labels |

The `tag action count` column counts tag incidences, not unique rows; a single
action can carry several labels.

## Ablations

| variant | mean regret | best-mean match | tail regret >=5 | delta vs public |
|---|---:|---:|---:|---:|
| public features only | 1.359796 | 0.644643 | 0.095357 | 0.000000 |
| public + all claim families | 1.125741 | 0.681071 | 0.072500 | -0.234055 |
| drop bidding risk | 1.146082 | 0.679464 | 0.074286 | -0.213714 |
| drop doubles/no-trump | 1.128410 | 0.678393 | 0.074643 | -0.231386 |
| drop 84 public | 1.141520 | 0.680000 | 0.073571 | -0.218276 |
| drop hidden public proxy | 1.117872 | 0.681429 | 0.071964 | -0.241924 |
| drop sequence/seat | 1.367813 | 0.647857 | 0.093750 | +0.008017 |

The dominant feature family is sequence/seat. Dropping it erases the gain and
slightly underperforms the public baseline. Bidding risk and 84 public structure
have smaller positive contributions. Doubles/no-trump is nearly neutral in this
mixed all-declaration split. The hidden public proxy is slightly harmful here;
it remains useful as a diagnostic/reporting surface, but should not be promoted
as a model feature without sharper belief calibration.

The `actual_policy` reference row is not a trainable baseline: it records the
source corpus's actual chosen actions and scores at `0.1226` mean regret. It is
kept only as an offline reference for the quality of the saved corpus actions.

## Leakage Boundary

Model features include public row context, candidate action facts, public
sequence/seat labels, public static bidding-risk labels where joined, public
84 bidder-structure labels where joined, declaration/action-local
doubles/no-trump labels, and public pressure proxies.

Excluded from features:

- oracle mean/regret, threshold mass, and lower-tail mass;
- hidden owners and full private opponent hands;
- `q_per_world`, `world_hands`, and future outcomes;
- 84 defender live-asset labels derived from full-deal opponent assets;
- generated auction partner/opponent pass values.

## Interpretation

This closes the phase-3 loop that began with the crash recovery. The book
vocabulary is not just decoration: joined public claim-family tags improve a
held-out action-ranking model and reduce tail mistakes. The improvement is
mostly tactical and sequence-local, matching the earlier empirical story.

The conservative claim ledger still matters. Model usefulness is not claim
truth. The result says the vocabulary is worth carrying forward into Gus/Burl
features and future generated tests, especially for pounce, closure, follow
control, count safety, and bidder lead-plan exceptions.

## Validation

```bash
python -m py_compile \
  w42/joined_claim_row_model_table/run_joined_claim_row_model_table.py \
  w42/joined_claim_row_model_table/validate_outputs.py

python w42/joined_claim_row_model_table/run_joined_claim_row_model_table.py \
  --output-dir w42/joined_claim_row_model_table \
  --max-iter 500 \
  --wandb-mode online \
  --wandb-group w42-joined-claim-row-model-table \
  --wandb-name t42-qtwb.4-joined-claim-row-model-table-v0

python w42/joined_claim_row_model_table/validate_outputs.py \
  --artifact-dir w42/joined_claim_row_model_table \
  --min-actions 75000 \
  --min-eval-decisions 5600 \
  --max-public-regret 10
```

## Links

[[w42]] | [[w42-claim-tag-model-probe]] |
[[w42-phase3-sequence-seat-counterfactuals]] |
[[w42-phase3-auction-bid-discipline-corpus]] |
[[w42-phase3-84-seed-mining-corpus]] |
[[w42-doubles-no-trump-legacy-mining]] |
[[w42-hidden-threat-legacy-mining]] |
[[w42-claim-analysis-synthesis-report]]
