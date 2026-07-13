# Cross-AI Agreement Analysis (t42-m2i7)

**Wave:** 1.4  **Parent epic:** t42-4zi6

## Question

Where do book detectors, the Gus row-model, scalar EV, and the
distribution-lens reranker disagree? Those decisions are the most
pedagogically valuable.

## Slice

- **Primary corpus:** `joined_claim_action_rows.csv` — 28 000 decisions,
  75 079 action rows, corpus_v2_train, seeds 0–99.
- **Dist-lens sub-corpus:** `branch_atlas_scaled_v0` / Wave 1.1 —
  280 decisions, seed 9430 (1.0% overlap with primary corpus).

## N

- 28 000 decisions (primary), 75 079 action rows.
- 280 decisions with dist-lens coverage.

## Sources

| Label | Definition |
|-------|------------|
| `ev_top_action` | `is_best_mean` from joined table (mean_regret == 0); exact EV oracle. |
| `gus_top_action` | `is_best_threshold` from labeled_handshape (proxy; actual model scores not saved). |
| `detector_endorsed` | Any ch03/ch04/ch05 positive-label from `paired_contrasts.csv`. |
| `dist_lens_top_action` | Any non-EV utility top-1 from Wave 1.1; deferred for 27 720/28 000 decisions. |

## Metric

Top-1 pick agreement rate (fraction of decisions where source A and source B
select the same candidate domino); mean_regret of each source's pick.

## Key Findings

See `summary.json` for headline numbers. Run `python3 run_cross_ai_agreement.py`
to reproduce.

## Caveats

- `gus_top_action` is a proxy (is_best_threshold), not the actual model output.
  Aggregate proxy-vs-EV agreement rate: 78.3%.
- `dist_lens_top_action` covers only 280/28 000 decisions (1.0%); deferred.
- Detectors cover ch03/ch04/ch05 only; 84-domain and bidding-domain detectors
  use different corpora and are not joinable here.

## Claim-ledger impact

`underpowered` — agreement analysis is diagnostic, not a direct claim test.
