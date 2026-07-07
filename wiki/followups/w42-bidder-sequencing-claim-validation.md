Reviewed against code on 2026-07-07 — no issues found.

Verified: all artifact paths under `w42/bidder_sequencing_claim_validation/` exist; findings-table numbers, count_pressure bucket stats (5.296599 → 2.908336 on n=113), baseline epochs, E[Q] N=10 figures, and commit `c603a0d9...` all match `summary.json` / `claim_summary.csv` / `bucket_proxy_stats.csv` / `claim_ledger_delta.json`.

- Source corpora (`gus/data/corpus_train_100.pt`, `corpus_eval_20.pt`) are absent as the page itself notes — unverifiable, but the page already flags this correctly.
- Cheap next probe: rerun `analyze.py`-style aggregation on any future run that retains row-level predictions, so paired Newcombe intervals replace the conservative independent-proportion CIs.
- The proposed `last_trump_spent_before_off_clear` detector in Next Checks remains the highest-leverage unbuilt piece for the reentry claim.
