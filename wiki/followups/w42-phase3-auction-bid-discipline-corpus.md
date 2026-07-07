Reviewed against code on 2026-07-07 — no issues found.

All headline numbers (0/14976, 12872/14976, 2104/14976, 337/1824, 77/192, 68/384 = 17.708%) match `w42/auction_bid_discipline_claim_tests/summary.json`; CSV row counts, risk-bucket means/rates, natural-bucket distribution, partner-high per-bid table, and opponent-pressure rates all recomputed from the CSVs and match; script CLI flags in the Validation block match `run_auction_bid_discipline.py` / `validate_outputs.py`.

- The W&B run (6cup1bat) is external and was not verified; the local summary.json records the same URL.
- Cheap next probe: rerun with 128 deals to tighten the thin `risk_gt_20` bucket (n=14) and the natural_32/33 cell (n=2), both too small for the reported rates to be stable.
- The 0/14976 bid-above-minimum result is structurally guaranteed given fixed hand/declaration labels (higher bid can only raise the make threshold); a follow-up could note this analytically rather than empirically.
