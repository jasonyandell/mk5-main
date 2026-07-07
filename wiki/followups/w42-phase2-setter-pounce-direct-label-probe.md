Reviewed against code on 2026-07-07 — no issues found.

- All headline numbers verified against `w42/gus_corpus_claim_deep_dive/paired_contrasts.csv` and `summary.json` (28,000 decisions / 75,079 actions; the three contrast rows match to rounding).
- Tiny-slice proxy numbers (+5.111 n=52, +4.353 n=4, n=0) match `w42/setter_pounce_direct_label_probe/tiny_report_slice.csv`; corpus field inspection matches `summary.json`.
- Follow-up: the "Required Phase-2 Data" table is now largely satisfied by the deep-dive corpus (bid_value, per-seat masks, declarations exist per the deep-dive page); a cheap next probe would be running the spec'd direct labels on real bid-margin / high-bid contexts (ch12) which remain uncovered.
