Reviewed against code on 2026-07-07 — no issues found.

- All headline numbers (p_set delta -0.0155 CI [-0.028, -0.003], EV delta -2.63 CI [-3.42, -1.84], 32.6% void-better, CVaR -0.213, N=276/500) match summary.json and manifest.json exactly; scripts match the described methodology.
- Caveat 1 (following/discard position untested) is already addressed by a sibling probe: `w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/` — the page could link to it.
- Cheap next probe: rerun at higher bid values (35/42) to test whether the negative void signal holds when the bidder has less slack, since bid_value=30 was fixed by the corpus snapshot conversion, not the probe.
