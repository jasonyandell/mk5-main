# burl-2000-harvest — audit 2026-07-07

Reviewed against code and result artifacts on 2026-07-07. All headline numbers verified: bucket table matches `HARVEST_SUMMARY.md` exactly (860/14/52/88/48 → strict pool 1062; BURL_BREAKS_CONSENSUS 299; FORCED_COMMIT 219 = 10.9% of 2000); corpus_index.jsonl is 2800 rows with 800 synthetic ILLEGAL (gi≥2000); D_required_first has exactly 2000 decision dirs; min300 corpus manifest confirms 1025 decisions / 2686 train / 665 val; PARITY_AUDIT.md and LENGTH_STATS_COMPARISON.md exist and back the truncation numbers.

## Corrections

- Page attributed the 11.4% / 1.2% / 43% contamination diagnostics to "v1 decisions" (the 1398-decision `031033` run); artifacts show they were computed on a batched 560-decision rerun paired against the sequential 560 (evidence: scratch/belief_trajectory_rollout/PARITY_AUDIT.md — 64/560 = 11.4%, 241/560 = 43%; LENGTH_STATS_COMPARISON.md — batched_560_rerun 1.2%). Added the n/560 figures and provenance note; substance of the diagnosis unchanged.

## Follow-ups

- The page's own suggestion of a per-decision McNemar test on the 560 overlap (seq vs v2 at 2048 tokens) still appears undone; it is the cheap way to close the residual −2.4pp BURL_BREAKS_CONSENSUS question.
- Wall-time and quarantine claims (5h46m, 0 fires, 733 max_turns_extensions) were not re-derived from events.jsonl — a one-liner over `trace_summary.json` files could confirm the 733 figure if it ever becomes load-bearing.
