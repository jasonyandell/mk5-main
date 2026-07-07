# burl-2000-harvest — audit 2026-07-07

Reviewed against code and result artifacts on 2026-07-07. All headline numbers verified: bucket table matches `HARVEST_SUMMARY.md` exactly (860/14/52/88/48 → strict pool 1062; BURL_BREAKS_CONSENSUS 299; FORCED_COMMIT 219 = 10.9% of 2000); corpus_index.jsonl is 2800 rows with 800 synthetic ILLEGAL (gi≥2000); D_required_first has exactly 2000 decision dirs; min300 corpus manifest confirms 1025 decisions / 2686 train / 665 val; PARITY_AUDIT.md and LENGTH_STATS_COMPARISON.md exist and back the truncation numbers.

## Corrections

- Page attributed the 11.4% / 1.2% / 43% contamination diagnostics to "v1 decisions" (the 1398-decision `031033` run); artifacts show they were computed on a batched 560-decision rerun paired against the sequential 560 (evidence: scratch/belief_trajectory_rollout/PARITY_AUDIT.md — 64/560 = 11.4%, 241/560 = 43%; LENGTH_STATS_COMPARISON.md — batched_560_rerun 1.2%). Added the n/560 figures and provenance note; substance of the diagnosis unchanged.

## Follow-ups

- ~~The page's own suggestion of a per-decision McNemar test on the 560 overlap (seq vs v2 at 2048 tokens)~~ — premise is false; see Review below. The viable paired designs are: v1-killed vs v2 (1392 aligned decisions, isolates the token-cap effect), or a new 2048-token batched rerun on the 900000-seed set (isolates batched-vs-sequential at equal cap). Either would close the residual −2.4pp BURL_BREAKS_CONSENSUS question.
- Wall-time and quarantine claims (5h46m, 0 fires, 733 max_turns_extensions) were not re-derived from events.jsonl — a one-liner over `trace_summary.json` files could confirm the 733 figure if it ever becomes load-bearing.

## Review (second pass, 2026-07-07)

- Auditor's core correction CONFIRMED: the 11.4% (64/560) and 43% (241/560) diagnostics were computed on the batched 560 rerun (`harvest_batched_20260425_010306`, batch=6, max_tokens=1024) vs sequential `harvest_20260424_133611`, not on the killed v1 run. Evidence: scratch/belief_trajectory_rollout/PARITY_AUDIT.md (header + "64/560 = 11.4%", "241/560 changed bucket (43%)"); LENGTH_STATS_COMPARISON.md (batched_560_rerun ≥2800 chars: 41 = 1.2%; sequential 0.0%, p99 1770 / max 2639).
- AMENDED (missed by first pass): quality-gates table column was still labeled "v1 (1024 tok, contaminated)" while carrying the 560-rerun's 11.4%/1.2% numbers; relabeled to "batched @1024 tok (560 rerun, contaminated)". Evidence: same two artifacts; the actual v1-killed run's at-cap rate is 1.0% (LENGTH_STATS_COMPARISON.md, batched_2000_killed row), not 1.2%.
- AMENDED (missed by first pass): the page's McNemar parenthetical claimed "560 decisions that overlap between sequential and v2" — false. Sequential 560 seeds are 900000–900019; v2 seeds are 0–71 (disjoint), verified from trace_summary.json across both harvests. Replaced with the real paired options: v1-killed vs v2 share 1392 aligned decisions (verified by matching seed/declaration/narrator_seat on decision_0..1397), or a new 2048-token rerun on the 900000 seeds. Followup's first suggestion corrected accordingly.
- Headline bucket numbers re-verified against harvest_batched_20260425_072910/HARVEST_SUMMARY.md (860/14/52/88/48; BURL_BREAKS_CONSENSUS 299; FORCED_COMMIT 219; ILLEGAL 800 synthetic).
