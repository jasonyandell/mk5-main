# gus-drama-atlas — audit

Reviewed against code on 2026-07-07 — no issues found.

Verified: commit `76355ac` exists; `gus/analysis/build_drama_atlas.py` implements the three quantities exactly as described (q_taken.std(), distinct-argmax fragility, 1−H/log(3) sharpness); headline numbers match `gus/analysis/tables/drama_summary_by_split.csv` (train variance mean 13.1 / median 14.4 / p99 29.1; sharpness 0.064/0.020; gus-mode agreement 72.4%) and `quadrant_summary.csv` (drama 73,411/280,000 = 26.2%; drama agreement 52.6%).

## Follow-ups

- The findings doc's proposed next step (hedge-player / signal-player relabeling of the 73k drama decisions as multi-target training labels) appears unpursued — a cheap probe given the parquet already exists.
- `drama_atlas_v2.parquet` and `add_real_drama_columns.py` (realised-regret join) exist but the wiki page only covers v1; a one-line pointer to the v2 realised-regret columns would help future readers.
- The lead-decision drama concentration (64.9%) directly motivates the concealment/signaling blindness point in [[belief-value-is-legibility]]-adjacent work; cross-linking would tighten the seam.
