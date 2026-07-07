# burl-perf-phase0 — audit 2026-07-07

## Corrections

- Page said the two baseline runs were at sha `1f11d28`; the run artifacts record sha `da25079` on branch `perf/bench` — `1f11d28` is the commit that recorded the ledger rows (evidence: `burl/eval/results/perf_20260427_013720_baseline-bf16.json`, `perf_ledger.csv`).
- Page said the 80% K1 match in run 2 came from gi=72 flipping `eq_delta_vs_bot` from −1.7 to +0.8 across the ≥0 threshold, and that "<0.5 Q-point deltas flip K1 sign about half the time." The artifacts show the flip was gi=36: run 1 signed_delta −11.49 (K1 fail), run 2 delta 0.0 / matched bot (K1 pass); gi=72 failed K1 identically in both runs at −1.73, and the −1.7→+0.8 numbers appear nowhere (evidence: `per_decision_grades` in both `perf_20260427_0137*/0138*` JSONs).

## Verified clean

- Measurement table (wall 79.5/79.9 s, p50 77.6/69.6, prefill 10,274/10,568, decode 87.5/79.8, peak 11.59 GB, K1 100/80, regret 0.0/−44.65) matches `burl/eval/results/perf_ledger.csv` exactly.
- Subset layout table (gi/game/decl/seat/trick/n_legal) matches `burl/eval/data/perf_subset_5.jsonl` row-for-row, including the corpus SHA256 fingerprint mechanism.
- All Pointers paths exist; `bench_decision_latency.py` does drive `GemmaLocalNativeBatched` + the `_init_decision_state → _prepare_step → _apply_step → _finalize` lockstep loop via a stats-recording subclass, as described.

## Follow-ups

- `harvest_batched.py` is imported from gitignored `scratch/belief_trajectory_rollout/` via a hardcoded absolute-path fallback in the bench; promoting it to tracked code (as the bench's own comment suggests) would make the harness reproducible from a clean checkout.
- Note run 1's 100% K1 match is self-graded (baseline vs itself), so only run 2 is a true run-vs-run stability read; a third run would cheaply confirm the 80–100% envelope.

## Review (second pass, 2026-07-07)

- Verified — corrections stand.
- sha correction re-derived: `burl/eval/results/perf_ledger.csv` rows 20260427_013720/013851 both record `sha=da25079, branch=perf/bench`; page frontmatter/ledger commit is `1f11d28`.
- gi=36 flip re-derived from `per_decision_grades` + `per_decision_rows` in both run JSONs: run 1 gi=36 `final_play=26, eq_delta_vs_bot=−11.49, k1_pass=false`; run 2 `final_play=1 (bot's play), delta=0.0, k1_pass=true`. gi=72 is `−1.7338, k1_pass=false` in both. K1 booleans differ only at gi=36 → 4/5 = 80%.
- The original page's "+0.8" has no eq-delta grounding anywhere in either JSON (the only ~0.83 value is a `step_stats[5].prompt_time`); its deleted "<0.5 Q-point deltas flip K1 sign about half the time" claim is likewise unsupported (observed deltas are −12.5/−11.5/−1.73/0.0/0.0). One phrasing nit: the followup's "−1.7→+0.8 numbers appear nowhere" is literal only for the flip pair — −1.73 itself does appear, stably, at gi=72, as the same sentence already states.
- Follow-up 1 re-confirmed still open: `harvest_batched.py` is untracked (`git ls-files` finds only chain-trace jsonls), `scratch/` is gitignored, and `bench_decision_latency.py:59-66` carries the hardcoded absolute-path fallback plus the "if/when promoted, this fallback can drop" comment. Follow-up 2 matches the ledger note "Phase 0 baseline (self-graded)".
- Frontmatter left at `1f11d28` is consistent with the 45a7e35 audit batch (pages generally did not bump `last_updated`); no edit damage found in tables, links, or pointers.
