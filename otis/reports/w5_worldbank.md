# Otis W5 — world-bank analysis (P6 + interaction structure)

> **⚠ Revalidated under sampler-contamination filtering (issue #52).** The numbers in
> the original body below were computed over the FULL stored world bank, which the
> pre-repair `WorldSamplerMRV` left 27–67% malformed per decision (duplicate tiles via
> injected domino 0-0, actor-hand leakage, wrong unseen set —
> `wiki/experiments/world-sampler-mrv-audit.md`). Every headline number below tagged
> **[SUPERSEDED §52]** is corrected in the **[Sampler-contamination revalidation
> (issue #52)](#sampler-contamination-revalidation-issue-52)** section at the end of
> this report; the old numbers are retained in place so the correction is reachable
> from the error. **The P6 verdict is unchanged: PASS.**

Playout-free rung of issue #49 (items 1+2, prediction P6). Everything below derives from stored `world_hands` [M,3,7] and `q_per_world` [M,7] in the eq corpus — no new game playouts. Instrument: `otis/analysis/worldbank.py`. CPU only.

## Method

- **Belief posterior**: gus student `gus/adapters/v3_consistency_10000g.pt` (voids transformer, belief head world-independent). For each decision `belief_logits[28,3]` -> `log P(rel-seat | domino)`; a sampled world's weight = `prod over hidden dominoes P(assigned seat | domino)`, normalized over the decision's M worlds. ESS = 1 / sum(w_norm^2). A uniform variant (w = 1/M) runs alongside.
- **Relative seats**: 0=left opp, 1=partner, 2=right opp (`gus/model/features.py:86`); belief axis, `world_hands` rows, and the placement code share this ordering.
- **P6 qualifying decision**: trick 0, actor holds the 3-2 (id 8), >= 100 sampled worlds.
- **Context vector**: joint placement of the other four count tiles (5-5, 6-4, 5-0, 4-1) over {left opp, partner, right opp, your hand}. Worlds clustered by exact context; clusters < 2% belief mass folded into `other`.
- **Cluster value**: belief-weighted mean of `q_per_world[:, a*]`, a* = argmax E[Q] over legal actions.
- **Bimodality**: clusters sorted by mean; among splits where both low/high groups carry >= 20% belief mass, take the one with the largest group-mean gap; bimodal iff that gap >= 10 points. (Existence reading of the P6 band; falls back to global-max-gap split when no split is mass-valid, which then reads as not-bimodal.)

## Band compliance (registered P6 band vs this implementation)

The registered band (`wiki/experiments/otis-v0.md` P6, written 2026-07-14 before W1):
"on **drama-directed root decisions** holding 3-2 (world-bank), fate/context clusters
show >= 2 modes separated by >= 10 points, each >= 20% belief mass, in >= 30% of
decisions; falsifier < 10%." The band is fixed; nothing below regrades it. Deviations,
precisely:

1. **Drama filter absent (drift).** The qualifying set is ALL trick-0 decisions where
   the actor holds the 3-2 — no drama filter. "Drama" has a registered refined
   definition in [[gus-drama-atlas]]: `marginal_eq_gap <= 1.0` (top1 - top2 E[Q] over
   legal) AND `count_unplayed >= 15` AND `n_legal >= 2`. (The atlas restricts that
   filter to mid/endgame phases, which is incompatible with "root"; the phase clause
   is dropped for the addendum.) Applying the remaining filter post hoc to the 300
   qualifying decisions (`scratch/otis-night/w5_drama_subset.json`): **77/300 (25.7%)
   are drama-directed; on that subset the bimodal fraction is 81.8% belief-weighted
   (63/77) and 68.8% uniform (53/77)** — the letter-of-the-band population also sits
   decisively on the PASS side. Non-drama decisions are slightly MORE bimodal (89.2%
   belief), so the missing filter inflated the headline by ~5 pp, not directionally.
2. **"Root" read as trick 0, all four seats.** Composition (d_idx: n, belief-bimodal,
   uniform-bimodal): 0: 70, 97.1%, 84.3% · 1: 40, 90.0%, 90.0% · 2: 100, 85.0%, 79.0%
   · 3: 90, 81.1%, 64.4%. Under the strictest reading (root = opening lead, d_idx=0)
   the fraction is the highest cell, 97.1%/84.3%. Every reading lands PASS-side.
   **[SUPERSEDED §52: d_idx=0 valid-filtered = 98.3% belief / 94.8% uniform, N=58 (12
   of 70 skipped for < 40 valid worlds). Still the highest cell; still PASS.]**
3. **">= 100 worlds" qualifier is implementation-added**, not in the band. Non-binding
   in practice: every v2 decision carries M=200 sampled worlds.
4. **"Modes" operationalized as a two-group split of exact-context clusters** (each
   group >= 20% belief mass, mass-weighted group-mean gap >= 10 points, existence over
   splits; clusters < 2% belief mass pooled into "other"). This is a reading of
   ">= 2 modes ... each >= 20% belief mass": groups may pool several small clusters to
   reach 20% (more permissive than requiring a single 20% mode), while the gap is
   measured between mass-weighted group means rather than mode peaks (more
   conservative). Thresholds were fixed before the run.

**Verification**: the bimodality decision was recomputed by hand from the printed
cluster tables for three CSV rows (`g41:s4:d1`, `g83:s28:d3`, `g29:s2:d9`); all three
reproduce the recorded gap and group masses to rounding (35.8 / 73.5 / 54.3 points).
The full otis suite passes (30 tests).

## P6 result

- Qualifying decisions N = **300** across **300** games (one per game: exactly one
  seat holds the 3-2 at trick 0). **[SUPERSEDED §52: N=281; 19 skipped for < 40 valid worlds.]**
- **Bimodal fraction (belief-weighted) = 87.3%** (262/300) -> **PASS (>=30%)** against
  the registered P6 band (>=30% PASS, <10% falsifier). **[SUPERSEDED §52: 86.8% (244/281) — still PASS.]**
- **Bimodal fraction (uniform weights) = 77.3%** (232/300). **[SUPERSEDED §52: 83.3% (234/281).]**
- Drama-directed subset (band letter, see above): 81.8% belief / 68.8% uniform, N=77.
  **[SUPERSEDED §52: 81.8% belief / 76.6% uniform, N=77.]**
- ESS distribution: median **13.7**, p25 7.4, p75 61.2 (of M=200 worlds); 64.7% of
  decisions have ESS < 20, 38% have ESS < 10. **[SUPERSEDED §52: median 59.3, p25 39.7,
  p75 94.8 — the low ESS was largely the contaminated worlds; filtering resolves the caveat below.]**

**ESS caveat — lean on the uniform variant.** _[SUPERSEDED §52 — this caveat is
substantially retired: after valid-world filtering median ESS rises to 59.3 and belief
and uniform verdicts move within 3.5 pp of each other. The original caveat is kept for
the record.]_ Median ESS 13.7 is below 20: the
belief-weighted per-decision numbers rest on ~5-15 effective worlds and are fragile
(single high-weight worlds can dominate a cluster; card 8 has ESS 2.4). The robust
anchor is the uniform variant, **77.3%**, which uses all 200 worlds equally and still
clears the 30% PASS threshold by 47 pp. Three observations support the conclusion
surviving the fragility: the belief and uniform verdicts agree on 86% of decisions
(36 belief-only, 6 uniform-only); the belief-weighted fraction is nearly flat across
ESS halves (88.7% below the median ESS vs 86.0% above); and the drama subset tells
the same story under both weightings.

## Findings — the shape of the ledger

Bimodality of the 3-2's context-conditioned value is the norm at trick 0, not the
exception. On 300 real decisions where the actor holds the 3-2, 77.3% show two
context-cluster groups at least 10 points apart with at least 20% mass each under
uniform world weights, rising to 87.3% under the gus belief posterior. The prevalence
is a property of the raw world bank — belief weighting sharpens the detected
fraction by ~10 pp by concentrating mass on plausible contexts, but does not create
it. The mode separation is large: among belief-bimodal decisions the group-mean gap
has median 24.8 points (p25 18.0, p75 33.6, max 73.5) — on a 42-point hand scale,
these are not marginal splits.

The context variable doing the work is the placement of the OTHER four count tiles.
The ledger cards (`otis/reports/w5_ledger_cards.md`) show the canonical shape: the
same 3-2-holding hand's best line is worth ~-41 points in contexts where the 5-5 and
6-4 sit behind the opponents and ~+31 where the partner holds the off count (card 1,
a 72-point swing at 26%/74% mass), and ~-38 vs ~+40 depending on whether the 5-0
sits with the left opponent or the partner (card 8). A single belief-averaged
marginal — the quantity #25's arena measured — reports one number for each of these
decisions and is structurally blind to the split; this instrument confirms on real
decisions that the split it averages over is 20-70 points wide most of the time.

Interaction lift concentrates where the big count tiles cross the opponent seats:
the top cell is 5-5@left-opp x 6-4@right-opp (lift -1.31), and 9 of the top 15 cells
involve the 5-5 or 6-4 at an opponent seat. But the lifts are an order of magnitude
smaller than the first-order effects: |lift| <= 1.3 points against first-order deltas
of 2-7 points (5-5@partner +6.6, 5-5@opponent -3 to -4). At trick 0/1 the count-tile
placement effect on E[Q(a*)] is first-order dominated; the joint structure worth
modeling at the root is the CLUSTERING of placements (which the bimodality measures),
not pairwise placement synergies. The 13 sign-opposing cells (of 90 qualifying) are
the honest residual — "count tile at opponent hurts UNLESS partner holds a high
count" — modest but real, and unrepresentable by a per-tile-independent pricer.

**Grade expectation**: on these numbers P6 will grade as **PASS** — the belief-weighted
fraction (87.3%), the uniform robustness check (77.3%), the band-letter drama subset
(81.8%/68.8%), and the strict-root subset (97.1%/84.3%) all clear the >= 30% PASS
threshold; none approaches the < 10% falsifier. Grading itself happens in W7.

## Interaction structure (issue item 1)

First-order delta(d@s) and pairwise lift(d1@s1, d2@s2) over the five count tiles at relative seats, per-decision centered and pooled over 2000 trick-0/1 decisions. Cells require >= 30 pooled worlds. Full top-50 in `w5_interactions.csv`.

- Sign-opposing 'count tile hurts UNLESS partner holds X' cells flagged: **13** (of 90 qualifying pairwise cells). **[SUPERSEDED §52: 17 of 90; the top interaction cell and first-order-dominance conclusion are unchanged — see revalidation §.]**

**Reading**: at trick 0/1 the count-tile placement effects on E[Q(a*)] are
first-order dominated — a count tile at partner helps (5-5@partner delta = +6.6),
at an opponent hurts (5-5@left/right delta ~ -3 to -4), and pairwise lifts are
small (|lift| <= ~1.3 points vs first-order magnitudes of 2–7). The joint head's
value therefore comes mostly from the marginals *at this stage*; the 13
sign-opposing cells are the honest residual where "count tile at opponent hurts
UNLESS partner holds a high count tile" — modest but real, and the kind of
structure that a per-tile-independent pricer cannot represent.

Top 15 cells by |lift| x mass **[SUPERSEDED §52 — this table is over the contaminated
bank; the valid-world top-15 is in `scratch/otis-night/w5_worldbank_generated.md` and
`scratch/otis-night/w5_interactions.csv`. Same top cell (5-5@left-opp × 6-4@right-opp,
lift −1.29); same first-order dominance.]**:

| # | tile1 @ seat | tile2 @ seat | lift | delta1 | delta2 | mass | n | flag |
|---|---|---|---|---|---|---|---|---|
| 1 | 5-5 @ left opp | 6-4 @ right opp | -1.31 | -3.97 | -1.99 | 0.0542 | 25057 | - |
| 2 | 5-0 @ right opp | 5-5 @ partner | -1.06 | 0.39 | 6.63 | 0.0650 | 27979 | sign_opposing |
| 3 | 3-2 @ right opp | 5-5 @ partner | 1.13 | 0.15 | 6.63 | 0.0593 | 25796 | - |
| 4 | 5-0 @ right opp | 6-4 @ left opp | -1.13 | 0.39 | -2.14 | 0.0540 | 26573 | - |
| 5 | 3-2 @ right opp | 5-5 @ left opp | -1.02 | 0.15 | -3.97 | 0.0598 | 28820 | - |
| 6 | 5-5 @ partner | 6-4 @ right opp | 1.10 | 6.63 | -1.99 | 0.0523 | 23299 | - |
| 7 | 5-0 @ left opp | 5-5 @ right opp | -0.96 | 0.76 | -3.07 | 0.0583 | 24744 | - |
| 8 | 5-0 @ left opp | 6-4 @ left opp | 1.06 | 0.76 | -2.14 | 0.0485 | 23605 | sign_opposing |
| 9 | 3-2 @ left opp | 6-4 @ right opp | -0.83 | 0.05 | -1.99 | 0.0521 | 24760 | - |
| 10 | 4-1 @ right opp | 5-5 @ left opp | -0.80 | 0.17 | -3.97 | 0.0535 | 24182 | - |
| 11 | 4-1 @ right opp | 5-0 @ partner | -0.66 | 0.17 | -1.09 | 0.0627 | 25603 | - |
| 12 | 3-2 @ left opp | 6-4 @ partner | 0.81 | 0.05 | 3.91 | 0.0504 | 24771 | - |
| 13 | 3-2 @ left opp | 6-4 @ left opp | 0.92 | 0.05 | -2.14 | 0.0424 | 22261 | sign_opposing |
| 14 | 5-0 @ left opp | 5-5 @ partner | 0.58 | 0.76 | 6.63 | 0.0667 | 27441 | - |
| 15 | 4-1 @ left opp | 6-4 @ left opp | -0.93 | 0.53 | -2.14 | 0.0400 | 21467 | - |

## Provenance

- Chunks streamed: corpus_v2_train_0-9_d0-9.pt, corpus_v2_train_10-19_d0-9.pt, corpus_v2_train_20-29_d0-9.pt
- Wall time: 6.3s
- Notes:
  - interaction sample truncated to first 2000 of 2400 trick-0/1 decisions (deterministic prefix)

Ledger cards for the 10 highest-mass bimodal decisions: `otis/reports/w5_ledger_cards.md`. Machine summary: `scratch/otis-night/w5_summary.json`. Drama-subset addendum (band-compliance recompute): `scratch/otis-night/w5_drama_subset.json`.

---

## Sampler-contamination revalidation (issue #52)

The entire analysis above was computed over the FULL stored world bank. The eq corpus
in `gus/data/` was generated with the **pre-repair `WorldSamplerMRV`**
(`wiki/experiments/world-sampler-mrv-audit.md`): a fraction of every decision's stored
worlds are malformed — duplicate tiles (injected domino 0-0), tiles that overlap the
actor's own hand, wrong unseen set. The world-bank instrument originally weighted and
clustered ALL of them.

**What changed.** `otis/analysis/worldbank.py` now applies **strict valid-world
filtering by default** (`--no-filter-valid` to disable). A world is VALID iff its
reconstruction is a real 28-domino deal: every hidden seat's initial hand
(recorded prefix plays ∪ sampled remaining tiles) has exactly 7 tiles and the full
4×7 deal is a permutation of all 28 dominoes — no duplicates, no actor-hand overlap,
tiles exactly the decision's unseen set. This is the established test
(`otis/tiedroll.py:_reconstruct_one`, `valid_world_indices` :237), lifted into
`world_validity(game, d_idx)`. Invalid worlds are dropped **before** clustering and
belief-weighting; per-decision valid fraction is recorded; a decision with **< 40 valid
worlds is skipped** entirely. `E[Q]` / `a*` come from the decision's stored,
world-independent E[Q] and are unchanged by filtering (only the per-world q at that a*
is re-sampled over valid worlds).

**Contamination measured (this corpus).** Per-decision valid fraction: **median 48%,
mean 55%, min 15%, max 100%** (p25 30%, p75 100%) over 300 P6-qualifying decisions —
consistent with the audit's 27–67% malformed. **19 of 300** P6 decisions and **32 of
2032** interaction decisions were skipped for < 40 valid worlds. Kept P6 decisions
retain a median of 100 valid worlds (min 40).

### OLD vs NEW — every headline number

| Metric | OLD (full bank, contaminated) | NEW (valid-world filtered) |
|---|---|---|
| Qualifying decisions N | 300 (300 games) | **281** (19 skipped) |
| P6 bimodal — belief-weighted | 87.3% (262/300) → PASS | **86.8% (244/281) → PASS** |
| P6 bimodal — uniform | 77.3% (232/300) | **83.3% (234/281)** |
| Drama subset (belief / uniform), N | 81.8% / 68.8%, N=77 | **81.8% / 76.6%, N=77** |
| Drama subset marginal-E[Q]-gap median | 0.41 | **0.23** |
| Strict-root d_idx=0 (belief / uniform), N | 97.1% / 84.3%, N=70 | **98.3% / 94.8%, N=58** |
| ESS median (of M=200) | 13.7 | **59.3** |
| ESS p25 / p75 | 7.4 / 61.2 | **39.7 / 94.8** |
| Interaction decisions pooled | 2000 | 2000 |
| Top interaction cell | 5-5@left-opp × 6-4@right-opp, lift −1.31 | **5-5@left-opp × 6-4@right-opp, lift −1.29** |
| Conditional-partner-rescue cells flagged | 13 of 90 | **17 of 90** |

### Reading

1. **The P6 verdict is robust to contamination: still PASS.** Belief-weighted bimodal
   fraction moves 87.3% → 86.8% (well clear of the 30% PASS line; nowhere near the 10%
   falsifier). Every subset lands PASS-side under filtering, and the strict-root and
   drama subsets rise or hold.
2. **The old ESS caveat is largely an artifact of the bad sampler.** Median ESS jumps
   **13.7 → 59.3**. The malformed worlds (notably those carrying injected domino 0-0)
   were taking extreme belief weights and collapsing the effective sample; on real
   worlds the posterior spreads across ~60 effective worlds. The belief and uniform
   fractions correspondingly converge (gap 10 pp → 3.5 pp), so the "lean on uniform"
   hedge is no longer load-bearing.
3. **The uniform fraction rises (77.3% → 83.3%)** because dropping malformed worlds
   removes low-value phantom contexts that had been diluting clean clusters — the
   bimodal signal is if anything *sharper* on real worlds.
4. **Interaction structure is unchanged in shape.** The top cell is the same
   (5-5 × 6-4 across the opponents, lift ≈ −1.3), lifts remain an order of magnitude
   below first-order deltas, and the sign-opposing "hurts-unless-partner" residual
   grows modestly (13 → 17 of 90). Conclusion — first-order-dominated at the root, the
   modelable joint structure is the clustering, not pairwise synergy — stands.

The ledger cards (`otis/reports/w5_ledger_cards.md`) are regenerated from valid-world
clusters only (each card now reports `n_valid` of `n_raw` sampled). Machine summary
with full `validity`, `drama_subset`, and `d0_subset` blocks:
`scratch/otis-night/w5_summary.json` (pre-filter snapshot preserved at
`scratch/otis-night/w5_summary_prefilter.json`). Per-decision valid fractions are in
`scratch/otis-night/w5_p6.csv` (new `n_raw`, `n_valid`, `valid_fraction`,
`marginal_eq_gap` columns). Unit test: `otis/tests/test_worldbank.py::test_world_validity_drops_malformed_worlds`.