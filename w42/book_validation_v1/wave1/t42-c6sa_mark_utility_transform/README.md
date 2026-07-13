# Wave 1.2 — Mark Utility Transform

**Bead:** t42-c6sa  
**Wave:** wave1  
**Parent epic:** t42-4zi6

## Question

If the project's existing E[Q] PDFs are scored under Chapter 10 mark/match utility instead of point EV, how often does the preferred action change, and which book-detector families explain those flips?

## Slice

- Corpus: `branch_atlas_scaled_v0`
- Seed: 9430 (single seed, 10 declarations, bid=30)
- N decisions: 280 (28 per game × 10 games)
- N legal actions: 773
- N worlds per decision: 1000
- Paired: no (per-decision ranking comparison under two objectives)

## Method

1. Load `q_per_world` tensors from `eq_pdf_s9430_d10_bid30_joint_1000s_v2.pt`.
2. For each decision, reconstruct final team-0 capture points per world:
   - `remaining_total = 42 - pre_t0 - pre_t1`
   - `remaining_t0[w,a] = (q_per_world[w,a] + remaining_total) / 2`
   - `final_t0[w,a] = pre_t0 + remaining_t0[w,a]` (clamped 0..42)
3. Apply the deterministic mark scoring transform from
   `w42/phase4_scoring_objective_tests/run_phase4_scoring_objective_tests.py`
   (functions `made_contract`, `score_hand_marks`, `mark_multiplier` reused verbatim).
4. Compute `mark_EV[a] = mean over worlds of mark_utility[w,a]`.
5. Compare `argmax(point_EV)` vs `argmax(mark_EV)` per decision.

**Per-world hook:** fully maintained. Mark utility is an (n_worlds, n_actions) tensor
before averaging. The transform is per-world, not a scalar post-hoc conversion.

## Findings

### Headline metrics

| Metric | Value |
|--------|-------|
| Top-1 flip rate | **23.2%** (65/280 decisions) |
| Mean EV-cost on flip | 1.06 points |
| Mean mark-gain on flip | 0.0013 marks |
| Endorsed flips (book detector active) | **54/65** (83%) |

### True objective flips vs tie-breaking flips

Of the 65 flips:
- **10 (15.4%)** have positive mark_gain — the mark-preferred action genuinely
  improves Team 0's expected mark outcome, at a point-EV cost.
- **55 (84.6%)** have mark_gain = 0 — the two actions are nearly tied under mark EV,
  and the flip is a surface-flattening artifact of the binary mark transform.

The 10 true objective flips have mean EV-cost = 1.46 points and mean mark-gain = 0.044 marks
(~4.4 percentage points higher probability of winning the hand under marks).

### Flip rate by declaration

| Declaration | Flip rate | Note |
|-------------|-----------|------|
| no-trump | **46.4%** | Highest; binary make/set threshold most sensitive to action choice |
| sixes | 32.1% | |
| threes | 28.6% | |
| twos | 28.6% | |
| doubles | 25.0% | |
| fives | 25.0% | |
| blanks | **7.1%** | Lowest; trump dominance makes mark threshold less action-sensitive |

### Flip rate by seat/role

| Seat role | Flip rate |
|-----------|-----------|
| bidder | **32.9%** |
| bidder_partner | 27.1% |
| right_setter | 18.6% |
| left_setter | 14.3% |

### Flip rate by score bucket

| Score bucket | Flip rate |
|--------------|-----------|
| mid_hand | **33.3%** |
| early_hand | 17.9% |
| near_threshold | 18.1% |

### Ch10 claim correlations

| Claim | Active decisions | Flip rate | Enrichment |
|-------|-----------------|-----------|------------|
| ch10-early-terminal-under-marks | 160 | 28.1% | 1.21x enriched |
| ch10-tournament-speed-tradeoff | 160 | 28.1% | 1.21x enriched |
| ch10-timed-marks-advancement-objective | 160 | 28.1% | 1.21x enriched |
| ch10-nonbidder-partial-points-erased | 80 | 26.3% | 1.13x enriched |
| ch10-set-severity-compression | 200 | 22.0% | 0.95x neutral |
| ch10-special-bid-mark-multiplier | 0 | — | Not activated (bid=30 only) |

### Top 5 surprising flips

1. **game 0, decision 18, blanks, bidder, mid_hand**: EV-cost 10.2 pts, mark_gain 0.0
   — Large point sacrifice to switch actions under marks. The blanks declaration and
   mid-hand position create a wide point-vs-mark divergence. Detector: `late_trick_threshold_closure`.

2. **game 1, decision 23, ones, bidder, mid_hand**: EV-cost 7.6 pts, mark_gain 0.0
   — Second-largest cost flip with zero mark gain. Pure surface-flattening artifact.
   Detector: `last_to_act_closure_policy|late_trick_threshold_closure`.

3. **game 4, decision 7, fours, bidder_partner, early_hand**: EV-cost 3.8 pts, mark_gain 0.070
   — Genuine objective flip (7% mark advantage). Partner opens a count-safe line early
   that sacrifices expected points but protects the make threshold.
   Detector: `last_to_act_closure_policy`.

4. **game 7, decision 17, doubles, left_setter, mid_hand**: EV-cost 3.7 pts, mark_gain 0.0
   — Doubles regime with `doubles_regime_plan` detector active. Large point sacrifice
   for no mark gain — surface flattening under doubles-as-trump.

5. **game 6, decision 4, sixes, left_setter, early_hand**: EV-cost 3.4 pts, mark_gain 0.028
   — Book-tagged `setter_lead_pressure` and `defender_damage_lead_class`. Genuine mark flip
   with 2.8% mark advantage for the setter at the cost of 3.4 points.

## Caveats

- Dataset is a single seed (9430), 10 declarations, bid=30 only.
- Mark multiplier = 1 throughout (bid=30 < 42). `ch10-special-bid-mark-multiplier`
  had zero activation; 84/126/168 multiplier effects cannot be observed here.
- Score state reconstructed from `offense_score_before` / `defense_score_before` columns;
  bidder team assumed = team 0 (offense) per atlas convention.
- 81.5% of flips have mark_gain = 0 — these are surface-flattening artifacts of the
  binary mark transform, not strategic preference changes the book discusses.
- Book detector endorsement uses `matched_position_detectors` from `decision_actions.csv`;
  `joined_claim_action_rows.csv` is a different corpus (corpus_v2_train, seeds 0-9)
  and does not overlap with branch_atlas_scaled_v0 (seed 9430).
- Evidence does NOT generalize beyond bid=30 or seed=9430 without further runs.

## Artifacts

| File | Description |
|------|-------------|
| `run_mark_utility_transform.py` | Analysis script (reuses phase4 transform functions) |
| `manifest.json` | Provenance + input SHAs + transform note |
| `summary.json` | Headline numbers, machine-readable |
| `action_mark_ev_scalars.csv` | One row per legal action with point_ev and mark_ev |
| `flip_rate_by_slice.csv` | Flip rate by declaration / seat / bid / detector-family / score-state |
| `detector_endorsed_flips.csv` | Top 54 endorsed flips (book detector active) |
| `per_ch10_claim_correlation.csv` | Per-Ch10-claim flip enrichment |

## Exact command

```bash
source forge/venv/bin/activate
python3 -u w42/book_validation_v1/wave1/t42-c6sa_mark_utility_transform/run_mark_utility_transform.py
```
