# Wave 4.0 — Utility-Argmax Divergence (t42-hmjr)

## Question

Wave 3.0 found EV=+0.77 / p_make≈0 on the ch05-void-creation-follow paired
contrasts (n=500). That is a **contrast-magnitude statistic**: it shows the
expected value of (void minus preserve) under EV is positive but under p_make
is approximately zero. It does **not** establish that an EV-greedy policy and a
p_make-greedy policy disagree on which action they pick from the legal-action
set at those snapshots. Wave 4.0 measures that directly: argmax over ALL legal
actions per snapshot under each of five utility lenses.

## Slice

- Corpus: `w42/book_validation_v1/wave2/snapshots/void_creation_follow/snapshots.jsonl`
  - 500 snapshots, schema `forge.eq.snapshot.v1`, sha256
    `2058eaed07242156f4caea15b7f504c1630d33077fb80a360ac2492fa4f8393a`
  - All bid=30 (mm=1 → p_make and mark_ev coincide)
  - All to-act players are setters (defense; team 1)
- Checkpoint: `forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`
- N pairs (all snapshots produced a valid argmax): **500**
- N void/preserve sub-pairs: **500**

## Method

For each snapshot:

1. Reconstruct game state with `GameStateTensor.from_snapshot`.
2. Run forge pipeline once (samples=100, device=mps, greedy=True) →
   `e_q[7]` (mean Q per action) and `e_q_pdf[7, 85]` (Q PDF per action,
   bins for Q ∈ [-42, +42]).
3. For each legal slot compute five utilities:
   - **EV** = `e_q[slot]`
   - **p_make** = `sum(pdf[slot, threshold_bin:])` where the threshold uses the
     forge bid-aware convention (`forge/eq/generate/actions.py::contract_threshold_bins`):
       - offense bin = 2*B → Q ≥ 2B-42 (bid=30 → Q ≥ 18)
       - defense bin = 85-2*B → Q ≥ 43-2B (bid=30 → Q ≥ -17)
     The to-act player in this corpus is always a setter (defense), so the
     defense threshold applies throughout.
   - **mark_ev** = `mm * (2*p_make - 1)` where `mm = max(1, bid // 42)`
     (bid=30 → mm=1, so mark_ev = 2*p_make - 1, an affine transform of p_make
     → identical argmax)
   - **CVaR_10** = mean of the lower-tail Q (cumulative pdf mass ≤ 0.10)
   - **robust_q25** = smallest Q with cumulative pdf mass ≥ 0.25
4. Argmax over legal slots per utility (tie-break: lowest slot index).
5. Disagreement matrix: 5×5 pairwise rate that two utilities pick different
   slots. Bootstrap CI (n=2000, percentile, seed=42).
6. Void/preserve confusion: restrict to the (void_slot, preserve_slot) pair
   from the original probe; tally how often each utility's argmax matches
   void / preserve / neither.

## Headline Numbers

| Pair | Disagree | Rate | 95% CI |
|------|----------|------|--------|
| **EV vs p_make** | **206/500** | **41.2%** | **[36.8%, 45.6%]** |
| EV vs mark_ev | 206/500 | 41.2% | [36.8%, 45.6%] |
| EV vs CVaR_10 | 215/500 | 43.0% | [38.6%, 47.4%] |
| EV vs robust_q25 | 148/500 | 29.6% | [25.8%, 33.8%] |
| p_make vs mark_ev | **0/500** | **0.0%** | [0.0%, 0.0%] |
| p_make vs CVaR_10 | 220/500 | 44.0% | [39.8%, 48.4%] |
| p_make vs robust_q25 | 192/500 | 38.4% | [34.2%, 42.8%] |
| CVaR_10 vs robust_q25 | 154/500 | 30.8% | [26.8%, 35.0%] |

**p_make = mark_ev exactly** at bid=30 (they share an argmax for every
snapshot). This empirically confirms the affine identity flagged by
Wave1.2 / Wave2.H — a useful sanity check.

### Void / preserve subset

| Utility | argmax = void | argmax = preserve | argmax = neither |
|---------|--------------:|------------------:|-----------------:|
| EV | 147 (29.4%) | 165 (33.0%) | 188 (37.6%) |
| p_make | 193 (38.6%) | 155 (31.0%) | 152 (30.4%) |
| mark_ev | 193 (38.6%) | 155 (31.0%) | 152 (30.4%) |
| CVaR_10 | 213 (42.6%) | 139 (27.8%) | 148 (29.6%) |
| robust_q25 | 203 (40.6%) | 153 (30.6%) | 144 (28.8%) |

Joint pattern of interest:
**EV → void AND p_make → preserve**: 29 / 500 = **5.8%**.
This is the canonical "EV pulls toward void-creation while p_make doesn't"
pattern Wave 3.0 surfaced, now measured at the policy-action level.

## Verdict gate

**Result: DISAGREE ≥ 5% with the canonical EV→void / p_make→preserve pattern
present at 5.8%.**

> Divergence is real at the policy-action level. Recommend scoping rung-2
> (utility-tunable searcher) as the next build.

The EV-greedy and p_make-greedy policies disagree on **41.2%** of snapshots
(95% CI [36.8%, 45.6%]) — an order of magnitude above the 5% gate. Wave 3.0's
contrast-magnitude split is **not** purely a magnitude-only artefact; the two
objectives select different actions at the policy level on this corpus.

A surprise: the void-subset breakdown shows **p_make picks void MORE often
than EV** (38.6% vs 29.4%), which inverts the intuitive Wave 3.0 framing
("EV likes void, p_make is neutral"). The story turns out to be: EV more
often picks a third action that is neither void nor preserve (37.6%), so the
EV vs p_make divergence is driven less by an EV→void / p_make→preserve split
and more by EV preferring "alternative" actions when both void and preserve
look mediocre. The 5.8% canonical-pattern slice is the part that matches the
literal Wave 3.0 framing; the remaining ~36% disagreement is composed of
other slot pairs.

## Caveats

- All snapshots have bid=30 → mm=1 → mark_ev is a 2·p_make - 1 affine and
  shares argmax with p_make. Disagreement between mark_ev and p_make is
  structurally zero on this corpus and would only appear at higher bids.
- All to-act players are setters (defense). The defense p_make threshold
  (Q ≥ -17 for bid=30) is much looser than the offense threshold (Q ≥ 18),
  which means p_make values are usually high and the distributional left
  tail (CVaR / Q25) is the more discriminating risk lens here.
- The argmax is deterministic but the underlying e_q/e_q_pdf comes from
  100 sampled worlds; per-snapshot argmax is sample-noise-bound at fine
  Q deltas. Snapshots where two slots have nearly equal utility may flip
  with re-seeding. We did not estimate this re-seed sensitivity here;
  the bootstrap CI captures sample-variance over snapshots, not over
  per-snapshot resampling.
- Snapshots are from the oracle-greedy legacy corpus; behavior on
  realistic mixed-policy data may differ.
- The robust_q25 utility is bin-quantised (smallest Q with cumulative mass
  ≥ 25%) so ties on the bin grid are common; its argmax is the noisiest
  of the five.
- **Architecture-decision measurement only.** No claim ledger row moves.

## Utility coverage

All five utilities recorded for every legal slot of every snapshot — meets
the Wave 3.0 utility-coverage requirement.

## Artifacts

| File | Contents |
|------|----------|
| `manifest.json` | provenance: snapshots SHA, checkpoint SHA, command, runtime |
| `summary.json` | headline numbers + gate verdict (machine-readable) |
| `per_snapshot_argmax.csv` | one row per snapshot: argmax slot + top-utility value per utility |
| `disagreement_matrix.csv` | 5×5 pairwise disagreement with bootstrap CIs |
| `void_subset_confusion.csv` | per-utility void/preserve/neither tallies |
| `analyze.py` | reproducibility script (single command runs the whole pipeline) |

## Reproducibility

```bash
python w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/analyze.py
```

Defaults: `--snapshots` points at the wave2 corpus, `--checkpoint` at the
3.3M domino-qval-large checkpoint, `--samples 100`, `--device mps`,
`--batch-size 1`. Runtime: ~210s on M5 Max.

## Blockers

None. One non-blocking note: batched pipeline calls (`--batch-size > 1`)
hit `index 28 is out of bounds: 1, range 0 to 28` in the world sampler when
heterogeneous-history snapshots are co-batched (likely a known behavior of
`generate_eq_from_snapshots` on snapshots at different decision indices).
The script falls back to per-snapshot calls automatically; runtime stays
well within budget so no fix attempted here.
