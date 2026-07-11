---
title: W42 Book Validation — Utility-Argmax Divergence (Wave 4.0)
kind: experiment
status: closed-on-completion
first_seen: local-2026-05-03
bead: t42-hmjr
parent_epic: t42-4zi6
---

## Question

Wave 3.0 [[w42-bookval-v2-utility-lens-synthesis]] flagged ch05-void-creation-follow as the most consequential utility-dependent verdict: paired contrasts gave EV delta = +0.77 (CI excludes zero) but p_make delta ≈ 0 (CI spans zero). That is a **contrast-magnitude statistic** — it shows the void-vs-preserve delta differs across utilities. It does not establish that an EV-greedy player and a p_make-greedy player would actually pick different actions at those snapshots.

This wave measures the policy-action question directly: argmax over **all legal actions** per snapshot under each of five utility lenses, then count pairwise disagreements. The result is a gating measurement for whether a multi-objective architecture (utility-conditioned model heads / utility-tunable searcher) is worth building.

---

## Method

**Inputs:** 500 snapshots from `w42/book_validation_v1/wave2/snapshots/void_creation_follow/snapshots.jsonl` (sha256 `2058eaed…`); checkpoint `forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`; 100 sampled worlds per snapshot; device=mps. All snapshots have bid=30 and to-act = setter (defense).

**Per snapshot:**
1. Reconstruct via `GameStateTensor.from_snapshot`.
2. Run `forge.eq.generate.pipeline.generate_eq_from_snapshots` once → `e_q[7]` and `e_q_pdf[7, 85]` for the to-act player.
3. For each legal slot (engine `legal_actions()`), compute five utilities:
   - **EV** = `e_q[slot]`
   - **p_make** = `Σ pdf[slot, threshold_bin:]` with the forge bid-aware threshold (`forge/eq/generate/actions.py::contract_threshold_bins`); for setter at bid=30 → Q ≥ −17.
   - **mark_ev** = `mm · (2·p_make − 1)` with `mm = max(1, bid // 42)`; bid=30 → mm=1.
   - **CVaR_10** = mean of pdf bins with cumulative mass ≤ 0.10.
   - **robust_q25** = smallest Q with cumulative pdf mass ≥ 0.25.
4. Argmax slot per utility (tie-break: lowest slot index).

**Disagreement statistic:** for every utility pair (a, b), `mean(argmax_a ≠ argmax_b)` over 500 snapshots. Bootstrap 95% CI: n_boot=2000, percentile, seed=42.

**Void/preserve subset:** the original probe's `find_void_and_preserve_slots` was run on each snapshot (all 500 produced a valid pair). For each utility, count whether its argmax matches `void_slot`, `preserve_slot`, or neither.

---

## Per-Utility-Pair Disagreement Table

| utility_a | utility_b | disagree | rate | 95% CI |
|-----------|-----------|---------:|-----:|--------|
| EV        | **p_make** | **206/500** | **41.2%** | **[36.8%, 45.6%]** |
| EV        | mark_ev    | 206/500 | 41.2% | [36.8%, 45.6%] |
| EV        | CVaR_10    | 215/500 | 43.0% | [38.6%, 47.4%] |
| EV        | robust_q25 | 148/500 | 29.6% | [25.8%, 33.8%] |
| p_make    | mark_ev    | **0/500** | **0.0%** | [0.0%, 0.0%] |
| p_make    | CVaR_10    | 220/500 | 44.0% | [39.8%, 48.4%] |
| p_make    | robust_q25 | 192/500 | 38.4% | [34.2%, 42.8%] |
| mark_ev   | CVaR_10    | 220/500 | 44.0% | [39.8%, 48.4%] |
| mark_ev   | robust_q25 | 192/500 | 38.4% | [34.2%, 42.8%] |
| CVaR_10   | robust_q25 | 154/500 | 30.8% | [26.8%, 35.0%] |

**Sanity check passed:** p_make and mark_ev share the same argmax on every single snapshot. At bid=30, mm=1, so mark_ev = 2·p_make − 1 — a positive-affine transform — and the argmax must coincide. This is the empirical confirmation of the Wave1.2 / Wave2.H affine identity at the policy-selection level.

---

## Void / Preserve Confusion Table

For each utility's argmax, was it the void slot, the preserve slot, or some third legal action?

| utility | argmax = void | argmax = preserve | argmax = neither |
|---------|--------------:|------------------:|-----------------:|
| EV          | 147 (29.4%) | 165 (33.0%) | 188 (37.6%) |
| p_make      | 193 (38.6%) | 155 (31.0%) | 152 (30.4%) |
| mark_ev     | 193 (38.6%) | 155 (31.0%) | 152 (30.4%) |
| CVaR_10     | 213 (42.6%) | 139 (27.8%) | 148 (29.6%) |
| robust_q25  | 203 (40.6%) | 153 (30.6%) | 144 (28.8%) |

**Joint pattern of interest** — "EV picks void AND p_make picks preserve" (the canonical Wave 3.0 framing): **29 / 500 = 5.8%**.

---

## Verdict

> **DISAGREE ≥ 5%. Divergence is real at the policy-action level. Recommend scoping rung-2 (utility-tunable searcher) as the next build.**

EV-greedy and p_make-greedy disagree on **41.2%** of snapshots (95% CI [36.8%, 45.6%]) — an order of magnitude above the 5% gate. Wave 3.0's contrast-magnitude split is **not** purely a magnitude-only artefact; the two objectives select different actions at the policy level on this corpus.

A subtlety worth noting in the recommendation: the void-subset breakdown shows **p_make picks the void slot more often than EV does** (38.6% vs 29.4%), which inverts the intuitive Wave 3.0 framing ("EV likes void, p_make is neutral"). The actual story:

- EV more often selects an action that is **neither void nor preserve** (37.6% — the highest "neither" rate of any utility).
- The literal "EV→void, p_make→preserve" pattern is 5.8% — meaningful, but it's only ~14% of the total EV/p_make disagreement.
- The remaining ~36% disagreement is composed of other slot pairs: e.g. EV→neither vs p_make→void, EV→void vs p_make→neither, etc.

So the architecture motivation stands — but the design framing should be "EV and p_make systematically pick different slots, with EV preferring third-option discards more often than the void/preserve dichotomy alone captures," not "EV likes void, p_make likes preserve."

---

## Caveats

- All snapshots are bid=30 → mm=1 → mark_ev coincides with p_make by construction. Disagreement between mark_ev and p_make would only emerge at higher bids; this corpus cannot test that.
- All to-act players are setters (defense). The defense p_make threshold (Q ≥ −17 at bid=30) is loose, so most legal actions have high p_make; CVaR_10 / robust_q25 are doing more of the discrimination here than they would on offense.
- Argmax is sample-noise-bound at fine Q deltas (100 worlds per snapshot). Snapshots with two near-tied slots may flip with re-seeding; the bootstrap CI quantifies snapshot-level variance, not per-snapshot resampling variance.
- Snapshots are oracle-greedy legacy corpus; behavior on mixed-policy data may differ. The verdict's recommendation to validate on the larger 10K-snapshot mixed corpus before committing to the rung-2 architecture build still applies as a follow-up.
- This wave is read-only on the claim ledger. No row is moved.

---

## Utility coverage

All five utilities recorded for every legal slot of every snapshot — meets the Wave 3.0 utility-coverage requirement.

---

## Artifacts

- `w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/per_snapshot_argmax.csv`
- `w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/disagreement_matrix.csv`
- `w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/void_subset_confusion.csv`
- `w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/summary.json`
- `w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/manifest.json`
- `w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/analyze.py` (reproducibility)

Reproducibility command:

```bash
python w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/analyze.py
```

Runtime: 207.9s on M5 Max (mps). Pipeline batched per-snapshot (the multi-snapshot batched call hits a `index 28 is out of bounds` issue when snapshots at heterogeneous decision indices are co-batched — automatic fallback, no functional impact).

---

## Provenance

Wave 4.0 measurement agent (t42-hmjr). Forge pipeline + bootstrap statistics. Inputs from Wave 2 corpus; conventions matched to Wave 2 reference probe (`w42/book_validation_v1/wave2/probes/t42-z31l_void_creation_follow/run_void_creation_follow_probe.py`).

---

## Links

- [[w42-bookval-v2-utility-lens-synthesis]] — Wave 3.0 finding this measurement gates
- [[w42-bookval-v1-wave2-void-creation-follow]] — original paired-contrast probe (t42-z31l)
- [[w42-bookval-v1-wave1-mark-utility-transform]] — Wave1.2 mark_ev ≡ p_make at bid=30 (now confirmed at argmax level)
- [[w42-bookval-v1-wave2-ch10-action-level]] — Wave2.H mark_ev affine identity at all bids
- [[w42-book-validation-campaign]] — parent campaign
- [[w42-book-claim-synthesis-and-ai-directions]] — campaign synthesis

---

## Audit (2026-07-07)

Two-pass audit against code and artifacts; no issues found.

- Cheap next probe: rerun on a higher-bid subset (bid ≥ 84, mm ≥ 2) to actually exercise the mark_ev ≠ p_make regime the bid=30 caveat says this corpus cannot test.
- Per-snapshot resampling variance (caveat 3) is checkable for ~$0: rerun 50 snapshots with a different world seed and count argmax flips.
