# WorldSamplerMRV measurement audit

This directory turns the unresolved `WorldSamplerMRV` note into an executable,
falsifiable measurement.  It does not assume that the surviving “~6.8 Q” number
was a clean sampler-only result.

## What it measures

For each fixed late-hand state, `audit.py`:

1. enumerates every remaining-hand assignment consistent with public play and
   voids (`enumerate_worlds_cpu`);
2. computes the exact probability of every completed world under MRV’s local
   choice rule;
3. runs the live CPU sampler under a fixed torch seed and checks it separately
   against the exact-uniform target and the analytically reconstructed legacy
   MRV probabilities;
4. compares uniform and MRV seat-by-domino marginals; and
5. evaluates every distinct output once with the Stage-1 checkpoint under the
   production remaining-only representation, then changes only world weights
   to measure action-EV shifts, rank changes, and exact regret; and
6. repeats Q evaluation with reconstructed full initial deals as an explicitly
   non-production encoding-sensitivity control.

The analytic recursion is the crucial control.  It distinguishes structural
bias from finite-sample noise.

## Historical claim and confound

The claim originated in commit `7321952` and survives in:

- `wiki/sources/7321952.md`
- `wiki/entities/burl.md`
- `wiki/topics/consumption-ledger.md`
- `wiki/questions/open.md`

The original session detail did not survive: selected action, sampler count,
RNG seed, and exact comparison code are absent.  At commit `7321952` there was
also an encoding asymmetry in Burl’s two paths:

- sampled worlds contain remaining opponent tiles only before
  `build_hypothetical_deals`;
- `_enumerated_worlds_tensor` packs played-known plus remaining opponent tiles.

That asymmetry can move model Q independently of world probabilities.  It was
fixed in the surrounding research session: both Burl paths now use
remaining-only worlds, with a regression test.  This audit makes that current
production representation primary and uses one canonical opponent slot order
so only probabilities change.  Full 4x7 reconstruction remains a labeled
sensitivity control.

## Run

Full audit, including the checked-in Stage-1 oracle:

```bash
python -m w42.world_sampler_audit.audit
```

Fast distribution/marginal-only pass:

```bash
python -m w42.world_sampler_audit.audit --skip-oracle --samples 20000
```

The runner writes JSON to stdout so recorded artifacts can be reviewed before
being checked in. `summary.json` preserves the reviewed pre-repair defect
measurement; `repair_summary.json` records the current implementation and its
fresh fixed-seed validation. A current runner invocation therefore agrees with
the repair artifact's live-sampler fields, not the historical summary's legacy
implementation-conformance fields.

Tests:

```bash
pytest -q w42/world_sampler_audit/test_audit.py
```

## Interpretation

- `uniform_vs_analytic_mrv_tvd > 0` is exact output-distribution error, not
  sample noise. It includes malformed outputs.
- `uniform_vs_analytic_mrv_valid_conditional_tvd > 0` isolates bias among valid
  outputs after conditioning away malformed mass.
- `analytic_mrv_dead_end_probability` records paths where no candidate exists;
  the legacy greedy tensor code injected domino `00` through all-false
  `argmax` there.
- `legacy_mrv_conformance.passes` means the live sampler still implements the
  audited historical MRV rule within a seven-sigma per-cell bound.
- `uniform_conformance.passes` means the live sampler matches exact-uniform
  enumeration within the same bound. After the repair the former must fail and
  the latter must pass.
- `max_abs_marginal_error` identifies the largest seat/tile ownership shift.
- `downstream_q.by_encoding.remaining_only_canonical` is the primary
  action-level consequence; `full_initial_control` measures sensitivity.
- A Q shift without an argmax flip is value distortion, not yet decision harm.
- An argmax flip with positive `exact_regret_of_mrv_choice_q` is direct
  downstream decision harm on that fixture.

The three fixed fixtures are a measurement panel, not a population estimate.
They can falsify “MRV is uniform” and demonstrate concrete downstream effects;
they cannot by themselves estimate corruption across historical corpora.

## Repair result

`repair_summary.json` records two replacement attempts rather than erasing the
failed intermediate. `uniform-rejection-v1` fixed validity and distribution on
the three late-state fixtures, then failed a real JudSearch state: only 924 of
17,153,136 labeled-seat partitions are valid (exact mass 5.39e-5), and a 40,960
proposal budget produced 3 of 10 requested worlds.

The surviving API fingerprints itself as `uniform-completion-dp-v1`. An int64
suffix dynamic program counts legal completions for all 8^3 remaining-capacity
states, then assigns each tile to a seat in proportion to the exact suffix count
behind that choice. The conditional probabilities telescope to one divided by
the root completion count, so complete assignments are uniform without a
world-level rejection loop. Branch draws use exact 62-bit integer rejection,
not float32 random keys; padding positions are identity transitions. At 100,000
samples per fixture it emits no invalid world, covers every exact world, and
matches exact-uniform frequencies within 2.83 standard errors per cell. The
low-mass JudSearch regression and the full arena test now pass. CUDA production
performance remains unmeasured on this CPU-only Mac.
