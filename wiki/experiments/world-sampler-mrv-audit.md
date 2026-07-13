---
title: WorldSamplerMRV audit
kind: experiment
first_seen: bc4eb386
last_updated: 4123b2d5
status: complete
---

Stage 0 of [[partnership-wall-research]] turns the surviving “~6.8 Q sampler
bias” note into an executable measurement. The result does not reproduce that
number cleanly. It finds a more concrete defect: `WorldSamplerMRV` can emit
malformed worlds.

## Question

Does `WorldSamplerMRV` sample uniformly from the worlds consistent with public
play and void evidence, and can any difference from exact enumeration be
separated from Monte Carlo noise and Q-input representation?

## Method

`w42/world_sampler_audit/audit.py` uses three fixed late-hand states. For each
it:

1. enumerates the exact remaining-hand world set;
2. calculates the full probability distribution induced by the tensor MRV
   implementation, including its no-candidate branch;
3. checks 100,000 seeded live samples against that analytic distribution;
4. compares ownership marginals and total-variation distance; and
5. evaluates each distinct output once with the checked-in Stage-1 checkpoint,
   then changes only its world weight.

The primary Q path uses the production remaining-only representation with one
canonical slot order. Reconstructed full initial deals are a labeled
representation-sensitivity control. This removes the confound in the original
[[burl]] comparison, where exact and sampled worlds were packed differently.
That Burl path now has a regression test enforcing representation parity.

## Result

| fixture | exact worlds | malformed mass | valid-only TVD | largest production Q shift | argmax flip |
|---|---:|---:|---:|---:|---:|
| historical seed 900013, play 20 | 12 | exactly 1/3; live 0.33449 | 0 | 4.619 | no |
| seed 900017, play 20 | 60 | 0 | 0.0333 | 0.00120 | no |
| seed 900029, play 20 | 30 | 0 | 0 | 0 | no |

On the historical state, MRV can select an active opponent with no remaining
candidate. The implementation does not reject or backtrack. `argmax` over an
all-false candidate row returns index zero, injecting domino `00` even though
the pool is `{20,22,31,32,44,63}`. The live invalid rate matches the exact
one-third mechanism; the largest live-versus-analytic residual is only 2.77
standard errors.

The resulting malformed mass moves the two legal actions in opposite directions
by `+4.416` and `-4.619` Q under the production encoding. The better action
remains `21`, so this panel demonstrates value distortion but no selected-action
harm. Conditional on valid outputs the historical fixture is uniform. The
second fixture proves a separate, smaller defect: greedy local MRV choices are
not generally uniform over legal completions even when every emitted world is
valid. The third fixture is the symmetry null.

## Epistemic classification

- The malformed-world and non-uniformity mechanisms are positive findings.
- The historical `~6.8 Q` number is confounded and retired as an effect
  estimate; it is neither confirmed nor cleanly contradicted.
- No argmax flip in three fixtures is a bounded null observation
  (superseded by the population scan: 20 flips in the 200 worst-mass states,
  regret up to `7.37 Q` — [[stage-0-closure]]), not evidence
  that historical corpora or match outcomes were unaffected.
- Failure on the low-valid-mass JudSearch state is a negative result for
  `uniform-rejection-v1` as a production repair.
- `uniform-completion-dp-v1` passes the defined sampler instrument gate. It is
  not a policy result.

## What this changes

- The “validity-guaranteed MRV sampler” claim is falsified.
- The old `~6.8 Q` statement is retired as a clean sampler estimate. Its action,
  sample count, RNG seed, and representation parity did not survive.
- The audit does **not** establish a population corruption rate or a C0 action
  loss. Three late states contain no argmax flip; that observation is bounded
  to the diagnostic panel. ([[stage-0-closure]] later established both: 2.51%
  distributional exposure over 32k reconstructed states, and argmax flips in
  20 of the 200 worst-mass states with regret up to `7.37 Q` — while the
  two-block C0 reproduction shows match-level conclusions survived.)
- Historical corpus and arena conclusions are not automatically void. Their
  exposure depends on state-specific dead-end probability, valid-world bias,
  Q gap, and consumer.
- New corpus generation and a fresh [[champion]] reproduction are blocked on a
  sampler that is both valid and uniform, checked against exact enumeration.

This is measurement cleanup, not evidence for CFR, a larger net, an LLM, a
symbolic plan library, or Jud v2. It narrows the first build to the sampler and
then a population exposure audit.

## Repair gate

A replacement passes only if it:

- emits no domino outside the pool and exactly partitions every pool;
- respects every void and hand-size constraint;
- matches uniform exact enumeration on tractable states, including the two
  biased fixtures above;
- proves feasibility before sampling and raises rather than returning a
  partial or fabricated world on any internal failure; and
- preserves the batched GPU-only production contract.

The first replacement, `uniform-rejection-v1`, passed the three audit fixtures:
uniformly random partitions conditioned on void validity are uniform. A broader
arena regression falsified it before release. In a real JudSearch state only
`924` of `17,153,136` labeled-seat partitions are valid (exact mass
`5.38677e-5`); 40,960 proposals produced only 3 of 10 requested worlds. Raising
the proposal budget was rejected because acceptance probability, not a tuning
constant, was the failure.

The surviving legacy-named `WorldSamplerMRV` fingerprints itself as
`uniform-completion-dp-v1`. An int64 suffix dynamic program counts completions
over tile position and the `8^3` three-seat capacity grid. Each tile goes to a
seat in proportion to the exact suffix count behind that branch, so the
conditional probabilities telescope to `1 / root_completion_count` for every
valid world. Padding is an identity transition, 62-bit integer draws avoid
float-key bias, infeasible inputs have root count zero, and the production path
never falls back from CUDA to CPU.

At 100,000 samples per audit fixture the final replacement emits zero invalid
worlds, covers all `12`, `60`, and `30` exact worlds, and matches exact-uniform
frequencies with largest per-cell residual `2.83` standard errors. The direct
unconstrained root count is `21!/(7!^3) = 399,072,960`. The combined sampler,
audit, and full JudSearch suite passes `46` tests with two CUDA tests skipped.
On CPU the final sampler takes `4.25 ms` for unconstrained `32 x 50`, `0.63 ms`
for the historical `1 x 50`, and `1.64 ms` for the low-mass `1 x 10` state where
rejection exhausted. [[stage-0-closure]] then measured the CUDA path on a
rented 4090 (`w42/world_sampler_audit/cuda_bench_2026-07-13.md`): every
device-parameterized regression passes, and the sampler is kernel-launch bound
(~30 ms/call flat across batch shapes) — ~7× slower than CPU at `32 x 50`,
paying off only through batch width (`858k` worlds/s at `256 x 100`, ≤63 MB).
Batch size, not device, is the throughput lever; small-batch consumers
(JudSearch n=10) belong on CPU.

Review of the shipped repair found one further defect in the same silent-bias
class it was built to kill: the MPS backend's int64 `gather` rounds the 62-bit
draws through float32, so on Apple Silicon every emitted world stayed valid
while the distribution collapsed — support `24/60` and chi-square `1.4e5` at
60,000 samples on the valid-bias fixture, worse than the legacy sampler it
replaced. All original validation ran `--device cpu`, which is why it passed.
Fixed at `4123b2d5`: candidate selection uses an exact `where` instead of
`gather` (bit-identical on CPU, so the checked-in fixed-seed artifacts remain
valid), and the dead-end and uniformity regressions now parameterize over
every available device (CPU/CUDA/MPS).

The repair closed the sampler gate; [[stage-0-closure]] (2026-07-13) then
closed the measurement questions it left open. The two-block C0 reproduction
landed inside its registered bands (`+0.385`/`+0.486` vs original
`+0.38`/`+0.42`), the P0 symmetry and challenger re-grades reproduced, the
CUDA suite passed, and the state-level exposure scan
(`w42/world_sampler_audit/exposure_scan/`) found `2.51%` of a 32,000-state
reconstructed late-state population carrying nonzero legacy malformed mass
(nonzero median `0.19`, max `0.83`; the exact-`1/3` fixture reproduces). The
repaired sampler costs ~2× wall time per C0 block on MPS. Decision-level harm
is measured: 20 argmax flips in the 200 worst-mass states, exact regret up to
`7.37 Q`, largest per-action shifts (`27 Q`) mostly cancelling in the argmax
([[stage-0-closure]], [[rank-vs-price]]).

## Artifacts and reproduction

- `w42/world_sampler_audit/summary.json`
- `w42/world_sampler_audit/repair_summary.json`
- `w42/world_sampler_audit/manifest.json`
- `w42/world_sampler_audit/audit.py`

```bash
python -m w42.world_sampler_audit.audit --samples 100000 --device cpu
pytest -q w42/world_sampler_audit/test_audit.py
```

## Links

[[partnership-wall-research]] [[consumption-ledger]] [[expected-q-value]]
[[burl]] [[forge]] [[champion]] [[the-wall]]
