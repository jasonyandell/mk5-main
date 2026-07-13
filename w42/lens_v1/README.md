# Lens v1 — Utility-Conditioned Q-Greedy Head-to-Head

**Bead:** t42-4ouu
**Parent:** t42-4zi6 (W42 book validation campaign)
**Status:** closed-on-completion
**Wiki page:** [`wiki/experiments/w42-lens-v1-utility-head-to-head.md`](../../wiki/experiments/w42-lens-v1-utility-head-to-head.md)

This is the architecture-decision payoff for [Wave 4.0](../book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/) (`t42-hmjr`). Wave 4.0 measured that EV-greedy and p_make-greedy disagree on **41.2%** of legal-action argmaxes on the bid=30 setter corpus. This experiment measures the next question directly: **when those greedy policies actually play hands head-to-head, which one scores more points?**

## What this is

A "Lens" player is a 1-step Q-greedy player parameterized by a utility function:

```python
def lens_choose_action(state, utility_name):
    e_q, e_q_pdf = forge.query(state)             # forge eq pipeline
    summarized   = utility(utility_name, e_q, e_q_pdf)
    return argmax_legal(summarized)               # tie-break: lowest slot
```

Five utilities, matched to Wave 4.0 conventions (see `lens.py`):

| name        | scalar definition                                       |
|-------------|---------------------------------------------------------|
| `ev`        | `mean(Q)` — forge `e_q[slot]`                           |
| `p_make`    | `P(Q ≥ contract_threshold)` (forge bid-aware bins)      |
| `mark_ev`   | `mm * (2*p_make − 1)` where `mm = max(1, bid // 42)`    |
| `cvar_10`   | mass-weighted mean of Q in lower-tail (cum ≤ 0.10)      |
| `robust_q25`| smallest Q with cumulative mass ≥ 0.25                  |

All five are "higher = better" because forge Q is POV-corrected by `query_model`.

## Why a custom simulator

The existing `forge.zeb.eval` "eq vs eq" batched path uses `forge.eq.generate.actions.select_actions`, which **hardcodes** p_make-with-EV-tiebreak as the action selector. We could not cleanly extend that helper to take a utility parameter without changing its call signature across the entire generate pipeline. So `parallel_match.py` reuses every other piece of the eq batched path (world sampling, deal building, tokenization, model forward, EV/PDF computation) and lifts only the action-selection step into a utility-aware function (`lens.argmax_under_utility`).

The simulator behaves identically to `_run_eq_vs_eq_batched` except for the action selector. Bid is forced to 30 on every hand to match the Wave 4.0 corpus and keep `mark_ev ≡ p_make`.

## Round-robin scope

Pairings (each 1000 hands, N=10 worlds per decision, paired-seed across both team assignments):

```
ev          vs p_make
ev          vs cvar_10
ev          vs robust_q25
p_make      vs cvar_10
p_make      vs robust_q25
cvar_10     vs robust_q25
```

`mark_ev` is excluded from the round-robin (Wave 4.0 confirmed argmax-coincidence with p_make on every snapshot at bid=30); recorded as a single sanity matchup that should produce identical play.

## Files

```
w42/lens_v1/
├── README.md                       # this file
├── manifest.json                   # provenance, command, runtime, device, dtype, samples
├── summary.json                    # full headline numbers, machine-readable
├── lens.py                         # utility wrappers + argmax_under_utility
├── parallel_match.py               # utility-aware parallel-hand simulator
├── round_robin.py                  # round-robin + sample-sweep + mark_ev sanity runner
├── fp_sanity.py                    # fp32 vs fp16 argmax-match sanity
├── analyze.py                      # re-derive verdict from per-hand CSV
└── results/
    ├── round_robin_n10.csv         # per-pairing summary
    ├── per_hand_margins.csv        # per-hand margins for each matchup
    ├── sample_sweep.csv            # ev vs p_make at N=10/50/100, 250 hands each
    ├── mark_ev_pmake_sanity.csv    # mark_ev vs p_make sanity (must tie)
    ├── fp_sanity.csv               # fp32 vs fp16 argmax-match table
    └── round_robin.log             # full stdout from the round-robin run
```

## Reproducibility

Round-robin (the production run):
```bash
python w42/lens_v1/round_robin.py --n-hands 1000 --n-samples 10 --device mps --base-seed 10000
```

fp16 sanity (small, ~5 sec):
```bash
python w42/lens_v1/fp_sanity.py --n-hands 30 --n-states-sampled 100 --n-samples 10
```

Re-derive verdict from CSVs only (no model needed):
```bash
python w42/lens_v1/analyze.py --results-dir w42/lens_v1/results
```

## Verdict

> **WINNER: Lens(ev) beats Lens(p_make) by +5.42 pts/hand** (95% CI [+4.03, +6.81]).

All 6 round-robin pairings have CIs that exclude zero. Implied total ordering:

```
ev   >   robust_q25   ≳   cvar_10   >   p_make
       (+2.49)         (+1.71)       (−2.97 from p_make)
```

Sample-sweep at N ∈ {10, 50, 100} on ev vs p_make confirms ev wins at every N. fp16 sanity is ≥99% argmax-match (round-robin still ran in fp32 for reproducibility on MPS). mark_ev vs p_make matchup mean-margin CI spans zero, consistent with the Wave1.2/2.H/4.0 affine identity (the literal margin-0 count is 1/200 — see wiki page for why this is expected, not a bug).

**Non-obvious consequence**: `forge.eq.generate.actions.select_actions` is essentially Lens(p_make) — i.e. the current production EV pipeline is using the **worst** of the four utilities tested. Switching that selector to ev-argmax is a one-line change worth a follow-up Zeb-Large evaluation.

See the wiki page for full results, sample-sweep table, fp16 sanity table, caveats, and verdict discussion: [`wiki/experiments/w42-lens-v1-utility-head-to-head.md`](../../wiki/experiments/w42-lens-v1-utility-head-to-head.md).

## Caveats

- **1-step lookahead only.** Lens is greedy — no search tree.
- **Forge's internal world-rollout opponents are NOT parameterized by utility.** Both teams' Q-distributions are computed under the same forge-default rollout policy, so this measures "which utility-greedy player wins?" not "what happens when everyone in the imagined game tree uses utility X?"
- **Bid=30 corpus only.** `mark_ev`'s affine-identity collapse only holds at bid=30.
- **Sample noise.** Argmax can flip at fine Q deltas; the bootstrap CI quantifies hand-level variance, not per-decision world-resampling variance.
- **Measurement instrument, not a claim verdict.** Promotion guard from `w42/book_validation_v1/AGENTS.md` applies — this experiment moves no central ledger row.
