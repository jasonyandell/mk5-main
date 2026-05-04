---
title: W42 Lens v1 — Utility Head-to-Head Round-Robin
kind: experiment
status: closed-on-completion
first_seen: local-2026-05-03
bead: t42-4ouu
parent_epic: t42-4zi6
---

## Question

Wave 4.0 [[w42-bookval-v3-utility-argmax-divergence]] established that EV-greedy and p_make-greedy disagree on **41.2%** of legal-action argmaxes on the bid=30 setter corpus — a real policy-level divergence, not just a contrast-magnitude artifact. Wave 4.0 said "rung-2 is justified" but did **not** answer the next question: when those two greedy policies actually play out the resulting hands head-to-head, **which one scores more points?** Same for cvar_10 and robust_q25.

This experiment is the first head-to-head measurement: drop a Q-greedy "Lens" player into a parallel-hand simulator, parameterize it by utility, run round-robin between the four interesting utilities (ev, p_make, cvar_10, robust_q25), report mean point margin per hand with bootstrap CIs.

This is a **measurement instrument**, not a claim verdict. The promotion guard from `w42/book_validation_v1/AGENTS.md` applies — no central ledger row is moved.

---

## Method

**Players.** A Lens(utility) player is a 1-step Q-greedy player. At each decision:
1. Forge eq pipeline → per-action Q-distribution (`e_q[7]`, `e_q_pdf[7, 85]`) using N world samples.
2. Compute the utility's scalar value per legal action from `(e_q, e_q_pdf)` (Wave 4.0 conventions; see `w42/lens_v1/lens.py`).
3. Pick `argmax` over legal actions; tie-break on lowest slot index.

**Five utilities** (functions taking the per-action Q-distribution, returning a scalar; higher-is-better for all five because forge Q is POV-corrected):
- `ev`           = mean of Q samples
- `p_make`       = P(Q ≥ contract_threshold) using forge bid-aware bins (defense bin = 25 at bid=30 → Q ≥ −17)
- `mark_ev`      = `mm * (2*p_make − 1)` where `mm = max(1, bid // 42)`
- `cvar_10`      = mass-weighted mean of Q in lower-tail (cum mass ≤ 0.10), with fallback to first mass-bearing bin
- `robust_q25`   = smallest Q with cumulative mass ≥ 0.25

**Simulator.** A custom parallel-hand simulator (`w42/lens_v1/parallel_match.py`) modeled on `forge.zeb.eq_player._run_eq_vs_eq_batched`. The existing batched eq-vs-eq path could not be cleanly extended because `forge.eq.generate.actions.select_actions` hardcodes p_make-with-EV-tiebreak (effectively Lens(p_make)). The simulator lifts that one design choice out and injects `argmax_under_utility(utility, e_q, e_q_pdf, ...)` from `lens.py`.

**Round-robin.** {ev, p_make, cvar_10, robust_q25} → 6 pairings × 1000 hands. `mark_ev` is excluded from the round-robin (affine-identical to p_make at bid=30 — Wave 4.0 confirmed argmax-coincidence on every snapshot); recorded once as a sanity matchup.

**Paired-seed.** The same shuffle seed produces the same initial deal+bidder+decl for both halves of a matchup. Half the hands play with team A in seats {0, 2}; the other half with team A in seats {1, 3}. Same seeds across both halves → paired-difference variance reduction.

**Bid forced to 30.** Wave 4.0's corpus is bid=30 only. We override `new_game`'s random bid to 30 in `_force_bid_30` so every hand is a bid=30 contract — keeps the corpus comparable to Wave 4.0 and keeps mark_ev ≡ p_make valid.

**Sample budget.** N=10 default (per `forge/zeb/OVERVIEW.md` lines 622-632: N=10 wins 55.3% vs Zeb-Large vs N=100's 55.7% — essentially tied, much faster). Confirmed for Lens-vs-Lens by a sample-sweep on ev vs p_make at N ∈ {10, 50, 100}, 250 hands each.

**fp dtype.** fp32. fp16 sanity check on 100 fixed game states (same world seed, same model, only precision differs) showed argmax-match ≥ 99% for all 5 utilities — fp16 is safe to use, but the round-robin runs in fp32 for reproducibility (MPS doesn't autocast inside `query_model` and the speed difference is small).

**Statistics.** Per-pairing 95% bootstrap CI on the mean point margin (n_boot=2000, percentile, seed=42). Decisive-hand rate = % hands with non-zero margin. A win rate = % hands where Team A scored more points.

---

## Results — Round-Robin

1000 hands per pairing, paired-seed (same shuffle seed across both team-assignment halves), N=10 worlds per Lens decision, fp32, MPS. Mean point margin per hand = (Team A points − Team B points) averaged over 1000 hands. 95% CI from percentile bootstrap (n_boot=2000, seed=42). "Decisive" = % of hands where margin ≠ 0.

| Team A      | Team B      | mean margin | 95% CI            | A win rate | decisive | CI ≠ 0 |
|-------------|-------------|------------:|:------------------|:----------:|:--------:|:------:|
| **ev**      | p_make      | **+5.42**   | [+4.03, +6.81]    | 59.5%      | 99.5%    | **yes** |
| **ev**      | cvar_10     | **+3.98**   | [+2.55, +5.45]    | 55.2%      | 99.7%    | **yes** |
| **ev**      | robust_q25  | **+2.49**   | [+1.09, +3.98]    | 56.2%      | 99.7%    | **yes** |
| p_make      | cvar_10     | −1.91       | [−3.35, −0.45]    | 44.3%      | 99.7%    | **yes** |
| p_make      | robust_q25  | −2.97       | [−4.52, −1.51]    | 44.3%      | 99.8%    | **yes** |
| cvar_10     | robust_q25  | −1.71       | [−3.15, −0.27]    | 47.4%      | 99.7%    | **yes** |

**Every pairing's 95% CI excludes zero.** Total ordering implied:

```
ev   >   robust_q25   ≳   cvar_10   >   p_make
       (+2.49)         (+1.71)       (−2.97 from p_make)
```

Margins between adjacent utilities in the ordering are 1.5 - 2 pts/hand (cvar_10 vs robust_q25, p_make vs cvar_10), so the ranking is consistent: ev dominates all three; robust_q25 beats cvar_10 by ~1.7; cvar_10 beats p_make by ~1.9.

The ev advantage is largest against p_make (+5.42) and smallest against robust_q25 (+2.49) — i.e. p_make is the worst utility on this corpus and robust_q25 is the second-best. This is at first surprising given that Wave 4.0's "intuition" framing was "EV likes void, p_make likes preserve" — the head-to-head answer is harsher: p_make plays meaningfully worse than EV at the actual point-scoring level.



---

## Sample-Sweep Confirmation

ev vs p_make at three sample counts, 250 hands each, paired-seed. Same `base_seed=10000` as the round-robin so the first 250 hands match the first 250 hands of the corresponding round-robin pairing.

| N    | mean margin | 95% CI            | A win rate | wall    |
|:----:|------------:|:------------------|:----------:|--------:|
| 10   | +6.18       | [+3.19, +9.10]    | 59.6%      | 20.9 s  |
| 50   | +4.54       | [+1.29, +7.66]    | 59.2%      | 26.6 s  |
| 100  | +3.13       | [+0.09, +6.34]    | 52.8%      | 58.4 s  |

All three N values confirm Lens(ev) wins ev-vs-p_make (CI excludes zero). Mean margin slightly drops as N increases (6.2 → 4.5 → 3.1), but the head-to-head ranking is preserved. **N=10 is the right default for this experiment** — the 1.6× speedup vs N=50 is a real budget win and the qualitative finding is robust to the larger sample budget.

A subtlety: the N=100 mean margin at 250 hands (+3.13) is closer to the round-robin's N=10 1000-hand result for ev-vs-robust_q25 (+2.49) than it is to ev-vs-p_make's 1000-hand result (+5.42). This mostly reflects the smaller hand count (250 vs 1000) — the CI is wide enough to overlap both numbers — not a real divergence between N=10 and N=100. If we want a tighter answer about how much N matters, the right next step would be N=10 vs N=100 at 1000 hands paired-seed; that's out of scope for v1.



---

## fp16 Sanity

100 fixed game states (sampled from a 30-hand fp32 ev-greedy rollout) evaluated as **a single batch** with both fp32 and fp16 model weights, same world-sampling seed, same input. Compared per-utility argmaxes.

| utility       | n states | argmax match | rate  |
|---------------|---------:|-------------:|------:|
| ev            | 100      | 99           | 99.0% |
| p_make        | 100      | 100          | 100.0%|
| mark_ev       | 100      | 100          | 100.0%|
| cvar_10       | 100      | 99           | 99.0% |
| robust_q25    | 100      | 100          | 100.0%|

All five ≥ 99%. **fp16 is safe to use** — but the round-robin runs in fp32 because (a) MPS doesn't autocast inside `forge.eq.generate.model.query_model` so the speed gain on Mac is small, and (b) reproducibility of saved per-hand records is cleaner under fp32. The finding is recorded for future runs on CUDA where fp16 pays off (~2x speedup).



---

## mark_ev ≡ p_make Sanity (bid=30)

200 hands of mark_ev vs p_make. Result:

| metric              | value                |
|---------------------|----------------------|
| mean margin         | +0.75 (CI [−2.36, +4.01]) |
| n hands with margin = 0 | 1 / 200          |

The mean margin's CI spans zero (consistent with the affine identity), but only 1/200 hands had a literal margin of 0. **This is expected, not a bug.** The Wave1.2/2.H/4.0 affine identity says: *given the same Q-distribution*, mark_ev's argmax equals p_make's argmax at bid=30. It does **not** say two independent matches with different random world samples will produce identical play.

In this matchup each player makes its own forge query at its turn, so the Q-distributions seen by mark_ev (when it plays seat 0) and by p_make (when it plays seat 0 in the paired half) are different random realizations even when the upstream game state is identical. The argmax of each individual Q-pdf coincides for mark_ev and p_make (Wave 4.0 confirmed 0/500 disagreement on identical pdfs); but the actions chosen by mark_ev and p_make in this matchup come from **different** pdfs, so they diverge.

The right argmax-level confirmation of mark_ev ≡ p_make at bid=30 is Wave 4.0's snapshot-level test (`disagreement_matrix.csv` shows 0/500 mark_ev-vs-p_make disagreements on identical Q-distributions). What the head-to-head matchup confirms is the weaker statement: **the mean margin between mark_ev-greedy and p_make-greedy is statistically indistinguishable from zero** — no systematic preference between them, consistent with their being the same policy in expectation.



---

## Verdict

> **WINNER: Lens(ev) beats Lens(p_make) by +5.42 pts/hand** (95% CI [+4.03, +6.81]). All 6 round-robin pairings are statistically distinguishable (every CI excludes zero). Implied total ordering: **ev > robust_q25 ≳ cvar_10 > p_make**. The Wave 4.0 architecture-decision question — "is utility-conditioning worth building?" — gets a clean experimental answer at the policy-action level: **EV-greedy is the strongest of the four lenses tested at 1-step lookahead on the bid=30 corpus; p_make-greedy is the weakest by ~5 pts/hand vs EV.**

The single most actionable consequence for downstream work: **a utility-conditioned head trained to mimic Lens(ev) is the natural rung-2 target.** A multi-utility head that conditions on {ev, p_make, cvar_10, robust_q25} is also defensible as research infrastructure (Wave 4.0 said it's worth building), but the head-to-head data says Lens(ev) is the best-scoring single lens by a clear margin — there is no "you should use p_make for safety" justification on this corpus.

A non-obvious finding: **`forge.eq.generate.actions.select_actions` is essentially Lens(p_make)** (it picks p_make-argmax with EV as tie-break). The current production EV pipeline is therefore using the **worst** of the four utilities tested. Switching `select_actions` to ev-argmax (with p_make as tie-break, or no tie-break at all — they almost never tie at fp32 resolution) is a one-line change worth a follow-up evaluation against Zeb-Large to see whether the +5.42 pts/hand head-to-head advantage of Lens(ev) over Lens(p_make) translates into a higher Zeb-Large win rate than the current 55.7% E[Q]-N=100 figure.



---

## Follow-up exploration: `disaster` utility

User question: "what if we treat any set outcome as equally bad (Q=−42) instead of distinguishing 'set by 5' from 'set by 25'? Why would that be a bad idea?"

A new utility `disaster` was added to `lens.py` (UTILITIES tuple) implementing exactly that: for each Q-bin below the seat's make threshold, replace the bin's Q value with −42; bins at/above threshold keep their continuous Q; take expectation under the PDF. So `disaster` matches EV above threshold and floors everything below threshold to the worst-case value.

Head-to-head (1000 paired-seed hands, N=10, fp32, MPS):

| matchup | mean margin | 95% CI | excludes zero |
|---|---:|:---|:---:|
| disaster vs **ev** | −1.55 | [−3.07, +0.05] | barely no (CI grazes zero) |
| disaster vs **p_make** | **+2.74** | [+1.24, +4.37] | yes |
| disaster vs **robust_q25** | +0.32 | [−1.13, +1.81] | no (tied) |

**Implied ordering with disaster inserted:** `ev ≳ disaster ≳ robust_q25 ≳ cvar_10 > p_make`. Disaster lands between EV and the risk-aware cluster.

**Interpretation:** disaster keeps the part of EV that matters most (continuous reward above threshold, which beats p_make's binary tie-breaking by +2.74 pts/hand) but loses the part of EV that matters slightly less (damage control on losing hands — distinguishing "set by 5" from "set by 25"). Net cost vs EV: ~1.5 pts/hand at n=1000, statistical separation just outside 95% confidence.

This is consistent with the earlier finding that EV's strength comes from using full Q-distribution information; throwing away one tail of the distribution costs only modestly because the above-threshold tail is doing most of the work. **Disaster is "not a bad idea" — it just doesn't get to use the damage-control information that EV uses.**

Artifacts: `w42/lens_v1/results/disaster_head_to_head.csv`, `w42/lens_v1/run_disaster.py`. Sanity test included in `run_disaster.py` confirms the utility correctly clips below-threshold bins (synthetic-pdf check).

## Caveats

- **1-step lookahead only.** Lens is greedy — no search tree, no opponent modeling beyond what's already inside the forge oracle's world-rollout assumption.
- **Forge's internal world-rollout opponents are NOT parameterized by utility.** Both teams' Q-distributions are computed under the same forge-default rollout policy (whatever the oracle was trained against). So this measures "which utility-greedy player wins?" not "what happens if every player in the imagined game tree uses utility X?"
- **Bid=30 corpus only.** Higher-bid contracts and structurally different threshold regimes are out of scope. The mark_ev affine-identity collapse only holds at bid=30.
- **Snapshot/world sample noise.** Argmax can flip at fine Q deltas, especially at N=10. The bootstrap CI quantifies hand-level variance (across the 1000 paired hands), not per-decision world-resampling variance.
- **Paired-seed reduces card-luck variance only.** It doesn't account for the variance from which hands the model misjudges in a systematic way; that's the utility-difference signal we want to measure, but it could still be biased by the specific 1000-hand corpus.
- **Measurement instrument, not a claim verdict.** Promotion guard from AGENTS.md applies — no claim ledger row moves based on this experiment alone.

---

## Utility coverage

All five utilities are implemented in `lens.py` and recorded for every decision in the fp16 sanity check. The round-robin runs only the four interesting utilities (mark_ev excluded as redundant at bid=30) but the underlying Q-distribution is the same — recomputing any other utility post-hoc requires only the saved per-hand records.

---

## Artifacts

- `w42/lens_v1/lens.py` — utility wrappers + `argmax_under_utility`
- `w42/lens_v1/parallel_match.py` — utility-aware parallel-hand simulator
- `w42/lens_v1/round_robin.py` — runner script (round-robin + sample-sweep + mark_ev sanity)
- `w42/lens_v1/fp_sanity.py` — fp32 vs fp16 argmax-match sanity
- `w42/lens_v1/analyze.py` — re-derive verdict from per-hand CSV
- `w42/lens_v1/results/round_robin_n10.csv` — per-pairing summary
- `w42/lens_v1/results/per_hand_margins.csv` — per-hand margins for every matchup
- `w42/lens_v1/results/sample_sweep.csv` — N-sweep on ev vs p_make
- `w42/lens_v1/results/fp_sanity.csv` — fp32 vs fp16 argmax-match table
- `w42/lens_v1/results/mark_ev_pmake_sanity.csv` — mark_ev vs p_make sanity
- `w42/lens_v1/manifest.json` — provenance, command, runtime, device, fp dtype, samples
- `w42/lens_v1/summary.json` — full headline numbers, machine-readable

Reproducibility command:

```bash
python w42/lens_v1/round_robin.py --n-hands 1000 --n-samples 10 --device mps --base-seed 10000
```

Re-derive verdict from CSVs only:

```bash
python w42/lens_v1/analyze.py --results-dir w42/lens_v1/results
```

---

## Provenance

Lens v1 build agent (t42-4ouu). Forge oracle + custom utility-aware simulator. Inputs: `forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt`. Conventions matched to Wave 4.0 (`w42/book_validation_v1/wave4/t42-hmjr_utility_argmax_divergence/analyze.py`).

---

## Links

- [[w42-bookval-v3-utility-argmax-divergence]] — Wave 4.0 measurement that gated this build
- [[w42-bookval-v2-utility-lens-synthesis]] — Wave 3.0 contrast-magnitude split
- [[w42-book-validation-campaign]] — parent campaign
- [[w42-book-claim-synthesis-and-ai-directions]] — campaign synthesis
