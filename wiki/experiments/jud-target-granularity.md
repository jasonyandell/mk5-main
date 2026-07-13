---
title: Jud Target Granularity — hand-level vs per-move at fixed capacity
kind: experiment
first_seen: local-2026-07-13
last_updated: local-2026-07-13
status: active
---

Does per-move supervision beat hand-level supervision at fixed capacity —
[[research-lane-selection]] Lane B, [[partnership-research-gates]] row 1, the
question [[w42-jud-v1]] left entangled. Everything below **Registered
predictions** was written before any evaluation ran; arm training had started
but produced no numbers when this page was committed (the page's `first_seen`
is the registration commit).

## Design

One code path, one corpus, one seed; the arms differ in exactly one flag.

- **Corpus:** 512 games of `margin:wp(r8)+lens:ev` self-play at base seed
  9,100,000 (5,653 hand snapshots — disjoint from every reserved eval block).
- **Arm H (hand-level):** `JudNet` (trunk 350-512-512-43, ~470k params — jud
  v1's capacity) on the 43-bin realized-points target, `--weight-decay 1e-4
  --seed 42`.
- **Arm HP (hand-level + per-move):** identical everything, plus the
  per-legal-action auxiliary head (512→7 linear) trained by masked MSE on
  **teacher-forced** per-action E[Q] labels (`e_q/42`, acting-seat
  orientation) computed on the corpus's exact recorded decision states,
  `aux_lambda 1.0`. Seeded trunk init and batch order are proven
  aux-independent (`champion/test_jud_aux.py`).
- **Signal identity:** this tests the **dense E[Q] ranking auxiliary** only —
  signal (a) of [[jud]]'s two-signal split. Signal (b), the policy-conditioned
  per-move realized continuation value, needs a rollout-to-hand-end labeler
  that does not exist tonight; it is the named follow-on arm, not this
  experiment.
- **Hard gate before HP trains:** teacher-forced label join coverage
  `aux_coverage() ≥ 0.999` on decision rows. The original bridge replays the
  oracle's own greedy line, not the recorded line — a low-coverage or
  misaligned join would poison the arm, so a coverage miss aborts rather than
  reinterprets.

Evaluation, in gate order (calibration alone is not passage):

1. Held-out ranking: aux-vs-label Spearman, pairwise ordering accuracy, top-1
   agreement with `argmax(e_q)`; arm H is read through its `judplay`-style
   post-state ranking on the same decisions.
2. Primary-head guard: HP's main-head val CE within `0.05` of H's.
3. Paired play-only marks (the gate metric): `judplay:H` vs `judplay:HP` and
   `judsearch:n10` with each head vs `lens:ev` — bid30 both sides, 256
   games/block, reserved seeds 7,000,000 and 9,000,000, repaired sampler
   ([[stage-0-closure]] baseline context: `judsearch(r4)` currently `-1.42`/
   `-1.54`; greedy `judplay(r4)` was `-2.73` on this protocol).

## Registered predictions (best guesses, recorded as guesses)

1. **R1 — ranking:** HP's aux head reaches ≥ 0.60 top-1 agreement with
   `argmax(e_q)` on held-out decisions and beats arm H's post-state ranking on
   Spearman and pairwise ordering by a clear margin (H's mechanism is the
   diagnosed wall, so this should be large). Falsifier: aux top-1 < 0.5 or no
   ranking separation between arms — dense supervision failed to teach local
   discrimination even in-distribution.
2. **R2 — guard:** HP main-head val CE within ±0.05 of H's. Falsifier: aux
   training degrades the primary head (then sweep `aux_lambda {0.3, 3.0}`
   before concluding).
3. **R3 — greedy marks:** `judplay:HP` beats `judplay:H` by **≥ +0.5
   marks/game** (point guess **+1.0**) on paired blocks. This is the
   load-bearing guess: the [[lamir1-ceiling]] mechanism says scalar-value
   noise flips argmax while ranking-trained heads preserve ordering — dense
   per-action supervision is exactly that countermeasure. Falsifier: Δ CI
   includes zero → per-move E[Q] auxiliary at this capacity is not the lever,
   and [[partnership-research-gates]] row 1 withholds the build (a clean,
   valuable negative — the capacity×target interaction then becomes the next
   question).
4. **R4 — search marks:** `judsearch:n10(HP)` lands in **[-1.2, -0.2]** vs
   `lens:ev` (improvement from `-1.4`ish, short of parity) — guess: search
   gains less than greedy because JudSearch already converts the leaf's
   calibration into a ranking, so the aux head's marginal value is smaller
   above search than under greedy argmax.
5. **R5 — the wall stands tonight:** no arm beats `lens:ev` at pure play.
   Registered so that if it falls, the claim was on record before the run.
6. **R6 — aux-direct consumer** (`judauxplay`, added before any grading):
   reading the aux head directly beats routing HP's gain through the 43-bin
   head — `judauxplay:HP` outperforms `judplay:HP`, guess **[-1.5, -0.3]** vs
   `lens:ev`. This consumer is an imitation diagnostic (it distills
   `argmax E[Q]`), so per [[partnership-research-gates]] its agreement with
   `lens:ev` is not passage; its value is separating ranking-transfer from
   trunk-regularization. Falsifier: `judauxplay:HP ≈ judplay:H` — the aux head
   ranks well in-distribution (R1) but the ranking does not survive live play
   states.

## Results

*(pending)*

## Links

[[research-lane-selection]] [[jud]] [[w42-jud-v1]] [[lamir1-ceiling]]
[[dense-q-supervision]] [[partnership-research-gates]] [[stage-0-closure]]
[[belief-weighted-jud-mcts]]
