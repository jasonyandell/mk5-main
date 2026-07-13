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
- **Hard gate before HP trains:** teacher-forced label join coverage = 100%
  of decision rows (note `aux_coverage()` is over all rows and caps near 0.5
  by design — the post-move evaluation half can never carry labels). The
  original bridge replays the oracle's own greedy line, not the recorded
  line — measured misalignment: only ~52% of decision rows joined. The gate
  is closed by `--teacher-forced` in `forge.cli.generate_eq_from_snapshots`
  (now a production capability: the pipeline advances the recorded line,
  E[Q] per state unchanged, fail-fast on any desync); proven 84/84 decision
  coordinates on a real corpus slice.
- **Pre-experiment ranking baseline** (context for R1): jud v1's `r4`
  main-head ranking against teacher-forced labels measures Spearman `0.128`,
  pairwise `0.568`, top-1 `0.473` (55 decisions, smoke) — the per-move
  discrimination wall, quantified at ranking level.

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

Training: both arms early-stopped at epoch 7 with byte-similar trajectories
(seed discipline held). Held-out ranking on 5,192 supervised decisions (test
split, teacher-forced labels):

| ranker | Spearman | pairwise | top-1 |
|---|---|---|---|
| arm H, main head (JudPlay-style) | 0.046 | 0.508 | 0.345 |
| arm HP, main head | 0.042 | 0.506 | 0.345 |
| arm HP, aux head (JudAuxPlay-style) | **0.236** | **0.603** | **0.517** |

- **R1 — PARTIAL.** The mechanism separates exactly as hypothesized: the aux
  head ranks far better than the main head, whose pairwise ordering is
  literally chance (0.508) — the hand-level wall in its starkest measured
  form. But the registered threshold missed: top-1 0.517 < 0.60. At this
  capacity (one 512→7 linear) and one 512-game corpus, dense per-move
  supervision buys a weak ranking, not an oracle imitation.
- **R2 — PASS.** Main-head val CE 2.5247 (HP) vs 2.5252 (H): the aux loss did
  not perturb the primary head — and notably also did not *help* it: the
  trunk-shaping ("dense supervision regularizes the encoder",
  [[dense-q-supervision]]) transferred nothing to main-head ranking
  (0.042 vs 0.046). At this scale the aux gain lives in the aux head alone.
- **Context observation (unregistered):** jud v1's `r4` head — trained on
  five cumulative self-play rounds — out-ranks tonight's single-corpus arm H
  (0.128/0.568/0.473 vs 0.046/0.508/0.345). Corpus volume moves the leaf's
  ranking; the H-vs-HP contrast is controlled for it, but absolute
  marks-vs-`lens:ev` numbers below are not comparable to the r4 protocol
  numbers.

Marks grading (R3–R6): *(battery running; block-1 arms and block-2 greedy/aux
arms are in; the two block-2 judsearch arms remain)*

Interim guess, registered mid-battery before the block-2 judsearch arms or
any paired analysis ran: block 1 showed `judsearch:HP` −1.64 vs
`judsearch:H` −1.98 while greedy showed nothing — my guess is the
search-consumer delta is **real but small**: combined game-paired
`Δ(HP−H) ≈ +0.25`, block 2 alone directionally positive with CI including
zero. If the paired CI includes zero combined, the whole experiment is a
clean negative at this capacity/corpus and the residual moves to the
capacity×target interaction ([[partnership-research-gates]] row 2).

## Links

[[research-lane-selection]] [[jud]] [[w42-jud-v1]] [[lamir1-ceiling]]
[[dense-q-supervision]] [[partnership-research-gates]] [[stage-0-closure]]
[[belief-weighted-jud-mcts]]
