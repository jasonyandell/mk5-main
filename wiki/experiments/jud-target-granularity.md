---
title: Jud Target Granularity — hand-level vs per-move at fixed capacity
kind: experiment
first_seen: 4cf66448
last_updated: 0dc9a51b
status: complete
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

Marks grading (all vs `lens:ev`, bid30 both sides, 256 games/block, marks_b
per game; paired Δ is game-paired HP−H with 10k-bootstrap 95% CI):

| consumer | block 7000000 | block 9000000 | paired Δ(HP−H) combined |
|---|---|---|---|
| `judplay:H` | `-3.270 [-3.555,-2.949]` | `-3.277 [-3.582,-2.945]` | — |
| `judplay:HP` | `-3.305 [-3.586,-2.996]` | `-3.414 [-3.699,-3.121]` | `-0.086 [-0.254,+0.092]` |
| `judauxplay:HP` | `-4.102 [-4.332,-3.867]` | `-4.180 [-4.414,-3.938]` | — |
| `judsearch:H` | `-1.980 [-2.297,-1.648]` | `-1.785 [-2.152,-1.410]` | — |
| `judsearch:HP` | `-1.641 [-1.980,-1.285]` | `-1.770 [-2.125,-1.402]` | `+0.178 [-0.041,+0.385]` |

The mid-battery guess (search delta "real but small", combined ≈ `+0.25`)
was **wrong per its own falsifier**: block 1's `+0.340 [+0.039,+0.637]` did
not reproduce (block 2 `+0.016 [-0.285,+0.320]`); the combined paired CI
includes zero.

## Verdicts

| prediction | verdict |
|---|---|
| R1 ranking | **PARTIAL** — aux head separates hard from the chance-level main head (0.603 vs 0.508 pairwise) but misses the ≥0.60 top-1 threshold (0.517) |
| R2 guard | **PASS** — main heads statistically identical (val CE 2.5247 vs 2.5252) |
| R3 greedy marks | **FALSIFIED** — paired Δ `-0.086 [-0.254,+0.092]` |
| R4 search marks | **MISS** — `judsearch:HP` `-1.64/-1.77`, outside `[-1.2,-0.2]`; paired Δ CI includes zero |
| R5 wall stands | **PASS** — every arm loses to `lens:ev` |
| R6 aux-direct | **FALSIFIED, worse than its own falsifier** — `judauxplay:HP` (`-4.1/-4.2`) plays *worse* than `judplay:H` (`-3.3`), not merely equal |

## Reading

**The dense per-move E[Q] auxiliary, at jud-v1 capacity on one 512-game
corpus, is a marks null** — [[partnership-research-gates]] row 1 does not
pass: the in-distribution ranking gain exists but carries to no consumer's
marks. Three mechanism-level facts survive the null and sharpen the wall:

1. **Ranking-label agreement does not order play strength.** The aux head
   agrees with `argmax E[Q]` far more than the main head does, and its direct
   consumer plays 0.9 marks *worse*. A weak ranking argmaxed directly is
   worse than a calibrated value priced over actual child states — the
   [[lamir1-ceiling]] noise-at-the-boundary mechanism, measured in a new
   form.
2. **The trunk-shaping hypothesis fails at this scale.** The dense signal did
   not move the main head at all (ranking 0.042 vs 0.046; greedy marks null)
   — the [[dense-q-supervision]] regularization story from Gus does not
   transfer to this architecture/corpus as-is.
3. **A representational asymmetry was left untested.** The aux head must
   predict all seven action consequences from the *parent* featurization;
   `JudPlay` evaluates each actual *child* state. The unrun arm that removes
   the asymmetry: supervise the **main head on child states** with per-move
   E[Q]-derived continuation values (the signal-(b)-shaped form of "per-move
   targets") instead of a separate parent-side ranking head. That, plus the
   corpus-volume confounder (r4's five-round corpus out-ranks tonight's
   single corpus before any aux signal), bounds this negative: **this
   experiment rules out the parent-side dense auxiliary at fixed capacity and
   fixed small corpus; it does not rule out per-move targets.**

Residual per [[research-lane-selection]]: the capacity×target interaction
([[partnership-research-gates]] row 2) and the child-state per-move form
above are the named follow-ups; the suggestive one-block search delta says a
search consumer remains the more promising reader of any future per-move
leaf.

Artifacts: heads at `scratch/lane-b/heads/`, labeled corpus at
`scratch/lane-b/eq/`, run summaries under `arena/results/lb_*`, ranking eval
+ paired analysis scripts in `scratch/lane-b/`.

## Round 2 — the two named residuals (registered before any round-2 training)

Same night, same protocol. Two more 512-game corpora (seeds 9,110,000 and
9,120,000; teacher-forced labels) give a 3× corpus. Arms:

- **H3** — hand-level only, 3× corpus (the volume lever alone).
- **HP3** — parent-side aux, 3× corpus (does volume rescue the aux?).
- **HC3** — **child-state per-move values**, 3× corpus: no extra head; the
  main head's `mean_points` on each recorded child row is regressed toward
  the parent decision's per-move oracle value (converted to declaring-team
  points), exactly the quantity `JudPlay` argmaxes. Consumer-aligned by
  construction.
- **HC1** — child-state values on the original 1× corpus (volume-controlled
  HC comparison).

Registered predictions (guesses on record):

- **V1 (volume):** H3 out-ranks arm H (Spearman ≥ 0.10 vs 0.046) and
  `judsearch:H3` improves on H's `-1.98/-1.79` (guess ≈ `-1.6`). Corpus
  volume is a real leaf lever, as the r4 comparison suggested.
- **C1 (child-state, the arm I believe in):** HC3's main head out-ranks H3's
  (pairwise ≥ 0.55) **and** `judplay:HC3` beats `judplay:H3` by ≥ +0.4
  game-paired; `judsearch:HC3` guess `[-1.5, -0.9]`. Falsifier: paired CIs
  include zero — then consumer-aligned per-move supervision at this capacity
  fails too, and the per-move-target family at v1 capacity is dead on both
  its parent-side and child-side forms (a strong, clean negative that moves
  the residual entirely to capacity interaction and opponents-in-rollout).
- **A2 (aux at volume):** HP3−H3 paired deltas remain null — the round-1
  null is mechanism, not data starvation.
- **W2:** the wall stands; no arm beats `lens:ev`.

### Round 2 training + ranking (final; marks battery running)

Training (4 arms parallel, ~3 min): volume is a real **calibration** lever —
val CE 2.525 (1×) → 2.139/2.135/2.149 (3× H3/HP3/HC3); HC3 trades a little
CE for MAE/ECE mix as the child-loss caveat predicted (pmf pulled toward its
mean). Held-out ranking on the same 5,192 round-1 test decisions:

| arm, ranker | Spearman | pairwise | top-1 |
|---|---|---|---|
| H3 main | 0.062 | 0.486 | 0.366 |
| HP3 main | 0.074 | 0.488 | 0.374 |
| HP3 aux | −0.045 | 0.523 | 0.310 |
| HC3 main | 0.059 | 0.485 | 0.365 |
| HC1 main | 0.055 | 0.515 | 0.351 |

- **V1 ranking clause — MISS.** 3× volume improved calibration and moved
  ranking nothing (0.486 pairwise ≈ chance). The r4-out-ranks-H observation
  was evidently not about raw volume — r4's five *on-policy self-play rounds*
  differ from 3× fresh champion self-play in distribution, not just size.
- **Aux collapse at 3× (unpredicted):** HP3's aux head ranks *worse than
  chance-adjacent* (Spearman −0.045 vs 0.236 at 1×). Best guess, recorded as
  a guess: early stopping reads the main-head CE, which converges faster at
  3×, so the aux head is stopped under-trained; an aux-aware stopping
  criterion or λ sweep would test this. Either way the parent-side aux is
  fragile, strengthening round 1's null.
- **HC3 shows no in-distribution ranking gain** even in exactly the space it
  supervises — child-value regression at λ=1.0 moved child-price ordering
  approximately nothing. The marks battery decides whether anything
  play-relevant changed anyway.

### Round 2 marks (final; game-paired deltas, 512 paired games each)

Absolute vs `lens:ev` (blocks 7M/9M): `judplay` — H3 `-3.305/-3.297`,
HP3 `-3.242/-3.305`, HC3 `-3.594/-3.246`, HC1 `-3.359/-3.055`;
`judsearch` — H3 `-1.762/-1.645`, HP3 `-1.695/-1.574`, HC3 `-1.848/-1.703`,
HC1 `-1.848/-1.766`.

| paired contrast | judplay | judsearch |
|---|---|---|
| H3 − H (volume) | `-0.027 [-0.238,+0.182]` | `+0.180 [-0.047,+0.402]` |
| HP3 − H3 (aux at volume) | `+0.027 [-0.152,+0.203]` | `+0.068 [-0.154,+0.295]` |
| HC3 − H3 (child-state) | `-0.119 [-0.307,+0.062]` | `-0.072 [-0.297,+0.154]` |
| HC1 − H (child-state, 1×) | `+0.066 [-0.104,+0.238]` | `+0.076 [-0.152,+0.305]` |

Round-2 verdicts: **V1 MISS** (both clauses — volume moved calibration only;
the search delta `+0.180` is directional with CI including zero).
**C1 FALSIFIED** decisively (the arm I believed in: greedy point estimate
*negative*). **A2 CONFIRMED** — the only guess that hit was the one
predicting a null: the parent-side aux stays null at 3× data.
**W2 PASS** — the wall stands; the night's best play stack is
`judsearch:HP3` at `-1.57/-1.70`, still behind `lens:ev` and behind r4's
five-round `-1.42`.

## Final reading

**The per-move-target family at jud-v1 capacity is dead in both its forms.**
Hand-level, parent-side dense auxiliary, child-state value regression, and 3×
corpus volume all price play within noise of each other under both consumers;
the registered predictions that asserted gains (R3, R4, R6, V1, C1) all
missed, and the one asserting a null (A2) hit. Combined with round 1's
mechanism facts (ranking-label agreement does not order play strength; the
dense signal neither shapes the trunk nor survives argmax), the residual for
[[jud]] v2 narrows to three levers this experiment could not touch at fixed
capacity: the **capacity×target interaction**
([[partnership-research-gates]] row 2), **on-policy loop data** (r4's
five-round cumulative corpus remains the best-ranking leaf of its size —
distributional, not volumetric), and **opponents-in-rollout** (v2's other
named cue, untested tonight). One soft trace survives for the search
direction: CE-lowering arms show a recurring ~`+0.18` judsearch-side paired
delta (HP−H round 1, H3−H round 2), never individually significant — if a
future leaf wants to claim search value from calibration, it needs ~4× the
games to resolve that size, and [[belief-weighted-jud-mcts]] J1/J2 remain the
registered consumers for such a claim.

## Links

[[research-lane-selection]] [[jud]] [[w42-jud-v1]] [[lamir1-ceiling]]
[[dense-q-supervision]] [[partnership-research-gates]] [[stage-0-closure]]
[[belief-weighted-jud-mcts]]
