---
title: Consumption Ledger
kind: topic
first_seen: afd4802
last_updated: b89ff635
status: active
---

The record of every mechanism tried against [[the-wall]] — ruled in, ruled
out, or partial, each verdict with its evidence — to be read before proposing
anything new. The prior-sweep rule: grep this ledger by mechanism before
registering any new lever — adjacency reading misses cross-family seams (the
Kubíček & Lisý warning on the [[gus]]/[[champion]] seam pre-explained
[[w42-jud-v1]]'s JS3 failure).

Distilled from [[the-wall-biography]] §2; organized by mechanism, not
chronology.

## Lossy abstraction over the game's structure

- **RULED OUT: count-centric abstraction (MCCFR).** Built for real (250k
  iterations, 172MB strategy, 135× CFD2 compression), killed because it
  *"couldn't learn suit-specific play"*; Jason: *"count centric was just not a
  good abstraction. dead end."* (`t42-tgr`, Dec 2025; [[pre-ml-ai-attempts]].)
- **RULED OUT: feature decompositions as a basis for play.** Three-axis
  decomposition failed at R²=23%; six folk-heuristics tested against the
  oracle, all six refuted. ([[the-analysis-epic]], Jan 2026.)
- **RULED OUT (by argument): vision-model-style clusterable latents.** *"the
  5-4 is not a 5, necessarily"* — 42 has role, not appearance. (2026-03-07.)
- **RULED IN: seed-invariant encodings (τ / trump_rank).** Raw-ID MLPs
  memorize seeds (test 0.040 vs 0.020 target); rank-relative-to-trump tokens
  generalize; survives in the production tokenizer. (`t42-wzsq`;
  `forge/eq/game_tensor.py`.)
- **RULED IN: deleting *irrelevant* state.** Score-removal cut the solver
  79M→2M states (~3.5GB→88MB VRAM) — the highest-leverage optimization found,
  and the opposite of lossy abstraction. (`t42-ze7i`.)

## Exact solving and live search over full state

- **RULED IN: offline backward induction as ground truth.** solver2: 10.3M
  states in 22s per seed on a 4GB GPU; still the Stage-1 generator.
  (`6f82f9f`, `ba07bc6`; [[forge]].)
- **RULED IN: PIMC-over-full-state as the substrate.** Won the Sept 2025
  elimination by refusing to abstract; alpha-beta to terminal cured the
  depressed android; [[expected-q-value|E[Q]]] is posterior-weighting on top
  of exactly this. (`3e063ff`, `t42-9ed`; [[pimc]].)
- **RULED OUT: live PIMC over full bidding.** 21-choose-7; *"intractable
  actually."* (085ffa71.)

## Aggregating per-world values (the E[Q] mechanism itself)

- **RULED OUT: naive averaging of perfect-info rollouts as a hand evaluator.**
  [[strategy-fusion|Strategy Fusion]]: E[max(score)] ≥ max(E[score]); *"will
  systematically advise you to bid too high."* Fix: E[Q] per action, then
  argmax. (adb6de51; `wiki/sources/pi-oracle-bidding-answer.md`; the ~0.1 pts/game magnitude is
  Skat/Bridge literature, never measured in 42.)
- **RULED IN: E[Q] as a computable, trustworthy, cheap signal.** `remaining +
  played_by` marginalization defines it; 12,325× vectorization, bit-for-bit
  verified, 6,493 games/sec. (`341dd53`, `0db7750`, `9098e5d`;
  [[expected-q-value]], [[eq-genesis]].)
- **RULED OUT: naive uniform 1k-world sampling as adequate.** Rigorous
  posterior+enumeration labels agreed only ~65% — marginalization changes a
  third of the labels. (2026-01-27, Jason's measurement.)
- **HARD FACT: the argmax-Q ceiling is ~74%.** Only 55.7% of states have a
  unique best action; coin-flip tie-breaking caps at 73.96% agreement — a game
  fact, not a training deficiency; breaking it requires a reason. (`d9402cf`,
  `docs/random-tiebreaker-ceiling.md @ 233b7dc5`; [[argmax-q-ceiling]].)
- **RULED IN: multi-sample voting.** Averaging noisy distilled evaluations
  improves decisions (regret 0.58→0.19, blunders 4.5%→0.52%); the champion
  E[Q] n=10 *is* this mechanism. (`t42-k54h`.)

## Distilling the oracle into a student

- **RULED IN: distillation itself works.** 97.8% policy accuracy / 0.072
  q-gap (`fc3acc7`); [[gus]] v3-10k at 0.551 regret / 76.07% bot-match is
  still the best fast student as of 2026-07-06. ([[student-distillation]].)
- **RULED OUT: value-head regression for bidding.** Plateaued at 7.4 points
  MAE; *"bidding needs simulation, not regression."* (`fc3acc7`.)
- **RULED IN: simulate-and-count for bidding.** `forge/bidding/`, vectorized
  135× in one night (0.33 → 44 hands/min).
- **RULED OUT: E[Q] as a direct policy teacher.** Policy-signal mix 25%→95%
  moved nothing past ~74% vs random at 3.3M params; "increase the oracle dose"
  is dead. Caveat: graded on the suspect vs-random metric
  ([[vs-random-eval-is-suspect]]). ([[full-teacher-eq-experiment]];
  `6081420`, `t42-4xvg`.)
- **RULED OUT: decision-time look-ahead over a distilled value function.**
  [[lamir1|LAMIR-1]]: every depth-1 variant lost to direct π_me; Kubíček &
  Lisý diagnosis — a distilled V cannot support look-ahead without CFR+
  ([[lamir1-ceiling]]). Corrected numbers: v-bootstrap regret 1.645, not the
  pre-fix 2.777 ([[gus-lamir1-mode-comparison]]). The K&L warning
  pre-explained jud's JS3 failure — the prior-sweep lesson.
- **RULED OUT (structurally): "distill X and see."** The [[candlewax]] rule:
  every distillation proposal must name its consumer and its licensed collapse
  first. The one that worked ([[w42-jud-v0]]'s bidder) named both — auction
  root, realized-outcome pricing — before training.

## Self-play reinforcement learning

- **RULED IN (qualified): AlphaZero-shaped self-play gets off the ground in
  42.** 50%→70.9% vs random in five days at 557K params; 76.3% peak at 3.3M.
  Real, checkpoint-backed. (`a7d6b5b`; `large-belief-recap.md`; [[zeb]].)
- **RULED OUT: that it reaches the champion.** No Zeb-descended policy ever
  beat E[Q] n=10 at pure play; Elo 1579 vs 1600 is neck-and-neck, not a win.
  (`afd4802`.)
- **RE-DIAGNOSED: the "capacity ceiling" wasn't capacity.** 6× params bought
  ~5 points — an information/architecture wall's shape; *"we were lying to the
  model."* (2026-03-11; [[alphazero-under-imperfect-information]].)
- **RULED OUT (by argument): continuing AlphaZero-Zeb.** *"distilling that run
  would just be distilling the heuristic I used to select >=18."*
  (2026-03-15; [[the-gestation]].)
- **HARDWARE FACT: MCTS is latency-bound, not compute-bound.** A rented B200
  gave ~2.2× over a 3050 Ti. (Feb 3–4.)

## LLM as the reasoner

- **RULED OUT (Sept 2025, Opus-era): LLM as raw play engine.** *"opus has
  repeatedly struggled to reason about the game."* First-hand, dated.
- **RULED OUT (at ~2B scale): weight-drilled curriculum play ([[lem|LEM]]).**
  Gemma→Qwen pivot, [[star|STaR]] to v10-maskfix: 86% comprehension, 55/100
  bot-match; ceiling diagnosed as STaR/capacity, not gradients. Closed clean.
  (be7efc4.)
- **RULED OUT (at this scale): tool-using LLM as a policy that beats the
  lookup.** [[burl]] validated the tool surface ([[iter3-rules-adapter]], 90%
  bot-match) and graded null on policy against E[Q]. Zero `burl/` commits
  since 2026-05-07.
- **RULED IN: rules-as-tools over rules-in-weights.** The one validated Burl
  win. (dbadb5f; [[rules-as-tools]].)
- **FLAGGED, PRE-REGISTERED, THEN WALKED INTO: self-fed traces plateau.**
  *"LLMs actually get dumber when fed their own content back"* (2026-04-04,
  left open) — then STaR went 30% → 42% → plateau 38–39% over 15 iterations.
  ([[star-10-iterations]].)

## The human canon as plan source

- **RULED IN: the book's content is trustworthy.** [[w42]] graded Roberson
  claim by claim — 64 ledger rows, phases 1–4, five waves, zero fabrication
  across ~120 pages. ([[w42-claim-ledger]], [[w42-phase4-final-claim-audit]].)
- **RULED OUT (as stated): the book's advice as a utility guide.** Wave 3.0's
  void-creation "ADOPT" was inverted by Waves 4.0/4.1 (the advice aligned with
  the *worst*-scoring utility); downgraded to ADOPT-DEFERRED. The reversal was
  never back-linked; it's real. ([[w42-bookval-v2-utility-lens-synthesis]].)
- **RULED IN: Lens v1 — EV beats p_make as the utility to optimize.** The
  family's most durable living output; the recommended one-line
  `select_actions` switch to ev-argmax was **never applied** — the code still
  computes Lens(p_make) as of `afd4802`.
  ([[w42-lens-v1-utility-head-to-head]].)
- **NEVER TESTED: multi-step book plans.**
  [[book-strategy-player|BookStrategyPlayer]] designed 2026-05-04, bead
  frozen, never built; chassis redirected to the auction (2026-06-09 design
  review, [[champion-design-review]]).

## Belief modeling

- **HARD NEGATIVE: Zeb's belief headline was hollow.** 72% top-1 overall was
  **39% on hidden state** — the number that mattered.
  ([[zeb-calibration-eval]]; reconfirmed `afd4802`.)
- **HARD NEGATIVE (Feb): belief decoupled from policy does nothing.** The
  belief head predicted ownership and never fed a move — *"2 things side by
  side only connected via loss."* (02-18T03:45; [[belief-feeding-policy]].)
- **RULED IN (June): belief feeding decisions, done right.** Champion #24 —
  [[gus]] belief conditioned on the completed auction — +2.59pp held-out, the
  first heavy-training measured win of the champion ladder; Gus survives as
  the champion's posterior engine. ([[w42-champion-auction-belief]].)
- **REFRAMED: belief's value is legibility, not marks.** The [[arena]]
  belief→marks nulls were measured on a PIMC-vs-PIMC harness structurally
  blind to concealment and signaling (Fable's insight) — the objective was
  mis-specified, not the belief. Legibility itself is an instrument (a
  verification surface), not a goal, per the author correction on
  [[the-wall-biography]].

## Realized-outcome value learning (jud)

- **RULED OUT: the parity plateau as a structural limit.** [[w42-jud-v0|jud
  v0]]'s registered prior ("parity with `net:wp` is the PIMC price of hidden
  information") was falsified by its own probe: **data starvation** in a tiny
  head; 3× on-policy data carried the learned bidder past the hand-tuned one.
  ([[w42-plateau-probe]].)
- **RULED IN: value-native reasoning at the auction.** `margin:wp`(head_8):
  +0.38 [+0.09,+0.67] and +0.42 [+0.12,+0.72] marks/game over the hand-tuned
  champion — the first learned component ever to beat it — saturating at
  ≈+0.3–0.4. ([[w42-plateau-probe]], [[champion]].)
- **RULED IN: the one-organ unification holds at the auction.** [[w42-jud-v1|jud
  v1]]'s info-state → 43-bin realized-points categorical prices contracts; the
  bidder survives the fold.
- **RULED OUT: greedy 1-ply value play as a move-ranker.** JP3 falsified; the
  self-play loop moves it zero. ([[w42-jud-v1]].)
- **PARTIAL: oracle-free search at the leaf.** `judsearch` recovers two-thirds
  of the play gap (−3.44 → −1.16, JS1 PASS +2.28) but not parity; neither more
  worlds (JS2, below band) nor a better-calibrated head (JS3, FAILED) closes
  the rest. [[the-wall|The wall]]'s current coordinates: a 470k MLP on
  hand-level Monte-Carlo labels cannot out-rank E[Q] n=10's per-move oracle.
  ([[w42-jud-v1]], [[jud]].)

## Evaluation itself

- **RULED IN: Bradley-Terry Elo over a pairwise matrix.** Anchored on E[Q],
  fanned to 28 T4s; first time everything sat on one scale. (`3a77bb6`;
  [[eval-matrix-bradley-terry]].)
- **SUSPECT AND KNOWN SUSPECT: vs-random with winner=more-points.** *"more
  points does not necessarily win the game!"* (02-09), plus 47.8/52.2 seating
  asymmetry in random-vs-random; every era-4 headline was graded against a
  non-win-condition objective, named in-window and never fixed in-window.
  Later eras moved to marks and full games to 7.
  ([[vs-random-eval-is-suspect]].)
- **RULED IN: pre-registration.** Every [[jud]] rung registered on GitHub
  before measurement; the plateau probe run to its falsifier. Should be the
  default loop.
- **UNTRACKED RISK, STILL OPEN: WorldSamplerMRV sampler bias.** Flagged as
  affecting "all historical Burl eval numbers and forge/eq training data,"
  parked as "worth a bead," never filed. Still dangling as of 2026-07-06.

## Process and tooling

- **RULED IN: big-bang no-legacy rewrites.** Three "3–4 week" epics closed in
  hours-to-days.
- **RULED IN: grok-don't-converge.** *"you all led me catastrophically
  astray."* ([[grok-not-converge]].)
- **RULED IN: the gofish ratchet testbed.** 5700 games/sec; proved the
  auto-research loop cheaply. ([[the-gestation]].)
- **RULED IN: the fleet pattern.** HF-as-message-bus, reputation scoring, CQRS
  monitor. ([[zeb-fleet-ops]].)
- **RULED OUT: build-automation swarms.** Claude Flow, *"broken beyond
  redemption"*, one attempt.
- **RULED OUT: vibe coding.** *"I read and understand all of the code and
  write most of it."*
- **RULED OUT: external RL frameworks.** Sample Factory, rejected in-session
  to protect the actual hypothesis.

## Links

[[the-wall]] · [[the-wall-biography]] · [[candlewax]] · [[ideated-not-built]]
· [[expected-q-value]] · [[jud]] · [[champion]] · [[gus]] · [[burl]] ·
[[lem]] · [[zeb]] · [[w42]]
