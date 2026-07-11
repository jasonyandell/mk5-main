---
title: "The Wall: A Biography"
kind: trail
first_seen: afd4802
last_updated: pending-this-ingest
status: active
---

Capstone of the seven-month archaeology, 2025-07-19 → 2026-07-06. Written over
the era memos (1–6) and the five era-6 reconciliation packets; every claim
traces to a dossier that was itself re-derived against the repo. This is the
wiki's long-form read on [[the-wall]] — the narrative companion to
[[consumption-ledger]] (the terse ruled-in/ruled-out record). Read it with
coffee; it was written the same way.

## 1. The one story

Seven months, six eras, five project families, one [[the-wall|wall]] — and
underneath all of it, one question that never changed its subject, only its
precision. The subject was set in 1980, on a grandmother's knee, and stated
plainly on Christmas Eve 2025: *"we are gonna get us an AI that's decent. been
dreaming about this since I was 12 literally. making a 42 ai. it ain't easy!"*
Everything else is the question sharpening.

**Era 1 (Jul–Dec 2025) asked: can the thing be built at all?** The answer was
the substrate — a pure-functional, event-sourced [[engine]] rebuilt from a
60k-line vibe-coded ancestor, rewritten with joy on a cadence Jason named
himself: *"it's like my hobby. like those mechanic guys with the project cars
in the garage... I'm on like mark 8 now."* And: *"I'm building like a crystal
palace in the sky over here because that's what's fun! so any little blemish
just won't do and I chase it down."* Two AI excursions ran here, with opposite
fates ([[pre-ml-ai-attempts]]). MCCFR was built for real and explicitly
killed — and the kill note is the era's most important sentence, because it
pre-registers the wall's objective function six weeks before the wall exists:
*"'Boring and competent' isn't worth the squeeze when we could get that with
fixed MCTS, and neural nets offer more upside for fun play."* [[pimc|PIMC]]
survived precisely because it refused to abstract. The era also diagnosed and
fixed, as a P1 bug, an AI that *"makes defeatist plays when losing"* — the
"depressed android." Hold that phrase. It comes back wearing a voice
modulator. And one more early datapoint, from 2025-09-25, dated and
first-hand: *"i really don't think an llm would do well... opus has repeatedly
struggled to reason about the game."*

**Era 2 (Dec 24 – Jan 8) asked: can the truth be computed?** It opened with
defeat — *"pimc is checking everything and takes minutes for one game... I am
pretty sure bidding is intractable actually. 21 choose 7. yikes"* — and in the
same breath, the reorganizing hunch: *"I wonder if play itself can be
bootstrapped from pimc starting at late game and bootstrapping earlier?"* The
[[suit-algebra]] fixed the representation (*"I want to actually represent
these actual rules actually correctly in code"*), the GPU solver ([[forge]])
made ground truth cheap (*"42 is like a funnel and chess is like a fractal. we
can solve it. we have solved it"*), and distillation produced its first honest
results ([[student-distillation]]): MLPs memorize, transformers generalize,
and *"bidding needs simulation, not regression."* Then
[[strategy-fusion|Strategy Fusion]]: the formal proof that averaging
perfect-information rollouts overstates achievable value — E[max] ≥ max[E] —
endorsed outright (*"yes to all"*). And on January 3rd, in one sentence, the
earliest in-record sighting of the wall, described but not recognized: *"it
thinks 2-2 is 'just as safe' to lead as your high trump even when someone else
might have a trump."* That's not an aggregation bug. That's a player with no
read and no plan. It got filed as an aggregation bug.

**Era 3 (Jan 9–31) asked: can the truth's one known lie be fixed?** The
[[the-oracle|oracle]] *"knows the right answer because the right answer was
answered before it was even asked via reverse induction"* — it sees four hands
and teaches like a liar. The fix was [[expected-q-value|E[Q]]], founded in a
single sentence Jason asked to be checked against a *"karpathy himself"*
standard: *"We train on games where every move is chosen by averaging oracle
values over all hidden worlds consistent with what's been publicly played so
far."* It shipped in days ([[eq-genesis]]), got 12,325× faster, and the
euphoria was earned: *"we did it buddy. we fixed strategy fusion. we have
unlimited training (!!!!!)."* But the era also spoke the wall's construction
permit, in one throwaway word, on January 10th: **"I also don't care about
signaling. yet. I want a damn good solid base player."** That scope cut chose
argmax over E[Q] as the consumption rule, and it was never revisited. The
[[argmax-q-ceiling|74% argmax-tie ceiling]] was measured the same week —
*"anything OTHER than 74 is wrong"* — celebrated as validation when it was
also the wall's first precise measurement: how far a coin-flip tiebreaker can
go before it needs a reason.

**Late January is the founding condition.** The doubt arrives only after the
triumph: *"maybe the most amazing thing I've ever built and I dunno what to do
with it, so let's friggin try stuff."* The January register was pure "we did
it buddy"; the "dunno what to do with it" is the sound of the deferrals coming
due.

**Era 4 (Feb 1–16) asked: can something learn to beat it?** — *"ok left
turn... why is alphago zero style approach not appropriate for 42 again?"*
[[zeb|Zeb]]: one million self-play games in five days, a self-healing Vast.ai
fleet run by one hobbyist (*"I'd never do this for work. ever. but for the
hobby project? let's gooo"*), Elo 1579 against the champion's 1600. And at 4am
on February 6th, the wall finally gets its true name ([[candlewax]]), mid-era,
before the confirming experiment even ran:

> *"there's no such thing as an e[q] optimal policy unfortunately. it's a
> distribution, an often lumpy, often smooth histogram of discrete scores, not
> something you can just pick 'best' for that application."*

> *"ultimately I have to pick a next move from that melted candle wax... and I
> have NO confidence that it selects moves well, with good judgement, only that
> it is extremely well informed about the statistical landscape."*

The [[full-teacher-eq-experiment|full-teacher experiment]] then confirmed the
prediction: cranking E[Q]'s policy signal from 25% to 95% moved nothing past
~74% vs random. The era closed with dignity (*"what if this is vaguely the
limit of alphago in 42?... we've learned a TON"*) and its real finding
delivered as a 3:45am shrug on the way out the door
([[belief-feeding-policy]]): *"so how do we get beliefs feeding policy? right
now we kind of have 2 things side by side only connected via loss
essentially."*

**Era 5 (Feb 17 – Apr 8) asked: what is the question, actually?** Fifty-two
days, zero commits, the loudest thinking in the project's life
([[the-gestation]]). The Zeb autopsy: *"n=10 e[q] beats it... we were lying to
the model and it got confused."* The sharpest statement of the wall anywhere
in the corpus, 2026-03-15: *"the wall was that these images, they're all
gorgeous rendering of suboptimal paths. distilling that run would just be
distilling the heuristic I used to select >=18 because that's required to win
42."* Not compute. Not data. **Consumption.** The cast was born here —
[[lem|Lem]] (*"Lem navigates it"*), [[burl|Burl]] (*"Burl sees the world"*),
Harl, LLem, the walker ([[ideated-not-built]]) — and so were the era's two
deepest asides: *"I honestly don't think it's so much complex as it is
subtle,"* and *"at the end of the game, I have changed. the model has not."*
And the reframe that everything after inherits: *"it's not a better player,
it's a better narrator... when my family talks about 42 games online they
mention the personalities more than the play."* That reframe carries the
author's correction, registered 2026-07-06: [[narration]] was never a goal —
it was a means. Jason: "an interpretable player is neat. something that can
talk 42? neat! not a main goal, though... one hope was that I could get a neat
talker, catch it saying actually-correct things, and then bootstrap that. I
later learned that's called STaR. notice how a player that sounds right is
just a means to an end, not an actual goal." Talking is the verification
surface that makes reasoning trainable ([[star]]); the narrator reframe was an
instrumental insight, not a goal-change. The goal never moved.

**Era 6 (Apr 9 – Jul 6) asked the question in its final form: who consumes the
oracle, and what is the plan that lets them beat it?** Five consumer
hypotheses, run in sequence, each honestly graded: LEM (a phone-class LLM
learns and narrates the game — closed clean at 86% comprehension, 55/100
bot-match, a STaR/capacity ceiling), Burl (LLM-as-tool-user — one real win at
90% bot-match via [[rules-as-tools]], null on policy against E[Q]),
[[gus|Gus]] (distill the oracle into a fast student — best-known student at
0.551 regret / 76.07% bot-match, every depth-1 look-ahead variant lost to
direct π_me), [[w42|W42]] (grade the family canon, Roberson's *Winning 42*,
claim by claim ([[the-book-enters]]) — trustworthy content, one real
utility-selection result, its promoted architecture never built), and
[[champion]]/[[jud]] (reason with the oracle at the auction). The era's own
summary sentence is Jason's, from the final working-tree commit:

> *"I saw eq, I said sure I could distill it. but for what purpose? no idea
> what to do with distilled melted candlewax."*

And then, in two July nights, the first genuine crack: [[w42-jud-v0|jud v0]]'s
`margin:wp` bidder — trained on realized self-play outcomes, its structural
prior registered and then falsified by its own plateau probe
([[w42-plateau-probe]]) — became **the first learned component ever to beat
the hand-tuned champion on marks** (+0.38 [+0.09,+0.67], +0.42 [+0.12,+0.72]).
[[w42-jud-v1|jud v1]] folded bid and play into one organ and the verdict came
back split, graded rung by rung against pre-registered predictions: **the
unification holds at the auction and is mechanism-limited at play.** The wall,
seven months after it was first sighted as a 2-2 lead, is now located to the
millimeter: a 470k MLP on hand-level Monte-Carlo labels cannot out-rank E[Q]
n=10's per-move oracle.

So the question's whole evolution, in Jason's own words: *"making a 42 ai"* →
*"can we solve it"* → *"a damn good solid base player"* (with signaling
deferred by one word, "yet") → *"nobody knows what 'better' really is"* →
*"no idea what to do with distilled melted candlewax"* → and the current form,
precise at last: something that **takes the E[Q] data, reasons with it, does
better, and has a plan that actually succeeds.** Note what got rejected along
the way: "plays like a person" — ill-specified, and Jason has said so. The
Stephen Hawking voice modulator complaint (*"obvious within seconds it isn't
like a person"*) is the symptom; the specification is the sentence above.

## 2. The ledger of ruled-out and ruled-in

*Organized by mechanism, not chronology. This is the section to read before
answering "what next"; its terse decision-support form is
[[consumption-ledger]]. Every entry carries its evidence.*

### Lossy abstraction over the game's structure

- **RULED OUT: count-centric abstraction (MCCFR).** Built for real — 250k
  iterations, 172MB strategy, 135× CFD2 compression — and killed because it
  *"couldn't learn suit-specific play (e.g., 'don't lead 5-0 when treys are
  trump')."* The compression worked; the thing needed at decision time didn't
  survive it. (`t42-tgr`, Dec 2025; [[pre-ml-ai-attempts]].) Jason's later
  verdict: *"count centric was just not a good abstraction. dead end. didn't
  correlate with good play at all."*
- **RULED OUT: feature decompositions as a basis for play.** Three-axis
  decomposition failed at R²=23%; six traditional folk-heuristics tested
  against the oracle and all six refuted. ([[the-analysis-epic]], Jan 2026.)
- **RULED OUT (by argument): vision-model-style clusterable latents.** *"the
  5-4 is not a 5, necessarily... you can't look at the 5-4 and draw a fuzzy,
  good enough line around it, like you can a road."* 42 has role, not
  appearance. (2026-03-07.)
- **RULED IN: seed-invariant encodings (τ / trump_rank).** Raw-ID MLPs
  memorize seeds (test 0.040 vs 0.020 target); rank-relative-to-trump tokens
  generalize. Survives into the production tokenizer. (`t42-wzsq`;
  `forge/eq/game_tensor.py`.)
- **RULED IN: deleting *irrelevant* state.** Score-removal cut the solver
  79M→2M states, ~3.5GB→88MB VRAM — the single highest-leverage optimization
  found, and the opposite of lossy abstraction: it dropped what provably
  doesn't affect the optimal action. (`t42-ze7i`.)

### Exact solving and live search over full state

- **RULED IN: offline backward induction as ground truth.** solver2: 10.3M
  states in 22s per seed, fits on a 4GB consumer GPU. Still the Stage-1
  generator ([[forge]]). *"42 is like a funnel and chess is like a fractal."*
  (`6f82f9f`, `ba07bc6`.)
- **RULED IN: PIMC-over-full-state as the substrate.** Won the Sept 2025
  elimination (over AlphaZero, CFR, ISMCTS) by refusing to abstract;
  alpha-beta to terminal cured the depressed android.
  [[expected-q-value|E[Q]]] is posterior-weighting on top of exactly this.
  (`3e063ff`, `t42-9ed`; [[pimc]].)
- **RULED OUT: live PIMC over full bidding.** 21-choose-7; *"intractable
  actually."* (085ffa71.)

### Aggregating per-world values (the E[Q] mechanism itself)

- **RULED OUT: naive averaging of perfect-info rollouts as a hand evaluator.**
  [[strategy-fusion|Strategy Fusion]]: E[max(score)] ≥ max(E[score]); *"will
  systematically advise you to bid too high."* Fix: compute Max(Average) —
  E[Q] per action, then argmax. (adb6de51; `wiki/sources/pi-oracle-bidding-answer.md`. Note:
  the ~0.1 pts/game magnitude is Skat/Bridge literature, never measured
  in 42.)
- **RULED IN: E[Q] as a computable, trustworthy, cheap signal.** `remaining +
  played_by` marginalization defines it (`341dd53`); 12,325× vectorization,
  bit-for-bit verified, 6,493 games/sec. Value computation never bottlenecks
  again. (`0db7750`, `9098e5d`; [[expected-q-value]].)
- **RULED OUT: naive uniform 1k-world sampling as adequate.** Rigorous
  posterior+enumeration labels agreed with it only ~65% — the marginalization
  mechanism changes a third of the labels. (2026-01-27, asserted from Jason's
  measurement.)
- **HARD FACT: the argmax-Q ceiling is ~74%.** Only 55.7% of states have a
  unique best action; an argmax player breaking ties by coin flip caps at
  73.96% agreement. Not a training deficiency — a fact about the game's tie
  structure. Breaking it requires a non-coin tiebreaker, i.e., a reason.
  (`d9402cf`, `docs/random-tiebreaker-ceiling.md @ 233b7dc5`; [[argmax-q-ceiling]].)
- **RULED IN: multi-sample voting.** Averaging noisy distilled evaluations
  genuinely improves decisions (regret 0.58→0.19, blunders 4.5%→0.52%,
  `t42-k54h`). It is also the champion: E[Q] n=10 *is* this mechanism.

### Distilling the oracle into a student

- **RULED IN: distillation itself works.** 97.8% policy accuracy / 0.072
  q-gap on H100s (`fc3acc7`); [[gus|Gus]] v3-10k at 0.551 regret / 76.07%
  bot-match is still the best fast student on 2026-07-06 (cited unchanged in
  `champion.md`). ([[student-distillation]].)
- **RULED OUT: value-head regression for bidding.** Plateaued at 7.4 points
  MAE; the commit says it plainly: *"bidding needs simulation, not
  regression."* (`fc3acc7`.)
- **RULED IN: simulate-and-count for bidding.** `forge/bidding/`, vectorized
  135× in one night (0.33 → 44 hands/min).
- **RULED OUT: E[Q] as a direct policy teacher.** The
  [[full-teacher-eq-experiment|full-teacher experiment]]: collapse the
  histogram to a per-action target, feed it to the policy head at 25% and then
  95% mix — play stays at ~74% vs random at 3.3M params. "Just increase the
  oracle dose" is dead. (`6081420`, `t42-4xvg`. Caveat: graded on the suspect
  vs-random / points-not-marks metric ([[vs-random-eval-is-suspect]]) — see
  the eval entries below.)
- **RULED OUT: decision-time look-ahead over a distilled value function.**
  Gus [[lamir1|LAMIR-1]]: every depth-1 variant — V-bootstrap, Q-bootstrap,
  π_opp-conditioned rollout, blunder-detect-and-route — lost to direct π_me.
  Diagnosis via Kubíček & Lisý ([[lamir1-ceiling]]): a distilled V cannot
  support look-ahead without CFR+. (Corrected numbers: v-bootstrap regret
  1.645, not the pre-fix 2.777; the correction is carried on
  [[gus-lamir1-mode-comparison]].) The K&L warning was on the wiki's
  Gus/champion seam and pre-explained jud's JS3 failure — the prior-sweep
  lesson.
- **RULED OUT (structurally): "distill X and see."** The [[candlewax]] rule,
  learned the hard way three times: every distillation proposal must name its
  consumer and its licensed collapse first. The one that worked
  ([[w42-jud-v0|jud v0]]'s bidder) named both — auction root,
  realized-outcome pricing — before training.

### Self-play reinforcement learning

- **RULED IN (qualified): AlphaZero-shaped self-play gets off the ground in
  42.** 50% → 70.9% vs random in five days at 557K params (`a7d6b5b`), 76.3%
  peak at 3.3M (`large-belief-recap.md`, W&B-sourced). Real,
  checkpoint-backed. ([[zeb]].)
- **RULED OUT: that it reaches the champion.** No Zeb-descended policy ever
  beat E[Q] n=10 at pure play — not in Feb, not since (`afd4802` names it "the
  play champion, as it has against every learned challenger since Zeb"). Elo
  1579 vs 1600 is neck-and-neck, not a win.
- **RE-DIAGNOSED: the "capacity ceiling" wasn't capacity.** 6× params bought
  ~5 points — that's an information/architecture wall's shape. Jason's own
  autopsy: *"we were lying to the model"* — imperfect information, mishandled.
  (2026-03-11; [[alphazero-under-imperfect-information]].)
- **RULED OUT (by argument): continuing AlphaZero-Zeb.** *"distilling that run
  would just be distilling the heuristic I used to select >=18."*
  (2026-03-15.)
- **HARDWARE FACT: MCTS is latency-bound, not compute-bound.** A rented B200
  gave ~2.2× over a 3050 Ti; the algorithm doesn't fit the hardware.
  (Feb 3–4.)

### LLM as the reasoner

- **RULED OUT (Sept 2025, Opus-era): LLM as raw play engine.** *"opus has
  repeatedly struggled to reason about the game."* First-hand, dated.
- **RULED OUT (at ~2B scale): weight-drilled curriculum play ([[lem|LEM]]).**
  Nine days, Gemma→Qwen pivot (Qwen 100% vs Gemma 60% on comprehension),
  [[star|STaR]] to v10-maskfix: 86% comprehension, 55/100 bot-match, ceiling
  correctly diagnosed as STaR/capacity, not gradients. Closed clean, handed
  off. (be7efc4.)
- **RULED OUT (at this scale): tool-using LLM as a policy that beats the
  lookup.** [[burl|Burl]] validated the tool surface —
  [[iter3-rules-adapter]], 90% bot-match via rules-as-tools — and graded null
  on policy against E[Q]. *"A reasoner that can look things up is not yet a
  reasoner that out-plays the lookup."* Zero `burl/` commits since 2026-05-07.
- **RULED IN: rules-as-tools over rules-in-weights.** The one validated Burl
  win; the adapter chain's terminal lesson. (dbadb5f; [[rules-as-tools]].)
- **FLAGGED, PRE-REGISTERED, THEN WALKED INTO: self-fed traces plateau.**
  *"LLMs actually get dumber when fed their own content back. How is the
  contradiction resolved?"* (2026-04-04, left open) — then STaR went 30% →
  42% → plateau 38–39% over 15 iterations. ([[star-10-iterations]].)

### The human canon as plan source

- **RULED IN: the book's content is trustworthy.** [[w42|W42]] graded Roberson
  claim by claim — 64 ledger rows, phases 1–4, five waves, zero fabrication
  across ~120 pages, conservative closes with explicit blockers.
  ([[w42-claim-ledger]], [[w42-phase4-final-claim-audit]].)
- **RULED OUT (as stated): the book's advice as a utility guide.** Wave 3.0's
  "EV supports the book's void-creation advice — ADOPT" was inverted by Waves
  4.0/4.1: the advice aligned with the *worst*-scoring utility; call
  downgraded to ADOPT-DEFERRED. (The reversal was never back-linked; it's
  real.)
- **RULED IN: Lens v1 — EV beats p_make as the utility to optimize.** The
  family's most durable living output ([[w42-lens-v1-utility-head-to-head]]).
  Note the recommended one-line `select_actions` switch to ev-argmax was
  **never applied**; the code still computes Lens(p_make) as of `afd4802`.
- **NEVER TESTED: multi-step book plans.** [[book-strategy-player|BookStrategyPlayer]]
  designed 2026-05-04, bead frozen, never built; chassis redirected to the
  auction by the 2026-06-09 design review.

### Belief modeling

- **HARD NEGATIVE: Zeb's belief headline was hollow.** 72% top-1 overall was
  **39% on hidden state** — the number that mattered.
  ([[zeb-calibration-eval]], reconfirmed by `afd4802`.)
- **HARD NEGATIVE (in Feb): belief decoupled from policy does nothing.** The
  belief head predicted opponent ownership and never fed a move — *"2 things
  side by side only connected via loss."* (02-18T03:45;
  [[belief-feeding-policy]].)
- **RULED IN (in June): belief feeding decisions, done right.** Champion #24 —
  [[gus|Gus]] belief conditioned on the completed auction — +2.59pp held-out,
  the first heavy-training measured win of the champion ladder, still live at
  the frontier. Gus survives as the champion's posterior engine.
  ([[w42-champion-auction-belief]].)
- **REFRAMED: belief's value is legibility, not marks.** The [[arena]]
  belief→marks nulls were measured against a PIMC-vs-PIMC harness structurally
  blind to concealment and signaling (Fable's insight). The objective for
  belief was mis-specified, not the belief. (Legibility here is an
  instrument — a verification surface, per the author correction woven through
  §1, §3, and §4 — not a promoted goal.)

### Realized-outcome value learning (jud)

- **RULED OUT: the parity plateau as a structural limit.** [[w42-jud-v0|jud
  v0]]'s registered prior — "parity with `net:wp` is the PIMC price of hidden
  information" — was falsified by its own plateau probe
  ([[w42-plateau-probe]]): the ceiling was **data starvation** in a tiny head.
  Scaled 3× on-policy, the learned bidder passed the hand-tuned one. The era's
  finest methodological hour.
- **RULED IN: value-native reasoning at the auction.** `margin:wp`(head_8):
  +0.38 [+0.09,+0.67] and +0.42 [+0.12,+0.72] marks/game over the hand-tuned
  champion — **the first learned component ever to beat it** — saturating at
  ≈+0.3–0.4. Bidding validates.
- **RULED IN: the one-organ unification holds at the auction.** [[w42-jud-v1|jud
  v1]]'s info-state → 43-bin realized-points categorical prices contracts; the
  bidder survives the fold.
- **RULED OUT: greedy 1-ply value play as a move-ranker.** JP3 falsified; the
  self-play loop moves it zero.
- **PARTIAL: oracle-free search at the leaf.** `judsearch` (belief-lift
  worlds, current-trick rollout, V_realized leaves) recovers two-thirds of the
  play gap (−3.44 → −1.16, JS1 PASS +2.28) — but not parity, and neither more
  worlds (JS2, below band) nor a better-calibrated head (JS3, FAILED) closes
  the rest. **The wall's current coordinates:** a 470k MLP on hand-level
  Monte-Carlo labels cannot out-rank E[Q] n=10's per-move oracle.

### Evaluation itself

- **RULED IN: Bradley-Terry Elo over a pairwise matrix**, anchored on E[Q],
  fanned to 28 T4s. First time everything sat on one scale. (`3a77bb6`;
  [[eval-matrix-bradley-terry]].)
- **SUSPECT AND KNOWN SUSPECT: vs-random with winner=more-points.** *"more
  points does not necessarily win the game!... there is no bid anywhere where
  getting <30 is a win"* (02-09) — plus random-vs-random not converging to
  50/50 (47.8/52.2 seating asymmetry). Every era-4 headline number is graded
  against a non-win-condition objective. Named in-window, never fixed
  in-window. Later eras moved to marks and full games to 7 (champion/jud).
  ([[vs-random-eval-is-suspect]].)
- **RULED IN: pre-registration.** Every jud rung registered on GitHub before
  measurement; the plateau probe run to its falsifier. This is the loop, and
  it should be the default.
- **UNTRACKED RISK, STILL OPEN: WorldSamplerMRV sampler bias** — flagged as
  affecting "all historical Burl eval numbers and forge/eq training data,"
  parked as "worth a bead," never filed. Still dangling as of 2026-07-06.

### Process and tooling (briefly, because they're load-bearing)

- **RULED IN:** big-bang no-legacy rewrites (three "3–4 week" epics closed in
  hours-to-days); grok-don't-converge (*"you all led me catastrophically
  astray"* — [[grok-not-converge]]); the gofish ratchet testbed (5700
  games/sec, proved the auto-research loop cheaply; [[the-gestation]]); the
  fleet pattern (HF-as-message-bus, reputation scoring, CQRS monitor;
  [[zeb-fleet-ops]]).
- **RULED OUT:** build-automation swarms (Claude Flow, *"broken beyond
  redemption"*, one attempt); vibe coding (*"I read and understand all of the
  code and write most of it"*); external RL frameworks (Sample Factory,
  rejected in-session to protect the actual hypothesis).

## 3. The pattern of the misses

Sharp, because the pattern is more valuable than any single instance — and
because Jason's standing preference is the sharp thing said plainly.

**1. The project keeps discovering the wall, filing it under a narrower
heading, and optimizing past it.** Count the sightings: December 2025
(*"'boring and competent' isn't worth the squeeze"* — the objective,
pre-registered, then forgotten when a strong number showed up); January 3 (the
2-2 lead — a player with no read, filed as an aggregation bug); January 10
(*"I don't care about signaling. yet"* — the consumption rule chosen by
scope-cut, never revisited); January 12 (the [[argmax-q-ceiling|74% ceiling]]
— the wall measured, celebrated as validation); February 6 ([[candlewax]] —
the wall named, at 4am, before the experiment); March 15 (*"distilling the
heuristic I used to select >=18"* — the wall generalized). Six sightings
before the founding condition ever said "I dunno what to do with it." The
wall was never invisible. It was repeatedly visible and repeatedly demoted to
a footnote of whatever was being built that week. The project's single most
expensive habit is treating its own sharpest sentences as asides.

**2. The depressed android became an ontological mystery the second time,
when it had been a P1 bug the first time.** In December 2025 an AI that
"isn't like a person" had a concrete cause (greedy rollouts) and a concrete
fix (search to terminal) ([[pre-ml-ai-attempts]]). When the identical symptom
returned as the Hawking voice modulator, it got treated as a property of E[Q]
rather than as the recurrence of a debuggable class. Same complaint, different
decade of response.

**3. The project predicts the null, then runs the experiment that can only
confirm it — and skips the one it is uncertain about.** February: *"not
something you can just pick 'best' for"* — then the
[[full-teacher-eq-experiment|full-teacher experiment]] picked 'best' from it
at two mix ratios and confirmed the prediction, while the actual open question
(how to use the distribution *without* collapsing it) went untested. April:
model-collapse flagged as an unresolved contradiction — then [[star|STaR]]
walked straight into the flagged plateau. The pre-registration discipline that
made jud's plateau probe ([[w42-plateau-probe]]) the era's finest hour is the
cure, and it arrived in month seven.

**4. The real finding keeps arriving as a throwaway on the way out the
door.** *"How do we get beliefs feeding policy?"* (3:45am, era 4's last note —
became [[gus|Gus]]/#24 ([[w42-champion-auction-belief]]), the frontier's live
belief engine, four months later). *"Subtle, not complex"* (era 5 — still the
best description of what E[Q] lacks, still without an instrument). *"At the
end of the game, I have changed. the model has not"* (the online-learning gap,
named perfectly, never touched). The asides outperform the roadmaps, and
nothing in the process promotes an aside to an experiment.

**5. The project measures against objectives it has already disowned — and
knows it while doing so.** Winner=more-points was flagged Feb 9 and every
era-4 headline was graded on it anyway ([[vs-random-eval-is-suspect]]).
Vs-random was disowned at the era-4 closeout — after the era was graded on it.
The belief→marks nulls were measured on a concealment-blind harness
([[arena]]). The lens fix (EV over p_make,
[[w42-lens-v1-utility-head-to-head]]) was validated and the one-line code
change never shipped. A measurement disowned but still reported is worse than
no measurement: it aims the next month's work.

**6. Deferrals accumulate into decisions nobody made.** "Yet" (signaling) was
never revisited. The browser AI — the literal 12-year-old's dream, *"distill
that into a policy network etc for the web game"* — fell off the table without
a decision; `actionSelector.ts` still offers `'beginner' | 'random'`, and no
neural player has ever shipped in the actual [[web-game]]. Legibility was
prototyped as a feature (the move-explainer, born from Jason's dad distrusting
a bot) and deferred like the rest — and it stayed what the author's correction
says it always was: an instrument, a verification surface for a
[[star|STaR]]-style bootstrap, never the objective (era 5's narrator reframe
read like a promotion; it wasn't one). Each deferral was correct triage in the
moment; the miss is that nobody logged the accumulation.

**7. Naming energy outruns mechanism.** [[lem|Lem]], [[burl|Burl]], Harl,
LLem ([[ideated-not-built]]) — a beautiful sharecropper lineage, weeks of
design — and the pipeline that shipped bypassed every intermediary:
[[expected-q-value|E[Q]]] itself was the judge and the world model all along.
The corollary in the record: "promoted goal in the present tense" is the
single most dangerous sentence shape in the corpus ("this is what ships,"
"already deployable," "exported to plunge as onyx" — none of them true). Plans
kept wearing the grammar of accomplishments.

**8. The record corrects itself forward and never looks back.** Zero
fabricated numbers across ~400 audited pages — genuinely remarkable — but ~70
pages sat `status: active` on dead lines, reversals (Wave 3→4, the perf-batch
retraction, the [[lamir1|LAMIR]] regret fix) were made and never linked from
the wrong page, and "worth a bead" became a resting state where risks go to
evaporate. A status field that's never falsified is decoration. A correction a
cold reader can't reach from the error is a correction that didn't happen.

## 4. What was never tried

Visible in the record as ideated-but-unbuilt or named-but-unexplored
([[ideated-not-built]] is the era-5 classification record). Each with where it
surfaced and why it bears on [[the-wall]]. Listed, not ranked — ranking them
is "what next," and that question is reserved.

- **Threat-class vocabulary.** Jason's own coinage, 2026-01-14: track *which
  threats* ("4s greater than 3") each opponent holds — "(7×3) things to
  combine rather than 21×3." Scoped as a sampling speed hack, never built
  ("hasn't made it to the implementation yet," 2026-01-27). It matters because
  it's the most plan-shaped idea in the whole record: a compression of hidden
  state into *strategically meaningful* units — a candidate vocabulary in
  which a plan is even expressible.

- **Consuming the E[Q] distribution as a distribution.** The 85-bin `e_q_pdf`
  has existed since `ef199b0` (Jan 24). Every consumer since has collapsed
  it — argmax, p_make, EV, a mean. The [[candlewax]] complaint is literally
  about the shape (*"lumpy spiky... long weird tails... the average score of a
  pdf is meaningless"*), and no experiment has ever consumed the shape. Risk
  posture, variance preference, tail avoidance — all live in those bins,
  untouched.

- **Opponents-in-rollout / opponent modeling.** [[jud]]'s rollouts assume the
  world; they don't model *how these opponents play*. Named explicitly in the
  jud v2 cue (`afd4802` era) and implicit in the March particle-filter thread
  (the likelihood-ratio opponent model, redirected and never built). It
  matters because a plan against nobody in particular is not a plan; the read
  is half of what the modulator lacks.

- **Forward planning at the leaf / multi-step search.** Era 5's
  proof-by-contradiction was announced — *"I'm trying to prove by
  contradiction that I do indeed need forward planning"* (03-21,
  [[the-gestation]]) — and never carried out. [[gus|Gus]] never went past
  depth-1. The ratchet's own stopping condition (*"if we ever find an
  algorithm... capable of forward planning, we can release the ratchet"*) is
  still armed. [[w42-jud-v1|jud v1]]'s leaf is a hand-level head; the v2 cue
  (bigger leaf, per-move targets) is the named-but-unbuilt continuation.

- **CFR+ over distilled values.** Kubíček & Lisý name it as the licensed way
  to get decision-time look-ahead from a distilled V. The [[lamir1-ceiling]]
  page listed it as an option; option 4 (self-play/value-native, no CFR+) was
  taken instead. The one theoretically-sanctioned path around the look-ahead
  ruling has never been walked.

- **Belief-weighting by counterfactual play evidence, at play time.** Deferred
  twice on principle in January (*"no heuristics... should arise naturally if
  it is correct"*). #24 ([[w42-champion-auction-belief]]) conditioned belief
  on the *auction*; nothing yet weights worlds by *how the play itself has
  gone* — who ducked, who signaled, who didn't. This is the uniform prior at
  the root of the feel-lessness, still mostly uniform.

- **Signaling and concealment as objectives.** The "yet" of 2026-01-10, still
  outstanding. The [[pimc|PIMC]]-vs-PIMC harness is structurally blind to both
  (Fable's point), so they have never even been *measurable*, let alone
  optimized. A harness that can see concealment is itself an unbuilt artifact.

- **A world model proper (Dreamer / LEWM).** Dreamer was seriously entertained
  and deliberately dropped (*"nah ok forget the dreamer for now... the world
  is the e[q] distribution"*, 03-18) — a real ruling, but by argument, not
  experiment ([[ideated-not-built]]). "lewm" is a conversation title only; the
  pixels-for-42 brainstorm behind it was never a design. The "E[Q] *is* the
  world" claim has never been stress-tested against a learned dynamics model.

- **The private-hand table.** Designed 2026-03-08: play *with* Claude
  honestly — *"I am a seat at the table with a private view"* — public events
  only, no peeking. Never built. It matters because it's the only proposed
  instrument that measures the thing the founding complaint is about: whether
  an agent at the table reads as having a plan, from inside the game.

- **BookStrategyPlayer / multi-step book plans.**
  [[book-strategy-player|BookStrategyPlayer]] designed 2026-05-04, bead
  frozen, redirected to the auction, never built. The book's *tactical claims*
  got graded ([[w42]]); its *plans* — the multi-trick shapes Roberson actually
  teaches — have never been executed by any agent and scored. The one direct
  test of "does a human-legible plan win tricks" never ran.

- **The narrator over the player.** Harl-narrates-over-Gus was designed and
  never executed; no wiki page shows [[lem|LEM]] ever consuming a [[gus|Gus]]
  decision. The era-5 phrasing — *"training a narrator not a policymaker"* —
  has never had its experiment: take the strong player's decision, produce the
  family-table explanation, and grade the explanation. Its value, per the
  author's correction, is as a [[star|STaR]] bootstrap source — catch the
  explanations that are actually correct, train on those — not as a
  legibility deliverable; a player that sounds right is a means, not a goal
  ([[narration]]). As a bootstrap surface it is still the most conspicuous
  untested bet in the record.

- **Online learning / within-game adaptation.** *"At the end of the game, I
  have changed. the model has not"* (2026-04-04). Named perfectly, never
  touched. Every trained artifact in seven months is frozen at inference.

- **E[Q]-distill as a per-move bootstrap with its consumer named.** The
  [[jud]] v2 cue: distill [[expected-q-value|E[Q]]] onto *per-move* targets,
  finally with the consumer (the search leaf) declared up front. This is the
  first distillation proposal in the project's history that passes the
  [[candlewax]] rule — and it is, as of 2026-07-06, unbuilt.

- **Publishing the corpus.** *"this may be the first ever collection of 42
  game logs. I should publish it"* (2026-01-11). Never done. Not
  wall-relevant, but it's in the record, and the 12-year-old would want it
  noted.

## 5. The state of the question, 2026-07-06

Here is the frontier, precisely, as of HEAD `afd4802`.

**The champion is unchanged.** [[expected-q-value|E[Q]]] n=10 — the noisy
mean over ten exact double-dummy worlds, evaluated per move — is undefeated at
pure play against every learned challenger since [[zeb|Zeb]]: Zeb (Elo 1579
vs 1600, neck-and-neck, never past), [[burl|Burl]] (tool surface validated,
policy null), [[gus|Gus]] (best fast student at 0.551 regret, every look-ahead
variant lost to its own direct policy head), and now [[w42-jud-v1|jud v1]]'s
play half. It has no plan, no strategy, no feel. It computes p(make) and EV
and stops. Seven months of challengers have made it more undefeated, not less.

**But [[the-wall|the wall]] now has a crack and a set of coordinates, and both
are new.**

The crack: **bidding validates.** [[w42-jud-v0|jud v0]]'s `margin:wp`(head_8)
— a learned head trained on realized 4-seat self-play outcomes, its structural
prior registered and falsified by its own plateau probe
([[w42-plateau-probe]]) — beat the hand-tuned champion bidder on marks: +0.38
[+0.09,+0.67] and +0.42 [+0.12,+0.72], saturating at ≈+0.3–0.4 marks/game. It
is the first thing in the project's life that takes the E[Q]-shaped data,
reasons with it, and does better. jud v1 then folded bid and play into one
organ and the bidder *survived the fold* — the unification holds at the
auction. The current best-player recipe is `margin:wp`(head_8) bidding with
`lens:ev` utilities over E[Q] play ([[champion]]), the first learned bidder
ever to beat `net:wp`.

The coordinates: **play is mechanism-limited at the leaf.** Greedy 1-ply value
play is a bad move-ranker (JP3 falsified; the loop moves it zero).
`judsearch` — belief-lift worlds, current-trick rollout, V_realized leaves —
recovers two-thirds of the play gap oracle-free (−3.44 → −1.16, JS1 PASS
+2.28), and then stops: more worlds don't close the rest (JS2 below band), a
better-calibrated head doesn't either (JS3 FAILED — the failure K&L's warning
on the Gus seam had pre-explained, [[lamir1-ceiling]]). The verdict sentence,
exact: *a 470k MLP on hand-level Monte-Carlo labels cannot out-rank E[Q]
n=10's per-move oracle.* Not "a learned player can't win" — *this* player,
with *these* targets, at *this* leaf, can't out-rank the oracle move by move.

**The question itself is in the sharpest form it has ever held.** Not "can I
build a 42 AI" (built), not "can I compute the truth" (computed, 12,325×
over), not "can I distill it" (distilled, three ways, to a 0.551-regret
student), not "can self-play beat it" (no), not even "who consumes the oracle"
in the abstract. The founding condition — *"maybe the most amazing thing I've
ever built and I dunno what to do with it, so let's friggin try stuff"* — has
been discharged by the trying. What remains is the goal stated in its final,
precise, non-ill-specified form: **something that takes the E[Q] data, reasons
with it, does better than E[Q], and has a plan — a strategy that actually
succeeds.** "Plays like a person" is explicitly off the table as a
specification; the modulator complaint is a symptom report, not a spec. One
component now meets the spec at the auction. No component meets it at play.

Around the frontier, the standing assets: Gus's belief engine live in the
champion (#24, auction-conditioned, +2.59pp,
[[w42-champion-auction-belief]]); Lens v1's EV-over-p_make result validated
with its one-line code switch still unapplied
([[w42-lens-v1-utility-head-to-head]]); the book's 64-row claim ledger closed
and trustworthy with its plans untested ([[w42-claim-ledger]],
[[book-strategy-player]]); the ratchet armed with its stopping condition unmet
([[the-gestation]]); the 85-bin histogram computed at scale and never once
consumed as a shape ([[candlewax]]); and one untracked risk (WorldSamplerMRV
sampler bias) still owed an issue. The named-but-unbuilt jud v2 cue — bigger
leaf, per-move targets, E[Q] distilled with its consumer declared,
opponents-in-rollout — sits in the record as an implication of the diagnosis
rather than a hope, which is a first.

That is the edge. What to do standing at it is a question Jason has reserved
until the wiki is organized, and this trail honors the reservation. The
biography ends here, at the wall, with a door drawn on it and the pencil set
down: the bidder crossed; the player located; the champion — flawless,
voiceless, forty-six years after the grandmother's knee — still waiting for
the thing that can explain why it plays.

## Related pages

[[the-wall]] · [[consumption-ledger]] · [[candlewax]] · [[ideated-not-built]]
· [[the-gestation]] · [[expected-q-value]] · [[jud]] · [[champion]] ·
[[w42-jud-v1]] · [[lem]] · [[burl]] · [[gus]] · [[zeb]] · [[w42]]
