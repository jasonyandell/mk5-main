---
title: A GPT on 42, read for its neurons (proposal)
kind: topic
first_seen: 2026-10-16
last_updated: 2026-10-16
status: active
---

## What Jason asked

2026-10-16: "I want to seriously try a gpt on 42 … I read an interpretability paper and
in it, he found that there was recursive generalization forming naturally … the neurons
gaining concepts. that's what I want for 42. seems to spring up from standard training …
this recursive generalization is exactly the pattern we see with the math of 42. what do
you think we should tokenize? I think the splat."

This page is the proposal. **Nothing here is built or measured yet.** The build is
issue #175.

## The source: "Neural Networks Are Already Clean"

Nick Cammarata, Goodfire, 2026-10-15 (`already-clean.goodfire.com`), subtitled "a working
hypothesis". A technical companion paper is announced. This section was first drafted from
secondary coverage, then corrected the same day against the essay's text, which Jason pasted
in (through the opening of "Cars and Animals"). The essay is © Goodfire and is **not** copied
into this repo. Short quotations are attributed below. The rest of the essay (its vision and
language-model sections) has not been read here yet.

- **The claim.** The canonical evidence for polysemanticity and superposition "was misread.
  What looked like arbitrary packing was mostly a motif I'll call recursive generalization."
  The motif: "as a neuron grows, it sprouts offshoots the way a tree branch sprouts
  sub-branches." Example: a neuron that detected "iy" grew to detect "diy", then "do it
  yourself", which does not contain "iy" at all. The idea first appeared in *Multimodal
  Neurons* (2021). It became checkable once neurons could be watched growing over training.
- **One learning principle, not two.** "Generalization groups similar things together and
  superposition groups unrelated things together." The essay keeps only generalization:
  "it turns out generalization is fractal." So explanations can be fractal too: "start with a
  basic explanation, then zoom in when you want to understand more."
- **Capacity makes concepts coarser, not tangled.** InceptionV1 was retrained across a 300×
  range of sizes and stayed legible at every scale. "With less room, the network doesn't fold
  unrelated things together; its neurons just cover coarser-grained concepts", like one car
  neuron covering many cars. With more room, that neuron differentiates into specific cars.
  (A secondary source's "layer 3b cut from 480 to 10" is not in the text read here.)
- **Architecture, not size, fixed the deep layers.** InceptionV1's last layer once looked
  almost uninterpretable. With ordinary architectural improvements for deep layers, and no
  change to size or data, 99–100% of its neurons are easily interpretable.
- **The canonical cases, reread.** InceptionV1's neuron 4e:55, "cats and cars", is a V-shape
  (pointed) neuron. Its circuit connects to V-shaped flames and fox faces, and its negative
  pole is round. The *Zoom In* "car hidden in dog neurons" circuit is shared pose, carried by
  over a hundred generalized pose neurons. Feature visualization had been reward-hacked into
  dogs.
- **The audit.** Preliminary audits of Qwen3-32B: about 90% of neuron poles have one main
  job, often with small related offshoots; about 7% have two related jobs; about 2–4% are not
  yet understood. Similar results hold in dense and mixture-of-experts models, up to
  trillions of parameters (Kimi K3). The exact numbers depend on thresholds.
- **The method.** The essay studies a network's own neurons, not a trained decomposition
  such as a sparse autoencoder. For each neuron it reads both poles (what drives it positive,
  what drives it negative). It reads the context around top activations, not only the token
  where the neuron fires; feeling detectors, for example, fire on commas. It checks steering
  (turn the neuron up and see what the model writes), upstream construction, downstream
  readers and growth over training. Understanding compounds "like a jigsaw puzzle or
  crossword": once most of the map is known, the unexplained remainder points at broken
  tools. That is how the essay found a double-negative regime in SwiGLU neurons.
- **The four steps, verbatim in substance.** (1) One learning principle, so ordinary science
  on the network can begin. (2) With modern models and tools that science goes better than
  expected. (3) Understanding compounds, so a network's own mechanisms belong in the
  toolkit. (4) "Highly speculative": this might be most of what interpretability needs.

What makes it relevant to 42: on this account clean concepts come from **ordinary training**.
No sparse dictionary or special loss is needed. If that is true, a small transformer trained
on 42 should grow neurons that can be read, and they should branch.

**What 42 offers that language does not: an answer key.** In a language model the crossword
is checked only against itself. In 42, many of the concepts a neuron might carry can be
computed exactly from the rules and the information state (suit membership, count still out,
proven voids). Proven math can therefore grade the neuron map. Because the models are small,
the essay's size sweep is cheap to run in full here.

## Why 42 is a good place to look: the mathematics is an answer key

Jason's thesis is that recursive generalization is the shape of 42's mathematics. What makes
42 unusual is that the mathematics is **proved**. The texas-42 repository states, and in
part kernel-checks in Lean, what a seat's information *is*, which parts of it matter for
what, and which symmetries hold. A language model's neurons can only be checked against
each other. A 42 model's neurons can be checked against theorems. Tier labels follow
texas-42's convention: [corpus] = the reconstructed foundation package (rec: texas-42
`ingest/texas-42-foundations-source-of-truth-v0.7-reconstructed/docs/20_MATHEMATICAL_FOUNDATION.md`;
Lean names are in texas-42 `lean/Texas42/`), [Lean] =
kernel-checked, [CONFIRMED] = exchange-adjudicated, [exploratory] = below every tier.

- **What the input is.** A seat's perfect-recall record is its own hand plus the public
  prefix: the auction with actors, the declaration and the plays in order, each attributed
  to its seat. Trick winners, led suits and scores are *derived* and must not become a
  second source of truth (rec INFO-01..03A, rec:1203-1257 [corpus]).
- **What the physics reads.** The mechanical projection keeps the played-by-seat sets and
  the public voids and deliberately drops the order of completed plays (INFO-07,
  rec:1298-1340 [corpus]). Equal projections give equal cells and fibers (CELL-05, CELL-14
  [corpus]; `ViewerCtx.losslessness`, `fiber_eq_iff_totalNF_eq` [Lean]). The reduced
  viewer kernel K = (declaration, own hand, pool, folded open trick, utility residue) is
  sufficient for the physics (FAC-02/PLAY-17, rec:4201-4240 [corpus]).
- **What belief may read.** Support is not belief (INFO-11 [corpus]). Play order can matter
  to a belief that reads plays as evidence. The 90-world witness has identical support and
  opposite optimal leads (STR-08/09 [corpus]; `ninety_world_witness` [Lean]). Under the
  uniform deal law with no action likelihood, the posterior is uniform on the fiber
  (BEL-07/07A [corpus]), and then the projection is enough.
- **Which symmetries hold.** Seat rotation is exact if it moves every seat-indexed object;
  odd rotations swap teams (SYM-01 [corpus]). Bidder anchoring is an exact gauge (SYM-02
  [corpus]). Reflection is not a symmetry (SYM-03 [corpus]). **S₇ pip relabeling is a
  symmetry of legality, not of value**: the count tiles break it. Only the 2↔3 swap
  preserves counts, and it transports the game only between declarations 2 and 3 (ALG-17..19
  [corpus]; `countPreserving_iff`, `swap23_transport_iff` [Lean]). Count-blind mechanics fall
  into exactly three classes: pip trumps, doubles-trump and no-trump (ALG-22/23 [corpus];
  x:004 [CONFIRMED]).

From this, candidate branchings that a network could grow, each with an exact label:

- **Role from pip.** Trumps are seven isomorphic legality structures that are *not* equal in
  value. A neuron that means "trump" across all seven pip declarations is the trunk; its
  branches are what the count tiles make different (a trump that is also a count tile, the
  6-4 under sixes against the 5-0 under fives). This is recursive generalization stated as a
  theorem about which merges are licensed. The suit algebra goes in, so the network is
  handed membership and rank. Whether one neuron serves every trump is the question.
- **Count.** 35 of the 42 points sit on five tiles ([[count-fate-ledger]]). "Count still
  out" → "count at risk this trick" → "count my side can still save".
- **Inference from the fall.** `shown-void` (a seat publicly failed to follow) is a viewer
  fact; `void` in a world is not ([[silas]]). A void is the branch of "did not follow".
  Earlier work saw a transformer infer voids from attention, where explicit void features
  added only +1.4 pp (`gus-v2-voids-1000g`).
- **Sufficiency.** Does the network compress the record toward the projection the
  mathematics says suffices? This is the sharpest test available, and it has an exact
  answer key (see H1 below).

## What the project already learned, and what it changes

The oracle and champion eras trained many networks on 42. Their lessons, with what each
changes here:

- **jud v1 was nearly this model** ([[jud]], `w42-jud-v1`): hand, canonical auction and
  play (350 dims), an MLP to a 43-bin categorical of the declaring team's points, cross
  entropy. It plateaued. Every information state of a hand shared **one** Monte Carlo
  label. At an opening lead the candidate states differed in 9 of 350 input dims, and
  **better test cross entropy gave worse play** (`jud-v2-retrain-probe`). *Changes:* the
  target is the whole splat, one per (state, move); low loss is never the success measure;
  move discrimination is graded separately.
- **Distill for what** ([[the-wall]], [[candlewax]]): any "distill X" proposal must name its
  consumer first. *Here the consumer is the neuron map, not play.* Imitating Walt yields at
  best "a cheaper equal" ([[scenario-player]]); that is acceptable because strength is not
  the goal.
- **Pricing is policy-conditional** ([[jud]]): a value head is honest only for the policy
  that produced its data. The splat is Walt's under one field net. *Changes:* never mix
  splats poured under different field nets.
- **Sorted hand slots leak** (`q0-positional-bias`): sorted hands gave slot 0 a 1.74-bit
  bias. *Changes:* own tiles enter as a set, with no positional embedding.
- **Tokenizers that worked** (`forge/zeb/observation.py`, `gus/model/tokenize.py`): one
  token per tile, a sum of factored embeddings, seat relative to the viewer (me, left,
  partner, right), chronological plays with trick index and position in trick. The public
  tokenizer had lost the trump-rank feature; Gus's transformer beat a flat MLP that overfit.
- **The suit algebra's τ needs the led suit** (`suit-algebra-spec`): tier (trump, follows,
  slough) is only defined once a trick is led. *Changes:* declaration-only facts (suit
  membership, rank in suit, count) go on every tile; the tier goes on play tokens, relative
  to that trick's lead.
- **A raw-ID MLP memorized seeds** ([[suit-algebra]]). *Changes:* split by deal; the
  raw-identity arm is a control, never the default.
- **Probes need ceilings** ([[belief-bayes-ceiling]]): belief top-1 is capped near 40%, and
  the cap moves with the decision index. *Changes:* every probe is graded against its own
  ceiling and chance level.
- **The only prior interpretability** (`gus-probe`): categorical concepts lived in
  embeddings, relational ones in the layers. Attention read as a chain from declaration to
  survey to move. There has been no neuron-level work before this.

## The design

### Input: one seat's view of a whole hand, causal

The model is a **causal transformer** (a GPT) over one seat's view of one hand. The sequence
is that seat's perfect-recall record, in the order the seat learns it:

1. the declaration, the bid and the bidder (seat relative to the viewer), and the auction
   in the canonical encoding (`champion/margin_net.py`), which guards against selection
   leakage;
2. the viewer's seven tiles, as a set (no positional embedding);
3. every play in order, each token carrying the tile, the seat relative to the viewer, the
   trick index and the position in the trick.

Each tile token's embedding is a sum: identity (one of 28), suit membership under the
declaration, rank in suit, count value (0, 5 or 10), and on play tokens the tier relative to
that trick's lead. These are the rules of [[suit-algebra]], which is Jason's call: small,
odd, and not worth the network's capacity to rediscover.

The causal mask keeps time honest: the prediction at a play sees only what had happened.
The viewer's own hand is the only private information, and no world ever enters (the fence,
below).

### Output (the value head): a splat table at every decision of the viewer

At each token where the viewer is to play, the model outputs a **28 × 43 table**: for each
tile, a distribution over the 43 final bidder-point totals (0..42), masked to the legal
tiles. One forward pass over a seat's hand gives the splats of every move at every one of
its decisions. A row of that table is a splat ([[splat-visualizer]]).

### Answer and loss

The answer is Walt's splat for each legal move at that decision, read from the pour
(`scenario/viz/export.py` `splat_bins` already bins one move's 512 outcomes into the 43
totals). The loss is cross entropy between each legal move's predicted distribution and its
splat, summed over legal moves and decisions. Cross entropy against an empirical
distribution is a proper scoring rule: its expectation is minimized by the true splat, so
the 512-world sampling noise costs variance, not bias. Splats from enumerated fibers
(`n_worlds` < 512) are exact.

### The fence, and the mistake it caught

A draft of this page (2026-10-16) fed the network a whole **world** (all four hands) and
trained it on per-world outcomes, averaging the predictions over the cell afterwards. The
target was honest, being Walt's plan played out, and the average is how a splat is defined.
Jason rejected it ("strategy fusion is a demon! it is fenced!"), and he was right on two
counts:

- **The neurons would see all four hands.** The concepts to be studied would form in a
  network that never has to infer a hidden tile. They would be a seer's concepts, which is
  the disease of the oracle era: students that play as if they could see every hand
  ([[the-wall]]). An honest target does not undo a clairvoyant input.
- **A per-world value is one argmax from fusion.** The fence
  ([[strategy-fusion-fence]]) makes the mistake structurally impossible; it does not trust
  discipline at the call site. A model with a world in its input is a clairvoyant API, and
  this project does not build new ones.

The rule for this work: **no world ever enters the input.** Worlds appear only inside the
answer, where Walt has already averaged them over the seat's cell. The first nanoGPT in this
project's history was trained on oracle move frequencies, blind to the transcript, and was
killed within about 36 hours ([[strategy-fusion]], "The first fix"). This design differs on
both counts.

## Two heads: the map and the navigator (Jason, 2026-10-16)

Jason: "policy head. I think history goes to the policy head. and we aren't dodging
history." And: "Zeb is merely superhuman at memory. it has no strategy there's nothing to
exploit. it's a map, basically. damn fine map don't get me wrong but it isn't a navigator."

The model has one trunk over the seat's history and two heads:

- **The value head is the map.** It predicts Walt's splat table, as above. Walt's splat is
  a map: perfect memory of what has fallen, every consistent world counted, one plan per
  cell. In the 250k pour it reads the record only through the projection (see the data
  section), so this head has no use for history beyond what the physics needs.
- **The policy head is where history goes.** At **every** play token, for any seat, it
  predicts that play from what the viewer knew just before it. This is next-token
  prediction, the objective a GPT is built on. For the viewer's own plays it is the
  viewer's policy. For the other seats it is a model of another mind read from its
  history: what partner is doing, what the bidder is planning, what a seat's earlier plays
  say about its hand and its intent. That needs order, style and plans across tricks:
  navigation, not mapping.

**What this changes in H1.** The two heads give a dissociation that can be pre-registered.
Value-head features should be invariant to play orders that preserve the projection.
Policy-head features should not be, wherever the players the head imitates read order.
One trunk, two predictions, both with an exact answer key.

**A navigator needs navigators to learn from.** A policy head trained on the pour learns to
predict Walt and the field net, which are maps; it can only learn map-following there. The
pour is the pretraining corpus, not the destination. The navigator corpus is people:

- **Plunge already records every human hand.** The D1 table `hands`
  (`plunge/migrations/0006_hands.sql`) keeps every hand a human played, solo or at the
  family table, including the takeback branches. `payload` holds the engine replay of every
  bid, call and play; `hand_players` names the human seats; the exact Walt build at the
  table is recorded (`plunge/src/history/legacy.ts:20-46`). Humans against Walt, with full
  history.
- **Takebacks are a separate signal.** A takeback branch is a human changing a decision
  after seeing what followed. It should be filtered from imitation data, and is worth
  studying on its own.
- **It is family data.** It is not "generated data" under the Hugging Face rule
  ([[publish-data-to-huggingface]]). Nothing from it is published, or leaves Jason's
  hands, without the consent of the people at the table. Only aggregates go into the wiki.

**History queries live in history space.** Jason, 2026-10-16: [[silas]] cannot quickly
express "how many hands made their bid missing the 5-5", because a world has no history. A
Silas world is a hidden deal at one frame. That question ranges over *completed histories*,
and is a query over a corpus of hands: SQL over Plunge's `hands` (the deal, the bidder, the
result), or over the pour's `deals`. The label pass this study needs evaluates Silas's
viewer predicates at every frame of every history. That pass writes exactly the
history-indexed table that makes such questions joins: "made their bid" ∧ "the bidder's
dealt hand lacks the 5-5", or "partner held the 5-5 and showed it at trick 2". The policy
head is a probability model over that same space of histories.

## The data, and what it really contains

[texas-42-zeb-pour-250k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k)
(CC0; [[huggingface-assets]]): 250,000 deals, 4,343,426 planned positions
([[metasplat-2026-10-13]]), a 512-world splat for every legal move (schema:
`scenario/atlas.py:248-274`, `scenario/hf.py:118-170`). Forced moves are not stored. They are
still in the record, and their splat is the position's.

What a careful reader of the code finds (2026-10-16):

- **The tail after a decided contract is random play.** The pour plans under `make`. Once
  the contract is decided, or the horizon is hit, the rest of the hand is played by the
  record-keyed dice tape, which is random legal play (#107; `scenario/planner.py:864-917`).
  So the 43-total splat is exact about make or set at 30, and past that point its shape is
  one random playout.
- **What the teacher reads.** The splat's seed and worlds come from (config seed, viewer,
  own hand, the *set* of tiles played, leader, open trick, banked points, voids)
  (`scenario/planner.py:639-643`). Worlds are uniform on the void-consistent fiber, all
  weights 1 (`scenario/metasplat.py:288-294`). The posterior draw (#111) and the stochastic
  field (#108) came after this pour. The other seats are played by the field net, whose
  features also drop the order of completed tricks (`scenario/field.py:52-64`). So **in this
  pour the target is a function of the mechanical projection, not of play order**: exactly
  the uniform-fiber case of BEL-07.
- **Two field nets.** Split s100001 was poured with the 5k-deal field net; s200001..s350001
  with the 50k net. Train on the four matching splits (200,000 deals); keep s100001 as a
  shift test.
- **Seat 0 always bids 30** and declares its best trump, so the auction carries no
  information yet. The tokens stay for later pours.
- **The noise.** pmake per move has a standard error of at most about 0.022 at 512 worlds;
  the plan's own seed adds 0.005–0.016 ([[metasplat-2026-10-13]]). 45% of the book-rule
  contrasts sit at an even make and 11% are ties, so sibling splats are often near-identical.
- **Size.** About 3.5 million decisions over 200,000 deals, each a (view, legal moves,
  splats) row; a seat's hand is one sequence of at most about 45 tokens. A model of a few
  million parameters trains on a single machine.

## Hypotheses, pre-registered before any training

Each is fixed in code and committed, with the hash cited, before the first count, per the
project's template (fly-connectome pilot, [[metasplat-2026-10-13]], `count-lapse`):
intervals by paired bootstrap clustered by deal, a chance level and a ceiling for every
number, nulls reported in full, "a tie is not agreement" ([[book-authority]]). Splits are by
deal. Arms: the trained model, the same network at random initialization, shuffled labels,
a linear probe on the input, and the raw-identity arm.

- **H1, sufficiency, and its dissociation.** The value head's teacher reads the record only
  through the projection (above); the policy head's targets are plays by players who may
  read order. Prediction: value-head features are invariant to projection-preserving play
  orders, and policy-head features are not wherever the imitated players read order. For
  the value head: if
  training finds what the teacher reads, the final layers should be **invariant to play
  orders that preserve the projection**: pairs of records with equal (played set, open
  trick, voids, leader, banked) but different order should sit closer than matched random
  pairs, and should move together over training. This is a direct test of whether gradient
  descent discovers a sufficient statistic that is a theorem (INFO-07, CELL-05 [corpus]).
  A null is as interesting as a pass: it would say the network keeps order the teacher
  never uses.
- **H2, one trump, many branches.** Under the suit algebra, a neuron pole that tracks
  "trump" (or "boss of trump") should fire across all seven pip declarations. Its offshoots
  should separate where the count tiles break S₇ (ALG-17 [corpus]). Track over checkpoints:
  the essay predicts the trunk first and the branches later.
- **H3, viewer concepts in neurons.** For each label below, the neuron poles that carry it,
  read the essay's way (both poles, context, steering, upstream and downstream), against
  the linear-probe baseline and the nulls.
- **H4, the size sweep.** The same architecture across a wide range of widths. The claim
  predicts coarser concepts that can still be read, not noise, at small widths, and
  branches that split as room grows.

### The answer key (labels)

All computed exactly from the viewer's information, in bulk, through [[silas]]'s Python
bindings (`scenario/core/src/python_silas.rs`; the 18 book rules over the whole pour took
79 s on 15 processes):

- **Viewer predicates** (`silas/README.md:367-404`): `boss`, `master`, `shown-void`,
  `trump`, `count`, `current-winner`, `led-context`, `to-make`, `to-set`, `banked`,
  `trick-number`.
- **The book prelude** (`silas/lib/book.silas`): `trump-out`, `top-trump`, `sure-trump`,
  `partner-winning`, `can-take-it`, `safe-trick`, `guards-count`, `dead-double`, `crisis`,
  `walker`, and the rest: the book's own concepts ([[book-authority]]).
- **The 18 book-faithful rules**, as "fires here" and as the allowed set.
- **Outcome predicates** `lay-down`, `certain-make`, `certain-set`. These are three-valued:
  the solver refuses when over budget. Refused rows are reported, never dropped silently.
- **Exact holder marginals** under the uniform fiber, Pr(tile in seat) = N(C^{d→s})/N(C)
  (CELL-10E, BEL-07A [corpus]). This is the belief the teacher actually uses in this pour.
- **Hidden facts for grading only**, never input: count fates (`otis/fates.py`), the real
  holder of a tile.

### Secondary measures

- **Splat fidelity** on held-out deals: KL to Walt's splat, error in pmake at the contract,
  and paired per-threshold contrasts with noise bars (`metasplat.contrasts`).
- **Discrimination**, separately from loss: regret of the model's argmax against the splats
  (`scenario/field_train.py:361-409`, in plan units and in points), against the field net
  (`margin_d250k_h128x64`; best points regret 0.26 at 512×256) and a uniform pick.
- **Play, last**: arena regret against Walt, with an interval and the chance baseline.

## Decisions for Jason

1. **The target's tail.** The make pour's margins past a decided contract are random play.
   (a) Train on the 43 totals as they are, knowing the tail is noise. (b) Train on a pour
   planned under `points`: the 5,000-deal `texas-42-walt-pour` exists (small; check how far
   its horizon plans before relying on its tail); a 200,000-deal re-pour is about a day of
   compute. (c) Factor the output into make or set at 30 plus a margin
   head trained only on the planned part. Recommended: (b), starting from the small pour as
   the pilot.
2. **The output space.** 43 totals, or the 256 count-aware outcomes (tricks won × which
   count tiles captured; score bins from texas-42 `walt/math/anytime_proof_state_score_v0.1.md`
   [exploratory]). The 256-way output makes count capture visible at the head: more labels
   for the neuron map, at six times the output size.
3. **A map teacher that reads plays.** In this pour the teacher's belief is
   the uniform fiber, so the network has no reason to read order or style. A later pour
   with the posterior draw (#111) or Zeb's play reading ([[reads-plays-2026-10-13]]) gives
   the value head a teacher whose belief depends on order, and H1 then predicts the
   *opposite* invariance result for that head.
4. **The navigator corpus.** How many human hands Plunge holds, whether the family consents
   to their use for training, and whether a policy head may be trained on them at all. If
   yes: pretrain on the pour, fine-tune the policy head on human hands, keep the data
   private, and publish only aggregates and, with consent, the trained net.

## Where it lives

A top-level `gpt/` package, CC0 like the rest of the science (`REUSE.toml`; MIT only if it
embeds Zeb). It joins `scripts/check` as its own Python suite. Every trained net and dataset
it cites goes to Hugging Face, CC0, before the page that cites it is committed
([[publish-data-to-huggingface]]). Issue #175 tracks the build.

## Next

- Jason decides the four questions above.
- Build (#175): the reader over the pour (extending `field_train.dataset` with a
  `[N, 28, 43]` target), the tokenizer, the causal model, the Silas label pass, the
  pre-registration commit, then training with checkpoints and the probe suite.
- Read the rest of the essay (its vision and language-model sections) and the companion
  paper once it lands, and fold in what bears on the probe suite.
