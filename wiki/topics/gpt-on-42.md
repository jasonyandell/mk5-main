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

Nick Cammarata, Goodfire, 2026-10-15 (`already-clean.goodfire.com`; a companion paper was
announced as pending). **This page was written from secondary coverage.** The sandbox that
wrote it could not reach the essay. Re-read the primary before quoting it anywhere else.
What the coverage agrees on:

- **The claim.** Most apparent polysemanticity is not superposition (unrelated features
  packed into one neuron). It is **recursive generalization**: a neuron learns one
  function, extends it to nearby cases, then sends sub-branches into adjacent territory.
  The cases it ends up covering look unrelated only if the branches between them are
  missed. The coverage's example is a neuron for the sound "iy". It extends to "diy", then
  to "do it yourself", which shares the concept but not the sound. The InceptionV1 neuron
  for cat faces and car fronts is reread as one for shared V-shaped geometry.
- **The distinction it draws.** Generalization groups similar things and teaches the
  network something. Superposition groups unrelated things and teaches nothing.
- **The squeeze.** InceptionV1 was retrained with layer 3b cut from 480 neurons to 10. Its
  neurons stayed interpretable and coarser: the branches merge back toward their trunk.
- **The audit.** A preliminary audit of Qwen3-32B estimates that about 90% of **neuron
  poles** (a neuron's positive and negative directions, read separately) have one main job,
  with 2–4% unexplained.
- **Its own caveats.** The essay calls this a working hypothesis and one step of its
  argument highly speculative. At the time of writing there was no independent replication,
  and commentators dispute that superposition is disproven.

What makes it relevant to 42: on this account clean concepts come from **ordinary training**.
No sparse dictionary or special loss is needed. If that is true, a small transformer trained
on 42 should grow neurons that can be read, and they should branch.

## Why 42 is a good place to look

Jason's thesis is that recursive generalization is the shape of 42's mathematics. The
project already has concrete branchings that a network could rediscover, each with a known
right answer to check a neuron against:

- **Suits from tiles.** [[suit-algebra]]: absorption (which tiles belong to a suit) and
  power (which suit beats which) are independent operations. All pip absorptions are
  isomorphic under S₇. A network shown raw tile IDs has to grow "is a six" first. That could
  branch to "follows sixes when sixes are led", then to "is trump when sixes are trump",
  then to the doubles suit under nello. Each step is a branch of the step before.
- **Count.** The five count tiles (5-5, 6-4, 5-0, 4-1, 3-2). A "5-5 is here" feature could
  branch to "count at risk this trick", then to "count my partner can save".
- **Inference from the fall.** A failure to follow suit reveals a void. A play reveals
  something about the hand it came from ([[reads-plays-2026-10-13]]). A "seat is void in X"
  feature is a branch of "seat did not follow X".
- **The recursion of the trick game itself.** Walt's rungs plan over continuations of
  continuations ([[scenario-player]]). Whether depth shows up as layered features is an open
  question, not a prediction.

These candidates are hypotheses to pre-register, not findings.

## The answer to "what to tokenize": not the splat as input

The splat is a move's distribution of final bidder-team margins over the mover's cell
([[splat-visualizer]], [[candlewax]]). Margins are even, so a splat lives on **43 values**
(bidder points 0..42). That fact settles the design:

> **A softmax over 43 margin tokens *is* a splat.** Make the margins the output
> vocabulary, and the splat is what the model predicts. It is not text the model reads.

- **Input tokens are the game as the mover saw it, and nothing else.** The sequence holds
  the information state: whose seat (relative: me, left, partner, right), the mover's seven
  tiles, the auction, the declaration, and every play so far in order. Then comes a query
  token for one legal tile. The sequence is exactly the cell's definition, so the model is a
  function of the information set. One output per information set (Lemma W, cited in
  [[splat-search-bounds]]) holds by construction.
- **Tiles go in as their 28 raw identities, not τ-encoded.** τ ([[suit-algebra]]) ranks a
  tile relative to trump. That is the concept handed to the model on a plate, and growing
  that concept is the thing to watch. The cost is known: a raw-ID MLP once memorized seeds
  instead of learning structure. Run a τ arm beside the raw arm as the control.
- **The target is the whole splat, not one sampled world.** Train with soft-label cross
  entropy against Walt's weighted per-world margins for that move. That is lower variance
  than predicting one world's ending, and it reaches the same optimum.
- **An auxiliary head predicts Walt's chosen tile** at the position: the policy, read beside
  the value.

Why not feed splats in? A model that reads splats learns to copy Walt's arithmetic. A model
that must produce them has to grow the concepts that make them.

### The fence

The splat is Walt's honest plan over a cell under a fixed field. Training on it is
sanctioned. A per-world double-dummy margin is not: that is the oracle-era mistake, and it
taught every student to play as if it could see all four hands ([[the-wall]],
[[strategy-fusion-fence]]). The first nanoGPT in this project's history failed exactly
there. It was trained on oracle move frequencies, blind to the transcript, and killed within
about 36 hours ([[strategy-fusion]], "The first fix"). This proposal differs on both counts.
The input is the transcript, and the target is a cell-level plan.

### The data already exists

[texas-42-zeb-pour-250k](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k)
(CC0; [[huggingface-assets]]) keeps the whole splat for every legal move at every planned
position. That is 4,343,426 planned positions ([[metasplat-2026-10-13]]), and several
times that many (position, legal move) rows. Each sequence is under about 60 tokens. A
model with a few million parameters is a CPU-scale experiment, not a GPU campaign.

Two scope limits are inherited from the pour:

- Its seat 0 always bids 30 and declares its best trump, so the auction is not learned.
- Past a decided contract, a margin comes from the tape, not the plan (issue #107).

## What to measure

Pre-register the following before the first training run, in the project's habit.

1. **The concept list and the probes.** For each candidate above, write down the exact label
   (computable from the information state, or from the deal for the hidden ones) and the
   test. The tests are neuron-pole top activations, plus a linear probe as the baseline the
   neurons must beat.
2. **Branching over training.** Save checkpoints along the run. For each concept-bearing
   neuron, trace what it covers at each checkpoint. The essay's claim predicts trunks first
   and branches later.
3. **The squeeze.** Retrain with one layer's width cut hard. The claim predicts coarser
   concepts that can still be read, not noise.
4. **Splat fidelity.** On held-out deals, measure the KL to Walt's splat and the error in
   pmake at the contract. The field net (`margin_d250k_h128x64`, [[scenario-player]]) is the
   baseline to beat or tie.
5. **Play, last.** Measure arena regret against Walt, with an interval and its chance
   baseline. A GPT that plays worse but whose neurons can be read is still the result asked
   for. A tie is not agreement ([[book-authority]]).

## Next

- Build (#175): tokenizer and dataset reader over the 250k pour, a small GPT, training
  with checkpoints, then the probe suite. The probes are the deliverable, not the player.
- Re-read the primary essay (and the companion paper once it lands) and correct this page
  wherever the coverage was wrong.
