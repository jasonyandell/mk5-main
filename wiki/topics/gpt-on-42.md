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
   test. The tests read each neuron pole (positive and negative), the context around its top
   activations, steering, and upstream and downstream weights. A linear probe is the baseline
   the neurons must beat. The neurons are the unit of study, not a decomposition. Use a modern
   block (pre-norm, a gated MLP); the essay finds architecture, not size, is what made deep
   layers legible.
2. **Branching over training.** Save checkpoints along the run. For each concept-bearing
   neuron, trace what it covers at each checkpoint. The essay's claim predicts trunks first
   and branches later.
3. **The squeeze, as a size sweep.** Train the same architecture across a wide range of
   widths (the essay used 300×). The claim predicts coarser concepts that can still be read,
   not noise, and branches that differentiate as room grows.
4. **Splat fidelity.** On held-out deals, measure the KL to Walt's splat and the error in
   pmake at the contract. The field net (`margin_d250k_h128x64`, [[scenario-player]]) is the
   baseline to beat or tie.
5. **Play, last.** Measure arena regret against Walt, with an interval and its chance
   baseline. A GPT that plays worse but whose neurons can be read is still the result asked
   for. A tie is not agreement ([[book-authority]]).

## Next

- Build (#175): tokenizer and dataset reader over the 250k pour, a small GPT, training
  with checkpoints, then the probe suite. The probes are the deliverable, not the player.
- Read the rest of the essay (its vision and language-model sections) and the companion
  paper once it lands, and fold in what bears on the probe suite.
