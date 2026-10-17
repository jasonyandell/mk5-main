---
title: Clean neurons for 42 — reading Zeb's own net, concept by concept, as it grows
kind: topic
first_seen: 2026-10-15
last_updated: 2026-10-17
status: active
---

> **The neurons gaining concepts. That's what I want for 42.** This recursive generalization is
> exactly the pattern we see with the math of 42.
>
> — Jason, 2026-10-15

## The source

Nick Cammarata, ["Neural Networks Are Already Clean"](https://already-clean.goodfire.com/)
(Goodfire). It is an informal research essay and working hypothesis; a companion technical paper is
announced but not yet published. The essay belongs to its author and Goodfire and states no
license, so this page links to it, summarizes it in our own words and quotes only short phrases.
This page itself is CC0, like the rest of the wiki. It was written from the primary essay
(2026-10-15).

**Its claim.**
- Most of the "polysemantic" mixing that made interpretability look hopeless was an artifact of
  ordinary, fixable problems: weak tools, dated architectures, short data, and generalization not
  yet followed.
- What mixing remains is mostly **recursive generalization**. A neuron forms around one concept,
  then grows offshoots along that concept's facets: "generalization is fractal". Its example is a
  neuron that answers "iy", then "diy", then "do it yourself".
- So a network's own neurons and circuits can be read in place, with no sparse autoencoder needed
  to unfold them.

**Its evidence.**
- Vision models (InceptionV1, and ConvNeXt across about 300× in size), plus a preliminary audit of
  Qwen3-32B. It reports about 90% of neuron poles with one main job, and calls such numbers "a
  broad pointer to the regime".
- Retraining runs that watch neurons form across checkpoints.
- A squeeze test: a layer cut from 480 neurons to 10 grew *coarser* concepts, not mixed ones.

**Its own limits.** The thresholds are judgment calls, some samples are small, the language-model
results are preliminary, and its scaling step is labelled "highly speculative". Treat it as a strong
hypothesis to test here, not a settled result to cite.

## Re-read 2026-10-17: what to carry into every test

The essay's own practices, in our words, beyond the claims above:
- **Superposition is the last hypothesis, not the null.** For a messy unit, suspect bad tools,
  bad explanations, narrow data or old architecture, and generalization not yet followed, in that
  order, before superposition.
- **Growth is local.** A unit forms on a narrow pattern and grows along its facets. Generalization
  groups related things; superposition would group unrelated ones.
- **Read a unit as a gate, a shared pattern, not a list of nouns.** Both poles matter: check the
  negative pole whenever a grouping is claimed.
- **Strong activations carry the work.** In the essay, ablating the weakest 75% of activations
  cost one point of accuracy, and ablating the strongest 7% cost almost all of it. Our audit
  reads the top 1%, which fits.
- **Test an explanation by steering the unit**, and watch formation across checkpoints.
- **Small models are coarse, big ones narrow**; decompositions help most on the small, coarse ones.

Where 42 stands against it: [[lem-squeeze-2026-10-16]] (narrow nets cleaner),
[[lem-formation-2026-10-17]] (family kept 0.82 vs 0.28, but one unit kind formed coarse first),
[[lem-width-tree-2026-10-17]] (counterparts share their family, along one concept's facets). The
untrained floor (0.15–0.28 clean) goes with every share. Steering and negative poles are #198.

## The scope clause: what this net can and cannot tell us

Jason, 2026-10-17: "I believe the recursive generalization is what is needed for 42. I also
believe our tools may be unable to express what they're finding, and it is likely that we haven't
yet tokenized the right concepts." He offers player-behaviour history as one thing we were blind
to. It shows the model is incomplete; it is not a claim that history is the missing piece. "We can
say what we learned from this model. We can't say we learned all of 42. Yet. But we will."

The essay backs this. Most of its "mixed" neurons were the observer's fault:
- short contexts made a comma neuron look polysemantic, until a longer one showed it tracked Colorado;
- narrow data hid another neuron's job;
- a tool that looked only backward missed what the Titanic neuron predicted;
- researchers "looked for the wrong kinds of things".

In Lem, the first three rungs of the essay's ladder have concrete counterparts:
- **the vocabulary** is the 70-concept Silas list;
- **the short context** is the field net's input: one position, in one hand, against uniform beliefs;
- **the narrow data** is the arena, which seats no one who models you across hands ([[book-assumptions]]).

This rhymes with Zeb, which is unbeatable until you exploit its uniform beliefs. A vocabulary as
expressive as [[silas]] can read as complete over the wrong inputs. So every Lem result states its
scope as "this net, these inputs, this vocabulary". A unit we cannot name is reported as "not named
in list vN", never as mixed. The inventory of what is missing is #202.

## Why it fits 42

The game's concepts generalize recursively in exactly this shape.
- **Mastery is recursive.** The double is master of its suit. Once it falls, the highest unplayed
  tile of the suit becomes master, and so on down. "Is the master" is one concept that re-anchors
  as tiles leave the game.
- **Count generalizes.** The 5-5 and 6-4 are worth ten, and the 5-0, 4-1 and 3-2 are worth five.
  "This trick carries count" is one concept with a facet for each count tile.
- **Voids propagate.** "Out of a suit" is learned from a failure to follow. It extends to "out of
  trumps", "can ruff" and "partner can ruff".
- **The suit algebra composes them** ([[texas-42]]). A tile's role depends on trump, on the suit
  led and on what is gone.

If a net trained on 42 grows neurons this way, its neurons *are* the game's concepts, and we can read
Zeb's reasons straight off them.

## Two experiments, cheap one first

**A. Audit the net Zeb already ships (this page).** The model organism is
`zeb/net/margin_d250k_h128x64.fnet`, the CC0 field net behind the published player. Its hidden
layers are 128 and 64 wide: 192 neurons and 384 poles, small enough to audit every one.
1. **Audit in place.** For each pole, take its top-activating positions from the 250k pour (public,
   CC0, on HF), then label it two ways:
   - *exactly*, by its correlation with [[silas]] predicates over the viewer's information (holds
     the master, count on the table, a seat's void, partner winning, trumps left);
   - *cheaply*, by Haiku readers working from short position cards, kept low in context and run
     many at a time.

   Score the share of poles with one main job, and name what the rest mix. Report every share with
   its interval, against a shuffled-label baseline as chance.
2. **Watch neurons form.** Retrain the same small net from the pour, keeping checkpoints. Test
   whether concepts branch as the essay predicts: "trump double", then "highest unplayed trump",
   then "master of any suit". Superposition and recursive generalization predict different growth:
   unrelated concepts sharing a neuron, or one concept spreading along its facets.
3. **Squeeze.** Shrink the hidden layer and ask whether the concepts get coarser or get mixed.
4. **Meet the book.** Match the clean neurons against [[w42]]'s concepts.
   - Where a neuron and a book rule coincide, Zeb holds the book's idea.
   - Where the book names a concept and no neuron carries it, either the net never needed it or the
     arena never asked for it ([[book-authority]], [[book-assumptions]]).

**B. Train a GPT on 42 and read it** ([#175](https://github.com/jasonyandell/mk5-main/issues/175)).
A transformer over the mover's information state, with the 43 margins as its vocabulary and Walt's
weighted per-world margins as its soft labels. It needs more build and more compute, and its
neurons are richer. Experiment A's concept list and probes carry straight over, so A is B's
pre-registration rehearsal.

**Fence.** Every probe and label reads the viewer's information only. Clairvoyant per-world values
never label a concept ([[strategy-fusion-fence]]).

**Cost.** Experiment A is CPU only, minutes per probe through `scripts/cpu-slot`, with Haiku doing
the bulk labelling.

## Prior work here

- [[gus-probe]] and [[gus-strategy-tags-probe]] were the oracle era's interpretability receipts. The
  tags probe found that the book's public-state tags help a tiny model, and it grew into the [[w42]]
  workstream. Those nets learned from the oracle, so their concepts carry its clairvoyance
  ([[the-wall]]). Zeb's field net does not.
- [[silas]] is the exact vocabulary the audit labels with.

## Status

Absorbed into the Lem program as its stage 1 ([[lem-the-teacher]], #176). The audit of the 384
poles (A1) is pre-registered in [[lem-field-net-audit-2026-10-15]], with the concept list, the
statistics, the Haiku readers and a first neuron-count curve; the probe kit is `lem/probe/`.
A2–A4 and experiment B are stages 3 and 4 of that program. **A1's result (2026-10-16):** 0.597
[0.581, 0.608] of live poles clean against a chance of 0.000, in the essay's direction though below its
"about 90%"; the play itself needs 165 of 192 neurons for 0.95 fidelity.

**The squeeze (A3), 2026-10-17:** narrow nets cleaner, not mixed (0.61 → 0.83 at 24 neurons; [[lem-squeeze-2026-10-16]]). The follow-ups are THE ALREADY-CLEAN TEST, #193 (#194–#200), the core thesis of [[lem-the-teacher]].
