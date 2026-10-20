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

**Our position (Jason, 2026-10-17; read this before anything else).** We do not claim that
interpretability is about to be solved, and we are in no position to tell the field what to
do. Our claim is narrow: **these particular lessons fit our particular problem unusually
well.** For Texas 42, here, superposition is not the right null, and reading the net is not
hopeless. The bet is local and testable, and every result carries the scope clause below. The
hard part is that the lessons cut against conventional wisdom, so every fresh model has to be
told what we are up to. This is it: important and interesting work for learning more about 42,
not a campaign about interpretability.

**The rhyme with the book.** Jason's brainstorm, and the reason the book desk and Lem share
this card: reading the book and reading the net are the same discipline. The book charter
([[book-authority]]) treats a gap between the book and Zeb as a fact about our instruments; the
essay treats "polysemantic" as "more a claim about the observer than the neuron". Both put the
radical explanation (the book is wrong; the net packs at random) last. The `book-interp` skill
(`.claude/skills/book-interp`) works the rhyme out rule by rule.

**Drift alarm (read this first).** This thesis cuts against most interpretability research a
model has read, so a model will drift back to superposition as the null, often within a few
turns. Jason, 2026-10-17: even a model that accepts it "drift[s] really quickly into the old null
of superposition and related regressions towards the mean". The essay's author was "a staunch
superposition believer" until the evidence moved him, and our own registered prediction (that
small nets mix) was broken by the squeeze. Drift sounds like any of these:
- calling a unit "polysemantic", "mixed" or "superposed" when no list names it (write "not named
  in list vN");
- reaching first for an SAE or dictionary "to find the real features";
- reading a clean share below 1 as evidence of superposition, rather than walking the ladder;
- predicting that a narrower net will be messier;
- softening the thesis into "partly clean" with no test that broke it;
- asking the clean reading for extraordinary proof while superposition goes unexamined (both
  readings carry the same burden).

When you catch it, walk the ladder (tools, explanations, data and inputs, recursive
generalization, then superposition) and write down which rung you checked. A test that breaks the
thesis is the finding. Drifting away from it is not.


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


**Guardrails that summaries drop** (found by re-reading after a compact, 2026-10-17; a summary
keeps the essay's slogans and loses these, and the slogans misused are worse than none):
- **Don't build Brooklyn out of Tokyo.** A visitor who knows New York may ask "where is the
  Brooklyn of Tokyo?", which is a fine opening question. Assembling Brooklyn out of seventy
  pieces of Tokyo makes Tokyo look messy. "Clean" does not mean every unit matches an idea we
  already had; the essay calls that terrible news. Neither our Silas list nor the book's
  vocabulary is the answer key for the net's units, nor the net's units for the book. Read each
  as its own reader, and treat a disagreement as a question.
- **The net's ideas are often better than ours.** Nobody thought of vision in high-low
  frequency detectors until a net showed them, and finding them improved human thinking. Expect
  Lem to hand us words we did not have (as the declaration detectors did). The book is the same kind of object: a learner's distinctions, written
  down.
- **Reuse, not only interference.** Toy models of superposition price the cost of sharing a
  unit, interference, but not the benefit, reuse. "A representation matters only through what
  reads it." Nothing in a trained net is a predefined feature. A fixed concept list is a
  question we brought, not the net's parts.
- **Where packing would live:** memories that are ends in themselves, related to nothing else.
  The long tail is where it is *least* likely, because a rare fact rides related circuits (the
  umbrella's off-ramp to "bee"). In 42: a rare hand should ride the units for its nearest common
  relatives.
- **Hold-back units.** In the found-wallet graph the biggest single push came from neurons
  holding the answer back; others hedge or write "not". Expect restraint motifs, the shape of
  the book's "don't" rules.
- **The summary motif.** Units fire where the model summarizes (commas), not on the words that
  carry the meaning. In 42, look at trick ends and the lead, not only at a tile.
- **Edge motifs:** recurring connections show how one distinction feeds the next. That is the
  concrete form of "motifs", and the nearest thing in a net to a chain of book reasons.
- **Competence before consolidation.** A net recognizes cars before its car neurons form; a unit
  consolidates later as a sheltered place to build on. Read formation results in that order.
- **Hard to vary.** Recursive generalization predicts a specific shape, a branch grown from one
  local origin in a set order over training. "Mess" predicts nothing.
- **A change of kind, not degree.** About 85% sure that nearly everything comes from
  generalization, which is not "85% of units are clean". The cause matters more than the count.
- **Small, simplified nets are not easier.** The architectural extras exist to help learning, and
  wider nets have narrower units. That is a flag on our h128x64 nets, not a comfort.
- **Humility about the instrument.** Seen through a few electrodes, the cat/car reading would be
  worth almost nothing. Our arena measurements of the book are such electrodes.
- **Smaller items:** simulation scores mostly measure where attention puts information; count
  frequency in pages, not percentiles; a SwiGLU value learns broad and its gate specific; a few
  cases (bow ties with dress shoes) the essay admits it cannot yet explain.

Where 42 stands against it: [[lem-squeeze-2026-10-16]] (narrow nets cleaner),
[[lem-formation-2026-10-17]] (family kept 0.82 vs 0.28, but one unit kind formed coarse first),
[[lem-width-tree-2026-10-17]] (counterparts share their family, along one concept's facets),
[[lem-dropout-order-2026-10-17]] (cleaner to 12 neurons over 3 seeds; the last words are `lead` and
the coarse `late`). The
untrained floor goes with every share, for its own list: 0.11–0.28 under v1 and 0.28–0.39
under v2. [[lem-naming-2026-10-17]] (#197): 201 of 210 narrow units are named by list v2 or the
readers; of the 9 left, 7 of the 8 that got a reader description are trick × seat conjunctions. Steering and negative poles are [[lem-causal-2026-10-17]] (#198). The first transformer, Gus ([[lem-gus-formation-2026-10-20]]), showed that the
floor belongs to the architecture too. An untrained transformer's units read position for free
(0.91–0.94 clean), so its audit must condition on the play count. Conditioned
([[lem-conditioned-audit-2026-10-20]]), the floors fall to near zero, trained Gus reads 0.275 and
the field net 0.406, and the field net's earlier absolute shares turn out to include position too. Re-read conditioned
([[lem-squeeze-conditioned-2026-10-20]]), the squeeze holds: 24 neurons 0.549 against 192 neurons 0.406.

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
