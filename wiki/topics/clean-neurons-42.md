---
title: Clean neurons for 42 — reading Zeb's own net, concept by concept, as it grows
kind: topic
first_seen: 2026-10-15
last_updated: 2026-10-20
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

## The work: what a day of Lem looks like

This is the part of the card the after-compact hook reloads, beside the essay's own words. It
says what to do, in the essay's terms. The guardrails further down say what to watch for.

**The stance.** We are naturalists in a rainforest, not a court. The network has already prepared
itself to be read by its own downstream readers; our job is to become one more reader. The
network is the answer, and the work is finding the right questions to ask it.

**The day's work is the essay's loop, run on one unit at a time:**
1. **Label a unit.** Both poles, top activations on held-out rows, its untrained twin beside it.
   Read it as a gate: what it lets through, what it keeps out, what it hands to whom. Look for
   the shared pattern (a V, a pose, "late in the trick") before a list of nouns.
2. **Follow the data through its circuit.** The inputs that vote for each pole, the readers that
   take each pole downstream, and where a position like this one goes next.
3. **Find the motif.** Gates, hold-backs, summary units, edges where one distinction feeds the
   next. A motif that recurs across units and nets is worth more than any one label.
4. **Watch it grow.** Formation across checkpoints: where it started, which facet it sprouted
   next, which reader came to depend on it.
5. **Test the explanation by using it.** Steer with matched doses and read where play goes.
   An explanation that can lose, and does not, is the one we keep.
6. **When it looks messy, look harder.** The noise usually points to the explanation: a missing
   word in the list, a missing input, a facet not yet followed. Write which one you checked.

**The artifact is a unit paper**, like the essay's thousand neuron papers for Qwen: one page per
unit, carrying the label, both poles, the circuit in and out, formation, the steering check, the
scope clause, and what it taught us about 42. Registrations and predictions still come before
runs; the question they ask is "what does this unit do, and how did it come to do it?"

**Read the source, not this card.** Before planning Lem work, and after every compaction, read
the essay's own text (the local copy, whole). This card is our summary, and summaries are what
drift.

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

Where 42 stands against it, as of 2026-10-20 (the instrument and the framing both changed that day):
- **The instrument.** Read raw activations and take the play count out in the labeller: a label counts
  only if it beats what the play count alone predicts ([[lem-fixed-conditioning-2026-10-20]], #233).
  Z-scoring activations within each play count was tried and retired, because it amplified near-silent
  strata. Every share is read against untrained twins of the same architecture, under the same
  instrument.
- **The squeeze, against list v2.** 24 neurons average 0.695 clean, against 0.542 at 192, over three
  seeds. One seed pair reverses, and the floors are 0.375 and 0.276. **The essay predicts squeezed
  units grow coarser, not cleaner**, and that bigger nets have narrower neurons. So "narrow reads
  cleaner" measures fit to our 70 concepts. It is not the thesis winning.
- **Coarser or mixed, the live question.** Read by how they grew ([[lem-unit-origin-2026-10-20]]), most
  broadened units sprouted their second concept or were born coarse, and something downstream reads
  the union. The two mixed-unit candidates were artifacts of the instrument
  ([[lem-lead-units-2026-10-20]]). Family-based statistics failed their own calibration and are
  withdrawn ([[lem-family-statistic-2026-10-20]], #229). "Family kept" is mostly the same label
  persisting.
- **Use.** Ablating a named unit hurts the play where its name says, 45 of 45
  ([[lem-causal-2026-10-17]]). Steering, read by cost, is null.
- **Gus**, a transformer, reads 0.312 against a floor of 0.014. Its units' words come with data: from one
  start, 4k deals form none, 50k form some and 500k more ([[lem-words-need-data-2026-10-20]]).
- **Process.** Every registration is reviewed by a fresh reader of the essay's full text before
  sealing (the clean-neuron skill).

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

**The unit papers (#235), 2026-10-20 to 21: nine units of the 16×8 seed-0 net, read whole.** Each was
registered after a raw read, reviewed against the essay's full text by a fresh reader, sealed, and
run. Together they read one circuit: how the net decides what to lead and what to throw.

| paper | unit | its verb, as found | its use (ablated where on, ranked against the layer on the same rows) | held |
|---|---|---|---|---:|
| [[lem-unit-partner-has-it-2026-10-20]] | L1 5 | "I play last", a dial on who has the trick | shares its push with the layer (beats it by 0.005) | 4 / 9 |
| [[lem-unit-big-double-2026-10-20]] | L1 13 | a gate shut when I hold the slot-14 double; its silence plays it | steered: −0.318 beyond the layer | 2 / 5 |
| [[lem-unit-lead-brake-2026-10-20]] | L1 0 | "lead the double, not the 6-5"; the counter-route of 13's bit | steered: −0.378 (6-5) / +0.431 (double) on leads | 0 / 5 |
| [[lem-unit-lead-doubles-2026-10-20]] | L2 3 | "lead an off double", dimmed by trump length | off-double leads −0.686 of 0.757, rank 1 of 8 | 3 / 5 |
| [[lem-unit-no-trump-yet-2026-10-20]] | L1 7 | a counter of trumps on the trick; open means trumping in is open | trumping in −0.411 of 0.797, rank 1 of 16 | 1 / 5 |
| [[lem-unit-theirs-2026-10-20]] | L1 14 | "not partner's trick; keep your count off it" | count given away 0.044 → 0.426, rank 1 of 16 | 3 / 5 |
| [[lem-unit-pull-trump-2026-10-20]] | L1 8 | a dial on trump still out against me, cut by the lead | forced on, trump leads −0.330 beyond the layer; ablated +0.235, rank 2 of 16 | 3 / 5 |
| [[lem-unit-count-on-it-2026-10-21]] | L2 4 | "feed partner's trick", a dial on who holds it; its tail is an opponent's trump | count on partner's trick 0.669 → 0.086 under ones–fours, rank 1 of 8 | 4 / 6 |
| [[lem-unit-top-double-2026-10-21]] | L2 1 | "lead the top off double" under fives and sixes, gated by "slot 14 holds no ten"; on sixes follows it lays the 5-5 on partner's trick | top-double leads 0.633 → 0.197, rank 1 of 8 | 3 / 3 graded (1 confirmatory missed) |

What carried across them:
- **Read steering by where play goes, against the layer on the same rows.** A unit tied to one
  tile or one decision showed a use specific to it (papers 2 and 4–6). Paper 1's unit, which reads
  a situation, did not.
- **The pole is not the job.** Read R names the deepest 1% (paper 5's `trump-led`, paper 1's
  `partner-winning`). The unit's decision lives at its on/off boundary, which can be a concept
  list v2 lacks (#236, #202).
- **Layer-1 units are graded sums.** Papers 5 and 6 missed predictions by treating them as bits.
  Read them in mean z with intervals, and grade only co-occurrence, use and growth.
- **Growth was never two concepts merging.**
  - Units died and were reborn as gates (paper 3).
  - Units were born whole (paper 4).
  - Facets were present at initialisation and refined by a late gate (paper 6).
- **When two predictions contradict, suspect the premise they share** (paper 3).
- **A unit's job can be split by declaration in the net's own frame** (paper 8). The push is the
  same in canonical slots, and count lands under it only where count sits in those slots. Another
  unit covers the other declarations: composition, not packing.
- **A unit can correct the layer** (paper 9). Three layer-2 units vote for slot 14's double, which is
  the 5-5, a ten, under blanks–fours. One unit, gated open where slot 14 holds no ten, votes for the
  top off double instead. Its gate is a pattern in the net's frame, not a list of declarations.
- **A branch grows from the push, later** (paper 9). The same push that picks the double on a lead
  lays the 5-5 on partner's trick on a sixes follow. That job reached half its final rate about
  120,000 steps after the lead job.
- **A rare off-set is a dial's tail, not a boundary** (papers 7–8). Its κ against a constant reads
  0.00 because the instrument is wrong; a graded word fits it (paper 8: R² 0.640).
- None of the nine read as mixing. Scope: one net, one position, uniform beliefs, no history.
