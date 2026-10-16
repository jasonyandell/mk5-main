---
title: Recursive generalization — learning reusable 42 concepts through ordinary training
kind: topic
first_seen: 2026-10-15
last_updated: 2026-10-15
status: active
---

<!-- SPDX-FileCopyrightText: 2026 Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Question and status

**Can ordinary, information-honest training produce reusable internal mechanisms for
Texas 42, and can those mechanisms generalize by extending and composing relationships
that the game's mathematics already expresses?**

This is an **IDEATED, unrun research proposal**, prompted by Jason's 2026-10-15 reading of
Nick Cammarata's Goodfire essay. It is not a trained model, a measured improvement, or a
claim that a particular neuron already represents a 42 concept. The proposed instrument
is a checkpointed neural player; better play remains the eventual test, not attractive
explanations. [[scenario-player]] supplies the current planning semantics; [[silas]]
supplies checkable descriptions and legal counterexamples.

## External motivation, bounded

Cammarata proposes that apparently mixed neuron responses often develop through locally
related extensions during training. His examples motivate inspecting native neurons and
circuits, including their development, rather than assuming an initial decomposition is
necessary. The essay explicitly presents a working hypothesis, not a universal result.
Goodfire lists its publication date as October 15, 2026. [1, 2]

The contrasting literature matters: toy models demonstrate superposition in their
specified settings, and dictionary-learning research supplies an alternative unit of
analysis. Neither establishes the representation a 42 player will learn. Native-neuron
inspection is the first experiment here, not a commitment to deny distributed
representations or to forbid sparse autoencoders. [3, 4]

## The proposed connection to the mathematics

The connection is **shared relational structure**, not a claim that learning dynamics and
game-tree recursion are the same operation.

[[suit-algebra-spec]] separates suit membership, following, and trick order. [[silas]]
adds ownership, quantified relationships, observations, and belief-conditioned queries.
[[scenario-player]] makes a single contingent plan across worlds that the acting seat
cannot distinguish. Together they suggest a possible progression of learned mechanisms:

    relative order in a suit
        → which opposing tiles can still defeat a candidate
        → which legal plays can remove those threats
        → which resulting winners can actually be reached
        → how those continuations preserve count or make the contract

These are **candidate descriptions**, not a training vocabulary. The experiment should
allow a model to group situations by a useful relationship that has no established name.
A tile's printed identity is one possible cue; its role in a continuation is another.

For example, two otherwise different positions might both favor retaining a tile because
spending it removes a later route to winning a count-bearing trick. That is a proposed
family, not a universal rule: following obligations, trump, seat order, partner behavior,
and the contract can destroy the analogy. Constructing both the family and its exceptions
is more informative than labeling every strong activation "protection."

Three distinct claims must stay separate:

| Claim | Evidence required |
|---|---|
| The game admits relational descriptions | Existing algebra and executable predicates |
| A trained model reuses a relationship across situations | Held-out transfer and controlled legal counterexamples |
| Its mechanism grows through successive related extensions | Training-checkpoint evidence plus causal tests; a final activation cluster is insufficient |

A finite game admits a complete deterministic description. The target is therefore not
"prove features cannot describe 42," nor an exact small quotient of every position. It is
**useful reusable structure without pretending that structure removes every contextual
exception**. The prior [[endgame-equivalence-census]] is a reason to test approximate,
policy-conditioned reuse rather than assert exact hidden-world interchangeability.

## Learning target: a seat's decision, not hindsight

The [[strategy-fusion-fence]] is binding. No perfect-information oracle values, no
per-world clairvoyant best replies averaged into labels, and no hidden hands in the
student's input.

Fix an acting seat, its information state I, a declared belief mu_I over compatible worlds,
and frozen information-honest policies F for the other seats. For a legal root action a,
the ideal reference object is

    Q_F(I, a) = sup over pi in Pi_I(a) E[u(Z) | mu_I, pi, F].

Here Pi_I(a) contains legal acting-seat continuation policies that start with a and make
one choice at each later information set; Z is the terminal record; u is the acting
team's contract-success indicator. F includes the partner: this is a unilateral planning
target under a stated field, not a claim of globally optimal partnership play.

The actual training label is **q_hat_Walt(I, a)**, the pinned finite-budget teacher's
estimate for each legal action, not Q_F itself and not a game-theoretic optimum. Record
belief construction, field identity, rung, sample sizes, horizon, floor, utility, seed,
and tie rule. A deliberately different utility, such as margin, is a separate arm.
Repeated teacher seeds characterize label instability rather than silently changing the
target. Validation must use fresh teacher randomness, not the samples used to fit labels.

Train a vector of legal-action make probabilities with masked binary cross-entropy or
squared error; choose the action by argmax under a recorded tie rule. **Prediction and
selection are different measurements.** Low value error alone does not establish good
choices. In particular, equal-looking actions should not be counted as decisive mistakes
when the independent teacher cannot resolve their difference.

A hidden-world predicate may be used for offline diagnostics, but its realized truth is
not information the seat can learn to recover when I is unchanged. A relevant observable
target is its probability under mu_I, with Silas's authority attached. Holding the encoded
seat information fixed while replacing the actual compatible hidden world must leave
both the student's output and its activations unchanged.

## Ordinary training first; capacity is an experimental variable

**Start with the existing field-net training lane, not a new research framework.** Its
implementation and prior results are documented at [[scenario-player]]. Inspect the
existing checkpoint as a feasibility check, then retain intermediate checkpoints in a
controlled retraining run. A final checkpoint can support interpretation; it cannot
reconstruct a mechanism's training history.

Use ordinary outcome/action-value supervision in the primary arm. Do not train on labels
named guard, walker, control, protection, or any proposed concept. Do not feed those
predicates in as features. Otherwise the experiment measures learning a supplied ontology,
not discovering a useful one.

The input must be a documented function of the seat's information: own tiles, the public
play record, contract/declaration, acting seat, and the score needed by the utility.
Include auction or match history only in arms whose teacher uses them; otherwise state
that scope. Unknown ownership is unknown, never filled from the generating deal.
Retain the distinctions needed for rank, count, order, and legal actions.

Proposed architecture comparison, not selected winners:

- The current small field network as the baseline, with its exact feature layout recorded.
- A wider residual MLP or tile-and-history transformer, provisionally 1–5 million parameters,
  trained on the same labels. The range is a starting budget, not an inferred requirement.
- A matched rule-feature arm only as an ablation. Features supplied by the encoder cannot
  subsequently be counted as discoveries of the network.

A single tiny model's failure is not a capacity-independent verdict. Conversely, extra
parameters are not evidence of understanding. Compare both equal-data and equal-compute
runs and report all attempted seeds. Keep the training distribution stationary in the
first mechanism-growth test; a backwards curriculum would confound spontaneous emergence
with deliberately staged exposure. Curriculum can be a separate later arm.

## Silas as the experimental workbench

The intended loop is

    train on decisions → inspect a candidate mechanism → state a hypothesis
        → construct legal tests and counterexamples → measure transfer and causal use.

The query-conditioned constructor is **implemented on an unmerged branch**, not assumed
available on main: PR #166, `codex/silas-world-constructor`, inspected at
`c05b8869392b8783fe660704eaef533bf7cd9e33`. Its reported `WorldSpace` and `OpeningSpace`
interfaces count disjoint ownership cells and support unranking and sampling. Its own
validation claims remain that PR's receipts, not results of this proposal. [5]

For a supported query q, the desired population is the compatible worlds in which there
exists a satisfying answer. Each world must be counted once, **not once per satisfying
role assignment**. Unsupported predicates, undefined answers, impossible histories, and
empty populations must remain explicit. Distinct-role constraints and existential
projection are part of the test, not convenient details to discard.

The constructor alone is not a training-distribution specification. Sample a position
family, a viewer/frame, and then its hidden completions under declared weights. Uniform
worlds within one frame do not imply uniform frames or uniform strategic families.
Conditioning on a hidden event for an experiment changes the population; it does not
entitle the acting model to know that event. Generate ordinary training records under the
declared information model and use targeted families primarily for held-out diagnostics.

Construct two kinds of challenge:

**Transfer:** preserve a candidate relation while changing tile identities, partner and
opponent holdings, the public record, or its embedding in a longer continuation.

**Boundary:** break the relation while retaining matched superficial statistics such as
hand size, count, and trump length where legal construction permits. Record unavoidable
co-changes; a legal matched pair is not automatically an isolated causal intervention.
Replay every example. One frame's exact query is a fact about that frame, not a universal
statement about all hands.

Do not use arbitrary pip permutations as game-preserving augmentation. Suit-incidence
symmetries alone do not preserve the full ranked, scored game; [[play-phase-algebra]]
explicitly distinguishes them. Recompute legality, outcomes, and teacher labels after
changes unless a full relevant symmetry has been proved.

## What the microscope must record

Keep model and optimizer checkpoints, training-step/data-order metadata, a frozen probe
bank, and the exact encoding and teacher manifests. For selected native hidden units,
record pre- and post-nonlinearity activations and downstream contributions. In gated
architectures also inspect gate, value, and their product; in residual networks inspect
small circuits or subspaces when an individual coordinate is insufficient.

Retrieve strong activations, weak activations, opposite-sign examples where meaningful,
and matched nonactivating controls. Display complete legal positions and continuation
contexts, not only the most salient domino. Discover candidate relationships on a
separate discovery split; freeze their definitions before confirmation. Silas can then
formalize a candidate or supply a counterexample. A concept need not be expressible by
the current predicate library to merit study, but it still needs a testable definition.

Use interventions as well as probes: conditional-mean ablation, activation patching
between matched legal examples, and targeted circuit interventions. Compare with random
units matched for layer, activation scale, and connectivity. Zeroing an activation may
create an unnatural state; an isolated degradation is not sufficient evidence. A linear
probe shows decodability, not that the policy uses what it decodes.

Within a run, track both unit coordinates and their changing response sets. Across seeds,
match response signatures or circuit behavior, not neuron indices. A proposed A → B → C
lineage must predict new examples at each extension and survive held-out tests; neither
smooth training curves nor post-hoc renaming establishes ancestry.

## Evaluation and advancement gates

Split by generating deal and constructed family **before** making descendants, alternate
views, or paired interventions. Keep all related records in one split. Maintain an
ordinary-distribution holdout and a separate structural-transfer holdout. Define held-out
relation compositions explicitly; random row splitting is not a compositional test.

Report four separate outcomes:

| Outcome | Main measurement |
|---|---|
| Prediction | Held-out loss, calibration, and independent teacher uncertainty |
| Decisions | Teacher-relative regret: max_a q_hat_eval(I,a) minus q_hat_eval(I,a_student) |
| Mechanism | Frozen-hypothesis transfer, counterexamples, and selective causal intervention effects |
| Playing strength | Paired games against pinned policies, with interval and runtime |

Teacher-relative regret is not regret against an optimal 42 strategy. Use paired,
deal-clustered uncertainty estimates; many decisions from one deal are not independent
replications. Correct for searching many neurons and hypotheses, or reserve a genuinely
untouched confirmation split. Include untrained, label-shuffled, and random-unit controls.

**Gate 0 — instrument:** load a known net, reproduce its outputs, verify information
invariance and legality, save checkpoints/activations, and recover a deliberately planted
or explicitly encoded positive control. Passing this validates the instrument, not
concept discovery.

**Gate 1 — learn and transfer:** retain a useful decision predictor, then test whether a
candidate mechanism transfers beyond its discovery families. Capacity and optimization
failures must be distinguished from negative transfer results.

**Gate 2 — causal confirmation:** predeclare at most three candidate mechanisms and their
counterexamples; replicate on fresh families and training seeds. Three seeds are an
initial replication floor, not a power calculation. Advance a mechanistic claim only when
its interventions have selective, predicted effects beyond matched controls.

**Gate 3 — useful player:** compare the selected candidate with the unchanged baseline at
matched decision cost, and separately with increased cost. A readable circuit without
better play is an interpretability result, not a strength result. No player promotion
follows from this note.

Every heavy run uses `scripts/cpu-slot -- <command>` and its 300-second cap. Make bounded,
resumable chunks; a smoke test is not a completed training or significance claim. Publish
new generated data and checkpoints to Hugging Face with manifests and licenses; keep
source, small specifications, and evidence links in GitHub. See [[standing-desks]] and
[[publish-data-to-huggingface]].

## What would change the conclusion

A causal mechanism that transfers through several verified relational extensions would
support this 42 hypothesis. It would not settle interpretability for all neural networks.
A well-trained model with no single-neuron story would motivate circuit-level analysis,
not immediate failure. Transfer without selective causal effects would remain a
representation result. Correlation only, seed/deal memorization, or disappearance on
matched legal controls would reject the proposed explanation. Inability to fit the target
would leave the representation question unresolved.

The practical commitment is small: **preserve the course of ordinary training and test
what grows, rather than naming all the pieces before the network has a chance to learn.**

## Provenance, attribution, and license

This original research proposal is CC0-1.0, following the repository's [license
line](../../LICENSE). Copyright attribution follows the project's existing convention.
The linked Goodfire and Transformer Circuits publications are cited sources, not vendored
content. No source figures, website code, model weights, datasets, or substantial passages
are copied or relicensed here. This notice grants no rights in those external materials.
Any future imported artifact needs its own license/provenance check. Changes to Zeb's MIT
training/runtime files retain that license; this CC0 note does not relicense them.

Repository context was inspected at `2cfc371c87db439dd4f55f2512b0b2df494198bf`; the
constructor's separate status is recorded above. No experiment was run for this note.

1. Nick Cammarata. *Neural Networks Are Already Clean*. Goodfire, 2026.
   https://already-clean.goodfire.com/ (accessed 2026-10-15).
2. Goodfire research catalog, publication-date record:
   https://www.goodfire.com/research (accessed 2026-10-15).
3. Nelson Elhage et al. *Toy Models of Superposition*. Transformer Circuits, 2022.
   https://arxiv.org/abs/2209.10652
4. Trenton Bricken et al. *Towards Monosemanticity: Decomposing Language Models With
   Dictionary Learning*. Transformer Circuits, 2023.
   https://transformer-circuits.pub/2023/monosemantic-features/
5. *Silas: construct query-conditioned worlds and opening families*, PR #166.
   https://github.com/jasonyandell/mk5-main/pull/166 (open and unmerged when inspected).

## Links

[[scenario-player]] · [[silas]] · [[suit-algebra]] · [[suit-algebra-spec]] ·
[[play-phase-algebra]] · [[endgame-equivalence-census]] · [[strategy-fusion-fence]]
