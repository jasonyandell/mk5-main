---
title: "Lem with nanoGPT: history, a population, and models that can teach"
kind: topic
first_seen: 2026-10-17
last_updated: 2026-10-17
status: active
---

## Purpose and status

The nanoGPT workstream implements stage 3 of [[lem-the-teacher]] (#178), with
the population (#177, #182) and [[clean-neurons-42]] as its experimental setting.
Jason commissioned it on 2026-10-17, including an Astra review of the mathematics
in `texas-42`, its intakes and outboxes, and its Lean sources. Learning enough to
explain every experiment is part of the deliverable.

Jason selected the observer-first lesson: [[lem-observer-2026-10-17]] implements
one tiny model, within-hand history, existing Pour A records and a worked-hand
viewer. Population construction and reward learning are deferred. Field-net
results are not GPT results.

The extended run completed 200,000 updates and selected step 4,000 by validation;
[[lem-observer-overbake-2026-10-17]] records its overfitting, frozen evidence and
interpretability handoff. Jason chose to leave unit reading to the interp work
and continue here with [[lem-observer-data-scale-2026-10-17]]: the same small
architecture, more observed deals, and comparisons at matched update counts.
That 50,000-deal run is complete: at 20,000 updates it gains 9.92 percentage
points [9.72, 10.11] in next-tile accuracy over the original corpus on 4,436
fresh shared test deals. The experiment scopes the result and preserves the
second formation trajectory for the interp handoff (#207).

The checkout is `/Users/jason/code/mk5-lem-nanogpt`, branch `codex/lem-nanogpt`,
initially at `aa0af91d`. The reference clone is `/Users/jason/code/nanoGPT`, upstream
[`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT), pinned at
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. Its README now recommends nanochat;
nanoGPT is retained for its compact implementation. Its MIT copyright and license
must accompany substantial copied code; the default CC0 annotation cannot
silently relicense it.

## What the book exposes

[[book-assumptions]] distinguishes remembering opponents (A1), partnership
conventions (A2), fallibility (A3), meaningful auctions (A4), styles (A6), and
responses to predictability (A9). Each needs a different experiment. A table
with random mistakes does not itself contain anyone who learns to exploit them.
Partners belong in the population: a convention needs a sender and a receiver.

Rules determine what can happen. A behavioral model determines how likely a
player is to choose among those possibilities. A population supplies differing
behavior; repeated encounters supply evidence for recognizing it. Finite history
does not identify arbitrary unseen policies.

An explicit player-model parameter is useful for a controlled experiment. A net
can instead infer behavior implicitly from its legal history. The requirement is
sensitivity to relevant evidence, not a mandatory type token. A hidden type can
be an evaluation label, but not an input to an arm described as inferring an
unknown player.

**Current-source correction:** `silas/GAPS.md` section 1 records that
`ReadingPredicate`, `(reading ...)`, and seated `crisis-read` are built (#128,
#162; [[crisis-read-reprice-2026-10-14]]). Older A2 prose still says guards cannot
read beliefs. Remaining examples include another seat's reading of the viewer's
decoy, reciprocal convention readers that would read themselves, and auction
readings needing context absent from a play frame. Name the missing construction;
"behavior cannot be expressed" is too broad.

## Mathematical foundation and proof boundary

[[two-recursions-and-transformers]] compares the mathematics' forward information
update and backward planning with the observer's actual attention and autoregressive
loss. It identifies the exact predictive-coordinate connection and the conditions
that next-tile prediction has not established.

The `texas-42` review is pinned to `9d6a5a2e`. Its controlling
[model-belief intake](https://github.com/jasonyandell/texas-42/blob/9d6a5a2e/walt/math/model_belief_base_player_v0.1_intake.md)
narrows the parent proposal where they differ. It adopts hidden state
`Xi = worlds × player types (× persistent random state)`: uncertainty includes
both the deal and the behavioral rules. The same observed action updates the
joint belief about both. Types persist over their declared horizon; resampling
one each action describes a different game. A shared convention can correlate
partners' types, so independence is an assumption to test, not a default theorem.

For a specified family, a schematic update is
`new belief(world,type) ∝ old belief(world,type) × P(observed action | world,type,history)`.
The actor's likelihood is evaluated from that actor's own information in the
candidate world. The viewer's next choice remains shared across possibilities it
cannot distinguish: merge public-action branches before maximizing. The intake
requires residual `Other` coverage for an incomplete model family and distinguishes
a Bayesian mixture, a set of plausible models, and an action-wise uncertainty
envelope. These are different objects with different guarantees.

The intake's reported finite rational checks and step-checked theorems are
historical evidence, not a new Lean certification. Its common-information
partnership bridge is explicitly future architecture. The rules alone do not
determine human behavior; exact answers under specified behavior are still possible.
This is not all proposal: `texas-42/walt/MAP.md` and `walt/FACTOR-BELIEF.md`
record experimental exact late-game mixtures and a positive model-fusion price.
They do not establish a general learned population player. The strategic-state
account retains mechanical state, continuation history and a joint physical/type
belief; dropping history or factoring that belief needs an additional argument.

The Lean audit inspected source and existing build artifacts, without rebuilding:

| Source in `texas-42/lean/Texas42/` | Inspected result | Boundary |
|---|---|---|
| `Information.lean`, `mech_not_injective` | Distinct histories can have the same mechanical projection | Mechanical state is not automatically sufficient information |
| `Witness.lean`, `ninety_world_witness`, `posterior_action_reversal` | Same 90-world support, different weights, reversed action ordering | Specified likelihoods and lowest-legal continuation, not equilibrium |
| `Belief.lean`, `likelihoodFrom_append`, `physicalBelief_support_isWorld` | History likelihood composition and posterior support inside the physical fiber | The kernel type itself does not enforce every behavioral-model obligation |
| `Strategic.lean`, `beliefVal_eq_exp_latentVal`, `bestResponse_eq` | Belief recursion equals latent expectation for a supplied process and strategy class | Concrete Straight-42 integration remains an explicit proof-plan obligation |
| `Trick1PerfectRecallNet/Recurrence.lean`, `sumBeforeMax_eq_freeProduct` | Abstract information-key policy algebra | Concrete replay/kernel correspondence is separate; fresh compilation was not checked |

M3 provenance needs reconciliation: its wiki reports an abstract gate complete,
but the last source commit `97ce321a` says the scaffold does not build and no
local M3 build artifacts were found. This is a provenance discrepancy, not a fresh
build failure. Current mk5 `scenario/lean/Stab/Rank.lean` proves finite settling
under stated dependencies; `Coherence.lean` bounds belief-shift regret; `Cycle.lean`
gives an abstract cyclic-response counterexample. None is a general self-play or
partnership-optimality theorem.

The outboxes reinforce the evidence boundary. `exchange/outbox/013-constellation-lean-stage1.md`
asks for source first and local compilation afterward; a requested theorem is not
a completed proof. `016-cheap-upper-witness-handoff.md` separates a lawful policy
lower witness from an information-relaxed upper. `017-second-rung-gluing-handoff.md`
records two repaired shortcuts: even random-legal play induces nonuniform belief
through differing legal-set sizes, and later arrival masses depend on earlier
focal actions. History and the distribution of reached positions are policy-dependent.
`018-fee-correlation-update.md` explicitly scopes its fixed uniform-field work as
exploratory. These upper-bound instruments are not learning targets; the
[[strategy-fusion-fence]] remains in force.

## Data available now

[[lem-pour-a-2026-10-16]] and the earlier
[`texas-42-zeb-pour-250k`](https://huggingface.co/datasets/jasonyandell/texas-42-zeb-pour-250k)
are public training sources. Pour A's card, raw array headers, exporter and
published card were checked on 2026-10-17.

| Object | Available information | Scope |
|---|---|---|
| Pour deals / raw plays | All 28 plays, initial deal, declaration, result | A replay source for lawful viewer histories; hidden hands are not model inputs |
| Pour positions | Mover's remaining hand, public prefix, legal/chosen actions, estimates | Unforced decisions, not complete population encounters |
| Pour splats | Weighted outcomes for each candidate under the planner's continuation | Model-relative targets, sampled except where enumerated, never clairvoyant values |
| [[tournament]] | Pinned participants, complete-game records, auctions and plays | Infrastructure to audit for a population corpus; mixed-partner training is not established |

Pour A has seat 0 bidding 30, no auction, independent deals and two
declaration-dependent field settings. It has within-hand history, but no persistent
cross-hand identities or deliberate population of styles. Its title says ten
declarations; the actual data covers nine playable declarations, 0–7 and 9.
Declaration 8 is absent. Preserve coverage-forced labels and separate metrics.

Its manifests say `utility=make`. `scenario/core/src/planner.rs::settled` stops
valuing the continuation once that contract is decided. Final margins must not
be presented as optimal points play or the results of playing to a different
bid. A 43-bin margin head needs an explicit answer to #175's target question.

Sources: `scenario/atlas.py::Pour`, `Position`; `scenario/hf.py::tables`;
`lem/pour/a.py`; `tournament/faceoff.py`; main checkout's
`scratch/lem/pour-a/card.json`, `merged/{pip,dtnt}/manifest.json` and array headers;
public [Pour A](https://huggingface.co/datasets/jasonyandell/texas-42-lem-pour-a).

## Three objectives

| Lesson | Training question | What success does not establish |
|---|---|---|
| Observer | From this viewer's information, what will the next seat play? | That the predicted move is good |
| Teacher student | What value/distribution does a named planner assign each candidate? | A universal value independent of the modeled table |
| Improving player | Which lawful policy earns the chosen return against this population? | Equilibrium or optimal partnership |

nanoGPT supplies next-token prediction. For observed action `a`, loss is
`-log p(a | visible prefix)`: putting more probability on what happened lowers
the penalty. Reward optimization is additional work, not a consequence of
repeatedly imitating generated games.

The proposed learning sequence is an observer over the existing pour, then a
fixed population with repeated encounters, then reward-based improvement. A
joint observer/value model is an alternative, with separate losses and metrics.
More history alone cannot make a history-blind teacher's targets sensitive to
player identity; the target and the population must actually reward that inference.

## Representation before size

### Candidate extension: predict the splat

Jason selected predicting the splat on 2026-10-17, after the
[[two-recursions-and-transformers]] comparison, and fixed the input boundary:
**the same viewer-visible input, with nothing appended**. The extension makes the
teacher's evaluated consequences explicit instead of supervising only the move
selected from them. The input/output boundary is selected; the distribution-loss
details remain a design, and no splat-output model has been implemented or trained.
The broader value-plus-policy design is already tracked in
[#175](https://github.com/jasonyandell/mk5-main/issues/175).

The raw splat is an `[action, scenario]` matrix of bidder-team final margins, with
the same scenarios shared across actions, their unnormalized weights, and the
sampled remaining hands. Each row follows the planner's information-consistent
continuation for that root action. These are model-relative planned outcomes,
not independently optimized clairvoyant values. The physical deal that actually
happened is a separate record, not the definition of a splat column.

**Uniform has two different domains.** Pour A's manifest specifications use the
default uniform draw over rule-compatible hidden hands, with no behavioral-history
posterior selected. Modeled choices still depend on the field: the pip split uses
the learned `margin_d250k_h128x64` net; doubles/no-trump uses a Walt rung whose bottom
field is random legal play. Response optimization changes the action distribution.
A uniform foundation therefore does not make the splat an unbiased account of all
players. Teacher configuration, objective and finite sampling remain part of its
meaning. Within a plan, observable field actions also partition scenario cells.

The selected input/output interface, **at the viewer's own decision**, is:

```text
INPUT:  original own hand + contract + public actor/tile prefix
        (exactly the existing tokens, ending at the current actor)
OUTPUT: for each candidate tile, a probability distribution over outcomes
        alongside the existing next-tile prediction
```

There is no `EVALUATE` token, candidate-action token, legal-action list, teacher
answer, or extra state feature in the input. "Same input" includes the viewer's
original seven tiles, which are private to that viewer, and the existing public
contract and history; it does not mean removing the own-hand tokens. The existing
encoder, 46-token vocabulary and causal visibility boundary remain unchanged.

Candidate identity belongs to an **output axis**. At the existing actor-token
position, the same representation can produce a `28 × 43` table: one distribution
over bidder-team totals for each tile. Splat supervision applies only to the
viewer's legal candidate rows with available teacher labels. Invalid candidate
rows are not outcome estimates; the mask selects loss terms and is not a model
input. The next-tile output continues to apply at every observed seat's turn.

Splats computed from another seat's private view must not be inserted as this
viewer's inputs. Walt supplies training targets outside the input sequence;
the trained student must produce its predictions without a Walt answer supplied
at use time. No predicted splat is appended to the next game-state input either.

For stored margin `m[a,j]` and scenario weight `w[j]`, define the target histogram

$$
p_a(s)=\frac{\sum_j w_j\,\mathbf1[(m_{a,j}+42)/2=s]}{\sum_j w_j}.
$$

The existing score convention converts exactly to 43 integer total outcomes.
Those outcomes can be categorical tokens: a distribution output trained against
the weighted histogram avoids pretending decimal probability strings have any
special meaning. Literal autoregressive serialization of counts/probabilities is
another candidate, requiring an explicit ordering and numerical precision.
Neither encoding is implemented here.

**This histogram is a deliberate summary of the splat.** It preserves each move's
outcome distribution, but discards which moves succeed in the same scenarios.
The raw paired matrix should remain the evidence source for any later paired
comparison target; arbitrary scenario order is not a meaningful sequence to learn.

**Current target qualification:** both manifest halves say `utility=make`, bid 30.
The planner stops optimizing when making/setting is decided; trace generation then
finishes the margin by random legal play. Consequently the whole histogram means
"this make-30 plan, with the specified random finish," not optimal points play.
Its mass at totals ≥30 reproduces the teacher's make probability. A different
threshold is an event under that same plan, not a policy optimized for another bid.
Predicting every candidate's make/set probability is the smaller, objective-aligned
alternative; retaining the full histogram teaches additional features of the stated
continuation process. The choice remains open under #175.

Source audit: `scenario/atlas.py::Position`, `in_plan_units`, `scenario/hf.py::tables`,
`scenario/planner.py::_invocation`, and `scenario/core/src/planner.rs::{settled,trace,playout}`
at `e02040b8`. The local merged Pour A manifests name source commits `9e0d3c04` (pip)
and `634323bc` (doubles/no-trump), matching the provenance in the observer's `data.json`.
No new samples or teacher values were generated for this design note.

### Existing observation contract

Start with explicit tiles, relative seats, declarations, boundaries and actions.
The observation contract matters more than the number of layers:

- A full-history sequence starts with the viewer's **initial** hand. Prepending
  its **current remaining** hand while supervising earlier plays leaks later own
  discards into earlier predictions. A current-position row may instead supervise
  only its next decision.
- Test noninterference: changing unavailable information at a fixed visible
  prefix must not change the tokens or output distribution.
- A causal mask hides the future, not secrets already in the prefix. A loss mask
  skips targets, not inputs. Actual hidden-hand legality cannot filter predictions
  about another seat. A playing adapter may enforce its own legal set, while
  reporting raw illegal probability separately.
- Split deals before constructing views, windows or symmetry variants. Keep
  encounters together. Unseen deals and unseen policy families are separate tests.
  Prior hidden tiles become inputs only when the recorded table rules reveal them,
  never retroactively at predictions made before that reveal.
- An end token does not block attention between unrelated episodes. Use episode
  boundaries in batching/attention. Retain forced moves as history, but report
  discretionary-move metrics so forced choices cannot inflate apparent skill.

Initially preserve nanoGPT's pre-normalized residual blocks, causal attention and
GELU MLP. Proposed teaching specimens are 1 layer / width 32, 2 / 64, and 4 / 128.
These are landmarks, not a controlled width experiment: that holds depth, data,
loss and training protocol fixed. Smaller need not mean easier to interpret.

Keep initialization and intermediate checkpoints, data/split hashes, tokenizer
and model versions, optimizer/sampler states, separate training/evaluation RNGs,
and genuine best/latest checkpoints. Stock nanoGPT's evaluation consumes the
training RNG; its checkpoints omit RNG/sampler states. These matter for comparing
formation trajectories. Sources: pinned `model.py`, `train.py`, and
`data/shakespeare_char/prepare.py`.

## A discriminating population experiment

Before scaling, use the 90-world witness as a representation challenge: retain
the actor-attributed auction histories that distinguish its posterior-sensitive
lead preferences, although its mechanical endpoint and world support agree.
It tests information retained by the encoding; it is not itself a population
training result. For a learned observer, add a tractable family of late-game
cases whose behavioral likelihoods and resulting decision values can be checked.

Use a few pinned, behaviorally distinct seats, including varied partners. Keep
assignments fixed through an encounter of several hands; draw new assignments
between encounters. Changing style every move erases the persistent thing history
is supposed to identify.

Compare held-out cases with current information alone, current-hand history,
previous hands plus current history, and known type as a diagnostic. For the
cross-hand contrast, remove or shuffle only previous-hand evidence, preserving
the current legal record. For a current-history contrast, preserve the same
rules information so remembering legality is not mistaken for reading a style.
The type-revealed optimum is an information upper under exact optimization;
a learned type-conditioned model is only a diagnostic, not a guaranteed upper.

Measure next-action log loss, calibration, discretionary-action prediction and
adaptation versus encounter length. Use encounter-level uncertainty and unfamiliar
policy families. Exact posterior labels are possible for a specified finite policy
family; generalization beyond it is empirical. Reward and the relevant book
assumptions need a separate playing evaluation under [[book-authority]].

Self-play needs an improvement rule and diversity mechanism. The
[AlphaStar league](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
retained strategies and trained exploiters against cycles and forgetting.
[PSRO](https://arxiv.org/abs/1711.00832) trains responses to mixtures of policies.
These motivate a league, not a convergence theorem for partnership 42. Preserve
full seat assignments and evaluate team combinations; one scalar rating cannot
describe all partner and opponent interactions.

## Reading the models and learning alongside them

Cammarata's [essay](https://already-clean.goodfire.com/) was read in full,
including appendices, from its author-provided text version. Its working lesson
is to investigate native mechanisms, trace how distinctions grow, and test
explanations with interventions. Small models can have broader units; top examples
alone can hide context or negative poles. The wider account is a hypothesis to
test here, not a theorem about all networks.

Apply the [clean-neuron skill](../../.claude/skills/clean-neuron/SKILL.md): inspect
tools, vocabulary, data/inputs and recursive generalization before positing
arbitrary mixing. An unnamed unit stays unnamed in the tested vocabulary. State
the actual sequence-model input scope and recalibrate the untrained floor and
uncertainty; field-net numerical corrections do not transfer by decree.

Read MLP preactivations and outputs, both poles, attention contributions and
residual updates. A label is a conjecture. Held-out counterexamples, ablation and
activation patching test it. Check instrumented attention against ordinary forward
logits before using it as a microscope.

Each milestone leaves a replayable worked example: visible position, tokens,
prediction before/after learning, an internal mechanism, and the effect of an
intervention. Introduce terms at use: an embedding represents a token; attention
selects earlier information; loss measures prediction error; a checkpoint preserves
a moment in learning. The learner should be able to explain the experiment's
question, evidence and limits before another objective is added.

## Decisions and later choices

1. First milestone: observer, selected and completed in [[lem-observer-2026-10-17]].
2. Initial scope: fixed-contract play from the existing pour; auction and match context deferred.
3. Population: persistent styles, partner systems, encounter/reset length and
   known-type diagnostic versus hidden-type evaluation.
4. Value target: make probability, continuation-dependent margins, or a new
   points/match-return teacher; retain field and teacher identity in every target.
5. Improvement algorithm: search/distillation or a named reward-learning method;
   policy retention/sampling and measured behavioral diversity.

The existing epics remain the durable ledger (#175, #177–#180, #182, #193, #202);
[#203](https://github.com/jasonyandell/mk5-main/issues/203) coordinates this workstream.
Register each subsequent experiment's predictions and controls before its run.
Shared planner, registry and check-runner changes follow [[standing-desks]]. Heavy
local runs use `scripts/cpu-slot`; training compute is priced before use.
