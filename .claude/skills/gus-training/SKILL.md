---
name: gus-training
description: Train, resume, evaluate, publish, and prepare interpretability specimens for Gus, the tiny history-conditioned Texas 42 GPT in mk5 lem/observer. Use for corpus squeeze, population reward learning, checkpoint comparisons, training telemetry, or Gus player improvement; not the historical oracle-era gus/ pipeline.
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

# Gus training

Build a better player while leaving an experiment another buddy can replay and
understand. Jason wants to inspect the actual inputs and outputs, and learn the
concepts well enough to explain them. Describe one real decision before hiding
it behind ML shorthand. Keep prediction, playing strength, and interpretation as
three separate claims.

Jason's 2026-10-19 steering makes the priority explicit: **learn to win more**.
Imitation supplies the starting policy and a control to beat. Lower imitation
loss, closer teacher agreement or better critic forecasts do not count as player
improvement. Reward-training updates must be driven by verified outcomes; a
candidate may need to disagree with its teacher to improve. Exact copying
reproduces a teacher's policy; this is not a theorem that every distilled student
is bounded by its teacher's measured strength.

Rehearsal, frozen-reference KL and imitation-retention thresholds are experiment
choices, not permanent requirements to remain teacher-like. Keep completed and
running registrations intact. Before the next registration, inspect whether
these choices obstruct the winning objective; compare changes explicitly rather
than silently removing them. Prediction/critic diagnostics are bounded support
work with a stated path back to actor training and fresh outcome confirmation.
Preserve clean inputs, fixed seats, legality, verifier authority and replay checks.
Jason may miss discussion windows; continue the authorized loop after the window,
incorporating his latest steering without requiring another reply.

## Find the live experiment

The repository is `/Users/jason/code/mk5-main`; the original Gus research worktree
is `/Users/jason/code/mk5-lem-nanogpt`. Locate the current checkout and artifacts
rather than assuming either branch is current. Read its `CLAUDE.md`,
`wiki/AGENTS.md`, and `wiki/entities/gus-tiny-gpt.md` first. Follow the entity's
current experiment routes. Work in the user's authorized worktree; don't edit
shared main or restart a completed campaign merely to answer a status question.

Implementation: `lem/observer/README.md`. Historical `gus/` and the oracle-era
forge students are a different lineage. Prefer the existing CLIs over new loops.
For execution, recovery, telemetry and publishing, read
[references/operations.md](references/operations.md).
For exact Silas endgame verifiers, GRPO/SFT/direct reward comparisons,
retention-aware rehearsal and bounded capacity experiments, read [references/rlvr.md](references/rlvr.md).
For every new model's repeatable static check and fixed H2H anchors, read
[references/benchmark.md](references/benchmark.md).
The full-hand outcome stage is implemented in `lem.observer.outcomes`, after
completed splat pretraining and exact-endgame comparisons. Read
`wiki/experiments/gus-full-hand-outcomes-2026-10-18.md` for the bounded matched
iterations, selection, confirmation and actual-game limits. Keep other seats
fixed and compare against rehearsal alone; register a new plan before changing
the recipe. Extend frozen evaluation lazily; external search is deferred.
The public-terminal credit ablation is recorded in
`wiki/experiments/gus-public-terminal-credit-2026-10-18.md`; neither horizon
established a stronger selected player in that comparison.
The independent-deal batch comparison and exact-inference-state reuse are in
`wiki/experiments/gus-independent-root-batches-2026-10-18.md`.
The update-scale diagnostic and matched learning-rate comparison are in
`wiki/experiments/gus-update-scale-2026-10-18.md`.
The exact G0/Zeb curriculum improves held-out finite-model endgames without
confirming stronger full-hand play; construction and the completed study are in
`wiki/experiments/gus-fixed-field-curriculum-2026-10-18.md`.
The completed staged comparison separates initial weights from the common
frozen reference, without a reward-specific full-hand gain; see
`wiki/experiments/gus-staged-outcomes-2026-10-18.md`. The bounded exposure
comparison uses short prefixes of continuous runs and matched rehearsal, without
establishing a fresh full-hand gain:
`wiki/experiments/gus-exposure-budget-2026-10-18.md`.
The completed value diagnostic supports a matched PPO/value comparison:
`wiki/experiments/gus-value-baseline-2026-10-18.md`. The completed
`wiki/experiments/gus-ppo-value-2026-10-19.md` comparison found no fresh playing
gain; its online critic forecast new batches worse than an earlier-batch prior.
Distinguish that fitting regime from the pooled diagnostic. The `--mode ppo`
estimator is implemented; a learned forecast is a baseline, never reward authority.
The pinned Demon is an evaluation competitor, never a source of labels.

## Recenter the research loop on current capabilities

Before selecting a new experiment, after a completed comparison, or when a
missing capability blocks progress, check the related Lem, Book and Silas work.
Use `wiki/trails/gus-book-silas-tools.md` as the route, then inspect their current
wiki findings, issue replies and relevant merged changes. Check available tool
entry points and source/CLI behavior before assuming a capability exists.
Distinguish available on main, branch-only, proposed and blocked; a saved trace
is not an implemented audit adapter, and a world constructor is not an outcome
verifier. Avoid repeating the full refresh for every short resume segment.

Jason requested ongoing progress checks with these collaborators as part of
this research loop. Read pending replies first; when new information is needed,
send a short, scoped question with a durable context link to the relevant desk
or session. Keep an unanswered request pending rather than repeatedly pinging
or restarting its terminal. If the desk is unavailable, use its latest evidence
and record that no response was received. This does not authorize new experiments
in another desk, changes to its work, or unrelated messages.

Before spending the next run, record what changed, which tool can answer the
question now, and whether that changes the selected lever or only its measurement.
Prefer an existing verifier, fixture or instrument over rebuilding it. Recheck
the user's fixed inputs, opponents, outcome objective and interpretation scope;
new capabilities alone do not change those controls. Preserve book concepts as
questions for the net, not an answer key (`book-interp`, beside `clean-neuron`).
File useful synthesis back into the wiki and carry unresolved dependencies into
the next registration. Continue a bounded justified experiment when ready; discuss
a genuine choice with Jason when the evidence leaves materially different paths.

## The companion is `clean-neuron`

Read the repository's **`.claude/skills/clean-neuron/SKILL.md`** before designing
an interpretability training specimen or writing a unit result. It implements
the already-clean test from Nick Cammarata's working hypothesis,
[Neural Networks Are Already Clean](https://already-clean.goodfire.com/).
The working bet is that capacity pressure can yield coherent, broader concepts;
it is not a promise that every small model will be easy to read.

“Too many games for a little model” means broad, distinct lawful experience at
fixed capacity with a declared exposure budget. It is different from repeating
a tiny corpus hundreds of times. Neither procedure alone establishes that
capacity has been exhausted, explains a neuron, or improves play.

Preserve initialization, early formation, periodic full checkpoints and native
activations. Follow `clean-neuron`'s ladder: tools → explanation vocabulary →
data/inputs/learning → recursive generalization → only then superposition.
An unnamed unit is not evidence of arbitrary mixing. For this GPT, state its
actual history-conditioned scope; do not import the field net's “no history”
sentence, numerical untrained floor, or 1.17 interval adjustment. Use a matched
untrained twin and uncertainty appropriate to correlated deals/sequences.
Training quality and causal unit evidence must accompany interpretation claims.

When applying an article's learning advice, inventory the actual architecture
first. Gus already has residuals, pre-LayerNorm, GELU and AdamW matrix decay.
`wiki/experiments/gus-modern-learning-recovery-2026-10-18.md` separates the
article's claims from the standard cosine continuation tested here. Do not
turn unspecified regularization into an attributed recipe, or confuse tuning
a visualization optimizer with changing the model. Read the full article's
`essay.md` text and appendices when a request depends on what it actually says.

## Keep the information contract visible

The original observer receives the viewer's **own original seven tiles**, fixed
bid 30, bidder, declaration, relative seats and the public actor/tile history.
The viewer is the acting seat when deployed as a player; pretraining also
predicts other seats from that viewer's information. Its context vocabulary has
46 tokens; its output has 28 tile logits. Own hand is private information
legitimately available to the viewer. Other hands are not inputs. The actor
token precedes the tile to be predicted.

- Show a literal encoded prefix, its decoded meaning, target/output distribution
  and chosen action when changing data, tokenizer or output semantics.
- Causal attention blocks future tokens, not secrets accidentally in the prefix.
  A later remaining hand or future actor schedule can leak future play.
- A legal-action mask constrains the acting player's output choice; it is not an
  extra input. For another seat's prediction, only publicly visible exclusions
  may define a deployable baseline; its hidden hand cannot repair predictions.
- Split physical deals before constructing four observer views, copies or
  augmentations. Reject overlap with training and reserved panels.
- Player identity, auction transcript, score, Nello and cross-hand memory are
  changes to the task, not already implemented abilities. The splat output head
  is implemented separately in `lem/observer/splat/`; original G0 and population
  releases do not contain it.
  Preserve Jason's selected clean inputs unless he changes that scope. A richer
  lawful output target does not require adding its teacher's internals to inputs.
- No clairvoyant value becomes a learning target. Lawful Walt forecasts are
  conditional on a specified behavioral field, not an optimal-play certificate.

## Current series: keep opponents fixed

Jason explicitly fixed the opponent dimension on 2026-10-18; the scope is in
`wiki/entities/gus-tiny-gpt.md`. During these learner/recipe investigations,
preserve all nonlearner policies, including the partner, their checkpoints,
mixture weights, belief semantics, bidder and deterministic budgets. Keep the
teacher and label-generation recipe pinned too. Existing exact-verifier and
Zeb benchmark controls are distinct; hold each unchanged within its comparison.

Vary learner training, size or decision coverage through declared comparisons.
Broader coverage means more situations under the same behavior law, not new
opponents. Keep matched fixtures; reserve fresh confirmation under the same law.
A frozen policy still responds to the new history a learner creates: do not
substitute a reference game's moves after histories diverge. A future arena
with evolving models/distillation is a separate direction. Do not rotate recent
Gus checkpoints into this series using the historical population recipe below.

## Choose the question before spending the run

**Pretraining:** imitate recorded next tiles. Preserve all raw 28-way predictions
for evaluation, including impossible probability. Sampling is with replacement;
`steps × batch / training_views` is an equivalent number of passes, not completed
shuffled epochs. The original corpus-squeeze protocol and results live in
`wiki/experiments/lem-observer-corpus-squeeze-2026-10-17.md`.

**Joint splat pretraining:** `lem/observer/splat/` keeps the exact encoder and
adds a 28 × 43 outcome head at the viewer's own actor positions. Read
`wiki/experiments/gus-splat-pretraining-2026-10-18.md` before reusing it. Weighted
margin histograms must reproduce every teacher make probability; supervise only
legal candidates with labels. Missing rows mean no supervision. The make-pour
histogram includes random post-settlement continuation, so play uses its make/set
tail, not expected points. It is a parallel categorical output, not a serialized
sequence fed back as input. Original tile-only and joint checkpoints have
distinct explicit loaders. Splats from another private view are never inputs.
Compare histogram KL (CE minus label entropy), consequential teacher regret,
calibration to the named teacher, and actual play separately.

**Historical population reward learning:** the separate initial recipe lives in `wiki/decisions/gus-population-iteration.md`: one
learner seat; each other seat independently 25% pinned Zeb and 75% uniform over
the latest five completed Gus generations. Freeze assignments within a hand and
the roster within a generation. Five is a sampling window, never an archive
deletion rule. Count behavioral differences; five nearby checkpoints need not
be five useful styles.

The first implementation uses legal-softmax exploration, terminal ±1 team reward,
policy-only clipped PPO and a baseline from previous completed batches. Preserve
actual sampled log probabilities, legal sets, rewards and replay prefixes; only
the learner's decisions enter the policy gradient. Old/new probabilities use
the same support. This is an initial tested implementation, not a validated
optimal recipe. Read `wiki/experiments/gus-population-depth-2026-10-18.md` before
reusing it: twenty generations and depth growth did not establish a fresh-panel
gain. Do not treat that as a capacity ceiling or silently repeat the campaign.

Before a new experiment, record the hypothesis, one main changed factor,
population, seed namespaces, deterministic sample/update budgets, selection
metric, minimum meaningful gain, stopping rule and independent confirmation.
Keep the latest learner, highest development score, material-gain anchor and
supported incumbent separate. Make any revised recipe explicit before results.
Reserve fresh confirmation before selection; never use it to choose again.

## Read the right scoreboard

| Question | Evidence |
|---|---|
| Did it predict recorded behavior? | Raw NLL, top-1, impossible mass; own/other and forced/discretionary slices; visible-uniform and untrained baselines |
| Did a candidate improve in the training setting? | Fixed paired development hands with frozen other seats; exploratory after repeated selection |
| Does it play a stronger game? | Fresh paired full-game H2H against named pinned opponents, shared bidder/contract, pair-level uncertainty and marks/game |
| Is the adapter working? | A short replayed smoke; no promotion claim from a handful of wins |
| What changed on the same positions? | Frozen benchmark rows, paired deltas and retained logits/native traces; repeated development evaluation |
| What does a native unit do? | Held-out activation evidence, formation and matched causal interventions under `clean-neuron` |

NLL rewards probability assigned to the observed tile; top-1 checks only whether
it was ranked first. Their disagreement is expected. Neither is team win rate.
All-play NLL includes other seats' hidden-hand uncertainty and forced moves.
For playing progress, inspect consequential own decisions as well.
Use the frozen benchmark before spending new games. Preserve its 234 scheduled
physical deals as training exclusions, including the reference game's unplayed
hands. Agreement with a recorded move is not optimality; its known final result
labels only the recorded continuation. The exact subset uses its separately
declared finite-model field and includes ties: report uniform-legal accuracy
and regret beside model scores. New-policy win rates require actual play on the
same schedule, and eventual promotion still requires fresh confirmation.
In particular, report legal-renormalized own-seat choices that are discretionary
and occur before public make/set settlement. Raw impossible mass is diagnostic;
the deployed legal mask prevents those tiles from being chosen. PPO trains a
sampled legal-softmax policy, while current panels and deployment use argmax:
measure both when testing an improvement, rather than assuming they move together.

Cluster uncertainty by physical deal or complete duplicate pair, not individual
tokens. Preserve the cross-play matrix beside ratings: population responses can
cycle. The existing tournament fits Bradley–Terry on an Elo scale; ratings are
conditional on the ladder, opponents and wrapper. The bid-30 model cannot
honestly enter unrestricted auctions. Route through `wiki/topics/tournament.md`
and the current Gus tournament experiment for the supported division.

## Leave a readable specimen

Pin source, tokenizer, dataset/split, field, population and checkpoint hashes.
Save optimizer/sampler/RNG state, full hands, literal learner tensors, old
probabilities, legal masks, outcomes and formation checkpoints. Independently
replay and reconstruct the inputs; test frozen-seat equality until the learner
first changes history. Preserve best, final and initialization separately.

For splat training, a pause preserves the fixed plan: SIGINT/SIGTERM,
`--stop-file`, or `--stop-after-seconds` checkpoints at a completed update.
Resume the same output with the same arguments and `--resume`; a changed recipe
uses `--fork-from` and a fresh output. The operations reference gives the exact
boundary and W&B recovery contract. Loading only published inference weights is
a warm start, not exact optimizer recovery.

Identity-appended residual blocks preserve an initial function, not evidence of
new learned depth. Verify initial logit equality and subsequent block updates.
The 2026-10-18 selected release is G10's function padded to fifteen blocks;
`gus-g017` through `gus-g020` are genuinely trained fifteen-block specimens.

Report what changed, what was measured and what remains unestablished. Publish
generated evidence to the existing public CC0 Hub destination before citing it
in the wiki; pin immutable revisions, verify fresh downloads, and keep the MIT
nanoGPT implementation attribution. Update the wiki and pass the repo's own
checks before merging. Leave future experiments as proposals unless authorized.

Improve this skill when real friction changes a decision: fix the narrow
instruction or operations note, rather than accumulating generic warnings.
