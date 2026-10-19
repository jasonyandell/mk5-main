---
title: "Gus: success-probability RLVR and lazy frozen evaluation"
kind: decision
first_seen: 2026-10-18
last_updated: 2026-10-19
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Selected direction

The selected path for [[gus-tiny-gpt]] is textbook reward learning on **verified
hand success**, with the existing architecture, clean inputs and other seats
fixed. Jason explicitly selected this direction and lazy frozen-set expansion
on 2026-10-18. External search and future-splat auxiliary training are deferred.
[Issue #220](https://github.com/jasonyandell/mk5-main/issues/220) records the completed
matched probability-reward experiment in [[gus-probability-reward-v3-2026-10-18]].
The full-hand terminal-outcome stage is implemented and measured in
[[gus-full-hand-outcomes-2026-10-18]] ([issue #222](https://github.com/jasonyandell/mk5-main/issues/222)).

This is post-training: the pretrained policy is improved using a verifier.
The objective is making or setting the bid30 hand, not unspecified match win
probability. Exactness is conditional on the named finite game model. The
three-style training field and the G0/Zeb evaluation field remain distinct and
fixed within their respective comparisons.

Jason reaffirmed on 2026-10-19 that the objective is stronger play and winning;
copying establishes a starting point. Teacher agreement, imitation losses and
critic accuracy are supporting measurements, never substitutes for verified
outcome gains. Reward training must be able to prefer a winning departure from
the teacher. Exact policy copying supplies no behavioral improvement by itself;
this is not a general upper-bound theorem for distilled students.

The implemented full-hand trainer already optimizes verified rewards, alongside
supervised rehearsal and frozen-reference KL. Its selection also requires tile
NLL and splat CE within declared bounds. Those restrictions can exclude a
different policy independently of its outcome score; whether they have obstructed
progress is an empirical question, not established by the current null result.
The next registration must justify their role against the winning objective.
Existing registrations and their results remain unchanged; future comparisons
may explicitly vary regularization or treat imitation drift as a diagnostic.
A critic-only diagnostic must remain bounded and lead back to a test of actor
improvement. The fixed G0 partner, pinned Zeb opponents, clean inputs, verified
rewards and independent confirmation remain controls. An absent discussion reply
does not block the authorized research loop after its steering window.

Jason's further steering expects behavioral drift after pretraining and questions
the continuing value of original mimicry. Outcome distributions are conditional
on continuation policies: changing Gus's later decisions can change its own
success distribution even with the other seats fixed. The original splat labels
still describe their named teacher-plus-finish law; they do not automatically
become current-Gus forecasts. Requiring agreement with them is a distinct
auxiliary objective, not verification of the new policy's consequences.

The current trainer already samples eight complete 28-play trajectories from
each training deal, with seven learner decisions per trajectory. In the latest
comparison, each arm collects 256 deals times eight draws: 2,048 complete hands.
Repeated draws hold the complete environment deal fixed while sampling Gus's
actions; hidden hands stay outside the policy input. Verified terminal make/set
outcomes train the move policy. The splat head is still supervised by original
teacher histograms through rehearsal, not refreshed current-policy histograms.

Branching from a particular public prefix, forcing each candidate move and
sampling the remaining hand repeatedly would be a separate action-value data
collection method. It is not the current whole-trajectory sampler. Such a design
must declare the hidden-deal belief, lawful continuation policies, sampled-action
coverage and cost budget; simulation may use hidden state, but the learner may
not. Training-time simulation does not require search inside the deployed player.
Full-match continuation remains future scope; the current contract is bid30 hands.

The v5 full-hand trainer now supports explicitly declared outcome-only selection
and zero-weight imitation in reward arms. `selection: outcomes` records original
loss drift without using it as an eligibility gate; `selection: retention` keeps
the declared loss bounds. The original frozen-reference KL can independently be
set to zero. The current-batch old-policy ratio clipping remains in both cases.
[[gus-unanchored-outcomes-2026-10-19]] tests that loss package while keeping
GRPO fixed. Outcome-only selection and fresh confirmation do not establish a
greedy gain. Both reward arms show a secondary sampled-policy gain against the
sampled parent, without an established unanchored advantage over anchored GRPO.
Confirm that distinction independently before changing the deployment decoder.

## Textbook choice for the current subproblem

[[gus-rlvr-silas-2026-10-18]] supplies an exact rational value `Q(I,a)` for
**every** legal root action: uniform legal-consistent hidden deals and the
fixed lowest/highest/uniform continuation mixture. The viewer has two tiles
left; its final move is forced. This is a one-decision contextual bandit.
There is no hidden-world maximization or new search inside Gus.

`lem.observer.rlvr.train --mode exact --reward success` now minimizes:

`L = -mean_I sum_a pi_theta(a|I) Q(I,a)`
`    + beta KL(pi_theta || frozen_parent) + lambda L_original_tasks`.

The first term directly differentiates finite expected return, the standard
policy-gradient objective. Enumerating two known rewards avoids unnecessary
action-sampling variance and a learned critic. The remaining terms are the
existing fixed-parent KL and supervised rehearsal, justified by the forgetting
observed in [[gus-retention-iterations-2026-10-18]]. Teacher splats are rehearsal
targets, not the verifier. Rational values become float32 only at training.
Source: [policy-gradient formulation](https://papers.neurips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html).

For two action logits, the reward-loss derivative is
`dL/dz_a = -pi_a (Q_a - sum_b pi_b Q_b)` before batch averaging.
At equal probabilities, an 80% versus 40% decision has forty times the reward
gradient of an 80% versus 79% decision. Ties have zero reward gradient. Adding
an action-independent baseline changes the loss value, not this gradient.

The existing `--reward best` remains the explicit binary-correctness control.
GRPO standardizes within a question; with two distinct reward levels this
removes the magnitude of the gap. Merely substituting probabilities into that
standardized surrogate does not test this hypothesis. Success rewards therefore
require exact mode. GRPO remains available for the original experiment; it is
not renamed to cover a different objective. Source: [DeepSeekMath / GRPO](https://arxiv.org/abs/2402.03300).

## Completed bounded comparison and next stage

[[gus-probability-reward-v3-2026-10-18]] completed the matched three-seed
comparison from the original splat parent, keeping architecture, question/replay
streams, update budget and KL/rehearsal coefficients fixed. Probability reward
did not pass the registered improvement criterion. Both predeclared seed42
representatives improved fresh finite-model regret with retention; brief game
checks did not establish transfer. The experiment page owns the measurements.

[[gus-full-hand-outcomes-2026-10-18]] broadens training to full-hand sampled
trajectories and verified terminal outcomes, with a matched rehearsal-only control. One learner seat,
fixed G0 partner and pinned Zeb opponents define the declared full-game stage.
Its actual outcome verifier remains distinct from the exact three-style
endgame approximation. Grouped samples justify a GRPO estimator when complete
continuation values are no longer cheaply available. Its two registered bounded
iterations retain the parent incumbent; the experiment page owns the measurements.

The next completed ablation, [[gus-public-terminal-credit-2026-10-18]], removes
reward-score terms after public make/set settlement while retaining the original
normalization, full KL and rehearsal. All three arms selected initialization;
the mathematically motivated cleanup did not establish stronger play.

The subsequent [[gus-independent-root-batches-2026-10-18]] used saved training
gradients to choose more independent deals per update, leaving loss weights
fixed. Empty batches fell from 6/32 to 1/32. Fresh greedy wins were 202/512 for
the larger batch versus 201/512 parent, with no confirmed gain or promotion.
It also added exact inference-state identities for safe reuse of lazy evidence.

[[gus-update-scale-2026-10-18]] increases learning rate at matched data and update
counts. Policy movement increases, but all selections retain initialization.
Its training-only feasibility probe shows that short endgames can cheaply use
the actual fixed G0/Zeb continuation policies for exact rational feedback.
The resulting [[gus-fixed-field-curriculum-2026-10-18]] implements a separate
verifier identity and completes a matched three-seed study. It improves fresh
finite-model endgame regret beyond rehearsal with retention, without establishing
a fresh full-hand gain. Uniform legal root belief remains an approximation;
the earlier three-style verifier is preserved. The root action is shared across
worlds and the later learner choice is forced. The experiment page owns the
measurements.

The completed [[gus-staged-outcomes-2026-10-18]] compares three starting policies
with a common frozen reference and a supervised-only control. It establishes no
reward-specific full-hand gain. Training-only diagnostics find substantial path
diversity and mostly retained endgame skill, with few mixed-outcome groups.
The completed [[gus-exposure-budget-2026-10-18]] tests four times the ordinary
on-policy exposure. Outputs move and imitation losses improve, but the registered
fresh full-hand gain does not appear. Zero-variance groups remain common despite
diverse paths. More exposure is not an automatic next step.

Full-hand recovery publishes the same completion state to `status.json` and
`report.json`; a previous pause no longer remains the live status. Completed
resumes validate checksums and preserve checkpoint and JSON artifact bytes.
Actual-signal recovery tests cover both reward horizons and rehearsal.

The frozen panel remains development evidence. Reserve new confirmation before
selection and extend actual continuations only where needed. Existing tests,
model sizes and qualitative hypotheses are not grounds for a general-player
promotion. The pinned Demon is a separate evaluation competitor; its internal
forecasts never become learning targets.

## Completed comparison: PPO with a success-value baseline

[[gus-ppo-value-2026-10-19]] completes the matched three-arm comparison. PPO
selects round 16 and wins 221/512 fresh greedy hands versus parent 218, GRPO 222
and rehearsal 221. Neither registered criterion passes. The critic fits its
current batch but predicts new batches worse than the earlier-batch success
prior (Brier .27745 versus .23917). No incumbent is promoted.

The completed [[gus-value-baseline-2026-10-18]] passes its registered prediction
criterion on disjoint physical roots using the original frozen representation.
It motivates a declared PPO/value comparison against GRPO and rehearsal. The
diagnostic spans changing learner snapshots: its pooled predictor is not an
on-policy critic, and its fitted head is not carried into actor training.

`lem.observer.outcomes --mode ppo` now implements that estimator with a separate
65-parameter linear head over the frozen reference's actor-position features.
`V(I_t)` estimates success under the current learner and fixed other seats;
terminal rewards still come from independently replayed game rules. The actor
remains 185,652 parameters with the original tokens. The critic begins with
zero weights and bias 0.5, uses its own Adam optimizer, and receives no hidden
hand, legal-mask feature or teacher forecast.

For these short complete hands, `A_t = G - V_old(I_t)`, where `G` is the verified
binary terminal success. This is the undiscounted GAE λ=1 case: the temporal
differences telescope, with zero value after termination. It needs no bootstrapped
target. Batch normalization replaces within-root group standardization; it is
a finite-sample PPO convention, not an unbiasedness guarantee. Old predictions
and advantages are captured before any fitting to the current batch and remain
fixed throughout the actor update. The critic subsequently regresses on current
rollout returns. PPO clipping, fixed-parent KL, rehearsal and actor optimizer
remain the existing controls. Complete-round recovery includes both optimizers
and all RNG states. The explicit value recipe is part of resume identity.

The standard components are PPO-Clip, return regression and a state-dependent
baseline. The frozen linear representation, retained KL/rehearsal terms and
fixed update budgets are declared experimental choices, not universal textbook
requirements. A finite learned estimate is neither an exact Silas probability
nor reward authority. Sources:
[PPO algorithm and value regression](https://spinningup.openai.com/en/latest/algorithms/ppo.html),
[generalized advantage estimation](https://arxiv.org/abs/1506.02438).

This tests a variance-reduction/credit estimator without assuming collapsed
exploration or a capacity shortage. A learned baseline need not vanish when
all eight sampled outcomes agree; that does not establish a useful gradient.
Fresh paired confirmation did not establish a playing gain. The pooled
prediction study and this tiny online critic use different fitting regimes;
usable frozen features do not by themselves establish a useful current-policy
baseline. A next comparison should first justify the critic fitting/stability
choice from forecast evidence, while preserving fixed seats, clean inputs and
verifier authority. It requires a fresh registration and Jason's steering window.

## Lazy immutable continuation evidence

`lem.observer.benchmark.continue_rows` appends supplements beside
[[gus-frozen-benchmark-2026-10-18]], leaving its manifest, rows and assets intact.
The initial supported scope is unsettled, two-legal-action penultimate-trick
rows in the two G0/Zeb anchor pairs. Silas enumerates the small legal root fiber;
a named seed samples a hidden deal uniformly. This law deliberately excludes
pre-root behavior likelihood and is separate from Zeb's internal belief model.

The baseline puts pinned Zeb in the viewer seat. A candidate replaces that
seat alone. Each of the other three seats keeps its original anchor G0 or Zeb
policy, including the partner. Their actions are recomputed after changed
history. Original dealer, bids, weights, deterministic planner budgets and
seating are retained. The network still receives only its own original seven
and public contract/history. Full hands and outcomes are audit sidecars.

Each unit is keyed by recipe, frozen row ID, question hash and sample index.
The root seed is derived from the common pairing recipe, row/question, sample
index and named stream; candidate weights are excluded from that seed. It is
independent of request order, process restarts and work slicing. Candidate
weights remain part of the separate result identity. Complete trajectories and
terminal success are checked by an independent rules replay before commit.

The recipe pins base manifest/runtime hashes, policy and weight identities,
additional source files, native binary bytes, platform, Torch build and CPU
float32/single-example execution. Seeds alone do not guarantee replay across
software or hardware changes; a changed runtime uses a new recipe and parity
check. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html).

Writes are atomic under per-unit locks. Failed/incomplete units have no completed
record; a later invocation recomputes them. Busy units are left for a subsequent
invocation. Existing valid records are skipped; `--recheck` recomputes them and
rejects any changed bytes. Corrupt records and recipe drift fail closed. Higher
`--samples`, added row IDs or independent `--start` slices append work. Sorted,
content-addressed snapshots preserve every earlier published view. Timings and
queue state are excluded from evidence bytes. Stop signals, a stop file and a
cooperative wall budget stop between complete hands; no timeout becomes a loss.

These are sampled actual-policy outcomes, not exact Zeb success probabilities.
Compare paired candidates at the same row/sample keys and report sample counts;
cluster broader uncertainty by original physical deal or complete anchor pair.
Retain sampled full-hand identities as future training exclusions alongside the
base panel's reserved deals. Add new provider scopes under explicit identities;
never rewrite existing evidence to make a larger panel look untouched.

The operational commands live in the `gus-training` skill's
`references/benchmark.md`; representation and unit claims follow `clean-neuron`.

Implementation and same-runtime replay evidence: [[gus-lazy-rlvr-smoke-2026-10-18]].
The matched multi-seed comparison is complete in [[gus-probability-reward-v3-2026-10-18]];
full-hand outcome post-training is measured in [[gus-full-hand-outcomes-2026-10-18]] with a matched rehearsal-only control
and the same clean inputs. Existing exact labels and measured terminal game
outcomes retain their distinct authority.


The completed [[gus-unanchored-outcomes-2026-10-19]] comparison removes the
teacher-loss package and selects all arms by outcomes. Fresh paired confirmation
does not meet its promotion criterion. This preserves winning as the objective
without claiming that permission to drift alone supplies useful policy credit.
Current-Gus splat calibration and action-conditioned continuation collection
remain separate capabilities; this comparison implements neither.


## Next bounded question: confirm the sampled policy

Jason authorized continuation on 2026-10-19. The next proposed comparison is
independent evaluation of the already selected checkpoints from
[[gus-unanchored-outcomes-2026-10-19]], before another training change.
Its sampled-policy gain was secondary; it cannot replace that study's failed
greedy endpoint. Policy-gradient learning optimizes expected return under an
action distribution. Taking its largest probability at every turn defines a
different policy; training gains need not survive that substitution.
[Spinning Up: stochastic policies and the RL objective](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html),
[PPO exploration and exploitation](https://spinningup.openai.com/en/latest/algorithms/ppo.html#exploration-vs-exploitation).

The proposed budget is 512 new physical deals, four independently seeded sampled
complete hands and one greedy hand per frozen checkpoint: parent, unanchored,
anchored and rehearsal, at most 10,240 complete trajectories. No retraining,
checkpoint reselection or temperature sweep. Seeds pair the models and exclude
model identity; repeat index must actually change the learner action stream.
The unchanged G0 partner, pinned Zeb opponents and bid30/pip task remain fixed.
All prior reserved deals, including the unanchored study and its tournament
schedule, enter the exclusion set before new roots are generated.

The primary replication question is unanchored sampled success versus parent
sampled success. A proposed meaningful-gain rule is at least 1 percentage point
with a paired 95% lower bound above zero. Compare mean outcome across the four
draws within each root, then bootstrap physical roots (20,000 draws, fixed seed);
2,048 sampled hands are not 2,048 independent deals. Report anchored and rehearsal
controls and greedy decoding separately. A claim of improved deployment also
requires at least 2 points over the greedy parent and 1 point over sampled
rehearsal, both lower bounds above zero. Passing sampled-parent replication alone
does not establish a better deployed player or trigger promotion. Every result
remains conditional on this training seed, selected weights and fixed field.

Use existing immutable lazy records and CPU float32 single-example inference,
with complete input/probability/outcome sidecars and independent rules replay.
Pin scripts, tensors, field, runtime and binary hashes before evaluation.
Proposed caps: 24 cooperative 45-second segments per unique inference state,
90-second outer caps, at most two live evaluators; exhausted budgets stop with
partial evidence and no strength claim. Audit all completed hands and recheck
16 units in reverse order byte-for-byte. Identical functions may share records
only under identical pinned recipes. This is a measurement step toward the
winning objective, not a new model architecture or deployed search.

The proposal completed as [[gus-sampled-confirmation-2026-10-19]] after verified
canonical delivery. Its sampled-policy replication and deployment-evidence
criteria both failed; see the experiment for root-clustered intervals. No new
training comparison is selected yet; changing decoding alone lacks support. The live refresh found no new
Lem/Book replies and Silas PR #166 still open. Existing outcome replay and native
captures suffice; the transformer causal bridge remains separate. No repeat
collaborator request was sent.


## Later charter clarification: learn the game and the transformer

Jason clarified after [[gus-leave-one-out-2026-10-19]] that learning Texas42
and personally understanding transformers hands-on are the primary goals.
A stronger player is a hoped-for consequence; lack of an aggregate win gain
does not exhaust the value of an experiment. Claims of stronger play still
require verified outcome evidence. This clarification does not alter completed
registrations, thresholds or reported failures.

Partner-aware training and whole games with random partners per hand were
brainstormed, alongside continued useful hill climbing. No new population,
input or experiment was selected. Existing clean inputs already distinguish
partner from opponents and retain public within-hand history. This permits
partner-history use but does not establish a learned behavioral model.
A fixed G0 partner does not test adaptation across partner types.

A possible progression separates role recognition, inference of a partner's
behavior from lawful history, coordinated choices, and persistent cross-hand
learning. A partner independently redrawn each hand provides no persistent
identity to learn across hands. Full games additionally require declared
auction/score/reward/memory semantics. These factors should not be silently
bundled into a single comparison. Attention maps can illustrate computation;
causal use of partner history needs a controlled test and appropriate replay
and legality checks. The transformer causal adapter remains separate work.

Uniform beliefs and uniform action choices are different objects. Walt's
belief weights hidden worlds; its response chooses a common information-lawful
policy to maximize expected utility against the stated behavioral field.
Uniform weights do not make that decision rule random or establish a minimax
or equilibrium guarantee. See [the mathematical definition](../../docs/math/walt.md)
and [[scenario-player]]. This statement does not establish the cause of any
particular observed exploit or import old deployment defaults into current pins.
