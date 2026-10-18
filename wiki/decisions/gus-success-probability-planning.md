---
title: "Gus: success probability, future splats and lawful planning"
kind: decision
first_seen: 2026-10-18
last_updated: 2026-10-18
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## Selected direction and implementation boundary

Jason selected **success probability** as the reward target and proposed future
splat prediction to encourage planning on 2026-10-18. The current
[[gus-retention-iterations-2026-10-18]] trainer still uses binary exact-best
rewards; the probability objective and future-planning experiment are not yet
implemented. [Issue #220](https://github.com/jasonyandell/mk5-main/issues/220)
tracks the bounded implementation and measurement work.

The objective is hand make/set success under the existing bid30 contract, not
an unspecified full-match win probability. Other seats' policies, teacher
weights, belief semantics and budgets remain fixed per [[gus-tiny-gpt]].
Beliefs can update after observations under that fixed law; the law itself does
not evolve. No self-play roster update is part of this direction.

## Post-training and its component methods

Training the existing pretrained Gus on these objectives is **post-training**.
Matching a pinned teacher's splats at future positions is supervised prediction
or distillation. Improving action choice from independently verified outcomes
is reward learning; learned teacher forecasts alone are not an exact verifier.
Explicit search at deployment is additional inference computation, not itself
training. Search targets can then be distilled into a fast student.

[Expert Iteration](https://arxiv.org/abs/1705.08439) is the closest established
search/generalization loop: search improves plans and a network learns them.
[MuZero](https://arxiv.org/abs/1911.08265) demonstrates a related learned model
used iteratively for planning-relevant predictions. Neither supplies an
off-the-shelf solution to 42's hidden information. Silas already supplies known
game mechanics; learning a replacement rules engine is not required here.

## Predict futures, then evaluate contingent plans

A bounded first comparison can examine legal actions through the next own
decision. Each branch contains a possible public observation history, evaluated
with the original clean encoder and same viewer's legitimate information.
Teacher splats at those future states are output targets, never extra evidence
in the original observed prefix. The current network already predicts splats
at recorded own-turn prefixes; unseen branch prediction and backed-up planning
are the additional questions, not merely another pass over later recorded rows.

The two recursions in [[two-recursions-and-transformers]] remain distinct:
advance information through possible observations; back up success values to
choose a contingent policy. Schematically, at a learner decision:

`Q(I,a) = sum_o P(o | I,a) V(T(I,a,o)); V(I) = max_a Q(I,a)`.

The fixed policies generate other seats' response probabilities. A branch
weight is probability mass, not one vote per winning leaf. A robust plan in
this study succeeds across probable hidden situations and responses under the
fixed model; no adversarial or worst-case guarantee is implied. Later learner
actions can depend on observations actually available then. They cannot depend
on hidden-world identity before those observations distinguish worlds.

Root belief, observation updates and leaf continuation policy must have coherent
semantics before claiming an exact Bellman backup. Independently recomputing
teacher forecasts under a different posterior or continuation policy does not
automatically give that consistency. Terminal verified outcomes anchor the
reward; teacher forecasts remain separately scored auxiliary targets.

Compare plain Gus, Gus with explicit bounded search, and a student trained on
improved plans. Report search cost separately. An external search gain belongs
to the combined system; internal planning in the one-pass network is a further
empirical claim for held-out continuation tests and `clean-neuron` interventions.
Multi-step prediction loss can encourage useful representations without proving
that the network discovered search.

## What the frozen scoreboard still needs

[[gus-frozen-benchmark-2026-10-18]] currently stores recorded outcomes, model
predictions and exact values for a restricted two-tile subset under the separate
three-style field. It does **not** contain measured Walt continuation success
from every position, or all alternative actions' future Zeb splats. The existing
splat net is approximate, not a demonstrated exact match to its teacher.

A versioned supplement should distinguish three labels at each selected root:

- The pinned Walt/Zeb teacher's forecast, with its model and continuation scope.
- The pinned Walt/Zeb baseline's make/set success when actually continued under
  a frozen legal hidden-deal distribution and fixed partner/opponents.
- Each candidate's paired continuation success on those same initial conditions.

Preserve the root information, hidden-deal/sample identities as audit sidecars,
policy/configuration hashes, deterministic budgets, seeds, terminal results and
sample counts. Report exact certificates where enumeration is feasible and
uncertainty for sampled estimates. Changed candidate actions require new lawful
continuations; the recorded baseline win is not their outcome. Training uses
separate deals, the repeated panel is development, and final confirmation stays
fresh. Preserve frozen v1 and publish the supplement under a new identity.

Matching a teacher's forecast does not entail beating its policy. Perfectly
matching its action values and taking the same argmax reproduces that decision.
Improvement must come from better continuation planning or another measured
learner change, verified through paired outcomes under the fixed controls.
The proposed scoreboard tests that claim rather than assuming it.
