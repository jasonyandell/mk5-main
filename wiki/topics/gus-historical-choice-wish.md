---
title: "Gus historical choice teaching wish"
kind: topic
first_seen: 2026-10-20
last_updated: 2026-10-20
status: active
---
<!-- SPDX-FileCopyrightText: Jason Yandell -->
<!-- SPDX-License-Identifier: CC0-1.0 -->

## IDEATED wish — not built

Conversation-derived, project-only wish: use continuation machinery to discover
better contingent decisions, then train [[gus-tiny-gpt]] on those decisions.
This is a proposed teaching direction, not a capability, experiment, or
empirical result. The research loop remains stopped; no training is authorized
by this note. It does not attribute any previous training result to a cause.

The target is a history-conditioned choice value:

`Q(h,a) = Pr(we succeed | observed history h, choose a, specified subsequent policies)`

Here `h` abbreviates the viewer's complete observable decision context: their
own hand together with the public history and other public context. Each such
history is paired with a decision target calculated using it. The targets keep
three roles distinct. Legality constrains the available action set. Under a
specified player model, action-choice likelihood reweights possible hidden
situations according to how likely that actor was to choose the observed
action. A continuation target evaluates future choices under specified
policies; by itself, it does not explain why the historical action was
selected. Silas's public history is not assumed absent; the open question is
how behavioral evidence and historical targets are represented and used.

At the root, the belief must account for the pre-root history as well as carry
its evidence forward through subsequent public observations. A fixed-policy
value evaluates a specified lawful continuation. An optimized value instead
chooses the best lawful contingent continuation. In either case, indistinguishable
worlds share the same choice; the policy may adapt after a public observation
separates the actor's information states. Hidden world state remains teacher
evidence and never enters actor inputs.

The teaching wish is a course of consequential forks with explanations of the
contingent choices still ahead: what the action commits to, what public evidence
can change the next decision, and how a sequence of choices shapes later
choices. The adopted formulation is: “what Gus really needs is to learn what a
sequence of choices means for the choices still ahead.” This is the assistant's
formulation adopted by Jason. A finite teacher already exists;
the full history-informed course and its integration into Gus are not claimed
to be implemented. See [[gus-continuation-teaching-tools]] for the inventory of
current tools and the stopped-loop boundary.
