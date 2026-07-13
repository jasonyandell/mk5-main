---
title: The Gestation (Era 5, 2026-02-17 .. 2026-04-08)
kind: trail
first_seen: 2026-07-06
last_updated: 2026-07-06
status: complete
---

## Overview

The Gestation is the 52-day span between the [[expected-q-value]]-era capacity-ceiling
close and [[lem]]'s first commit. Zero commits land on any ref in this window — verified
three independent ways (`git log --all`, a full-log `awk` date filter, a reflog grep, all
empty) — and zero beads are created, closed, or touched. It is also the most
idea-dense stretch in the project's archaeology: roughly fifteen substantive Texas-42
conversations, in which the entire next year's cast of names — [[lem]], [[burl]], Harl,
LLem, walker — gets born out loud, with nothing committed to disk. See
[[era5-gestation|conversation digest]] for the full quote trail.

The era opens against a specific, git-verified negative result: commit `6081420`
(2026-02-16, the last commit before the gap) closed the [[full-teacher-eq-experiment]]
at ~74% vs random for a 3.3M-parameter [[zeb]], described in its own message as
"a capacity ceiling for this approach" — see [[alphazero-under-imperfect-information]]
for the full era-4 build that produced it. Everything in this window is a reaction to that
close.

## The autopsy of Zeb

2026-02-18, Jason pastes the live `ZebModel`/`ZebEmbeddings` code and probes the
belief-head plateau while it is still training: *"I wonder if the policy head has to be
scalar logits. truly there is not one value in this hidden information game... the value
head... does great with a loss under 0.1... and belief converges fast. not many cycles
and belief caps out and can't break through."*

2026-03-11 states the closing verdict on the AlphaZero line of attack: *"n=10 e[q] beats
it... I think the root of that was imperfect information. when training the oracle we
tried to keep the hand the same and pick 10 random distributions of other player
dominoes hoping it would generalize but it cratered... I left that experience with the
intuition that we were lying to the model and it got confused."* The ruling: Zeb's
plateau was diagnosed as an imperfect-information training artifact, not a capacity or
compute limit.

## The wall, stated as role-not-appearance

2026-03-07 (`vision-models-vs-conceptual-compression-in-dominoes`) draws the distinction
that structures everything after it — a domino is not a stable visual category: *"the 5-4
is not a 5, necessarily. nor a 4 necessarily. sometimes it's a mid 5, sometimes it's a
trump... you can't look at the 5-4 and draw a fuzzy, good enough line around it, like you
can a road."* Same day, [[candlewax]] is used as an argument before it has the name:
*"the 42 potential is not gaussian at all. it is lumpy spiky and has long weird tails...
the average score of a pdf is meaningless if the PDF is a non gaussian spiky melted
candlewax."*

2026-03-15 (`upgrading-zeb-with-world-models-and-reasoning`, 134 turns to 03-24) produces
the sharpest statement of the wall in the whole corpus, rejecting continued
AlphaZero-Zeb: *"the wall was that these images, they're all gorgeous rendering of
suboptimal paths. distilling that run would just be distilling the heuristic I used to
select >=18 because that's required to win 42."* The wall is a **consumption** problem —
whose answer is worked out three different ways this era and never nailed at the
consumer end. See [[ideated-not-built]] for the mechanisms this diagnosis produced.

## The perception/decision split — Burl and Lem

Same 2026-03-15 thread, immediately after the wall statement: *"it's kind of related to
the world model idea. just... the world is the e[q] distribution which... I suppose kind
of IS the world of 42 in a way."* The decision/navigator model is named on the spot:
*"ok write this up in an artifact as Lem: March 2026"* (2026-03-15T23:12:19). Six days
later (2026-03-21T01:20:16), the perception model is named in the same East-Texas
sharecropper scheme that produced [[zeb]]: *"burl is a type of wood in addition to a name
which somehow fits and my fingers wanted to say burl so burl it is."* The design tagline —
*"Burl sees the world. Lem navigates it"* — becomes the era's headline answer to the wall.

Both names are **coined here, not built here**. As of 2026-03-31: *"can't wait to build
burl"* (still future work). The [[lem]] and [[burl]] that actually ship, starting
2026-04-09 and 2026-04-18 respectively, are real repo entities built on different
premises than the era-5 design docs describe — see [[ideated-not-built]] for the full
account of what was designed here versus what those later, same-named projects became.

`[[walker]]` is also coined in this thread (2026-03-18T01:27:37), for a low-value domino
that quietly wins late tricks because nobody can follow suit: *"you will see the 6-5
could cost you, you could see the 2-1 is a walker."*

## The honest doubts

2026-03-29 (`building-ai-robust-to-uncertainty-beyond-games`) generalizes the stumbling
block out of 42 and names a problem the era never gets an instrument for: *"I think I'm
asking lem to develop instincts... but personalities are heuristics. boo. ultimately they
have to be variables of some metric and I select the metrics and yuck."* And the reframe
that makes the imperfect-information wall feel ordinary: *"a strategy that is robust
under uncertainty is just.. a strategy. everyday stuff. you're never sure anything is
really going to work in life. there isn't any perfect information. so it's a surprising
stumbling block."*

2026-03-31 (`fine-tuning-models-through-game-failure-loops`) coins **LLem** as a pun
("an LLM for lem. l l lem. lol yay") and states, unresolved: *"I honestly don't think
it's so much complex as it is subtle."*

## The reframe: player to narrator

2026-04-03 (`questions-in-planning-and-rlaif-reward-models`, 58 turns) works out a
"perfect judge" architecture end to end — information state → policy bundle →
hidden-world sampler → terminal utility → GRPO reward — under the placeholder *"we will
call it P for now and rename it later,"* then christens it **Harl** in the same
East-Texas naming session that produced Zeb and Burl.

The next day (2026-04-04, `ratchet-mechanism-research-paper`) states the era's most
consequential move: *"the value there is not a better player... it's not a better
player, it's a better narrator... when my family talks about 42 games online they
mention the personalities more than the play."* This re-points "better" from win-rate
toward legibility — the direct source of the standing project frame that belief's value
is legibility, not marks. The same conversation names the online-learning gap and leaves
it open: *"at the end of the game, I have changed. the model has not... There was other
research that LLMs actually get dumber when fed their own content back. How is the
contradiction resolved?"* No resolution is recorded in-window.

## The first concrete build plan

By 2026-04-07 (`llm-star-vs-eq-ev-loop`) the ideation converges to a concrete kickoff:
grade an LLM's last-real-decision against [[expected-q-value]] n=10 greedy-by-EV,
pointed directly at the real `forge/eq` repo path. This is the direct seed of
[[lem]]'s STaR harness and, later, [[jud]]. The window closes 2026-04-08 on Gemma
fine-tune economics; one day later, `a8bccfa` lands and [[lem]]'s repo trail begins.

## What actually ran

The one thing in this "gestation" that executed: **gofish**, a card-game testbed built
with a collaborator to prove the auto-research ratchet loop cheaply before betting it on 42 —
5700 games/sec on a 3050 Ti, a rebel policy beating a tuned heuristic 56-57%
(2026-03-25). The ratchet's stopping condition is stated the same week (2026-03-22):
*"if we ever find an algorithm that can crack 42... capable of forward planning, we can
release the ratchet and stop hill climbing."*

## Corrections to the received story

- **[[lem]] was not founded in this era.** Its first repo artifacts (`lem/narrate/`,
  `lem/rules/primer.md`) land at `a8bccfa`/`6bb8a40` on 2026-04-09 — the day after this
  window closes. Any dating of LEM's founding inside 2026-02-17..2026-04-08 is a date
  migration.
- **"Crystal Palace → Crystal Forge" is not a rename.** Both names predate this window:
  "Crystal Palace" was already the rule-engine nickname (`569feb0`, 2025-12-21); "Crystal
  Forge" was already the ML pipeline name (`2559818`, 2025-12-30). A 2026-03-12 chat
  draft reused "Crystal Palace" loosely as a validation-target label; no causality runs
  between the two names in either direction.
- See [[ideated-not-built]] for the full corrected account of Harl, LLem, "lewm", and the
  autonomous-research architecture, and for which of this era's designs the shipped
  pipeline actually used.

## Net effect

The Gestation did not break the consumption wall. It reframed the founding question —
from "what do I do with this amazing champion" to "how do I make this champion legible
to a person" — which every later era ([[belief-trajectory]], [[gus]], [[jud]]) inherits.
See [[candlewax]] for the wall's precise later statement and [[ideated-not-built]] for
what this era tried and did not finish.

## Links

[[ideated-not-built]] [[lem]] [[burl]] [[zeb]] [[expected-q-value]]
[[candlewax]] [[jud]] [[era5-gestation]]
