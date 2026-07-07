---
title: Ideated, Not Built — Era 5's Unbuilt Record
kind: topic
first_seen: afd4802
last_updated: afd4802
status: complete
---

## Overview

During [[the-gestation]] (2026-02-17 .. 2026-04-08), zero commits landed on any ref
while roughly fifteen substantive Texas-42 conversations designed a whole generation of
architecture. This page is the "what was never tried" record: every name and design
this window produced, classified explicitly as **IDEATED**, **BUILT**, or **RENAMED**
against git/beads/disk evidence, with the honest gap between what was designed and what
the project actually shipped later under the same or different names.

Convention followed throughout: **IDEATED** = discussed/designed in conversation only,
zero repo trace (commit, bead, or file) anywhere in the window. **BUILT** = a repo
artifact exists. **RENAMED** = one name was formally replaced by another; **substitution**
= a design's *role* was later discharged by a different, pre-existing mechanism, not a
straight rename.

## World models for Zeb — IDEATED, world-model framing explicitly retracted

**Question asked**: could a world-model-flavored, forward-reasoning system replace or
upgrade [[zeb]]'s AlphaZero-style self-play — "something with that ability to reason
forward in some space" (2026-03-15T21:58:01)? Sharpened same-day into: given that E[Q]
distributions are already a cheap, high-fidelity ground-truth signal, *"what we need to
learn is how to navigate these blobs"* (22:10:52).

**What happened**: a single 134-turn thread (`upgrading-zeb-with-world-models-and-reasoning`,
2322e051, 2026-03-15 to 2026-03-24). Dreamer (an actual world-model RL system) was
seriously entertained — *"I still think dreamer will be good for us... explore forward
and be able to learn why you have this distribution"* (23:17:57) — then explicitly
dropped nine days later: *"nah ok forget the dreamer for now. simple. we finally cash in
on eq and self play a model 2 navigator... I bet it beats e[q] n=10, which alphazero Zeb
couldn't do"* (2026-03-18T01:48:29). This is a real ruling-out, not an unnoticed road.

**Status**: IDEATED. Zero commits, beads, or files anywhere in the 52-day window (three
independent methods, confirmed). The Dreamer/world-model framing itself is IDEATED-then-
RETRACTED — no successor artifact exists for it.

## "lewm" — a conversation title, not a design

**Question asked** (2026-03-24, `lewm-world-model-relevance-assessment`, 427836eb):
prompted by an arxiv paper, could a pixel/frame-based world-model architecture (Atari-
style, robotics-style) be adapted to 42 by turning it into an animated real-time game so
existing techniques would have pixels to train on? Jason frames it explicitly as *"how
can I friggin jam this concept into my space to see if it's useful"* (17:25:03) — a
brainstorm, not a plan.

**What happened**: the thread pivots almost immediately back to the already-named
[[burl]]/[[lem]] design (the full Burl artifact is re-pasted verbatim), reaffirming that
42's state is *conceptual*, not physical/pixel-based (17:43:06): *"world models need a
world and a world in these definitions is a thing you can touch, see, inspect."* No code,
artifact, or experiment resulted.

**Status**: IDEATED — and even that overstates it. The term "lewm" never appears in the
conversation's body; it occurs exactly once, as the conversation's auto-generated title
(a portmanteau of "Lem" + "World Model"). It is not a design Jason proposed and set
aside; it is a title labeling a one-off relevance-assessment brainstorm. No page should
credit "lewm" as an entity.

## The hill-climbing proof — a think-piece, consumed as an argument

**Question asked** (2026-03-21, `hill-climbing-algorithm-and-forward-planning-necessity`,
68746f82): can 42 be solved by a "hill-climbing" model — a single simple function,
without forward search — if followed correctly? Jason requests proof-by-contradiction
explicitly: *"I want to prove that there is a hill climbing algorithm for 42... I'm
trying to prove by contradiction that I do indeed need forward planning"* (01:36:16),
grounded in the repo's existing "42 algebra" formalism.

**What happened**: an 8-turn conversation producing a proof artifact whose content is
not preserved in the available evidence trail (only human turns are captured). Jason's
own reaction — *"you took a nearly incoherent ramble and refined it into a proof"*
(03:13:54) — confirms a proof was produced and satisfied him, but its formal content is
asserted, unverified. The same session pivots directly (same 100-minute window) into the
[[burl]] design paste — the hill-climbing argument was consumed as a reason to keep
investing in forward-planning architecture, not shipped as a standalone artifact.

**Status**: IDEATED. No wiki page, bead, or commit exists for "hill-climbing" as a topic
prior to this page. Its content did not get renamed into anything findable; it fed the
Burl/Lem split directionally, nothing more.

## Autonomous-research architecture (the ratchet, pre-Karpathy-loop framing) — IDEATED, domain-general

**Question asked**: two same-day conversations (2026-03-12, `autonomous-ai-experiment-generation-framework`
f30297a2, and `three-tier-autonomous-ml-research-architecture` 61f968d1) ask whether an
autonomous research loop — one mutable artifact, one frozen eval harness, one scalar
metric, a git-ratchet keeping improvements and discarding regressions — could be
generalized across domains and run without a human in the loop per-experiment, explicitly
citing "Karpathy's recent auto research" (03:29:13) and proposing a three-tier
worker/meta-orchestrator/human architecture. Texas 42 (drafted as "Crystal Palace" in this
thread) is named as only one validation target among several (home automation, phone-use
tasks, PR build-time speedups, Claude.md tool-use efficiency).

**What happened**: pure specification/critique conversations, no code. Neither the
three-tier architecture nor the "generate 50 experiments" framework was ever built under
this name or any other findable name. A repo-wide grep for "crystal palace / three-tier /
meta-orchestrator / worker tier" turns up only unrelated homonyms (the north-star "crystal
palace in the sky" metaphor in `CLAUDE.md`, a pre-existing rule-engine nickname, a
game-rules "three-tier trick-rank function," and an unrelated networking authority
structure) — none instantiate this design.

**What did survive**: a much smaller, single-tier, no-orchestrator instance of the same
instinct — **gofish**, a card-game testbed built with a collaborator specifically because it is
cheaper than 42 to iterate on: *"gofish is a shockingly great testbed for some of the 42
ideas we've been kicking around... we ironed out how to do the auto research loops with
the git ratchet"* (2026-03-21T22:10:44), benchmarked at 5700 games/sec on a 3050 Ti, a
rebel policy beating a tuned heuristic 56-57% (2026-03-25T00:59:01). This is the one thing
in the entire Gestation that actually ran. Its stopping condition was also stated
in-window: *"if we ever find an algorithm that can crack 42 (imperfect information,
competitive/collaborative), that's capable of forward planning, we can release the
ratchet and stop hill climbing"* (2026-03-22T16:27:47).

**Status**: IDEATED (the three-tier architecture itself). **NOT RENAMED** — "Crystal
Palace" was not renamed into "Crystal Forge." Both names predate this conversation: "Crystal
Palace" was already the rule-engine nickname (`569feb0`, 2025-12-21) and "Crystal Forge" was
already the ML pipeline name (`2559818`, 2025-12-30), each roughly two to three months
before this 2026-03-12 chat. Jason reused an existing repo term loosely in a throwaway
sentence; the causality runs neither direction. **BUILT** (smaller, single-tier form only):
gofish.

## Harl / LLem — the judge and the narrator, superseded by the oracle

**Question asked** (2026-04-03, `questions-in-planning-and-rlaif-reward-models`,
e5b097fe, 58 turns): could a trained whole-hand judge — an RLAIF/GRPO reward model
sampling hidden worlds and terminal utility to score an entire information-state-to-action
policy bundle — be built, and could a small LLM (a Gemma narrator) be fine-tuned to
imitate that judge's choices well enough to become "a better narrator... better teacher"
rather than "a better player"?

**What happened**: the judge was worked through end to end under the placeholder *"we
will call it P for now and rename it later"* (03:58:25), then named **Harl** the same
night in a naming session (*"harl. oh and it was Christmas 2025. please draw up a harl
artifact,"* 04:39:04). The narrator/imitator half was floated the same night: *"I'm
thinking the ratchet still on Gemma but have harl as the goal so it tries a move and if
it agrees with harl then that's the star strategy vs whether it wins or not. we are
training a narrator not a policymaker"* (04:57:30) — the first appearance of STaR applied
to 42 in this window. **LLem** is a real, pre-existing name (coined 2026-03-31: *"an LLM
for lem. an LLM for lem. l l lem"*) that Jason explicitly ties to the Harl-narrator idea
the next day (2026-04-04): *"one idea is that LLem is more training for a model that
learns to make the same conclusions and harl... it's a better narrator."* Whether LLem
was ever the *settled, exclusive* name for a Harl-trained narrator, versus a looser
umbrella for "an LLM version of Lem," is not resolved by the corpus either way.

Nothing was committed to the repo under either name. A repo-wide grep for `harl`/`llem`
returns zero real hits in the working tree at any point in history (only coincidental
substrings inside base64-encoded binary blobs).

**What the shipped pipeline actually used**: `lem/gemma_star/star_harness.py` (first
added 2026-04-10, one day after `lem/`'s founding commit) grades Gemma's candidate moves
directly against **the E[Q] oracle's actions** — not against a trained GRPO reward model.
The judge role Harl was designed to fill was discharged by the pre-existing E[Q] oracle
directly. Jason's own 2026-04-05 remark, reading the drafted STaR pipeline doc, notices
the drift in real time: *"that's a meta effect far beyond what we were aiming for with
harl."*

**Status**: Harl — IDEATED only; **substitution, not rename**: Harl (proposed trained
judge) → E[Q]-oracle-as-grader inside the STaR harness (BUILT). Confidence moderate,
inferred from role continuity plus Jason's own remark, not an explicit "replacing Harl"
statement. LLem — a confirmed pre-existing name, IDEATED, never built as a standalone
artifact; its role was absorbed into `lem/gemma_star/`'s STaR pipeline (BUILT, but not
under either name). Both names were still used loosely three weeks into era 6
(2026-04-23: *"things are rolling along on the burl/lem/harl/etc front. it's fractured
quite a bit as I try things"*) — not abandoned so much as blurred into the broader
Lem/STaR effort.

## STaR / rStar — contradiction stated, then walked into

**Question asked** (2026-04-04, `ratchet-mechanism-research-paper`, reading an
rStar-style paper): could a model improve just by trying things, self-fed, without
external training signal? Jason names the online-learning gap precisely: *"the idea that
it can get better just by trying things at all is fascinating... on 42 I keep thinking
about how I learn and LLMs don't.. without actual training... at the end of the game, I
have changed. the model has not."*

**What happened**: in the same breath, Jason flags a contradiction he does not resolve:
*"There was other research that LLMs actually get dumber when fed their own content
back. How is the contradiction resolved against this new article?"* No resolution is
recorded in-window; the question is left in question form, not answered.

**Status**: the contradiction was pre-announced and then walked into. The STaR run that
followed it (era 6, `lem/gemma_star/`) plateaued at 38-39% pass rate over 15 iterations
under K1-only grading — the pattern the flagged contradiction predicted. The honest move
would have been to treat the contradiction as a gate before running, not a footnote after
reading it.

## What this record shows about the wall

None of these designs closed the [[expected-q-value]] consumption question. What they did
was rule things out with real arguments (continuing AlphaZero-Zeb; a literal Dreamer-style
world model; a pixel-native world-model architecture for a concept-shaped game) and
propose consumer hypotheses (perception/decision split, trained judge, narrator-not-player)
that the shipped pipeline mostly bypassed in favor of the pre-existing E[Q] oracle doing
double duty as both world model and judge. Two of the era's own diagnoses — "personalities
are heuristics... I select the metrics and yuck" (2026-03-29) and "I honestly don't think
it's so much complex as it is subtle" (2026-03-31) — were never turned into a measurable
lever in-window. See [[candlewax]] for where the consumption question stands now, and
[[the-gestation]] for the full narrative this record supports.

## Links

[[the-gestation]] [[lem]] [[burl]] [[zeb]] [[jud]] [[candlewax]]
[[expected-q-value]] [[sources/claude/era5-gestation]]
