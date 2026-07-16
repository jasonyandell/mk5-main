---
title: "Strategy Fusion — E[max(score)] ≥ max(E[score]), the diagnosis, and the E[Q] founding sentence"
kind: topic
first_seen: 2026-07-06
last_updated: 2026-07-15
status: complete
---

## Overview

Strategy fusion is the failure mode of a value model trained on fully-observed game states
when that model is then asked to act under partial information: it plays as if it can see all
four hands, because it can. Formally: **E[max_play(outcome)] ≥ max_play(E[outcome])**.
Averaging a perfect-information oracle's per-world best play over sampled hidden-hand
completions is not the value of the imperfect-information game — it is strictly an upper
bound on it (citing Frank & Basin 1998; magnitude studies via Long et al. 2010). This page
covers the whole arc: the Jan 3–5 diagnosis and formal naming (era 2, [[breakthrough-and-oracle]]),
the Jan 9–11 abandoned first fix, the process-discipline correction that preceded the real
fix, and the founding of [[expected-q-value]] as the answer.

An earlier, related symptom — a defeatist "depressed android" that dumped count when losing,
diagnosed and fixed in December 2025 by replacing greedy rollouts with alpha-beta search to
terminal — is a precursor worth naming: the project had already found and fixed one case of
"technically correct, doesn't feel like a person" before strategy fusion surfaced as a second,
harder instance of the same class of bug. No dedicated wiki page for the December fix exists
yet; this note stands in until one does.

## The naming (Jan 3, 2026)

First posed to the project on 2026-01-03 as a one-line research request over a pasted Gemini
critique Jason endorsed outright: *"Averaging the results of Perfect Information games (PIMC)
does not equal the value of the Imperfect Information game. It is strictly an Upper Bound...
Path A calculates the Expected Value of Perfect Play. Reality requires the Maximum Value of
Expected Play... Your 'Path A' will systematically advise you to bid too high."* (conv
adb6de51, "yes to all").

Jason had already felt the bug in the model before the formal name arrived, and held the line
on describing it precisely against a looser paraphrase: *"it doesn't confuse those two. it
thinks 2-2 is 'just as safe' to lead as your high trump even when someone else might have a
trump"* — and, pushed further: *"close but again a distinction. with trumps 6, it considers
the 2-2 a good initial lead. it isn't low vs high trump confusion. it is 'good lead in
isolation is just as good as your trump, maybe better' but it is categorically worse in the
general case."* (conv adb6de51, 2026-01-03). This is not a bidding-magnitude complaint; it is
the precise shape of "averaged perfect-play advice looks locally fine and is globally wrong."

He set hard constraints on the research himself: the DP oracle stays source of truth,
approximations must never invent probability mass, and speculative Bayesian
opponent-modeling language was cut on sight — *"that smells like solutioning."* There is a
fitting two-day commit gap here, Jan 3–4: zero commits repo-wide. The conversation *was* the
work.

## The synthesis (Jan 5, 2026)

`a81fe27` ("Add PI oracle bidding research synthesis") landed `wiki/sources/pi-oracle-bidding-{question,answer}.md`
plus eight per-model raw-response files from a multi-model deep-research pass. The headline
prescription: **use the oracle as an evaluator, not a policy teacher — compute
`max_trump(E[V])`, not `E[max_trump(V)]`.** Fix the strategy (trump/action) before aggregating
over sampled worlds; never let the oracle choose per-completion.

The cited magnitude — *"Strategy fusion affects ~15% of games... average loss ~0.1 points per
game vs. Nash equilibrium"* — is sourced to external trick-taking-game studies (Skat/Bridge,
Long et al. 2010), **not measured in Texas 42**. The research doc's own "What Remains
Unaddressed" section lists an in-domain empirical measurement as an open item. Treat the
~0.1-pts/game figure as external-domain, not a 42 result.

A day before the Jan 3 conversation, `189da59` ("Add marginalized Q-value training pipeline
for imperfect-info play," bead t42-elle, 2026-01-02) had already built the concrete
implementation-side answer: generate training data with multiple opponent-hand distributions
per hand, averaging Q *per action* rather than averaging per-world best strategies. The
prescription and the implementation converged — and directly seed the "founding sentence"
below, six days later.

## The diagnosis continues (Jan 9)

The 97.7%-accurate perfect-information oracle carried the named disease forward from era 2.
Jason's own framing, verbatim (2026-01-09T00:55:30):

> "the weakness of the oracle is a weakness in the model, it knows all... it makes these plays
> because they work. but it can't consider that the enemy has the high trump right now as even
> a concept. it knows the right answer because the right answer was answered before it was even
> asked via reverse induction."

The oracle cheats by seeing four hands, so it plays like a god and teaches like a liar.

## The first fix, and its self-caught failure

The opening idea was an autoregressive nanoGPT trained on move frequencies — tokenize millions
of oracle plays, let cross-entropy learn the robust move. Beads were scaffolded and the
approach built, then killed inside ~36 hours the moment Jason pushed on what the model could
actually see (2026-01-10T21:00:38):

> "bid amount is important we are trying to apply a global value to the 4-3 without seeing what
> dominoes have already been played!!!! holy shit buddy that's WAY off."

Frequency-CE doesn't know *when* information has narrowed the game. "who played what... that's
like 9/10ths of the value of the graveyard." A value function blind to the transcript is a
value function that can't play.

## Grok, don't converge

Before proposing anything new, Jason names a failure mode of the process itself
(2026-01-11T00:13:52 and 2026-01-10T21:24:04):

> "I'm not getting reliable signals from LLMs on this topic. I've used you, ChatGPT, codex... you
> all led me catastrophically astray."

> "we got here by refining a plan back and forth until it converged among multiple LLMs and then
> learned we were solving nothing valuable. in order to prevent us from doing that again, I need
> to actually grok this stuff not just LLMs converge."

North star, same night: "good solid AI. thats the goal. good solid AI. keep that front and
center nothing else matters more." This discipline is captured as its own page:
[[grok-not-converge]].

## The founding sentence

Working it through — and transparently pasting in a ChatGPT consult ("sorry buddy ChatGPT
framed that like something existed but I was just trying to figure out what to do") — the
formulation converges into a single sentence, asked to be checked against a "karpathy himself"
standard (2026-01-10T23:56:04):

> "We train on games where every move is chosen by averaging oracle values over all hidden
> worlds consistent with what's been publicly played so far."

Asked as a proposal, not asserted as architecture, and hedged on novelty: "I know it's popular
and all but I think it might be a good fit." This sentence is the founding statement of
[[expected-q-value]] in this codebase.

## The scope cut that footed the wall

In the same breath, the boundary that turns out to be the whole story (2026-01-11T00:04:42):

> "I also don't care about signaling. yet. I want a damn good solid base player."

This single cut chose argmax over E[Q] as the consumption rule, and it was never revisited
in-era. The consumption question — belief-weighting, signaling, threat-class, personality — got
named four separate times across the era and deferred or cut every single time. Nobody logged
that the deferrals were accumulating into a decision. See [[argmax-q-ceiling]] for how this
reads once the 74% ceiling lands, and [[expected-q-value]] for the built mechanism.

## A note on later framing (dating correction)

A retrospective description of this same founding event — "maybe the most amazing thing I've
ever built and I dunno what to do with it, so let's friggin try stuff" — is sometimes read
back into this window. It does not belong here: an exhaustive name-sweep of this era's
conversational record found zero occurrences of "amazing thing," "founding condition," or any
doubt-register language. The January register is unshaded triumph ("we did it buddy. we fixed
strategy fusion. we have unlimited training (!!!!!)", 2026-01-11T06:13:07). The doubt is a later
description of what was, at the time, pure amazement — and that gap between the two framings is
itself part of the story: the wall was invisible in January precisely because the result felt
so clean.

## The sharpened statement — "eq is not 42" (2026-07-15)

Jason's era-8 restatement of this page's diagnosis, stated while designing
[[otis]]'s lesson extractor: eq, conditioned as hard as possible, is still
playing a different game — one where the opponent *already knows* it doesn't
have to protect anything, and can therefore make a play with terrible odds in
the actual game, with total confidence, and win doing it — in the
perfect-information version. The insurance economy (guards, protection,
information value) exists only *between* information states; within-world
backward induction is played in a game where protection is not a concept. So
E[Q] does not under-price protection — it is computed where the concept is
absent, and belief-conditioning the sampling sharpens *which* fictions are
averaged, never that each is a fiction. The operational consequence — the
oracle bootstraps and referees physics but may never grade a lesson — is
recorded at [[count-fate-ledger]] (The argument's referees).

## Links

[[breakthrough-and-oracle]] [[the-oracle]] [[pimc]] [[rank-vs-price]] [[jud]]
[[expected-q-value]] [[argmax-q-ceiling]] [[grok-not-converge]] [[zeb]] [[eq-genesis]]
[[otis]] [[count-fate-ledger]]
[[era2-breakthrough-oracle|conversation digest]]
[[era3-eq-era|conversation digest]]
