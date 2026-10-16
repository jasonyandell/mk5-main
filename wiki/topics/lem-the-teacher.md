---
title: Lem, the teacher — the generative player, and the program to read it
kind: topic
first_seen: 2026-10-15
last_updated: 2026-10-16
status: active
---

> **The north star is to learn the game of 42 as completely as it can be learned. That's
> policies and beliefs and dynamics... interpretability just sings to me. I claim that there is
> no small set of features that covers 42 and I want to prove myself wrong. I want to train a
> great net up, use interpretability tools and say "hey look at that it's just control and
> qvatz, a word we had to make up".**
>
> — Jason, 2026-10-15

## What Lem is

Lem is the project's generative player: a net trained on 42 that learns policies, beliefs and
dynamics, and models the other seats. It is also the subject of the project's interpretability
work. Jason's line for the PRs: "Lem, who will grow to be our noble teacher."

Lem is an educational play, not a strength play (Jason, 2026-10-16): "It doesn't have to
produce something fun to play against. I just want the meanest nastiest models with lots of
diverse competitors so we see what features arise and can follow them." The population is the
instrument, and the features that arise under it are the result.

The name is reused on purpose. The old [[lem]] (the Little Expert Model, 2026-04, a phone-class
language model taught the game by STaR) is finished and stays as history; its page links here.
Jason: "we are just gonna go through our old names because they're good names and it kinda
echoes."

The Lem desk ([[standing-desks]]) owns this page, the `lem/` tree, the five epics below, and
the Lem Desk notebook (registered as `links/lem-notebook` in the feed).

## The two claims this program tests

1. **The feature claim.** Jason's bet is that no small set of features covers 42, and he wants
   to be proved wrong. The falsifiable form: train a great net, then measure how many features
   explain its play, as a curve of feature count against fidelity. A knee at small k says the
   game is a few named concepts; no knee says it is not. Either answer is a result.
2. **The other-modeling claim.** The book has whole classes of problems around modeling the
   other seats ([[book-assumptions]] A1, A3, A4, A6), and "we can't uniform our way to modeling
   others". The first measured case is the overcall: against the Demon that contests, Zeb
   0.4.1's refit wins 57.0% [53.1, 60.9] where 0.4.0 won 32.6% [28.9, 36.1] on the same seeds,
   and against a table that mostly passes the same refit gives back about 3 points
   ([[rematch-zeb-041-vs-demon-2026-10-15]], [[match-aware-bidding-2026-10-14]]). Jason:
   "strategy... un-fusion. which is right? it depends! model of other is a parameter." A
   hand-tuned constant per opponent type does not scale; a net that infers the table in
   context is the scalable form.

## The five stages

Each stage stands on its own and ships its own result.

### 1. Tools on what exists: the field-net audit

Audit the net Zeb already ships, `zeb/net/margin_d250k_h128x64.fnet` (CC0; hidden 128 and 64,
so 192 neurons and 384 poles). Label every pole two ways:

- **exactly**, by its co-occurrence with [[silas]] predicates over the viewer's information
  (the master, count on the table, a seat's shown void, partner winning, trumps left, what the
  bidding side still needs);
- **cheaply**, by Haiku readers working from short position cards (explain the top-activating
  positions, then score held-out cards; the readers run through the `claude` CLI, each call
  fresh and low in context).

Every share is reported with its interval and against a shuffled-label baseline. The products
are the **concept list** and the **probe kit** (`lem/probe/`), which stages 3 and 4 reuse
unchanged. The design and the pre-registration are [[lem-field-net-audit-2026-10-15]]; the
idea it grew from is [[clean-neurons-42]].

### 2. The population

Every game draws its table at random, so the net must infer who it faces, and who it plays
with: **partners come from the population too**, so a seat reads its partner as well as its
opponents. The engine is **mixed-partner self-play** in a league (league training, PSRO style;
#182): every Lem checkpoint and every configured seat joins the always-on ladder
([[tournament]]), and the hand-written seats are the bootstrap. Jason: "the Silas/pure LLM ideas
will bootstrap nicely but won't scale like self play of mixed partners will." The bootstrap:

- **parameterized Zeb fields**: play-ε (`field_eps`), `void_miss` ([[fallible-field-2026-10-15]]),
  the count lapse ([[count-lapse-2026-10-15]], #172), both overcall tunings (0.4.0's and
  0.4.1's), and the book player;
- **Silas opponents written by a language model** (Jason: "silas yes"): each one a readable
  program, priced in the arena and kept by it (`silas/examples/populations/` is the seed:
  `draw-trump`, `marginal-30`, `optimist-2`/`-3`, `set-bid`, `ten-count-35`);
- **the Demon's plays as behavior.** Its plays are fine as data about a table; its per-world
  values never enter anything ([[strategy-fusion-fence]]).

**Ratings that survive cycles.** The clean hierarchy is already gone: the overcall tunings of
Zeb 0.4.0 and 0.4.1 each beat the other's table (rock and paper; [[rematch-zeb-041-vs-demon-2026-10-15]]),
and Jason expects scissors: "it wasn't the k-levels loop either, just regular policies." The
ladder keeps Bradley–Terry and adds the full pairwise payoff matrix with Nash-averaged ratings,
which are invariant to redundant players, so non-transitivity is shown rather than averaged
away (#182). The arena runs all the time, idling only when no participant has joined for a
while; #173's work-unit budgets made a game about 3× cheaper, and the FAIR CLOCKS rule
(`CLAUDE.md`, #181) keeps every game replayable.

### 3. The nets

- A GPT over the information state that predicts **every seat's next play** (the model of
  others) and **Zeb's value for our seat** (the splat, as in #175's design). Never a world.
- Checkpoints kept along the run, and a scaling family (widths and data sizes), so stage 4 can
  watch concepts form and ask where the knee moves. Every arm of the family runs under one
  pipeline and one loss, the old LEM's lesson ([[lem]], "Lessons learned").
- Data generation is CPU (the vast.ai CPU fleet, through `scripts/cpu-slot` locally). Training is
  the M5 Max GPU or rented B200s.
- The league is the data source as well as the judge: each checkpoint joins the ladder, its
  games (with partners and opponents drawn from the population) feed the next round of
  training, and the model of others is learned from a table that keeps changing (#182).

### 4. Interpretability

The neuron audit of stage 1 run on each checkpoint; concepts branching across checkpoints (the
recursive-generalization prediction of [[clean-neurons-42]]: "trump double", then "highest
unplayed trump", then "master of any suit"); the **feature-count curve**; table-type neurons
(does a neuron read "this table overcalls"?); and the unnamed features, each one a candidate
for a word the game needed and did not have.

### 5. The player

In-context adaptation, priced in the arena ([[tournament]]) against the whole league, with the
payoff matrix and Nash averaging as the scoreboard (#182):

- against each specialist of stage 2: nearly win at both kinds of table (the overcall's two
  tunings are the first pair);
- against the Demon;
- on the book's other-modeling problem classes ([[book-assumptions]] A1, A3, A4, A6), reported
  under [[book-authority]]: a gap is the scope of our model, never a verdict on the book.

## The fence

Lem reads a seat's information and nothing else. Clairvoyant per-world values never label a
concept, never train a head, and never enter the player. The Demon is a sanctioned opponent and
a source of plays only ([[strategy-fusion-fence]]; enforced by
`src/tests/architecture/strategy-fusion-fence.test.ts`).

## Budget

Jason: "if we are talking hundreds of dollars lets do this. if we are talking thousands of
dollars lets think about it. if we are talking more than that lets fantasize."

| item | rough price | tier |
|---|---|---|
| stage 1, the audit | CPU minutes through `cpu-slot`; Haiku readers about 1,200 CLI calls | free |
| stage 2–3 data, about 300 CPU-hours on vast.ai CPU boxes | tens of dollars | do it |
| stage 3 training, a few B200-hours (or the M5 Max) | tens of dollars | do it |
| a scaling family that needs tens of B200-hours | low hundreds | do it, priced first |

Every spend is priced before it is made. Anything over about $100 goes to Jason as a feed ask
before it is spent. Credentials live in the Keychain (`docs/SECRETS.md`); none is ever printed
or written down.

## Where the code lives, and its license

The code is the top-level `lem/` tree; stage 1 is `lem/probe/`, its tests `lem/tests/` (the
`py-lem` suite of `scripts/check`). The old LEM's code was removed at b4a3e5b7 under the no-legacy
rule; the wiki and Hugging Face keep that history ([[lem]]).

License, on the line of [LICENSE](../../LICENSE): the science is CC0, so the probe kit, the
data generation, the training code, the nets and the interpretability results are CC0. Whether
a shipped Lem player follows Zeb (MIT) is the Mayor's call when there is one to ship.

## Prior work

- [[clean-neurons-42]]: the audit-first plan this page absorbs; Cammarata's "already clean"
  thesis as a 42 experiment.
- #175: the GPT-on-42 proposal (tokenizer, small GPT, probe suite); stage 3's trunk and heads
  follow its design, with the population of stage 2 added so the model of others has
  something to infer.
- [[gus-probe]] and [[gus-strategy-tags-probe]]: the oracle era's interpretability receipts. Gus's
  embeddings clustered doubles and count; the book's public-state tags cut a tiny model's regret
  from 2.01 to 1.18 Q-points. Both nets learned from the oracle, so their concepts carry its
  clairvoyance ([[the-wall]]); Zeb's field net does not.
- The old [[lem]] and [[burl]]: language models taught the game (86% comprehension, a 55/100
  bot-match plateau; then tool use). Lem the net is the other direction: the game in weights,
  read back out.
- [[scenario-player]] (Walt and Zeb), the teacher of stage 3's value head; [[silas]], the
  vocabulary every label is written in.

## Issues

One epic per stage: #176 (stage 1, the audit), #177 (the population), #178 (the nets), #179
(interpretability), #180 (the player). #182 is the league: the arena as Lem's population
engine, across stages 2, 3 and 5.

## Links

[[lem]] · [[clean-neurons-42]] · [[lem-field-net-audit-2026-10-15]] · [[scenario-player]] ·
[[silas]] · [[book-assumptions]] · [[book-authority]] · [[strategy-fusion-fence]] · [[the-wall]] ·
[[standing-desks]] · [[tournament]]
