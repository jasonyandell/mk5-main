---
title: The Web Game (Era 1 — prologue)
kind: entity
first_seen: b25480b
last_updated: pending-this-ingest
status: retired
---

## What it is

The web game is the Texas 42 project's founding substrate: a pure-functional,
event-sourced TypeScript engine (`state = replayActions(config, history)`),
built solo across roughly eight ground-up rewrites between 2025-07-19 and
2025-12-23. It predates [[forge]], [[gus]], and [[burl]] entirely — no E[Q],
no oracle, no LLM play-engine exists anywhere in this window. Its job was to
get the rules, the state model, and a working non-ML AI right, and it
succeeded: the engine (`src/game/`), the [[layer-system]], and
capability-based multiplayer (`src/multiplayer/`, [[multiplayer-pattern]])
it produced are still current today — current reference at
[[engine-architecture]] ([[sources/claude/era1-web-game-prologue|conversation digest]]).

This page is a retired-era hub. It exists to route to what Era 1 actually
built, and to preserve two things later eras needed but under-filed: the
project's stated objective ("fun play," not competence) and a debuggable
personality-defect diagnosis ("depressed android") — both registered here,
both rediscovered much later in the [[candlewax]] consumption problem.

## History does not start at commit zero

The first in-window commit, `b25480b` ("starting point," 2025-07-28), is
almost entirely deletion (53 files, +624/−10747) of a prior, already-existing
build. Jason described that predecessor five days earlier: *"I vibe coded a
giant app to play Texas 42. 60k lines of code and markdown specs of dubious
completeness. it was built tdd so it has a boatload of tests."*
([[sources/claude/era1-web-game-prologue|conversation digest]]) At least one full rewrite
predates the visible git history. "Eight rewrites" describes a cadence, not a
single messy project — the era's grammar is delete-and-rebuild, repeated.

An attempt to automate the rebuild itself — driving a "texas42-v2" rewrite
through Claude Flow's swarm orchestrator — failed on first invocation
(`❌ Failed to run SPARC mode: Deno is not defined`, 2025-07-25) and was
abandoned same-day: *"ok this was a disaster. Claude Flow is just broken
beyond redemption."* It was never retried. The method that won instead was
solo, dense commits, every line read — stated directly a month later: *"I
don't like vibe coding. I tried it for a while and learned it is not
effective... I read and understand all of the code and write most of it."*
(2025-10-24)

## The engine, the rewrites, and the Layer system

By 2025-08-20 a pure-functional core is in place; the state lives in the URL
(*"the state is encoded entirely in the URL as well as state object. game
engine is pure functional transitions,"* 2025-08-11) — the direct origin of
CLAUDE.md's "pasted URLs" test doctrine. `docs/arch-r1.md`, a 700-line
event-sourcing plan (2025-08-22), is deleted whole five weeks later
(2025-09-27); the shipped architecture has no verified lineage to it.

October is the architecture crucible. Adding nello as the first rule
*variant* breaks the "variants are just rule-function overrides" design,
because nello changes control flow (declared at trump-selection, the partner
never plays, the hand can end early): *"If I have to modify the game engine
then the variant system isn't complete."* (2025-10-25) The session ends
unresolved. The next day, `dec33ae` (2025-10-26), a `GameLayer` interface
lands — `nello.ts`, `plunge.ts`, `sevens.ts`, `splash.ts` all born in the same
commit. By 2025-11-23: *"nello and sevens are now composed rulesets... [base,
nello, sevens] array of rules + getNextAction and BAM it's perfect."*

Mid-November is where the rewrite cadence becomes legible as fast, not
reckless: the HandOutcome discriminated-union epic (8 sub-tasks) closes in
~19 minutes (2025-11-16); the first "Crystal Palace" dedup epic, estimated
"3-4 weeks," closes in 3 days (Nov 18-20); the Layer-unification epic (22
phases, "big bang migration, no backward compatibility") closes across a
five-hour session on 2025-11-24, with the last phase spilling to the next
morning. This is also the session where Jason writes the words later
canonized as CLAUDE.md's North Star:

> *"it's like my hobby. like those mechanic guys with the project cars in the
> garage that they work on every weekend... I'm on like mark 8 now. like
> fundamental rewrites and it gets better every time and now it's sweeeet."*
> *"I'm building like a crystal palace in the sky over here because that's
> what's fun! so any little blemish just won't do and I chase it down."*
> (2025-11-16, [[sources/claude/era1-web-game-prologue|conversation digest]])

A second, unrelated "Crystal Palace" epic (suit-system unification,
2025-12-20/21) closes the era: `rules-base.ts` becomes the single source of
truth, the `suitAnalysis` cache is deleted ("~50% of allocations are for
suitAnalysis which is never read by AI"), and `checkHandOutcome` drops from
O(28) to O(1) (`7529489`, 2025-12-24, the era's last commit). The two
"Crystal Palace" epics are a name collision, not one initiative — five weeks
apart, different scope.

## The two AI excursions

The first working AI (2025-08-29) is a hand-strength/lexicographic heuristic,
no search — *"when leading I play the strongest, when enemy is winning and I
can't beat them, I play the weakest."* Jason's dad contributes a whole
feature on top of it: PerfectsApp, a "play just one hand" mode with
seed-difficulty filtering, simulating 100 games per seed to keep only
challenging ones. PerfectsApp is built, iterated (the project's first
non-Jason commit), then deleted whole in the December Crystal Palace sweep.

The real algorithm question is asked and resolved in a single evening
(2025-09-25): AlphaZero is raised and self-rejected within the same
conversation (*"there is not perfect information... at the start of the game
you can only see your own dominoes"*); CFR, ISMCTS, and neural-nets-from-
scratch are ranked and set aside on deployment-constraint grounds (Cloudflare
Workers, 128MB, no persistent storage); PIMC (Perfect Information Monte
Carlo — sample opponents' hands, run minimax as if you could see everything,
average) wins by elimination. LLMs are ruled out of the play-engine seat the
same evening, by direct observation: *"i really don't think an llm would do
well. I've got a pretty substantial game going and opus has repeatedly
struggled to reason about the game."* This is the dated origin of the
project's later "trained-out thinking" theme — three months before [[gus]]
or [[forge]] exist.

PIMC itself is not implemented until the era's final week: `3e063ff`
(2025-12-21) replaces greedy heuristic rollouts with alpha-beta minimax to
terminal, curing what Jason named the **"depressed android"** — an AI that
*"makes defeatist plays when losing — dumping count because 'we're losing
anyway.'"* A companion P1 bead removes all temporal fakery from the AI ("no
fake delays or timing" — the earlier AI had been given fake thinking-time to
seem human). PIMC-minimax on the pure event-sourced engine, played through
capability-filtered views, is exactly the baseline [[pimc]] and the E[Q] era
inherit. Full detail: [[pre-ml-ai-attempts]].

The other December excursion, MCCFR, is built for real and then explicitly
killed — see [[pre-ml-ai-attempts]] for the retirement note that states the
project's objective function a month before it was needed:

> *"'Boring and competent' isn't worth the squeeze when we could get that
> with fixed MCTS, and neural nets offer more upside for fun play."*

## Legibility, named early

Triggered by Jason's dad reporting an online 42 bot "he thought cheated,"
a design thread for a Monte-Carlo move-explainer opens 2025-11-28 — an
attempt to restore the game's social/trust dimension through explanation
rather than strength. It stays a UI-adjacent feature in this era; it is the
earliest anchor for the much later "belief's value is legibility, not marks"
line.

## Frameworks evaluated, none adopted

Colyseus, PartyKit, boardgame.io, Nakama, and Cloudflare-Workers-as-host were
all evaluated in conversation across the era; none left a dependency in 249
commits. See [[multiplayer-lineage]] for the full arc and the surviving
Socket/Room/GameClient pattern.

## The book's first appearance

Dennis Roberson's *Winning 42* enters the project as a citation on
2025-07-26, durable today in [[rules-of-42]] (in `docs/rules.md` until the docs→wiki consolidation). An extraction protocol to mine
its hands into test data was designed in detail (2025-08-18) and never
executed in this era. See [[the-book-enters]].

## What this era pre-registered for the wall

Three things later eras rediscovered instead of reading here:

- **The objective.** "Fun play, not competence" was written down in December
  2025 — before E[Q] existed — and not carried forward as the project's
  stated goal.
- **The failure mode.** "Depressed android" (defeatist play under a losing
  position) was diagnosed and fixed as a debuggable defect in December 2025.
  The identical symptom recurring later reads as a mystery only if this
  fix is forgotten.
- **The abstraction lesson.** MCCFR's count-centric compression (135×) lost
  suit-identity and produced competent-but-hollow play. Any lossy summary of
  Texas 42 risks the same failure, one level up.

## Related pages

[[the-book-enters]] · [[multiplayer-lineage]] · [[pre-ml-ai-attempts]] ·
[[pimc]] · [[engine]] · [[texas-42]] · [[sources/claude/era1-web-game-prologue|conversation digest]]
