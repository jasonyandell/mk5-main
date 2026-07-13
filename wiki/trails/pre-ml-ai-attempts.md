---
title: Pre-ML AI attempts — AlphaZero ruled out, PIMC survives, MCCFR built-then-retired
kind: trail
first_seen: 2026-07-06
last_updated: 2026-07-11
status: retired
---

## Three separate questions, not one program

1. **2025-09-25** — could AlphaZero, "a really popular ml topic before LLMs
   had their heyday," be trained on Texas 42 for the play phase alone
   (bidding deferred)? Jason surfaces the objection himself, same
   conversation: *"there is not perfect information... at the start of the
   game you can only see your own dominoes."*
2. **2025-09-25, continuing the same evening** — given imperfect information
   and a hard deployment ceiling (Cloudflare Workers, 128MB, no persistent
   storage, "20-100 games/month"), which known search/learning approach fits:
   could PIMC (sample opponents' hands, run minimax as if you could see
   everything) beat the existing heuristic ranking AI, within a vanilla-
   TypeScript, no-server-state budget?
3. **2025-12-07 .. 2025-12-20** — could a count-centric information-set
   abstraction ("Texas 42 is fundamentally about the 5 count dominoes") make
   Monte Carlo Counterfactual Regret Minimization (MCCFR) tractable enough to
   train an offline strategy table that beats heuristic rollouts, the way
   abstraction made CFR cheap enough to ship for poker?

## AlphaZero / MCTS-for-imperfect-info — IDEATED, never built

No code. The exploratory conversation asks about OpenSpiel, Cloudflare
Workers hosting, and minimax variants "for unknowns"; no implementation
follows. The idea resurfaces two months later only as an inert
future-features placeholder bead, closed with no code artifact ever cited.
Do not read that placeholder as a spec for what PIMC or MCCFR became — they
are independent, earlier-dated lines. AlphaZero was ruled out purely on
architecture (imperfect information, deployment memory ceiling), before any
consumption question could even be asked.

The same evening, LLMs are ruled out of the play-engine seat by direct
observation, not architecture: *"i really don't think an llm would do well.
I've got a pretty substantial game going and opus has repeatedly struggled to
reason about the game."* This is the dated origin of the project's later
"trained-out thinking" theme — three months before [[gus]] or [[forge]]
exist.

## PIMC (Perfect Information Monte Carlo) — the survivor

IDEATED 2025-09-25 as the recommended approach over AlphaZero/MCTS, CFR, and
Information Set MCTS, on stated deployment-constraint grounds. Not
implemented as minimax-to-terminal until the literal last two days of the
era: `3e063ff` (2025-12-21) deletes the 238-line `rollout-strategy.ts` and
replaces greedy per-trick rollouts with alpha-beta minimax, curing what
Jason named the **"depressed android"** — an AI that made defeatist plays
when losing ("dumping count because 'we're losing anyway'"). A companion
bead removes fake AI "thinking time" entirely — the earlier AI had been
given artificial delays to seem human; it was ripped out as a defect, not a
feature.

**Terminology drift, self-corrected in-repo.** The tracker called this same
search-based AI "MCTS" for about two weeks (bead closed 2025-12-02) before a
direct self-correction: *"This is PIMC (Perfect Information Monte Carlo), NOT
MCTS."* (bead filed 2025-12-14) Any narrative citing early-era "MCTS" for
this project is citing a mislabel — it was always PIMC.

PIMC-minimax on the pure event-sourced engine is the baseline [[pimc]] and
the later E[Q] work inherit — it survives *by refusing to abstract*: full
state, sampled worlds, search to terminal.

## MCCFR / CFD1 / CFD2 — built, measured, explicitly retired

Fully built within this era: a real trainer, a custom CFD1/CFD2 compact
binary strategy-serialization format, a 172MB trained-strategy artifact at
250k iterations. Measured: 50K games gave 1,223,043 canonical unique states
vs. 37,659 count-centric states (32.5× compression); the raw 171.5MB JSON
strategy compressed to 1.27MB via CFD2+gzip (135×), with 96,007 valid nodes
preserved round-trip.

It is then explicitly killed, twelve days after its first commit, with the
single most load-bearing sentence in this era:

> *"the count-centric abstraction proved too lossy. The strategy couldn't
> learn suit-specific play (e.g., 'don't lead 5-0 when treys are trump')...
> CFR is punted. 'Boring and competent' isn't worth the squeeze when we
> could get that with fixed MCTS, and neural nets offer more upside for fun
> play."*

The postmortem is retained at `wiki/sources/mccfr-exploration.md`, which
records the direct quality comparison: *"Play quality: Noticeably worse than
simple heuristic rollouts."* Do not carry MCCFR forward as live or resumable
infrastructure — this postmortem is its only surviving trace.

## What this ruled in / out

**Ruled out:** lossy game-specific abstraction as a path to good play
(MCCFR's compact regret table couldn't keep suit-identity); AlphaZero for
the play phase (architecture, not evidence); LLM-as-raw-play-engine (direct
observation, Sept 2025).

**Ruled in:** PIMC-over-full-state as the thing to beat, precisely because
it sidesteps abstraction; "fun play" as the stated true objective, a month
before the later [[candlewax]] consumption problem needed an answer to the
same question.

The general lesson MCCFR's kill note contains — *any* compression that drops
suit-identity produces competent-but-hollow play — is a direct, early,
one-level-down precursor to the later finding that scalar E[Q] summaries can
drop exactly the strategic texture that makes play read as considered rather
than mechanical.

## Related pages

[[web-game]] · [[pimc]] · [[multiplayer-lineage]] · [[candlewax]] ·
[[era1-web-game-prologue|conversation digest]]
