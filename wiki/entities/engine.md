---
title: TypeScript Game Engine
kind: entity
first_seen: 8d26e0d
last_updated: local-2026-07-06
status: active
---

## What it is

The TypeScript game engine is the authoritative implementation of [[texas-42]] rules, state
transitions, legality checks, and visible-information accounting. It lives at
`src/game/core/` (`src/core/` never existed in this repo — `git log --all -- src/core`
returns zero commits at any point; `src/game/core/state.ts` has git history back to the
repo's earliest commit `f989134`). (burl/OVERVIEW.md @ 8d26e0d)

Its origin is [[web-game]] (2025-07 .. 2025-12): the pure-functional,
event-sourced core, the `GameLayer` variant system (nello/plunge/sevens/splash),
and PIMC-minimax as the working AI were all built there, pre-dating [[forge]],
[[gus]], and [[burl]] by months.

## Distinct from Forge

The engine is distinct from [[forge]], which hosts the perfect-information solver, E[Q]
framework, [[zeb]], and visualizers. The engine handles the rules and state; Forge handles
the search and learning infrastructure built on top.

## Role for Burl

[[burl]] wraps the engine as a set of epistemic tool calls exposing facts about the current
state:

| Tool | What it exposes |
|---|---|
| `is_legal(dom)` | Legality of playing a domino, with reason if not |
| `is_trump(dom)` | Whether a domino is trump under the current declaration |
| `unseen()` | Dominoes not in the narrator's hand and not yet played |
| `void_audit(player, suit)` | Whether a player has been proven void in a suit |
| `trump_declared()` | The current trump declaration |

The engine validates every committed play. If [[burl]] commits an illegal play, the engine
rejects it and Burl gets another turn. Legal-move compliance is a software invariant, not
a trained behavior. (burl/OVERVIEW.md @ 8d26e0d)

## Role for LEM

LEM did not depend on the engine directly per the LEM OVERVIEW. Suit/trick logic
(`forge/oracle/tables.py`: `can_follow`, `led_suit_for_lead_domino`, `trick_rank`) was
imported by [[narration]] via Forge, but the TypeScript engine as a distinct entity was
not named as a LEM dependency.
