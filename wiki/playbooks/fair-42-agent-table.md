---
title: Fair 42 Agent Table
kind: playbook
first_seen: local-2026-05-25
last_updated: local-2026-05-25
status: active
---

## What it is

Fair 42 agent table is a human-in-the-loop play setup for [[texas-42]]. The
human sits in P0 cheater/referee mode, sees the whole deal, talks through the
position, and makes their own moves. Three agents sit in P1-P3 and receive only
their legal seat views.

The point is mostly play. Secondary goals are learning which [[burl]] tools are
usable at game speed, improving wrappers and prompts, and collecting friction
that makes the next hand smoother.

## Pilot scope

The first playable slice deliberately skips bidding. P0's team sets the bid and
trump, then the table plays the hand.

The setup is not a full automation project. The human plays. Agents make moves
only when their seat is on turn, and the referee/broker remains the authority on
state, legality, and visibility.

## Fairness contract

Each agent seat receives only:

- its filtered `GameView`
- its own hand and legal actions
- public trick, score, bid, and trump context
- brokered tool results scoped to its legal perspective
- public book/wiki strategy material

Agents must not inspect raw `Room` state, shuffle seeds, hidden hands, deal
override files, or other seats' private transcripts. If hidden information leaks
accidentally, the seat should stop using it, notify the referee, and log
friction.

## Skill contract

The seat skill is `.agents/skills/forty-two-table-player/SKILL.md`, mirrored to
`.claude/skills/forty-two-table-player/SKILL.md`.

The skill tells agents to:

- read `wiki/AGENTS.md` and only the relevant [[burl]], [[tool-orchestration]],
  [[rules-as-tools]], and [[w42]] pages
- read recent friction before play
- use `scripts/send_message.py` for tool requests, move commits, and tool
  proposals
- use `scripts/friction.py` to append rough edges to
  `scratch/42-table/friction.jsonl`
- prefer changing the send-message or broker wrapper over direct low-level Burl
  calls with many parameters

## Tool posture

Good tools answer compact seat-scoped questions: `legal_plays`, `state_brief`,
`board_snapshot`, `play_brief`, `trick_winner_if`, `what_beats_what`,
`contract_progress`, and `count_dominoes_remaining`.

Bad tool use asks an agent to reconstruct hidden implementation details or call
low-level [[burl]] / [[forge]] functions directly with many positional
parameters. That shape is slow, brittle, and hard to learn from.

When a repeated pattern appears at the table, the preferred move is to propose a
small wrapper such as `candidate_briefs`, `risk_budget`, `book_hint`, or
`count_safety`, then log the friction and keep playing.

## Friction loop

Friction is a first-class table signal. At seat start, agents read recent
entries. During play, they log small pain points with observed behavior,
desired behavior, workaround, and a proposed prompt/tool/skill update.

If a friction item repeats, the next agent should propose a small change instead
of silently working around it again. The skill improves by accumulating these
table receipts and turning repeated rough edges into wrapper or prompt updates.

## Related

[[burl]] · [[burl-lab]] · [[tool-orchestration]] · [[rules-as-tools]] ·
[[w42]] · [[post-commit-q-and-a]] · [[book-strategy-player]]
