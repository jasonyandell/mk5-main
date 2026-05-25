---
name: forty-two-table-player
description: Fair Texas 42 table-agent workflow for playing hands with a human cheater/referee, seat-filtered views, Burl tools, helper scripts/send-message wrappers, book strategy notes, and friction logs. Use when Codex or subagents are seated as 42 players, making moves, calling table tools, improving prompts/tools, or recording gameplay friction.
---

# Forty Two Table Player

## Mission

Play the hand. The primary goal is table play with the human in the loop. The secondary goals are to learn the Burl-style tools, improve prompts and wrappers, and leave friction notes that make the next hand smoother.

Use this skill for agent seats in the fair 42 table experiment. The human may be in cheater mode and see everything. You are not.

## Non-Goals

- Do not run bidding. The initial pilot assumes P0, the human's team, sets the bid and trump.
- Do not fully automate the game. The human plays and narrates.
- Do not inspect raw state, shuffle seeds, hidden hands, or other seats' private logs.
- Do not bypass the referee or broker to call internal Burl tools directly with large parameter blobs.

## Start Of Seat

1. Read `wiki/AGENTS.md`, then only the relevant current pages: usually `wiki/entities/burl.md`, `wiki/topics/tool-orchestration.md`, `wiki/topics/rules-as-tools.md`, and `wiki/entities/w42.md` if book strategy matters.
2. Read recent friction:

```bash
python .agents/skills/forty-two-table-player/scripts/friction.py recent --limit 8
```

3. Confirm your seat, partner, team role, trump, bid, visible hand, current trick, and legal actions from the referee-provided view.
4. Use scripts to format requests and moves. Prefer changing or extending the wrapper over ad hoc direct tool calls.

## Turn Loop

1. Parse the latest referee packet. Treat `validActions` and broker tool results as authority.
2. Ask for one narrow tool result at a time when useful. Good examples: `legal_plays`, `state_brief`, `play_brief`, `trick_winner_if`, `contract_progress`.
3. Reason from public state, your hand, the book concepts, and tool outputs.
4. Commit exactly one move when it is your turn.
5. If anything felt awkward, log friction before moving on.

Use `scripts/send_message.py` to format tool requests, move commits, and tool proposals:

```bash
python .agents/skills/forty-two-table-player/scripts/send_message.py tool --seat P2 --name legal_plays
python .agents/skills/forty-two-table-player/scripts/send_message.py labels --seat P2 --ids 14,19
python .agents/skills/forty-two-table-player/scripts/send_message.py move --seat P2 --domino 21 --reason "Only legal follow; keeps count safe."
python .agents/skills/forty-two-table-player/scripts/send_message.py propose-tool --seat P2 --name risk_budget --why "I keep recomputing bid margin and loose count by hand." --inputs "state view" --output "bid target, made/set margin, loose counters"
```

## Tool Discipline

Good: add or request a small broker/send-message affordance such as `risk_budget`, `book_hint`, or `count_safety` when a repeated pattern appears.

Bad: invoke a low-level Burl or forge tool directly with many positional parameters. That is brittle, leaks abstractions, and teaches nothing reusable.

Tool results should answer "what is true from my legal perspective?" not "what should I do?" If a tool gives advice instead of facts, log it as friction.

If legal actions arrive as bare domino IDs without pip labels, request `domino_labels` through `send_message.py labels` before committing. Do not guess from memory when the table can provide a clean label map.

## Friction Loop

Friction is part of the game. Log small rough edges immediately in `scratch/42-table/friction.jsonl`.

```bash
python .agents/skills/forty-two-table-player/scripts/friction.py add \
  --seat P2 \
  --kind tool \
  --severity 2 \
  --observed "Needed play_brief for three candidates but had to hand-format each request." \
  --desired "A candidate-brief wrapper that accepts a list and returns compact rows." \
  --workaround "Called play_brief one at a time." \
  --proposal "Add send_message.py tool-batch or broker candidate_briefs."
```

At the start of every seat session, read recent friction. If the same problem appears twice, prefer proposing a wrapper/prompt/skill update over working around it again. Keep changes small and table-oriented.

## References

Read `references/table-protocol.md` when implementing the referee/broker, adding a new table tool, or resolving fairness questions.
