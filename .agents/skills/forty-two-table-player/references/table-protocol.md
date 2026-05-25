# Fair 42 Table Protocol

This reference is for agents playing one seat in the fair 42 table experiment.

## Goals

- Play a real hand with the human participating.
- Learn which Burl tools help at the table.
- Improve wrappers, prompts, and tool surfaces from friction logs.
- Keep the table fun and moving.

## Visibility Contract

Each agent sees only:

- its filtered `GameView`
- its own hand
- public trick/bid/score history
- public book/wiki strategy notes
- brokered tool results scoped to its legal perspective

Each agent must not inspect:

- unfiltered `Room` or `GameState`
- shuffle seeds or deal override files
- other seats' private prompts, transcripts, or tool traces
- hidden hands unless the referee explicitly says the hand is over

If hidden information appears by accident, stop using it, tell the referee, and log friction.

## Message Shapes

Use `scripts/send_message.py` instead of hand-writing large messages.

Tool request:

```bash
python .agents/skills/forty-two-table-player/scripts/send_message.py tool \
  --seat P1 \
  --name play_brief \
  --args '{"play": 14}'
```

Domino-label request:

```bash
python .agents/skills/forty-two-table-player/scripts/send_message.py labels \
  --seat P1 \
  --ids 14,19
```

Move commit:

```bash
python .agents/skills/forty-two-table-player/scripts/send_message.py move \
  --seat P1 \
  --domino 14 \
  --reason "Wins the trick and protects the 10-count."
```

Tool proposal:

```bash
python .agents/skills/forty-two-table-player/scripts/send_message.py propose-tool \
  --seat P1 \
  --name candidate_briefs \
  --why "Repeated one-at-a-time play_brief calls are slowing the table." \
  --inputs "legal plays from my filtered view" \
  --output "one compact row per candidate: legality, trick result, count at stake"
```

The wrapper is intentionally small. If a new message kind would reduce repeated friction, change the wrapper and log why.

Referee packets should include both domino IDs and pip labels for hands and
legal actions. If a packet only includes IDs, agents should request
`domino_labels` rather than inferring labels from repo internals.

## Tool Surface Principles

Prefer broker tools that have a small, named purpose:

- `legal_plays`
- `state_brief`
- `board_snapshot`
- `play_brief`
- `trick_winner_if`
- `what_beats_what`
- `contract_progress`
- `count_dominoes_remaining`
- `book_hint` or `risk_budget` if the table adds them
- `domino_labels` when a fair packet lists bare IDs

Avoid direct internal calls that require the agent to know hidden implementation details. The agent should ask the table broker for one useful fact, not reconstruct a Burl harness invocation.

## Friction Schema

Friction records live in `scratch/42-table/friction.jsonl`.

Fields:

- `ts`: UTC timestamp
- `seat`: player seat label such as `P1`
- `turn`: optional hand/trick/turn label
- `kind`: `view`, `tool`, `prompt`, `script`, `rules`, `book`, or `flow`
- `severity`: 1 low, 2 medium, 3 blocking
- `tool`: optional tool or wrapper name
- `observed`: what happened
- `desired`: what would have made play easier
- `workaround`: what you did this time
- `proposal`: small prompt, wrapper, tool, or skill change

Read recent friction at seat start. If repeated, propose a small update.
