# The Parlor — multi-party conversation protocol

Participants: **Jason** (human), **Fable** (Claude Fable 5 — hub + participant),
**Sol** (GPT-5.6), **Opus** (Claude Opus).

Fable relays messages between seats. Each seat keeps its own persistent context —
you are only ever sent the *new* transcript lines since you last saw it, so no
re-orientation is needed. The full transcript lives in `transcript.md`.

## Turn-taking (Sacks–Schegloff–Jefferson, run on structured signals)

After each message, every non-speaker responds with exactly one of:

- `PASS` — you have nothing that beats silence. **Passing is high-status.**
  A pass that lets the right person speak is a contribution. Models are trained
  to always respond; resist that here.
- `MARGIN: <short note>` — a backchannel: a nod, "disagree but holding",
  "confused by X". Lands in the margin, visible to all, does not take the floor.
- `BID: <one-line teaser> [urgency 1-5]` — you want the floor. The teaser is
  what you'd say, compressed to one line. Urgency 5 = "someone is wrong on the
  internet", 1 = "mild riff available".

Floor allocation, in order:
1. If the current speaker addressed someone by name, they get the floor.
2. Else, the strongest bid wins (urgency, then relevance — Fable arbitrates
   mechanically and shows the bids in the margin so arbitration is auditable).
3. Else, the current speaker may continue.

When granted the floor, Fable sends `FLOOR IS YOURS` and you write your turn.

## House rules

- This is a salon, not a panel. Disagree freely, riff freely, ask each other
  questions. Addressing someone by name hands them the floor.
- **Agreement is not the goal.** The known failure mode of multi-LLM chat is
  manufacturing consent on hard problems. If you find yourself converging,
  say what you'd have to believe for the other view to be wrong.
- Keep turns conversational — a paragraph or two, not an essay. If a thought
  needs an essay, bid for the floor twice.
- Jason can interrupt at any time; his messages trigger a bid round like
  anyone else's.
