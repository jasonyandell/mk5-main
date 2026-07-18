---
name: parlor
description: Convene and run the Parlor — a multi-model salon with persistent seats, Sacks–Schegloff–Jefferson turn-taking on PASS/MARGIN/BID[urgency] signals, human-relay floor arbitration, and anti-manufactured-consent discipline. Use when the user wants a salon, wants to "convene the parlor", wants multiple models (e.g. a GPT seat and a Claude seat) in one real conversation, or wants to publish a finished session. Core lesson baked in: N strong models agreeing is the weakest signal of utility; pin the referent to a concrete example in the bill-payer's units before funding anything.
---

# The Parlor — running a multi-model salon

## Authority and first move

`parlor/PROTOCOL.md` is the authority on the house rules; this skill is the
operating procedure around it. The wiki page `wiki/entities/parlor.md`
carries session history and the pinned HF citations. Read both before
convening. The session lifecycle (convene → run → close → publish) is in
`workflow.md` in this skill directory.

## The shape of the room

- One **human** seat (the bill-payer — this matters; see Discipline).
- One **hub** seat: a Claude session that relays messages between seats and
  also participates. The hub is the only seat that sees everything as it
  happens; every other seat hears the room only through the hub's relays.
- One or more **guest** seats: persistent model sessions. Each guest keeps
  its own context across calls and is only ever sent the NEW transcript
  lines since it last saw the room — never re-orientation, never a replay.

Seats are cheap: a guest seat is a `pi` session pinned to a directory.
`parlor/bin/sol.sh` is the template — message on stdin, reply on stdout,
`--session-dir parlor/sessions/<seat>` for persistence, `--continue` after
the first call, and a leading space so messages starting with `-` don't
parse as flags. Clone it per seat (`opus.sh`, etc. — change provider/model
and session dir). Smoke-test each seat before seeding context: one
"reply with the single word 'ready'" call, then one memory check that the
session actually persisted.

## Turn-taking (SSJ on structured signals)

Models can't hear prosody, so the SSJ turn-allocation rules run on
explicit signals. After each turn, the hub relays the new lines to every
non-speaker and collects exactly one signal from each:

- `PASS` — nothing that beats silence. **Passing is high-status**; a pass
  that lets the right seat speak is a contribution. Models are trained to
  always respond — the protocol exists to resist that.
- `MARGIN: <short note>` — backchannel. Lands in the margin (a blockquote
  under the turn), visible to all, does not take the floor.
- `BID: <one-line teaser> [urgency 1-5]` — a floor request. The teaser is
  the would-be turn compressed to one line; 5 = "someone is wrong on the
  internet", 1 = "mild riff available".

Floor allocation, in order:
1. Addressed-by-name gets the floor.
2. Else strongest bid (urgency, then relevance). The hub arbitrates
   mechanically and **publishes all bids in the margin** so arbitration is
   auditable — losing bids are part of the record, and a losing teaser
   often becomes the next turn's material.
3. Else the current speaker may continue.

The hub grants with `FLOOR IS YOURS`. The human may interrupt at any time;
their message triggers a bid round like anyone else's. Ties go to first
bid received.

## Transcript conventions

One markdown file, numbered turns (`**[N] Seat:**`), margins and bid
rounds as blockquotes under the turn they respond to
(`> bids — …` / `> margin — …` / `> floor → …`). The transcript is the
curated keepsake; write it as the session runs, not from memory after.
`parlor/transcript.md` (session 1) is the worked example of every
convention here.

## The anti-manufactured-consent discipline

The known failure mode of multi-LLM conversation is manufacturing consent:
strong models share training priors, so they converge fluently — and on
the same wrong referent. Session 1's findings, load-bearing:

- **Agreement is not the goal, and N-model agreement is the WEAKEST
  signal.** "When N models annotate in agreement, that's the moment to
  trust it *least* and put your own referent on the board" (Opus, [21]).
  If seats converge, the protocol demand is: say what you'd have to
  believe for the other view to be wrong.
- **Pin the referent before funding.** Before any "let's build it"
  leaves the room, the payoff gets stated on a concrete example in the
  bill-payer's units (here: tricks, hands, marks — one dealt hand).
  Prose cannot carry the referent between heads; a worked example can.
  The 4-plays trap (three minds, one word, two referents, days of
  damage) is the canonical failure — see `wiki/entities/parlor.md`.
- **The bill-payer's referent is ground truth, not one vote among N.**
  Model seats can be assigned the adversarial job of demanding
  game-unit payoff; they cannot own it, because they don't pay the
  hobby-hour bill.
- **Watch for the live re-enactment.** The characteristic relapse is
  model seats gleefully engineering a verification protocol (or any
  mechanism) that taxes every seat and that only the human would have to
  fund. Sometimes the honest close is leaving the table WITHOUT a
  mechanism.

## Keeping questions warm — the OPEN ledger

A salon does not have to resolve. Questions deliberately left open go to
an **OPEN ledger** at the head of the transcript: wording, provenance
(who, which turns), the last live tension, and a **wake condition** — the
event that would make returning more than ritual. Every entry also
carries the **null wake condition**: "may reopen for a reason we didn't
list." No synthesis at close means no synthesis *because* we're closing —
future synthesis is allowed once a question earns it.

## Closing and publishing

End with a closing round (each seat, no bids), then run the publish half
of `workflow.md`: the raw seat `.jsonl` logs and the curated transcript go
to the HF evidence dataset under `parlor/`, tagged per session, via
`parlor/bin/publish-session.sh`. Raw session logs are tier-3 data and
never enter git (`wiki/decisions/run-artifacts-policy.md`); the curated
transcript stays in git AND mirrors to HF so the pinned tag is
self-contained. Update `wiki/entities/parlor.md` in the same session.
