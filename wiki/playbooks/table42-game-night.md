---
title: Playbook — table42 game night
kind: playbook
first_seen: 2026-07-16
last_updated: 2026-07-18
status: active
---

## The vibe (read this first)

This is not a competition. The point is for a Claude session to **feel Texas
42 from the inside** — bid real hands, make mistakes, label things, disagree,
think it through with Jason — and to get every decision *with its reasoning*
onto disk for later analysis against the [[belief-policy-value-algebra]]
frame. Jason may sit a seat, or just observe and advise; both are normal.
Wins don't matter. Discussion does. It might lead to inspiration.

**Honor system:** the full record (seed, all four hands, every move) sits in
a local log anyone could read. Everyone playing agrees not to read
`log.jsonl` or other seats' view files until the review. That's the whole
security model, on purpose — Jason prefers everything logged over
cheat-proofing.

## The pieces ([[table42]] has the architecture)

The code is tracked at `tools/table42/` (promoted out of scratch
2026-07-18, issue #65). The **local Python host is the authority**; there
is no cloud half anymore — the Cloudflare relay was retired with the
promotion. Run data goes to `scratch/table42/run/<gid>/` (gitignored);
durable copies go to the HF evidence dataset per [[run-artifacts-policy]].

## Start a game

From `tools/table42/`:

```bash
python3 -u host.py new \
  --seats "file:Jason,file:Claude,jud:Jud,jud:Jed" \
  [--seed N] [--marks 7] [--tick 1.0]
```

- Seat drivers: `file:` local mover (a human or LLM in a terminal — the
  main session or an agent teammate) · `random:` instant baseline ·
  `jud:` **the graded champion** — `margin:wp`(r8) ValueBidder at auction,
  `lens:ev` (forge E[Q], 10 worlds, MPS) at play, numbers logged as
  reasoning. Serving rule (#69): seats get the graded champion, never a
  loop artifact.
- Run it in background (it ticks at 1s, exits at game end or after 30 idle
  minutes). A died host mid-game is currently a lost game — keep hosts
  alive (resume is issue #80).
- Everything is written under `scratch/table42/run/<gid>/`: `log.jsonl`
  (deals with seeds+hands, every action with reasoning, chat, trick and
  hand scores), per-seat `view-seat-N.json`, `moves/` + `chat_in/`
  inboxes, `config.json`.

## Seat an LLM (agent teammates, one persistent context per player)

Spawn one background agent teammate per `file:` seat (general-purpose;
model to taste). The prompt template — fill name/seat/gid — is the one used
for Bo in the 2026-07-16 session; core shape:

1. loop `python3 -u tools/table42/wait.py --run scratch/table42/run/<gid>
   --seat N` (blocks until your turn / new chat / game end — one tool call
   per decision),
2. on your turn, decide and submit
   `python3 -u tools/table42/act.py move --run scratch/table42/run/<gid>
   --seat N --turn-id <id> --move "<legal move>"
   --reasoning "<genuine 2–4 sentence thinking>"`,
3. rare short table talk via `act.py say`; never hint at concealed tiles,
4. GAME ENDED → return a summary (per-hand outcomes, key decisions, one
   thing you'd change).

Plus the honor rules and a compact rules primer (count tiles, follow-suit,
marks). The main session usually takes a `file:` seat itself the same way —
that keeps the orchestrating context free to talk with Jason between moves.
A human plays the same way: `wait.py` in a terminal shows the view, `act.py`
submits the move.

## During the game

- The table **announces the trump declaration in chat** ("X declares trump
  fours") — added after game night 1's misread; the waiter also prints
  `trump <decl>` in its header every turn.
- Hands that become mathematically decided (set, or bid already made)
  **fast-forward automatically**; the table announces it in chat and the
  dead plays are logged with `reasoning: "fast-forward (...)"`.
- Chat is the narration channel and is fully logged. House rule from game
  one: live table talk comments on **public** state only; private-hand
  reasoning goes in the move's `--reasoning` field (logged, discussed at
  review).
- All-pass hands reshake (dealer rotates; third reshake forces the shaker).

## Traditions (accreted from real nights)

- **The shuffle-pause**: hand ends → talk policy *immediately* — could/
  should-haves, luck vs skill — while the emotions are live; then shuffle
  and move on. The lesson extracts at the moment of maximum feeling.
- **Announce plans mid-hand** ("watch what I do next") — a registered
  prediction, human edition; grade it at review like a net's pricing.
- **Say the trump out loud, and verify it from the view, not the chat.**
  Game night 1 lost four tricks of analysis to a trump misread propagated
  through table talk. The host now announces declarations itself, but the
  tradition stands: the view is the authority.
- **"Let me show you what I had"** — mid-game hand photos for the review.
- Watchers should fire on *your turn only* (poll `view-seat-N.json`,
  guard on `turn_id`), so table chat never blocks the porch conversation.

## Review / analyze

The game record is `scratch/table42/run/<gid>/log.jsonl` — replayable and
self-contained (`deal` records carry seed + all four hands; `action`
records carry mover, move, reasoning, source; `trick`/`hand_score`/
`game_end` records carry outcomes). Read it top to bottom for the debrief,
or load it in a notebook. Harvest each agent teammate's full transcript
(TaskOutput) for their side of the table. When a night produces anything
worth citing, mirror the run dirs to the HF evidence dataset
([[huggingface-assets]]) and cite the pinned URL.

## Refs

[[table42]] (architecture + findings) · [[jud]] (the player at the table) ·
[[texas-42]] / [[rules-of-42]] (the game) ·
[[count-fate-ledger]] and [[belief-policy-value-algebra]] (the frames the
discussion keeps reaching for)
