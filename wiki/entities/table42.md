---
title: table42 — the four-seat online table
kind: entity
first_seen: 2026-07-16
last_updated: 2026-07-16
status: active
---

## What it is

A four-seat Texas 42 table on Cloudflare, built 2026-07-16 so Jason and a
Claude session could play a full game from opposite teams with model-brained
partners — the first realization of the [[engine]]'s never-built online mode
([[multiplayer-pattern]]: "designed so a WebSocket/Durable-Object wiring could
reuse Room and GameClient unchanged; that online mode was never built").
Purpose: feel the game from the inside, narrate hands live, and record
everything for post-game analysis against the
[[belief-policy-value-algebra]] frame.

Live at `https://table42.jasonyandell.workers.dev`. Code (~700 lines) in
`scratch/table42/` of the `table42` worktree — gitignored; keep-or-promote is
issue-tracked (see below).

## Architecture

The Worker is a thin, seat-token-gated HTTP shell around the engine's own
composition — no rule logic is reimplemented:

- **State** = `HeadlessRoom(config)` + `replayActions(history)` per request,
  exactly the [[engine-architecture]] event-sourcing invariant. D1 stores
  `(config, history, seq)` per game plus `chat` and `access_log` tables;
  optimistic concurrency via the `seq` column (409 on conflict).
- **Views** go through `getVisibleStateForSession` ([[multiplayer-pattern]]
  capability filtering), plus a table42-specific scrub: `shuffleSeed` and
  `dealOverrides` are removed from every view — `FilteredGameState` carries
  them by default because the Svelte app is local single-player, and knowing
  the seed is knowing the deal.
- **Actions** are validated by regenerating the seat's menu and matching with
  the engine's own `actionsMatch` semantics (type/player/bid/trump/dominoId,
  meta-insensitive), then appended to history.
- **Clients** are all plain pollers against their own seat token:
  - `public/index.html` — Jason's browser UI (vanilla JS, 2.5s poll,
    click-to-play, table-talk chat).
  - `cli.ts` — a Claude session's seat: `view | act | say | wait | reveal`.
  - `jebhost.py` — the partner bots ("Jeb"/"Jed"), each decision routed to a
    regular Claude model (Opus 4.8) via headless `claude -p` in an empty
    scratch cwd; ~6s per decision. Trivial acknowledgments (agree/complete)
    skip the model call.

## No-cheating properties

- Per-seat bearer tokens; no unfiltered endpoint exists; every view/act is
  access-logged in D1.
- `/reveal` (full config, history, final state, chat, access log) refuses
  until `phase === 'game_end'` or both human seats post `/end` in chat.
- The Claude seat's entire play runs through its session transcript — every
  command on the record for post-game audit. Symmetric caveat: the transcript
  necessarily contains that seat's hand and reasoning, so the human must not
  read the session while a game is live.

## Running a game

From `scratch/table42/` in the worktree (deps: repo `npm install`, `wrangler`,
Keychain token `cloudflare-table42`, `claude` CLI authed):

1. **Deploy** (only after code changes):
   `CLOUDFLARE_API_TOKEN=$(security find-generic-password -s cloudflare-table42 -w) CLOUDFLARE_ACCOUNT_ID=eb6564e57c2aebe97bbc5d33a0ffe5cb wrangler deploy`
   (schema: `wrangler d1 execute table42 --remote --file=schema.sql -y`; D1 id
   `cc513c60-7115-4f97-a14a-2c14188fd8f9`).
2. **Create a game**: `POST /api/new` with
   `{"names":["Jason","Claude","Jeb","Jed"]}` → gameId + four seat tokens.
   Teams are seats 0&2 vs 1&3. Write `{base, gameId, seats}` to
   `scratch/table42/.game.json` (gitignored; holds all four tokens).
3. **Hand the human seat 0's URL**: `/?g=<gameId>&t=<token0>`.
4. **Start the jebs**: `python3 -u jebhost.py` (background; reads
   `.game.json`, plays seats 2–3, logs to `logs/jebhost.log`, exits at game
   end).
5. **Play the Claude seat** via `npx tsx cli.ts view` / `act pass` /
   `act play 6-4` / `act trump sixes` / `say <text>`; a session Monitor
   polling the view endpoint emits "MY TURN" / chat events.
6. **Debrief**: after game end (or mutual `/end`), `npx tsx cli.ts reveal`
   writes `logs/reveal-<gameId>.json` with the full deal, history, chat, and
   access log.

Smoke harnesses: `rand.ts` (all-random seats; validated a full 339-move game
to 7 marks, view latency flat ~150–250ms at full history) and `smoke.ts`
(PIMC-brained seats).

## Findings

- **[[intermediate-ai]] bidding is minutes-per-decision headless.** The
  shipped `BeginnerAIStrategy` bid evaluator rolls out every candidate bid ×
  `biddingSimulations` with **minimax to terminal from a fresh 28-tile
  position** (`rolloutToHandEnd` = `minimaxEvaluate`); one bid decision pinned
  a core at 100% for 6+ minutes. This is why table42's partner bots are
  model-brained rather than PIMC-brained; the original PIMC poller survives as
  `jeb.ts`.
- Worker CPU is a non-issue: full-history replay through the layer system
  stays ~150–250ms wall per request over a complete game.

## Links

[[engine]] · [[multiplayer-pattern]] · [[engine-architecture]] ·
[[intermediate-ai]] · [[client-implementation]] ·
[[belief-policy-value-algebra]] · [[texas-42]]
