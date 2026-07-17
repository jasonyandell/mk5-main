---
title: table42 — the four-seat table
kind: entity
first_seen: 2026-07-16
last_updated: 2026-07-16
status: active
---

## What it is

The table where Claude sessions (and Jason, when he wants a seat) play real
Texas 42 — built 2026-07-16 so the game could be felt from the inside,
narrated live, and fully recorded for analysis against the
[[belief-policy-value-algebra]] frame. How to run a game night:
[[table42-game-night]]. Code in `scratch/table42/` of the `table42`
worktree — gitignored; keep-or-promote is issue #65.

## Architecture (v2 — the local host is the authority)

- **`host/host.py`** — a Python game authority standing on the project's
  real Python rules: [[forge]]'s zeb engine for play
  (`forge/zeb/game.py`: `legal_actions`/`apply_action`, live per-trick
  `team_points`) plus the arena's auction machinery (`arena/auction.py`:
  `legal_bids`, `score_hand`, marks) and `deal_from_seed`/`hand_seed` for
  reproducible deals. It runs the full marks race, reshakes pass-outs,
  **fast-forwards mathematically decided hands** (set or made — checked on
  live trick points), and writes everything to `run/<gid>/log.jsonl`:
  seed + all four hands per deal, every action **with the mover's
  reasoning**, chat, tricks, scores. 1s tick; 30-minute inactivity exit.
- **Seats** are pluggable movers: `file:` (a local inbox — the main session
  or a persistent agent teammate per player, driven through `wait.py` /
  `act.py`), `cf:` (a human on the relay page), `random:` (baseline).
  A [[jud]] seat is the intended fourth driver (not yet wired).
- **Cloudflare relay** (`worker/`, `public/relay.html`) — dumb glass, no
  game logic, no engine: the host pushes per-seat filtered view blobs up
  (D1 `relay_views`); the browser renders them and appends moves/chat
  (`relay_moves`) that the host polls down each tick. **Hidden hands never
  enter Cloudflare in any form.** Spectator views carry public info only.
- **Honor system**: the local log is complete and readable; players agree
  not to look until review. Chosen deliberately over v1's token-gating —
  full logging beats cheat-proofing for this purpose.

## v1 (retired same day)

The first build inverted the trust shape: the Worker was the authority,
bundling the TypeScript engine and replaying `(config, history)` from D1
per request, with capability-filtered views and access-log auditing. It
worked — full games validated end-to-end — but every logic change was a
deploy, logs lived in D1, hidden state sat in the cloud, and Jason called
the HeadlessRoom substrate half-baked for this use. Code preserved in
`scratch/table42/v1/`; its game logs in `scratch/table42/logs/`.

## Findings

- **[[intermediate-ai]] bidding is minutes-per-decision headless** — each
  candidate bid × sims rolls minimax to terminal from a fresh 28-tile
  position; measured 6+ minutes for one bid decision. Why v1's bots were
  swapped from PIMC to model brains.
- **The zeb engine hosts interactive play cleanly**: a full random game
  (auction → 7-mark race, fast-forward included) runs in ~0.6s CPU; the
  arena's auction module slotted in unchanged.
- v1 measurements: worker replay-per-request stayed ~150–250ms across a
  339-move game; Cloudflare's bot check 403s Python's default urllib
  User-Agent (set a custom one).

## Links

[[table42-game-night]] · [[engine]] · [[forge]] · [[jud]] ·
[[intermediate-ai]] · [[belief-policy-value-algebra]] · [[texas-42]]
