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
  `act.py`), `cf:` (a human on the relay page), `random:` (baseline), and
  `jud:` — [[jud]] v1 in the seat (wired 2026-07-16): ValueBidder over
  JudNet with the marks-to-7 utility at auction, JudPlay (argmax
  E[bidding-team pts], defenders minimize) at play, and every move logs its
  actual numbers — P(make)+utility per candidate bid, E[pts] per candidate
  play — as the reasoning field.
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

### Game night 1 (2026-07-16, game `0716-222859`)

Jason + jud vs Claude (Fable session) + jud; three hands played to marks
3-0 Jason&Jud, abandoned at hand 4's auction near midnight. Full record
in `host/run/0716-222859/log.jsonl`; the seat's first-person account is
[[playing-from-the-inside]]. What the table produced:

- **Belief contagion between language-users.** Jason misread hand 3's
  declaration (fours) as sixes, said "goes in 6s" in chat, and Claude —
  whose waiter didn't print the decl field — inherited the false premise
  and *defended it against four tricks of contradicting evidence*,
  diagnosing an engine bug before doubting the premise. The two nets never
  wavered. Zero-support policy priors don't update, they misperceive: a
  declarer leading naked off-suit had no mass in Jason's human-trained
  prior, so his brain edited a public fact instead. b = B(π) demonstrated
  on a *scoreboard-visible* variable.
- **jud's partner-model is field-fragile at the auction.** Hand 3: jud bid
  31 in fours on two trumps missing the boss (P(make)=0.68 claimed); it
  made 39-3 only because the human partner had passed holding
  4-4 + 5-5 + 15 count — a hand no jud-shaped partner ever passes, so the
  head's pass-semantics were trained on a species the partner doesn't
  belong to. Probe filed:
  [issue #66](https://github.com/jasonyandell/mk5-main/issues/66).
- **Count-consolidation is a live jud-ism, both directions**: last-seat
  argmax repeatedly banked loose count into already-won tricks (Jed feeding
  Claude's boss 6-5 the 6-4 and 5-0 in hand 2; Jud welding 6-4 onto
  Jason's 5-5 for a 26-point trick in hand 3). Looks like partner loyalty;
  is pure value pricing. Predictable once you know the policy — "the better
  your field model, the less of the world is luck."
- **Family vocabulary imported**: *rathouse luck* (Jason's grandfather —
  wrong play rescued by the world; the outcome leak with a folk name, the
  thing [[count-fate-ledger]]'s referees formalize), its inverse (right
  play chosen by a non-load-bearing reason), the *high trump / good lead /
  walker* taxonomy (walker = trash promoted by history — belief-relative,
  time-varying), and the family bid scale (30 "doesn't stink" / 31 "some
  confidence" / 32 "I'm serious") — a graded-confidence code on exactly
  the magnitude channel jud's cheapest-positive-utility bidding leaves
  dead.
- **Inside-feel results**: discretion is scarce (Claude made two free
  choices in all of hand 1 — plans die of forcedness, not wrongness);
  bid-reach must be counted in points not tricks (hand 2: five controlled
  tricks cap at 30, bid 32, scored 29 — set by own arithmetic); a strong
  hand is one whose value is *insensitive* to the unseen (strength =
  variance reduction across b, felt at the hand-2 auction).

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
