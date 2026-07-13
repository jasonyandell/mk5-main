---
title: Multiplayer lineage — evaluated frameworks, distilled pattern
kind: topic
first_seen: 2025-08-19
last_updated: 2026-07-11
status: retired
---

## The question

Could an existing, battle-tested multiplayer framework be adopted wholesale
for [[web-game]], or would the game's specific needs (pure-functional core,
capability-filtered views, URL-replayable state) force a bespoke, minimal
layer instead? Jason at the first decision point: *"I've got this design for
a colyseus layer on my game. question is, should I commit to colyseus or
not?"* (2025-08-02) By late November the question sharpened to "how simple
can this actually be" — the framing of the epic that eventually shipped.

## What happened

**Colyseus** was prototyped informally in July and shelved before this
window opens: *"prototyped colyseus... just a rough prototype that I
tossed."* (2025-08-02) No Colyseus artifact exists anywhere in the era's git
history.

**The architecture arc runs independently of any single framework choice.**
On 2025-09-30 Jason designs a capability-based, variant-as-transformer
multiplayer architecture from a pasted real `GameEngine` class — this
conversation already contains the CLIENT/SERVER/MULTIPLAYER/CORE four-layer
diagram matching the shipped [[multiplayer-pattern]]. On 2025-10-24, implementing
the offline parts and trying to add nello as the first variant, the naive
"variants are rule-function overrides" design breaks: nello changes control
flow (skips trump-selection, skips the partner's turn, ends the hand early).
*"If I have to modify the game engine then the variant system isn't
complete."* The session ends unresolved 2025-10-25 ("these variants are
different games with shared mechanics") and is picked up again once the
`GameLayer` interface lands (see [[web-game]]).

**PartyKit** is named and evaluated 2025-09-09, the outcome of an Aug 2-3
survey that also considered boardgame.io and Nakama. **Cloudflare Workers as
host/backend** is evaluated 2025-09-20, apparently superseded by raw
Cloudflare Workers + Hono + SSE/WebSocket by 2025-09-23/24. Cloudflare's
"vibesdk" app builder is evaluated 2025-09-24 with no confirmed adoption.
Neither PartyKit nor Cloudflare-Workers-as-host leaves a dependency, config,
or source file anywhere in the era's 249 commits. The Cloudflare Workers
128MB/no-persistent-storage constraint does feed forward, though: it is one
of the stated reasons the same week's AI-architecture conversation rules out
heavier search/learning approaches in favor of PIMC (see
[[pre-ml-ai-attempts]]).

**The landing.** Epic `t42-don`, "Simplify Multiplayer Architecture" (closed
2025-11-25), states the before/after directly: *"NetworkGameClient: 550 lines
→ 40 lines... Total multiplayer code: ~50% reduction,"* explicitly "roll
forward / clean break / NO backwards compatibility," and explicitly
"inspired by PartyKit/Colyseus/boardgame.io." Five child beads close the same
day (delete old multiplayer code, create the new simple pattern, refactor
Room, update the game store, fix tests). Two security bugs — unfiltered-state
leaks before JOIN — are caught and fixed same-day. A separate, later,
genuinely optional consensus layer (agree-trick/agree-score) follows two
days after, all closed 2025-11-27, wired into Room config but not documented
inline in the era's multiplayer doc (docs/MULTIPLAYER.md @ 233b7dc5).

## Terminal status

**RENAMED / distilled, not adopted-then-abandoned.** None of Colyseus,
PartyKit, or Cloudflare-Workers-as-host left a surviving dependency. What
survived is their *pattern*: the shipped Socket/Room/GameClient design was
stated "inspired by PartyKit, Colyseus, and boardgame.io"
(docs/MULTIPLAYER.md @ 233b7dc5). The landing artifact —
`src/multiplayer/`, `src/server/` — is current today; current reference at
[[multiplayer-pattern]]. Deploy is GitHub Pages (`d991529`,
2025-08-25), not Cloudflare Workers.

## Related pages

[[web-game]] · [[pre-ml-ai-attempts]] · [[sources/claude/era1-web-game-prologue|conversation digest]]
