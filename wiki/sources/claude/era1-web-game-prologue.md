---
title: "Source digest: claude.ai era 1 — Web Game Prologue (2025-07-19 .. 2025-11-30)"
kind: source
first_seen: 2026-07-06
last_updated: 2026-07-06
status: complete
---

## Provenance

Mined from `claude.ai` conversation exports, window 2025-07 through 2025-11 (199-file baseline: Jul 34 + Aug 63 + Sep 41 + Oct 25 + Nov 36; December excluded from this window). 58 candidate files matched a Texas-42 keyword pass and were read in full. Every quote below is Jason's own words, cited as (date, "conversation title", uuid-prefix). This digest backs [[web-game]] and its leaf pages ([[engine]], the layer system, [[pre-ml-ai-attempts|the MCCFR excursion]], [[pimc]], [[multiplayer-lineage]]).

---

## 1. A prior build already existed before this window

By 2025-07-23 ("Texas 42 Component Museum", ba563793), Jason describes a pre-existing build:

> "I vibe coded a giant app to play Texas 42. 60k lines of code and markdown specs of dubious completeness. it was built tdd so it has a boatload of tests."

This predates any rewrite attempt visible in this window — at least one full rewrite (the "vibe-coded original") existed before 2025-07-23, asserted in Jason's own words, unconfirmed against a commit artifact in this scout.

## 2. The claude-flow disaster (2025-07-25)

Two same-day conversations ("Automated Texas 42 Game Development", 87feea3c; "Ultra Mega Vibe 42", 86957711) document an attempt to drive a "texas42-v2" rewrite through Claude Flow (a swarm/hive-mind orchestrator) from a plan file `fourtytwo-v2.md`, mining assets from a prior `~/fourtytwo` directory. It failed immediately:

> "❌ Failed to run SPARC mode: Deno is not defined"

> "ok this was a disaster. Claude Flow is just broken beyond redemption. I still want to find a way to orchestrate and kick off all of these tasks." (repeated three times, same day)

## 3. The North Star quote — "project car" (2025-11-16)

The same conversation UUID as the claude-flow disaster (86957711, "Ultra Mega Vibe 42") was revisited four months later, on 2025-11-16, producing the exact language later canonized in CLAUDE.md's "North Star" section:

> "it's sweet. it's like my hobby. like those mechanic guys with the project cars in the garage that they work on every weekend and dream of taking it to the track but have no near term dates in mind. it has become a nice hobby. gamewise I'm overengineering it to death because I got a couple of things very wrong very early. and because it is easy to bury code problems with LLM code generation. but I pulled it together and am on like mark 8 now. like fundamental rewrites and it gets better every time and now it's sweeeet."

> "oh there are so many bosses. I'm building like a crystal palace in the sky over here because that's what's fun! so any little blemish just won't do and I chase it down."

This is the direct verbatim source of CLAUDE.md's "crystal palace in the sky" / "project car" framing, confirmed as Jason's own words. "Mark 8" here is consistent with CLAUDE.md's later "8th major overhaul."

Same exchange, the origin of the temporal/event-sourcing bug lore:

> "I finally finally found the temporal bug I've had since forever. temporal stuff the way i did it was like kryptonite for the event sourcing that was otherwise amazing. but it never got perfect and I didn't suspect setTimeout because how else are you going to have the AI wait etc etc."

And the description of the first-generation heuristic AI, in Jason's own words:

> "I computed strength by evaluating each domino for what it can beat and what can beat it in every possible scenario given the trumps... when leading I play the strongest (can beat most), when enemy is winning and I can't beat them, I play the weakest. or if partner is winning i play weakest."

A single-hand play mode with seed-difficulty filtering is described the same day:

> "I made a thing to simulate 100 games for every seed and discard seeds unless the win ratio was in the challenging range."

(This became the PerfectsApp feature — built, iterated, later retired; see [[engine]].)

## 4. Framework churn across rewrites

- 2025-08-10/11 ("game is in solidjs with vite and playwright already.")
- 2025-08-26 ("codegen this from the state transitions and the svelte" — Svelte mentioned)
- 2025-09-09 ("I'm using svelte." — confirms Svelte as of that date)
- Multiplayer transport churn: Colyseus considered/prototyped (Jul-Aug) → boardgame.io / Nakama / partykit all evaluated (Aug 2-3) → PartyKit chosen by name (Aug 9) → Cloudflare Workers + Durable Objects chosen by 2025-09-20, then superseded by raw CF Workers + Hono + SSE/WebSocket by 2025-09-23/24. The "vibesdk" (Cloudflare's AI app builder) was evaluated 2025-09-24 with no confirmation it was adopted.

State-in-URL / debug-first testing philosophy, stated directly (source of "pasted URLs" in CLAUDE.md's unit-test description):

> "the state is encoded entirely in the URL as well as state object. game engine is pure functional transitions. debug screen is reporting exact things from state object." (2025-08-11)

## 5. The house rule "short-circuiting" (2025-08-04)

> "the way my family plays is if you can't win a bid (or you've already made enough points to win the bid), we stop playing the hand at that point and award the winner their mark(s). is there a name for this? in my head I call it short circuiting."

A house rule, named by Jason, not (as far as this scout found) an official N42PA term.

## 6. AI strategy arc — direct lineage to PIMC and the layer system

- 2025-08-08: first mention of a Rust solver attempt — "I made this in rust and can explore a stupidly huge search space. can I come close to solving the game?" — plus the first appearance of particle-filter/Bayesian belief-tracking discussion for opponent-hand inference (conceptual ancestor of the later belief-tracking work; no named entity yet in this window).
- 2025-09-11: a precomputed `getDominoStrength()` lookup table (auto-generated, "DO NOT EDIT") and a `SuitAnalysis`/`SuitCount`/`SuitRanking` structure — both interfaces reappear verbatim in a pasted `types.ts` on 2025-11-26, confirming they persisted in the real codebase across at least ten weeks.
- 2025-09-25, a single evening produced four related conversations systematically ruling out AlphaZero/MCTS-for-perfect-info, CFR, and neural-nets-from-scratch in favor of PIMC (Perfect Information Monte Carlo), explicitly because of Cloudflare Workers' 128MB/no-persistent-storage constraints. Why LLMs were ruled out as the play-engine:

  > "i really don't think an llm would do well. I've got a pretty substantial game going and opus has repeatedly struggled to reason about the game."

- Also 2025-09-25: a back-of-envelope for a perfect-solve database (~500M canonical positions after symmetry reduction, "1-20 hours" on a laptop) is proposed as an alternative to NN training — an idea-form ancestor of the later forge/oracle shard pipeline, unnamed here.
- 2025-11-23: confirmation that nello and sevens were successfully implemented as composed rulesets:

  > "nello and sevens are now composed rulesets. you have [base, nello, sevens] array of rules + getNextAction and BAM it's perfect."

  This is the direct verbal ancestor of CLAUDE.md's "Unified Layer system with two surfaces."
- 2025-11-26 through 11-29: PIMC/tree-search work is live and being measured — "65 percent is the new floor" (win rate vs a target, 2025-11-26); concrete throughput ("1.5 minutes for 3k hands at 5000 sims and depth 1", 2025-11-27); an explicit rejection of a Rust port in favor of staying in TypeScript ("I had been considering 'vibe porting' it to rust... but I think that would take a lot longer than even depth 5 games", 2025-11-27); and a design thread for a Monte-Carlo move-explainer aimed at restoring the game's social/trust dimension (2025-11-28).
- A verbatim `types.ts`/`GameState` dump pasted 2025-11-26 shows `BidType` already including `splash`/`plunge` as "compositional, enabled by layers" and `TrumpSelection.type` including `'nello' | 'sevens'` — by this date the layer system was fully live in the actual repo, not just a proposal.

## 7. The layer-system architecture struggle

Three conversations form a tight, dated arc:

1. 2025-09-30 ("Multiplayer game engine for dominoes"): starting from a pasted real `GameEngine` class, Jason designs a capability-based, variant-as-transformer multiplayer architecture. Key line: "variants are just different rule functions" is proposed and refined into "variants are stored in game state and applied at runtime," discussing CRDTs and algebraic effects (considered and explicitly set aside). This produces a formal spec (continued the next day) that already contains the "CLIENT LAYER / SERVER LAYER / MULTIPLAYER LAYER / CORE GAME ENGINE" four-layer diagram matching docs/MULTIPLAYER.md's Room/Socket/GameClient description.
2. 2025-10-24 ("Multiplayer game engine for dominoes" follow-up): Jason returns having implemented the offline parts, tries to add nello as the first variant, and discovers the naive "variants are rule-function overrides" design breaks because nello changes control flow (skips trump-selection, skips partner's turn, ends the hand early). Direct quote:

   > "If I have to modify the game engine then the variant system isn't complete."

   These conversations explicitly reference the live repo at `github.com/jasonyandell/mk5-main` — confirming `mk5-main` was already the repo name by 2025-10-24.
3. 2025-10-25 ("Nello variant implementation challenges"): the unresolved tension is stated cleanly — "these variants are different games with shared mechanics" — and the session ends without a solution, handed off for a future session. (Section 6 above shows the problem was solved by 2025-11-23.)

## 8. Naming collision friction

> "since the US President is also named Trump, I get 'can't respond to this request' errors a lot and it breaks my workflow." (2025-08-05)

A real, dated operational annoyance, not a design decision.

## 9. Task-tracking predates bd/beads

2025-08-27: Jason describes using GitHub Projects, finding it "unwieldy," and wanting Claude Code to manage a todo list directly via GitHub issues — the earliest in-window evidence of the issue-tracking problem that eventually led to beads (and later, beads being retired in favor of GitHub issues again).

## 10. Ideas registered but not confirmed built in this window

- A Roblox client (2025-08-03) — explicitly "wild tangent," "just curious," never pursued further in-window.
- "Contract 42" (bridge-style, same hands dealt to all players, no cheating) and an NYT-style daily-puzzle hand mode (2025-08-27/28) — registered as GitHub todos, no confirmation of implementation in this window.
- An SMS-based play format and PWA distribution (2025-11-27) — the PWA idea lands ("dude. pwa sounds perfect"), no in-window confirmation of shipping.
- Alloy/formal-methods verification of the rules (2025-10-26/27) — "just for fun," explicit daydream, no evidence of follow-through.
- Vibe-porting the engine to Rust for search throughput — explicitly considered and rejected 2025-11-27 in favor of staying in TypeScript.

## 11. Practice evolution: from "vibe coding" to reviewed authorship

> "for the memory system, I don't like vibe coding. I tried it for a while and learned it is not effective. but I'm very much enjoying LLM assisted coding where I read and understand all of the code and write most of it." (2025-10-24)

> "with the 42 fun times it's way more like reviewing my teams work really carefully than being the primary active hand... this has actually been a big motivator for me to go functional. less code means less can go wrong." (2025-11-23)

This documents, in Jason's own words and dated, the shift CLAUDE.md's "No legacy" / "every line of code is a liability" ethos codifies — from a 60k-LOC vibe-coded original (section 1) to a smaller, hand-reviewed, functional-core codebase by late November.

---

## Files read in full but yielding no standalone quotable insight beyond what's captured above

Design/UI/tooling detail sessions: 2025-07-22 (web Texas 42 domino game design), 2025-07-24 (domino game probability calculation), 2025-07-26 (complete game rules; digital game architecture; declarative rule architecture; game rules specification — four separate sessions), 2025-07-27 (game flow navigation), 2025-07-31 (trump hand analysis), 2025-08-04 (game insights), 2025-08-08 (game rules research, cited partially above), 2025-08-17/18 (game development; a treatise example-mining pipeline — tooling only, low evidence of downstream use in this window), 2025-08-22/23 (event-architecture visualization artifacts), 2025-08-27 (design consultation, art/design sourcing, no build outcome), 2025-09-14 (a speedrun-tool tangent), 2025-09-20 (Cloudflare Workers evaluation, cited above), 2025-10-01 (multiplayer architecture specification variants, cited above), 2025-11-28 (pathfinding and test cleanup, a two-line todo re: seed-based test fixtures).
