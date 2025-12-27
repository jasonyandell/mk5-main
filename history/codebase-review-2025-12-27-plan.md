# Codebase Review Plan of Attack — 2025-12-27

This plan is derived from the review epic `t42-21ze`. Use beads for execution tracking; this document describes sequencing and rationale.

## Guiding Principles (from CLAUDE.md)

- Correctness and architectural unity over short-term convenience.
- Executors/rules should delegate through `ExecutionContext.rules` (avoid hard-coded invariants like “4 players”).
- Keep the client dumb: server/kernel should compute derived fields; client should consume them without re-deriving rule logic.
- “No legacy”: delete/replace dead paths instead of carrying compatibility scaffolding.
- Run `npm run test:all` before closing any beads issue.

## Phase 0 — Baseline & Guardrails

- Confirm current behavior for each target area before refactors (especially URL replay + special contracts).
- Add/adjust targeted tests only when needed to lock in behavior (avoid broad refactors without safety nets).

## Phase 1 — Fix correctness and architectural invariants (P1)

1. `t42-qsg6` — Fix trick completion assumptions in kernel/view-projection  
   Outcome: derived/view layers never hardcode `currentTrick.length === 4`; nello (3-play tricks) and future variants behave correctly.

2. `t42-9an8` — Make actionToId/actionToLabel exhaustive (fix URL 'unknown' events)  
   Outcome: URL serialization/replay is total over the action set we expect to serialize; no `unknown` action IDs appear in production URLs.

3. `t42-zl13` — Fix hints layer capability + requiredCapabilities semantics  
   Outcome: hints are metadata, not an action-execution gate. Either define and grant `see-hints` properly or change the filtering model so actions remain executable while hints are conditionally visible.

4. `t42-umsi` — Update URL tooling scripts to match current url-compression (remove legacy d=base64)  
   Outcome: the `CLAUDE.md` “generate test from URL” workflow matches the actual URL format produced by the app today, without depending on legacy params.

Recommended dependency ordering: do `t42-9an8` before (or alongside) `t42-umsi`, since tooling and URL compression both depend on stable action IDs.

## Phase 2 — Unify duplicated logic (P2/P3)

5. `t42-6hv5` — Unify action equality/matching logic (avoid JSON.stringify comparisons)  
   Outcome: one shared action-key/equality function is used for authorization checks and transition matching; fewer drift risks.

6. `t42-wutc` — Deduplicate capability builders and tighten playerIndex typing  
   Outcome: remove redundant code and make “playerIndex is 0–3” a boundary-validated invariant, not a cast.

7. `t42-nw1n` — Deduplicate 'which player executes this action' logic  
   Outcome: consistent execution-player inference across HeadlessRoom, gameStore replay, and other tooling.

8. `t42-bxxp` — Remove ad-hoc minimal GameState constructors with magic defaults  
   Outcome: analysis/rules helpers don’t hand-roll partial `GameState`; they use a single factory or reuse `createSetupState/createInitialState` to avoid drift.

9. `t42-u5oc` — Clean up stores (avoid await void, internal client access, subscribe/unsubscribe getters)  
   Outcome: store APIs are consistent (sync vs async), boundaries are respected, and “get current store value” uses `get()` instead of transient subscriptions.

10. `t42-8s1f` — Consolidate scoring helpers (avoid duplicate isGameComplete/getWinningTeam)  
   Outcome: fewer ambiguous exports and less import confusion; clearer semantics for “hand complete” vs “match complete”.

## Phase 3 — Process cleanup (P3)

11. `t42-43w4` — Retire markdown checklist planning workflow (use bd instead)  
   Outcome: no competing task systems; any remaining markdown plans are treated as reference docs only (ideally in `history/`), with beads as the source of truth.

