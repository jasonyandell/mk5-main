# Codebase Review (Non-tests) — 2025-12-27

Scope: all non-test code in this repo (primarily `src/` and `scripts/`). Excluded: `src/tests/**`, Playwright reports, `node_modules/`, build artifacts, and other generated output.

Issue tracking: all actionable items are filed in beads and linked to the epic `t42-21ze`. This document is a narrative index, not the task system.

## Executive Summary

The codebase mostly follows the “pure core + layered composition + dumb client” direction described in `CLAUDE.md` and `docs/ORIENTATION.md`, but there are several high-impact mismatches:

- **Layer invariants are violated in the “dumb client” derived/view layers** (hard-coded 4-play tricks), which breaks variants like nello.
- **Action identity is not treated as a total function** (`actionToId` can return `unknown`), which undermines URL isomorphism and replay tooling.
- **A capability-gated “hints” layer currently conflicts with the capability system** (undefined capability + gating semantics), and can remove actions instead of hiding metadata.
- **Legacy URL tooling scripts still exist and conflict with the current URL format**, including the “generate test from URL” workflow called out as critical in `CLAUDE.md`.
- **A markdown-checkbox task workflow exists (`scripts/implement-plan.sh`) despite bd being the required tracker**, creating duplicated tracking systems.

## Issues Filed (Beads)

Parent epic:

- `t42-21ze` — Codebase review: redundancy/unification sweep (non-tests)

Correctness / Architecture (prioritize first):

- `t42-qsg6` — Fix trick completion assumptions in kernel/view-projection  
  Files: `src/kernel/kernel.ts`, `src/game/view-projection.ts`, (UI consumes these via `src/stores/gameStore.ts`)

- `t42-zl13` — Fix hints layer capability + requiredCapabilities semantics  
  Files: `src/game/layers/hints.ts`, `src/multiplayer/types.ts`, `src/multiplayer/capabilities.ts`

- `t42-9an8` — Make actionToId/actionToLabel exhaustive (fix URL 'unknown' events)  
  Files: `src/game/types.ts`, `src/game/core/actions.ts`, `src/game/core/url-compression.ts`

Tooling / Workflow correctness:

- `t42-umsi` — Update URL tooling scripts to match current url-compression (remove legacy d=base64)  
  Files: `scripts/replay-from-url.js`, `scripts/get-state-from-url.js`, `scripts/encode-url.js`, `scripts/decode-url.js`, plus any docs that reference the old format

Unification / Cleanup (reduce duplicated logic and drift risk):

- `t42-6hv5` — Unify action equality/matching logic (avoid JSON.stringify comparisons)  
  Files: `src/multiplayer/authorization.ts`, `src/kernel/kernel.ts`

- `t42-wutc` — Deduplicate capability builders and tighten playerIndex typing  
  Files: `src/multiplayer/capabilities.ts`, `src/server/Room.ts`

- `t42-nw1n` — Deduplicate 'which player executes this action' logic  
  Files: `src/server/HeadlessRoom.ts`, `src/stores/gameStore.ts`

- `t42-bxxp` — Remove ad-hoc minimal GameState constructors with magic defaults  
  Files: `src/game/core/rules.ts`, `src/game/ai/utilities.ts`

- `t42-u5oc` — Clean up stores (avoid await void, internal client access, subscribe/unsubscribe getters)  
  Files: `src/stores/playerConfigStore.ts`, `src/stores/seedFinderStore.ts`, `src/stores/gameStore.ts`

- `t42-8s1f` — Consolidate scoring helpers (avoid duplicate isGameComplete/getWinningTeam)  
  Files: `src/game/core/state.ts`, `src/game/core/scoring.ts`, `src/game/core/actions.ts`, `src/game/index.ts`

Process / Consistency:

- `t42-43w4` — Retire markdown checklist planning workflow (use bd instead)  
  Files: `scripts/implement-plan.sh`, `docs/rules-gherkin-plan.md`

## Additional Notable Observations (Not Yet Filed)

These are smaller or higher-risk changes that I did not file as separate issues (to avoid clutter), but they are worth noting:

- **Legacy cleanups in UI code**: `src/App.svelte` contains “cleanup legacy override element” logic (`theme-color-overrides` vs `theme-overrides`) and an unusually aggressive forced reflow sequence. This may be necessary, but it’s a maintenance hotspot that could use either simplification or a doc explaining why it’s required.
- **Optional chaining on non-optional fields** (e.g., `state.currentBid?.type`) appears in multiple layers; it reads like state can be missing when it can’t, and makes it harder to trust type invariants.
- **Multiple ways of “getting current store state”** exist (subscribe/unsubscribe patterns instead of `get()`), and similar patterns appear in UI code too (`SeedFinderModal.svelte` vs `seedFinderStore.ts`).

