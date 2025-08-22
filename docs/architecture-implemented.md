# Implemented Architecture (Current State)

## Architecture Overview
- Svelte UI consumes store-derived state/actions; the store delegates game logic to a modular engine via src/game/index.ts.
- UI handles view switching and UX; rules, state evolution, scoring, and URL compression live behind the engine API.

## Layers
- UI Layer
  - Files: src/App.svelte (onMount→gameActions.loadFromURL(), handleKeydown→gameActions.undo(), reactive view switching and flash), src/lib/components/*
  - Role: Render play area/action panel, manage gestures/debug/flash, dispatch user intent to store.
- Store/Orchestration Layer
  - Files: src/stores/gameStore.ts (gameState, gamePhase, availableActions; gameActions.executeAction()/loadFromURL()/undo())
  - Role: Maintain initial+current state and history, derive available actions, validate/recompute, update URL, notify controllers.
- Game Engine Layer
  - Files: src/game/index.ts (public API); delegates to ./core/*, ./controllers, ./constants
  - Role: Types/rules, valid actions/transitions, pure action execution, player view, scoring, domino utils, URL compression.

## Data Flow
- UI → store (gameActions) → engine (derive/execute) → store updates state/URL → UI re-renders; UI may read scoring helpers for ephemeral display (e.g., calculateTrickWinner()).

## Key Benefits
- Decoupled: UI logic separated from rules via src/game/index.ts.
- Safe interactions: UI gates via availableActions derived from engine transitions.
- Shareable/replayable: URL compression enables deterministic restore from seeds and actions.
- Modular/testable: Rules, scoring, transitions, and compression are small, focused modules.

## Testing
- Deterministic Playwright E2E: tests fix seed via d= URL and testMode=true; sample: src/tests/e2e/basic-gameplay.spec.ts and helper src/tests/e2e/helpers/game-helper.ts

## Pointers
- Store/data-flow details: docs/stores-and-data-flow.md
- Engine API surface: docs/game-engine-api.md
- Testing (deterministic Playwright): docs/testing.md

