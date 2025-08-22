# Stores and Data Flow (Implemented)

## Exposed Stores (src/stores/gameStore.ts)
- gameState: writable current state; initialState snapshot; actionHistory list; validation error reporting
- availableActions: derived from getNextStates($gameState), filtered for privacy; testMode exposes all
- playerView: derived getPlayerView($gameState, $playerId)
- gamePhase: derived $gameState.phase
- uiState: derived from gameState+availableActions for deterministic UI toggles

## Game Actions (src/stores/gameStore.ts)
- executeAction(transition): push to history; set new state; validateState(); updateURLWithState(); notify controllers
- loadFromURL(): parse d= param; expandMinimalState(); replay actions via getNextStates(); update stores; validate; notify controllers
- loadState(state): replace initial+current; clear history; update URL; notify controllers
- resetGame(): re-seed; clear history; update URL; notify controllers
- undo(): pop last transition; recompute state from initial+remaining actions; validate; update URL

## Data Flow
- UI → gameActions.* → engine helpers (getNextStates/executeAction/URL) → store mutates writable(s) → UI re-renders
- URL save/load uses encodeURLData/decodeURLData; testMode=true yields deterministic behaviors for E2E.

