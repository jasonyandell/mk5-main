## Player Perspective Layer (Design Spec)

Purpose: Provide secure, deterministic, player-specific views and actions on top of the existing pure game engine without leaking hidden information.

Goals:
- Multi-perspective state generation (0–3) from a single full GameState
- Player-specific action filtering that honors turn and phase
- Simple API for UI and services to request a player’s current view and valid actions
- Seamless integration with GameEngine, applyAction/getValidActions, and URL compression
- Pure functions first; optional thin class wrapper for convenience and debugging

---

### Key Concepts

- Canonical State: The full GameState owned by GameEngine or in a pure flow.
- Perspective State: A sanitized, player-specific view with no hidden info from other players.
- Neutral Actions: Actions that do not belong to a specific player (e.g., complete-trick, score-hand, redeal). UI may auto-execute these or restrict to a host.

---

### Type Definitions (proposed)

These types live in a new module (suggested path: src/game/core/perspective.ts). They compose existing types from src/game/types.ts and intentionally remove hidden info.

```ts
import type { GameState, Domino, Trick, Bid, TrumpSelection, GameAction, SuitAnalysis } from '../types';

// Public info about a player as seen by another player
export interface PublicPlayerInfo {
  id: number;
  name: string;
  teamId: 0 | 1;
  marks: number;
  handCount: number;            // number of tiles remaining in hand (no contents)
  isDealer: boolean;
  isCurrentPlayer: boolean;
}

export interface PlayerSelfInfo {
  id: number;
  name: string;
  teamId: 0 | 1;
  marks: number;
  hand: Domino[];               // full contents (self only)
  suitAnalysis?: SuitAnalysis;  // self only
}

// Sanitized, player-specific game state
export interface PlayerPerspectiveState {
  playerId: number;
  phase: GameState['phase'];
  dealer: number;
  currentPlayer: number;
  trump: TrumpSelection;        // public once declared; when type: 'none', still public
  currentSuit: number;          // -1 if no trick in progress
  bids: Bid[];                  // public bidding history
  currentBid: Bid;              // public winning/current bid
  winningBidder: number;        // -1 when not known
  currentTrick: Trick['plays']; // public plays of the ongoing trick
  tricks: Trick[];              // completed tricks (plays, winner, points, ledSuit)
  teamScores: [number, number]; // public per-hand running score
  teamMarks: [number, number];  // public marks
  gameTarget: number;
  tournamentMode: boolean;

  // Player views
  self: PlayerSelfInfo;
  players: PublicPlayerInfo[];  // 4 entries, self included but with summary view
}

export interface PlayerPerspective {
  state: PlayerPerspectiveState;
  availableActions: GameAction[]; // filtered to the given player only (by default)
}
```

Notes:
- For other players, only handCount is exposed; no hand contents or suitAnalysis. Hand counts can be derived deterministically (7 – cards played by that player).
- The legacy `hands` property is not surfaced in perspective.

---

### Implementation Overview

Pure functions (stateless; do not mutate inputs):

```ts
// 1) Derive per-player hand counts from canonical state
export function computeHandCounts(state: GameState): number[];

// 2) Build a sanitized perspective state for a single player
export function getPlayerPerspectiveState(state: GameState, playerId: number): PlayerPerspectiveState;

// 3) Filter valid actions to only those the given player may perform
export function getValidActionsForPlayer(state: GameState, playerId: number, options?: { includeNeutral?: boolean }): GameAction[];

// 4) Convenience aggregate: all four perspectives at once
export function getAllPerspectives(state: GameState, options?: { includeNeutral?: boolean }): PlayerPerspective[];
```

Details:
- computeHandCounts:
  - Start from 7 (GAME_CONSTANTS.HAND_SIZE) for each player in a fresh hand; during the hand, derive from actual state: simply `state.players[i].hand.length` (preferred, as the engine maintains it faithfully). This also works across redeal/new hand boundaries.
- getPlayerPerspectiveState:
  - Copy public fields as-is from GameState.
  - Build `players` with summaries: {id, name, teamId, marks, handCount, isDealer, isCurrentPlayer}.
  - Build `self` with full hand and optional suitAnalysis.
  - Important: Do not copy other players’ `hand` or `suitAnalysis`.
  - Ensure the returned object is newly constructed (immutability).
- getValidActionsForPlayer:
  - Start with `getValidActions(state)`.
  - Default policy: return only actions where `('player' in action) && action.player === playerId`.
  - Optional `includeNeutral`: when true, also include actions with no `player` field (e.g., `complete-trick`, `score-hand`, `redeal`). UI may gate these by role (host) or auto-execute them.
- getAllPerspectives:
  - Map over playerId ∈ [0..3], building `state` via getPlayerPerspectiveState and `availableActions` via getValidActionsForPlayer.

---

### API Surface

Two integration modes are supported:

1) Pure functional usage (recommended)
```ts
import { getAllPerspectives, getPlayerPerspectiveState, getValidActionsForPlayer } from './core/perspective';

const state = engine.getState();
const perspectives = getAllPerspectives(state);
const p0 = getPlayerPerspectiveState(state, 0);
const p0Actions = getValidActionsForPlayer(state, 0);
```

2) Thin wrapper class for convenience and debug (optional)
```ts
export class PerspectiveService {
  constructor(private engine: GameEngine) {}
  getFor(playerId: number, options?: { includeNeutral?: boolean }): PlayerPerspective {
    const state = this.engine.getState();
    return {
      state: getPlayerPerspectiveState(state, playerId),
      availableActions: getValidActionsForPlayer(state, playerId, options),
    };
  }
}
```

Both approaches preserve determinism and do not mutate engine state.

---

### Player-specific Action Filtering

- Bidding phase: only the currentPlayer may bid/pass; others receive [] by default.
- Trump selection: only winningBidder may select-trump.
- Playing phase: only currentPlayer may play; `complete-trick` is neutral and only appears when 4 plays are present; by default it is excluded unless `includeNeutral` is set.
- Scoring phase: `score-hand` is neutral; excluded by default.
- Bidding all-pass: `redeal` is neutral; excluded by default.

This keeps the client UX simple and secure: a seat only sees what it can do.

---

### Developer Debugging Interface

Provide a helper to visualize what each player sees without leaking hidden data:

```ts
export function buildPerspectiveReport(state: GameState): string {
  const lines: string[] = [];
  for (let p = 0; p < 4; p++) {
    const view = getPlayerPerspectiveState(state, p);
    lines.push(`P${p} hand(${view.self.hand.length}): ${view.self.hand.map(d => d.id).join(' ')}`);
    lines.push(`P${p} actions: ${getValidActionsForPlayer(state, p).map(a => a.type).join(', ')}`);
  }
  return lines.join('\n');
}
```

This can be logged in dev builds or surfaced in a hidden debug panel.

---

### Integration With GameEngine

- Read-only: Use `engine.getState()` to derive perspectives.
- Execution: Clients submit actions back to the host/controller. Optionally, a guard can verify the action is in `getValidActionsForPlayer(state, actorId)` before calling `engine.executeAction(action)`.
- Undo: After `engine.undo()`, recompute perspectives; no special handling needed as perspectives derive from state.

Optional guard (pure):
```ts
export function canPlayerExecute(state: GameState, playerId: number, action: GameAction): boolean {
  const allowed = getValidActionsForPlayer(state, playerId, { includeNeutral: true });
  return allowed.some(a => JSON.stringify(a) === JSON.stringify(action));
}
```

---

### Integration With URL Compression

- Save/share: Use existing `compressGameState(state)` and compress action IDs via `compressActionId(actionToId(action))`.
- Restore: Rebuild full state using `expandMinimalState(minimal)` then replay decompressed actions via `decompressActionId` + a parser back to GameAction (mirror of actionToId).
- Perspectives are then computed from the reconstructed canonical state. No per-player data is stored in the URL, preventing leakage.

---

### Testing Strategy

- Visibility correctness:
  - Ensure perspective for player X never includes other players’ hand contents or suitAnalysis.
  - Ensure handCount equals `state.players[i].hand.length` for all players.
- Action filtering:
  - For each phase, verify only the rightful player sees actionable items by default.
  - With `includeNeutral=true`, neutral actions appear only when valid.
- Determinism & purity:
  - Repeated calls with the same state produce deep-equal perspective outputs.
  - No mutation of input state (freeze state in tests to catch accidental writes).
- Undo/redeal:
  - After `executeAction`/`undo`/`redeal`, perspectives recompute consistently.

---

### Implementation Notes

- Keep perspective builders pure; rely on `clone`-like creation of outputs without copying hidden fields.
- Avoid referencing deprecated `hands` on state in perspectives; rely on authoritative `players[i].hand` maintained by the engine.
- Performance: The data volume is small; building four perspectives on each render is acceptable. If needed, memoize by state identity.

---

### Minimal Work Plan (for implementation)

1) Create src/game/core/perspective.ts with the types and functions above.
2) Add unit tests for visibility and action filtering.
3) Integrate PerspectiveService where UI needs seat-specific data.
4) Optionally add a dev-only report/logger utility.

This layer stays strictly on top of the existing pure engine, enables secure multi-client UX, and dovetails with URL compression and history/undo without additional state.

