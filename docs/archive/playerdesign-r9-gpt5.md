# Player Perspective Layer — R9 Minimal Unified (gpt5)

Correct by construction. Every line of code is a liability. A single, pure, leak‑proof whitelist projection with host‑gated neutral actions and deterministic reproducibility. Core stays tiny; transport/dev helpers live outside the core.

## Non‑Negotiables
- Pure, stateless, deterministic
- Whitelist projection only; never clone/hide full GameState
- Leak‑proof by type: only self can contain a hand; others expose counts only
- Neutral actions are visible only when host === true
- Reconstructable from seed + action log; no private data in transport
- UI derives convenience flags from core fields (no drift‑prone duplicates)

## Minimal Types (leak‑proof by construction)
```typescript
export interface PublicPlayerInfo {
  id: number;
  name: string;            // public label
  teamId: 0 | 1;
  handCount: number;       // count only
}

export interface PlayerSelf {
  id: number;
  hand: Domino[];          // full contents for self only
}

export interface PlayerView {
  // Core context
  playerId: number;
  phase: GameState['phase'];
  dealer: number;
  currentPlayer: number;
  trump: TrumpSelection | null;     // null until declared
  currentSuit: number | null;       // null if none
  bids: Bid[];                      // public history only
  currentTrick: Trick['plays'];     // public plays only
  tricks: Trick[];                  // completed tricks
  teamScores: [number, number];
  teamMarks: [number, number];
  gameTarget: number;

  // Player‑specific
  self: PlayerSelf;                 // self hand only here
  players: PublicPlayerInfo[];      // others are public‑only; no hand field

  // Actions for this player at this moment (host‑gated neutrals)
  validActions: GameAction[];
}
```

Notes:
- No isDealer/isCurrentPlayer flags; derive in UI (e.g., view.dealer === view.playerId)
- By type, players[] cannot carry private data; only self.hand may contain dominoes

## Single Core Function + Helper (signatures)
```typescript
// Build view by explicit whitelist; do not read/copy other players’ hands
export function getPlayerView(
  state: GameState,
  playerId: number,
  host = false
): PlayerView

// One place to classify neutral actions
function isNeutral(a: GameAction): boolean
```

Required outcomes (not code):
- Build PlayerView from explicit fields only; no cloning of GameState then hiding
- self.hand = state.players[playerId].hand
- players = state.players.map(p => ({ id: p.id, name: p.name, teamId: p.teamId, handCount: p.hand.length }))
- validActions = getValidActions(state) filtered to actions owned by playerId; if (!host) filter out isNeutral(a)
- No caches; no mutation; same state → same view

## Reproducibility (Core)
```typescript
export interface GameStatePackage {
  seed: string | number;
  actions: GameAction[];
  version?: number; // optional for migration safety
}

export function reconstruct(pkg: GameStatePackage): GameState
```
Constraints:
- Package contains seed + action log only; never private data beyond what’s required for deterministic reconstruction

## Out‑of‑Core Utilities (Optional)
Keep these outside the core module. They must not be persisted with private data and are not required by the core.
```typescript
// URL helpers (transport)
export function toURL(seed: string | number, actions: GameAction[]): string
export function fromURL(s: string): { seed: string | number; actions: GameAction[] }

// Dev helper (debug UI only)
export function allViews(state: GameState): PlayerView[] // [0..3].map(pid => getPlayerView(state, pid, false))
```
Guidelines:
- Dev helpers are not persisted, not sent over network, not embedded in URLs
- Advanced dev tools (replay/validateSecurity) may live in a separate dev‑only module if needed; keep them minimal and clearly non‑core

## Svelte Wiring (example)
```typescript
// One derived store for correctness‑by‑construction rendering
export const currentView = derived([gameState, currentPlayerId, isHost],
  ([$s, $pid, $host]) => getPlayerView($s, $pid, $host === true)
);
```
Usage:
- Render controls only by mapping over currentView.validActions; do not render buttons that aren’t in the list
- Derive booleans in UI, e.g., const isDealer = currentView.dealer === currentView.playerId

## Security Invariants (must hold)
1. Hand privacy: only self.hand is present; players[] exposes handCount only
2. Determinism: same GameState → same PlayerView
3. Stateless: no caches, no retained views
4. Neutral gating: neutral actions appear only when host === true
5. Transport safety: URLs/persisted payloads never contain private data

## Tests (lean, sufficient)
- Leak‑proofing: players[] has no hand; self.hand equals engine hand for playerId; nullables correct
- Neutral gating: getPlayerView(state, pid, false).validActions excludes neutral; host=true includes when appropriate by phase
- Reconstruction determinism: reconstruct(pkg) yields identical PlayerViews (modulo self.hand) across seats when recomputed from the same seed + actions
- UI correctness by construction: components render actions exclusively from currentView.validActions; no ad‑hoc guards or can()

Example (illustrative only):
```typescript
test('no leakage + neutral gating', () => {
  const s = initGame('seed');
  for (let pid = 0; pid < 4; pid++) {
    const v = getPlayerView(s, pid, false);
    v.players.forEach(pp => expect((pp as any).hand).toBeUndefined());
    expect(v.self.id).toBe(pid);
    expect(v.self.hand).toEqual(s.players[pid].hand);
    expect(v.validActions.every(a => !isNeutral(a))).toBe(true);
  }
});
```

## Implementation Target
- One file, ~70–90 LOC for core (types + getPlayerView + isNeutral + reconstruct wrapper)
- External utilities (URL helpers, dev‑only helpers) live in separate modules

## Migration Notes
- From R7‑cc: replace players: { hand?, handCount } with self + players split; keep validActions in the view; extract URL/debug helpers out of core
- From R8‑gpt5: unchanged core semantics; keep the self/players split and separation of concerns; continue to derive UI booleans

