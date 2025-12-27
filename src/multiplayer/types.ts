/**
 * Multiplayer Types - Consolidated from game/multiplayer/types.ts and shared/multiplayer/protocol.ts
 *
 * This file contains all shared types for the multiplayer system:
 * - Capability system (authorization)
 * - Player sessions
 * - Game state wrappers
 * - View types for clients
 * - Result type helpers
 */

import type { GameAction, GameState, FilteredGameState } from '../game/types';

// ============================================================================
// Capability System
// ============================================================================

/**
 * Capability tokens control what a session can see or do.
 * They are composable and purely descriptive.
 *
 * Execution capabilities (gate action execution):
 * - act-as-player: Can execute actions for a specific player
 *
 * Visibility capabilities (gate state/metadata visibility):
 * - observe-hands: Can see player hands
 * - see-hints: Can see educational hints in action metadata
 */
export type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-hands'; playerIndices: number[] | 'all' }
  | { type: 'see-hints' };

/**
 * Utility: create a capability matcher for comparisons.
 */
function capabilityMatches(candidate: Capability, target: Capability): boolean {
  if (candidate.type !== target.type) {
    return false;
  }

  switch (candidate.type) {
    case 'act-as-player':
      return candidate.playerIndex === (target as typeof candidate).playerIndex;
    case 'observe-hands': {
      const targetCap = target as typeof candidate;
      if (candidate.playerIndices === 'all' || targetCap.playerIndices === 'all') {
        return candidate.playerIndices === targetCap.playerIndices;
      }
      return JSON.stringify([...candidate.playerIndices].sort()) ===
             JSON.stringify([...targetCap.playerIndices].sort());
    }
    default:
      return true;
  }
}

export function hasCapability(session: PlayerSession, target: Capability): boolean {
  return session.capabilities.some(cap => capabilityMatches(cap, target));
}

export function hasCapabilityType<C extends Capability['type']>(
  session: PlayerSession,
  type: C
): boolean {
  return session.capabilities.some(cap => cap.type === type);
}

// ============================================================================
// Player Session
// ============================================================================

/**
 * Player session information for multiplayer games.
 * Separates identity (playerId) from seat position (playerIndex).
 */
export interface PlayerSession {
  playerId: string;              // Unique identifier: "player-0", "ai-1", "alice", etc
  playerIndex: 0 | 1 | 2 | 3;   // Seat position in the game
  controlType: 'human' | 'ai';   // Who controls this player
  isConnected?: boolean;         // Connection status (optional, defaults to true)
  name?: string;                 // Display name (optional)
  capabilities: Capability[];    // Capability tokens determining visibility/authority
}

// ============================================================================
// Multiplayer Game State
// ============================================================================

/**
 * Multiplayer game state stores pure GameState and filters on-demand.
 * Authority stores pure state; filtering happens per-client in createView().
 */
export interface MultiplayerGameState {
  gameId: string;                           // Unique game identifier
  coreState: GameState;                     // Pure GameState (NOT filtered)
  players: readonly PlayerSession[];        // Immutable player sessions
}

/**
 * Request to execute an action on behalf of a player.
 */
export interface ActionRequest {
  playerId: string;  // Player identity (e.g., "player-0", "ai-1")
  action: GameAction;
}

// ============================================================================
// Client View Types
// ============================================================================

/**
 * Server-computed derived fields for dumb client consumption.
 * These fields are computed by the kernel using ExecutionContext.rules
 * to avoid any client-side rule evaluation.
 */
export interface DerivedViewFields {
  /** Is the current trick complete? Uses rules.isTrickComplete for layer-aware completion (e.g., nello = 3 plays) */
  isCurrentTrickComplete: boolean;

  /** Trick winner for current trick (-1 if incomplete) */
  currentTrickWinner: number;

  /** Tooltip metadata for each domino in hand, pre-computed using rules */
  handDominoMeta: HandDominoMeta[];

  /** Current hand points per team, computed from completed tricks */
  currentHandPoints: [number, number];
}

/**
 * Server-computed metadata for a domino in hand.
 * Includes playability and tooltip info derived from rules.
 */
export interface HandDominoMeta {
  dominoId: string;
  isPlayable: boolean;
  isTrump: boolean;
  canFollow: boolean;
  tooltipHint: string;  // e.g., "Trump", "Follows suit", "Can't follow"
}

/**
 * Complete view of game for a client.
 * This is everything a client needs to render the game.
 */
export interface GameView {
  /** Current game state */
  state: FilteredGameState;

  /** Pre-calculated valid actions for current player */
  validActions: ValidAction[];

  /** Pre-calculated transitions for UI rendering */
  transitions: ViewTransition[];

  /** Player information */
  players: PlayerInfo[];

  /** Game metadata */
  metadata: {
    gameId: string;
    layers?: string[];
  };

  /** Server-computed derived fields (no client-side rule evaluation needed) */
  derived: DerivedViewFields;
}

/**
 * Valid action with additional UI hints.
 */
export interface ValidAction {
  /** The actual game action */
  action: GameAction;

  /** Human-readable label */
  label: string;

  /** Optional grouping for UI organization */
  group?: string;

  /** Keyboard shortcut hint */
  shortcut?: string;

  /** Is this a recommended action? */
  recommended?: boolean;
}

/**
 * View transition - action with UI metadata, no leaked state.
 */
export interface ViewTransition {
  /** Unique ID for this transition (for keying) */
  id: string;

  /** Human-readable label */
  label: string;

  /** The actual game action */
  action: GameAction;

  /** Optional grouping for UI organization */
  group?: string;

  /** Is this a recommended action? */
  recommended?: boolean;
}

/**
 * Player information for UI.
 */
export interface PlayerInfo {
  playerId: number;
  controlType: 'human' | 'ai';
  sessionId?: string;
  connected: boolean;
  name?: string;
  avatar?: string;
  capabilities?: Capability[];
}

// ============================================================================
// Result Type
// ============================================================================

/**
 * Result type for operations that can fail.
 */
export type Result<T> =
  | { success: true; value: T }
  | { success: false; error: string };

/**
 * Helper to create success result.
 */
export function ok<T>(value: T): Result<T> {
  return { success: true, value };
}

/**
 * Helper to create error result.
 */
export function err<T>(error: string): Result<T> {
  return { success: false, error };
}
