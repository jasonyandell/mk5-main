/**
 * Minimal GameState factory for rule calculations and analysis.
 *
 * This is a separate module to avoid circular dependencies.
 * The core/state module imports from core/actions which imports from layers,
 * and layers/base imports from core/rules. If rules.ts imported from state.ts,
 * it would create a circular dependency.
 */

import type { GameState, TrumpSelection, Play, Player, LedSuitOrNone } from '../types';
import { GAME_CONSTANTS } from '../constants';
import { EMPTY_BID, NO_LEAD_SUIT, NO_BIDDER } from '../types';

/**
 * Creates a minimal but valid GameState for rule calculations and analysis.
 *
 * Use this factory instead of constructing GameState objects inline.
 * Provides sensible defaults for required fields while allowing targeted overrides.
 *
 * Common use cases:
 * - Trick winner calculation (override trump, currentTrick, currentSuit)
 * - Hand analysis during bidding (override trump, players with hands)
 *
 * @param overrides Partial GameState with fields to override
 * @returns A complete, valid GameState
 */
export function createMinimalState(overrides: {
  trump?: TrumpSelection;
  currentTrick?: Play[];
  currentSuit?: LedSuitOrNone;
  players?: Player[];
  phase?: GameState['phase'];
  currentPlayer?: number;
  dealer?: number;
  tricks?: GameState['tricks'];
  teamScores?: [number, number];
  teamMarks?: [number, number];
} = {}): GameState {
  const playerTypes: ('human' | 'ai')[] = ['human', 'ai', 'ai', 'ai'];
  const theme = 'business';
  const colorOverrides: Record<string, string> = {};

  const defaultPlayers: Player[] = [
    { id: 0, name: 'Player 1', hand: [], teamId: 0, marks: 0 },
    { id: 1, name: 'Player 2', hand: [], teamId: 1, marks: 0 },
    { id: 2, name: 'Player 3', hand: [], teamId: 0, marks: 0 },
    { id: 3, name: 'Player 4', hand: [], teamId: 1, marks: 0 },
  ];

  return {
    initialConfig: {
      playerTypes,
      shuffleSeed: 0,
      theme,
      colorOverrides
    },
    theme,
    colorOverrides,
    phase: overrides.phase ?? 'playing',
    players: overrides.players ?? defaultPlayers,
    currentPlayer: overrides.currentPlayer ?? 0,
    dealer: overrides.dealer ?? 0,
    bids: [],
    currentBid: EMPTY_BID,
    winningBidder: NO_BIDDER,
    trump: overrides.trump ?? { type: 'not-selected' },
    tricks: overrides.tricks ?? [],
    currentTrick: overrides.currentTrick ?? [],
    currentSuit: overrides.currentSuit ?? NO_LEAD_SUIT,
    teamScores: overrides.teamScores ?? [0, 0],
    teamMarks: overrides.teamMarks ?? [0, 0],
    gameTarget: GAME_CONSTANTS.DEFAULT_GAME_TARGET,
    shuffleSeed: 0,
    playerTypes,
    actionHistory: []
  };
}
