/**
 * Shared helper functions for rule set implementations.
 *
 * These pure utility functions are used by rule sets to avoid duplication
 * and to eliminate modulus arithmetic in favor of semantic helpers.
 */

import type { GameState, Bid } from '../types';
import type { HandOutcome } from './types';
import { getNextPlayer as getNextPlayerCore } from '../core/players';

/**
 * Get partner of a player (opposite seat).
 *
 * In Texas 42, teams are 0-2 and 1-3, so partner is always +2 seats.
 * Uses existing player relationship logic to avoid modulus arithmetic.
 */
export function getPartner(playerIndex: number): number {
  return getPlayerAtOffset(playerIndex, 2);
}

/**
 * Get player at offset using existing getNextPlayer logic.
 * Avoids modulus arithmetic by using core player navigation.
 */
export function getPlayerAtOffset(from: number, offset: number): number {
  let current = from;
  for (let i = 0; i < offset; i++) {
    current = getNextPlayerCore(current);
  }
  return current;
}

/**
 * Get player's team (0 or 1).
 */
export function getPlayerTeam(state: GameState, playerIndex: number): number {
  const player = state.players[playerIndex];
  if (!player) {
    throw new Error(`Invalid player index: ${playerIndex}`);
  }
  return player.teamId;
}

/**
 * Check if bidding team must win all tricks (or must NOT win any).
 *
 * Used by: plunge, splash, sevens (must win all), nello (must win none)
 *
 * This shared helper eliminates code duplication across 4 rule sets.
 *
 * @param state Current game state
 * @param biddingTeam Team ID (0 or 1) of the bidding team
 * @param mustWin true for "must win all tricks", false for "must win none"
 * @returns HandOutcome
 */
export function checkMustWinAllTricks(
  state: GameState,
  biddingTeam: number,
  mustWin: boolean
): HandOutcome {
  // Check each completed trick
  for (let i = 0; i < state.tricks.length; i++) {
    const trick = state.tricks[i];
    if (!trick || trick.winner === undefined) continue;

    const winnerTeam = getPlayerTeam(state, trick.winner);
    const biddingTeamWon = winnerTeam === biddingTeam;

    // For "must win all": fail if opponents won
    if (mustWin && !biddingTeamWon) {
      return {
        isDetermined: true,
        reason: `Defending team won trick ${i + 1}`,
        decidedAtTrick: i + 1
      };
    }

    // For "must win none": fail if bidding team won
    if (!mustWin && biddingTeamWon) {
      return {
        isDetermined: true,
        reason: `Bidding team won trick ${i + 1}`,
        decidedAtTrick: i + 1
      };
    }
  }

  return { isDetermined: false }; // Not determined yet
}

/**
 * Check hand outcome for trick-based special contracts.
 *
 * Used by: sevens, nello, plunge, splash
 *
 * Trick-based contracts have different termination logic than point-based contracts:
 * - Trick-based: Outcome determined by which team wins tricks (not points)
 * - Point-based: Outcome determined by points scored (standard marks bids)
 *
 * This is why trick-based contracts can't use checkStandardHandOutcome() from
 * handOutcome.ts - that function uses point-based logic (teamScores, remainingPoints)
 * which doesn't apply to contracts where points are irrelevant.
 *
 * Pattern:
 * 1. Check for early termination via checkMustWinAllTricks()
 * 2. Check if all 7 tricks are complete
 * 3. Return not determined if neither condition is met
 *
 * @param state Current game state
 * @param biddingTeam Team ID (0 or 1) of the bidding team
 * @param mustWin true for "must win all tricks", false for "must win none"
 * @returns HandOutcome
 */
export function checkTrickBasedHandOutcome(
  state: GameState,
  biddingTeam: number,
  mustWin: boolean
): HandOutcome {
  // Check for early termination based on trick winners
  const earlyTermination = checkMustWinAllTricks(state, biddingTeam, mustWin);

  if (earlyTermination.isDetermined) {
    return earlyTermination;
  }

  // Check if all 7 tricks complete
  if (state.tricks.length >= 7) {
    return {
      isDetermined: true,
      reason: 'All tricks played'
    };
  }

  // Not determined yet - continue playing
  return { isDetermined: false };
}

/**
 * Get highest marks bid value from bids so far.
 *
 * Used by: plunge, splash to determine automatic bid value.
 * Plunge/Splash bids are automatic - they jump over the current high bid.
 */
export function getHighestMarksBid(bids: Bid[]): number {
  return bids
    .filter(b => b.type === 'marks' || b.type === 'plunge' || b.type === 'splash')
    .reduce((max, bid) => Math.max(max, bid.value || 0), 0);
}

/**
 * Count doubles in a hand.
 * Used by plunge (4+ doubles) and splash (3+ doubles) requirements.
 */
export function countDoubles(hand: { high: number; low: number }[]): number {
  return hand.filter(d => d.high === d.low).length;
}
