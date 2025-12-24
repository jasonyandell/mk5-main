import type { GameState } from '../types';

/**
 * Result of checking if hand outcome is determined early.
 * Uses discriminated union to make invalid states unrepresentable.
 */
export type HandOutcome =
  | { isDetermined: false }
  | { isDetermined: true; reason: string; decidedAtTrick?: number };

/**
 * Checks if the hand outcome is mathematically determined
 */
export function checkHandOutcome(state: GameState): HandOutcome {
  // Can't determine outcome during bidding or trump selection
  if (state.phase !== 'playing' && state.phase !== 'scoring') {
    return { isDetermined: false };
  }
  
  // Already in scoring phase
  if (state.phase === 'scoring') {
    return { isDetermined: true, reason: 'Hand complete' };
  }
  
  const bid = state.currentBid;
  
  // No bid or invalid bid - can't determine outcome
  if (!bid || bid.player === -1 || bid.type === 'pass') {
    return { isDetermined: false };
  }
  
  const bidPlayer = state.players[bid.player];
  if (!bidPlayer) {
    throw new Error(`Invalid bid player index: ${bid.player}`);
  }
  const biddingTeam = bidPlayer.teamId;

  const [team0Score, team1Score] = state.teamScores;
  const biddingTeamScore = biddingTeam === 0 ? team0Score : team1Score;
  const defendingTeamScore = biddingTeam === 0 ? team1Score : team0Score;
  const currentTrick = state.tricks.length + 1;

  switch (bid.type) {
    case 'points': {
      // Points bid (30-41)
      const bidValue = bid.value!;

      // Bidding team has made their bid
      if (biddingTeamScore >= bidValue) {
        return {
          isDetermined: true,
          reason: `Bidding team made their ${bidValue} bid`,
          decidedAtTrick: currentTrick
        };
      }

      // Defending team has set the bid
      // Note: "can't make" check (maxPossible < bidValue) is mathematically
      // equivalent since maxPossible = 42 - defendingTeamScore
      if (defendingTeamScore > 42 - bidValue) {
        return {
          isDetermined: true,
          reason: `Defending team set the ${bidValue} bid`,
          decidedAtTrick: currentTrick
        };
      }

      break;
    }

    case 'marks': {
      // Standard marks bid logic (suit/doubles/no-trump trump)
      // Special contracts override via their layers
      //
      // Note: "can't win all" check (biddingScore + remaining < 42) is
      // mathematically equivalent since remaining = 42 - biddingScore - defendingScore
      if (defendingTeamScore > 0) {
        return {
          isDetermined: true,
          reason: 'Defending team scored points on marks bid',
          decidedAtTrick: currentTrick
        };
      }

      break;
    }
  }
  
  return { isDetermined: false };
}