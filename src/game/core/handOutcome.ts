import type { GameState } from '../types';
import { createDominoes, getDominoPoints } from './dominoes';

export interface HandOutcome {
  isDetermined: boolean;
  reason?: string;
  decidedAtTrick?: number;
}

/**
 * Calculates the maximum possible points that can still be earned from unplayed dominoes
 */
function calculateRemainingPoints(state: GameState): number {
  // Get all played dominoes
  const playedDominoes = new Set<string>();
  
  // Add dominoes from completed tricks
  state.tricks.forEach(trick => {
    trick.plays.forEach(play => {
      playedDominoes.add(play.domino.id.toString());
    });
  });
  
  // Add dominoes from current trick
  state.currentTrick.forEach(play => {
    playedDominoes.add(play.domino.id.toString());
  });
  
  // Calculate remaining points from counting dominoes
  let remainingPoints = 0;
  const allDominoes = createDominoes();
  
  allDominoes.forEach(domino => {
    if (!playedDominoes.has(domino.id.toString())) {
      const points = getDominoPoints(domino);
      remainingPoints += points;
    }
  });
  
  // Add remaining tricks (1 point each)
  const tricksPlayed = state.tricks.length;
  const remainingTricks = 7 - tricksPlayed;
  remainingPoints += remainingTricks;
  
  return remainingPoints;
}

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
  const defendingTeam = biddingTeam === 0 ? 1 : 0;
  
  const [team0Score, team1Score] = state.teamScores;
  const biddingTeamScore = biddingTeam === 0 ? team0Score : team1Score;
  const defendingTeamScore = biddingTeam === 0 ? team1Score : team0Score;
  
  const remainingPoints = calculateRemainingPoints(state);
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
      
      // Bidding team cannot possibly make their bid
      const maxPossibleScore = biddingTeamScore + remainingPoints;
      if (maxPossibleScore < bidValue) {
        return {
          isDetermined: true,
          reason: `Bidding team cannot reach ${bidValue} (max possible: ${maxPossibleScore})`,
          decidedAtTrick: currentTrick
        };
      }
      
      // Defending team has set the bid
      if (defendingTeamScore > (42 - bidValue)) {
        return {
          isDetermined: true,
          reason: `Defending team set the ${bidValue} bid`,
          decidedAtTrick: currentTrick
        };
      }
      
      break;
    }
    
    case 'marks': {
      // Marks bid - bidding team must win all 42 points
      if (defendingTeamScore > 0) {
        return {
          isDetermined: true,
          reason: 'Defending team scored points on marks bid',
          decidedAtTrick: currentTrick
        };
      }
      
      // If bidding team lost any points, they can't win
      if (biddingTeamScore + remainingPoints < 42) {
        return {
          isDetermined: true,
          reason: 'Bidding team cannot win all 42 points',
          decidedAtTrick: currentTrick
        };
      }
      
      break;
    }
    
    case 'nello': {
      // Nello - bidding team must lose all tricks
      // Check if bidding team has won any tricks
      const biddingTeamTricks = state.tricks.filter(trick => {
        if (trick.winner === undefined) return false;
        const winnerPlayer = state.players[trick.winner];
        if (!winnerPlayer) {
          throw new Error(`Invalid trick winner index: ${trick.winner}`);
        }
        return winnerPlayer.teamId === biddingTeam;
      }).length;
      
      if (biddingTeamTricks > 0) {
        return {
          isDetermined: true,
          reason: 'Bidding team won a trick on nello',
          decidedAtTrick: currentTrick
        };
      }
      
      break;
    }
    
    case 'splash':
    case 'plunge': {
      // Splash/Plunge - bidding team must win all tricks
      // Check if defending team has won any tricks
      const defendingTeamTricks = state.tricks.filter(trick => {
        if (trick.winner === undefined) return false;
        const winnerPlayer = state.players[trick.winner];
        if (!winnerPlayer) {
          throw new Error(`Invalid trick winner index: ${trick.winner}`);
        }
        return winnerPlayer.teamId === defendingTeam;
      }).length;
      
      if (defendingTeamTricks > 0) {
        return {
          isDetermined: true,
          reason: `Defending team won a trick on ${bid.type}`,
          decidedAtTrick: currentTrick
        };
      }
      
      break;
    }
  }
  
  return { isDetermined: false };
}