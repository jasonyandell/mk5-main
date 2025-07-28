import type { GameState, Play, PlayedDomino, Trump } from '../types';
import { BID_TYPES } from '../constants';
import { getDominoValue, getDominoPoints, getDominoSuit } from './dominoes';
import { getBidComparisonValue } from './rules';

/**
 * Checks if a domino follows the led suit (contains the led suit number)
 */
function dominoFollowsLedSuit(domino: { high: number; low: number }, leadSuit: number): boolean {
  return domino.high === leadSuit || domino.low === leadSuit;
}

/**
 * Checks if a domino is trump
 */
function isDominoTrump(domino: { high: number; low: number }, numericTrump: number | null): boolean {
  if (numericTrump === null) return false;
  
  // Special case: doubles trump (numericTrump === 7)
  if (numericTrump === 7) {
    return domino.high === domino.low;
  }
  
  // Regular trump (contains trump suit number)
  // In tournament rules, doubles belong to their natural suit unless doubles are trump
  if (numericTrump === 7) {
    // When doubles are trump, only doubles are trump
    return domino.high === domino.low;
  } else {
    // When specific suit is trump, only dominoes containing that suit are trump
    return domino.high === numericTrump || domino.low === numericTrump;
  }
}

/**
 * Determines the winner of a completed trick (overloaded for different interfaces)
 */
export function calculateTrickWinner(trick: Play[], trump: number): number;
export function calculateTrickWinner(trick: PlayedDomino[], trump: Trump): number;
export function calculateTrickWinner(trick: Play[] | PlayedDomino[], trump: number | Trump): number {
  if (trick.length === 0) {
    throw new Error('Trick cannot be empty');
  }
  
  const leadPlay = trick[0];
  const leadSuit = getDominoSuit(leadPlay.domino, trump);
  
  // Convert trump to numeric value for comparison
  const numericTrump = typeof trump === 'number' ? trump : 
                      (typeof trump === 'object' && 'suit' in trump) ? 
                      (typeof trump.suit === 'number' ? trump.suit : 
                       // Handle string suits
                       (function(suit: string) {
                         const suitMap: Record<string, number | null> = {
                           'blanks': 0, 'ones': 1, 'twos': 2, 'threes': 3, 
                           'fours': 4, 'fives': 5, 'sixes': 6, 'no-trump': 8, 'doubles': 7
                         };
                         const result = suitMap[suit];
                         return result !== undefined ? result : 0;
                       })(trump.suit)) : trump;
  
  let winningPlay = leadPlay;
  let winningValue = getDominoValue(leadPlay.domino, trump);
  let winningIsTrump = isDominoTrump(leadPlay.domino, numericTrump);
  
  for (let i = 1; i < trick.length; i++) {
    const play = trick[i];
    const playValue = getDominoValue(play.domino, trump);
    const playIsTrump = isDominoTrump(play.domino, numericTrump);
    
    // Trump always beats non-trump
    if (playIsTrump && !winningIsTrump) {
      winningPlay = play;
      winningValue = playValue;
      winningIsTrump = true;
    }
    // Both trump - higher value wins
    else if (playIsTrump && winningIsTrump && playValue > winningValue) {
      winningPlay = play;
      winningValue = playValue;
    }
    // Both non-trump - must follow suit and higher value wins
    else if (!playIsTrump && !winningIsTrump && 
             dominoFollowsLedSuit(play.domino, leadSuit) && 
             playValue > winningValue) {
      winningPlay = play;
      winningValue = playValue;
    }
  }
  
  return winningPlay.player;
}

/**
 * Calculates points in a trick
 */
export function calculateTrickPoints(trick: Play[]): number {
  return trick.reduce((total, play) => total + getDominoPoints(play.domino), 0);
}

/**
 * Calculates marks awarded at end of hand
 */
export function calculateRoundScore(state: GameState): [number, number] {
  if (!state.currentBid || state.winningBidder === null) {
    return state.teamMarks;
  }
  
  const newMarks: [number, number] = [...state.teamMarks];
  const biddingTeam = state.players[state.winningBidder].teamId;
  const opponentTeam = biddingTeam === 0 ? 1 : 0;
  const biddingTeamScore = state.teamScores[biddingTeam];
  
  const bid = state.currentBid;
  
  switch (bid.type) {
    case BID_TYPES.POINTS:
      const requiredPointsScore = bid.value!;
      if (biddingTeamScore >= requiredPointsScore) {
        // Point bid made - award 1 mark to bidding team
        newMarks[biddingTeam] += 1;
      } else {
        // Point bid failed - award 1 mark to opponents
        newMarks[opponentTeam] += 1;
      }
      break;
      
    case BID_TYPES.MARKS:
      const requiredMarksScore = 42; // Mark bids always require 42 points
      const markValue = bid.value!;
      if (biddingTeamScore >= requiredMarksScore) {
        // Mark bid made - award bid value in marks to bidding team
        newMarks[biddingTeam] += markValue;
      } else {
        // Mark bid failed - award bid value in marks to opponents
        newMarks[opponentTeam] += markValue;
      }
      break;
      
    case BID_TYPES.NELLO:
      // Nello: bidding team must take no tricks
      const biddingTeamTricks = state.tricks.filter(
        trick => trick.winner !== undefined && 
        state.players[trick.winner].teamId === biddingTeam
      ).length;
      
      if (biddingTeamTricks === 0) {
        newMarks[biddingTeam] += bid.value!;
      } else {
        newMarks[opponentTeam] += bid.value!;
      }
      break;
      
    case BID_TYPES.SPLASH:
    case BID_TYPES.PLUNGE:
      // Special contracts: bidding team must take all tricks
      const nonBiddingTeamTricks = state.tricks.filter(
        trick => trick.winner !== undefined && 
        state.players[trick.winner].teamId === opponentTeam
      ).length;
      
      if (nonBiddingTeamTricks === 0) {
        newMarks[biddingTeam] += bid.value!;
      } else {
        newMarks[opponentTeam] += bid.value!;
      }
      break;
  }
  
  return newMarks;
}

/**
 * Checks if game is complete (any team reached target marks)
 * Overloaded to handle both tuple and individual team marks
 */
export function isGameComplete(teamMarksOrTeam0: [number, number] | number, gameTargetOrTeam1?: number, gameTarget: number = 7): boolean {
  // Handle signature: isGameComplete(team0, team1, gameTarget = 7)
  if (typeof teamMarksOrTeam0 === 'number' && typeof gameTargetOrTeam1 === 'number') {
    const team0 = teamMarksOrTeam0;
    const team1 = gameTargetOrTeam1;
    return team0 >= gameTarget || team1 >= gameTarget;
  }
  
  // Handle signature: isGameComplete([team0, team1], gameTarget)
  const teamMarks = teamMarksOrTeam0 as [number, number];
  const target = gameTargetOrTeam1 || 7;
  return teamMarks[0] >= target || teamMarks[1] >= target;
}

/**
 * Gets the winning team (0 or 1), or null if game not complete
 */
export function getWinningTeam(teamMarks: [number, number], gameTarget: number): number | null {
  if (!isGameComplete(teamMarks, gameTarget)) return null;
  
  return teamMarks[0] >= gameTarget ? 0 : 1;
}

/**
 * Calculates team scores from individual player scores
 * Players 0,2 are team 0, players 1,3 are team 1
 */
export function calculateGameScore(playerScores: [number, number, number, number], biddingPlayer?: number, markValue?: number): [number, number] {
  const team0Score = playerScores[0] + playerScores[2];
  const team1Score = playerScores[1] + playerScores[3];
  
  return [team0Score, team1Score];
}

/**
 * Calculates final game score summary
 */
export function calculateGameSummary(state: GameState) {
  const winningTeam = getWinningTeam(state.teamMarks, state.gameTarget);
  
  return {
    winningTeam,
    finalMarks: state.teamMarks,
    handsPlayed: Math.max(...state.teamMarks),
    gameTarget: state.gameTarget,
    tournamentMode: state.tournamentMode
  };
}