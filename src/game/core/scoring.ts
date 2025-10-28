import type { GameState, Play, PlayedDomino, TrumpSelection, LedSuitOrNone, LedSuit } from '../types';
import { isEmptyBid } from '../types';
import { BID_TYPES } from '../constants';
import { getDominoValue, getDominoPoints } from './dominoes';

/**
 * Checks if a domino follows the led suit (contains the led suit number)
 */
function dominoFollowsLedSuit(domino: { high: number; low: number }, leadSuit: LedSuit): boolean {
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
  // When specific suit is trump, only dominoes containing that suit are trump
  return domino.high === numericTrump || domino.low === numericTrump;
}

/**
 * Determines the winner of a completed trick (overloaded for different interfaces)
 */
export function calculateTrickWinner(trick: Play[] | PlayedDomino[], trump: TrumpSelection, leadSuit: LedSuitOrNone): number {
  if (trick.length === 0) {
    throw new Error('Trick cannot be empty');
  }
  
  const leadPlay = trick[0];
  if (!leadPlay) {
    throw new Error('Cannot determine winner of empty trick');
  }
  
  // Convert trump to numeric value for comparison
  const numericTrump = trump.type === 'not-selected' ? null :
                       trump.type === 'suit' ? trump.suit! :
                       trump.type === 'doubles' ? 7 :
                       trump.type === 'no-trump' ? 8 : null;
  
  let winningPlay = leadPlay;
  let winningValue = getDominoValue(leadPlay.domino, trump);
  let winningIsTrump = isDominoTrump(leadPlay.domino, numericTrump);
  
  for (let i = 1; i < trick.length; i++) {
    const play = trick[i];
    if (!play) {
      throw new Error(`Invalid trick play at index ${i}`);
    }
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
             leadSuit >= 0 && dominoFollowsLedSuit(play.domino, leadSuit as LedSuit) && 
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
  if (isEmptyBid(state.currentBid) || state.winningBidder === -1) {
    return state.teamMarks;
  }
  
  const newMarks: [number, number] = [...state.teamMarks];
  const winnerPlayer = state.players[state.winningBidder];
  if (!winnerPlayer) {
    throw new Error(`Invalid winning bidder index: ${state.winningBidder}`);
  }
  const biddingTeam = winnerPlayer.teamId;
  const opponentTeam = biddingTeam === 0 ? 1 : 0;
  const biddingTeamScore = state.teamScores[biddingTeam];
  
  const bid = state.currentBid;
  
  switch (bid.type) {
    case BID_TYPES.POINTS: {
      const requiredPointsScore = bid.value!;
      if (biddingTeamScore >= requiredPointsScore) {
        // Point bid made - award 1 mark to bidding team
        newMarks[biddingTeam] += 1;
      } else {
        // Point bid failed - award 1 mark to opponents
        newMarks[opponentTeam] += 1;
      }
      break;
    }
      
    case BID_TYPES.MARKS: {
      // Standard marks bid: require 42 points
      const markValue = bid.value!;
      const requiredMarksScore = 42;
      if (biddingTeamScore >= requiredMarksScore) {
        newMarks[biddingTeam] += markValue;
      } else {
        newMarks[opponentTeam] += markValue;
      }
      break;
    }
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
export function calculateGameScore(playerScores: [number, number, number, number]): [number, number] {
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
    gameTarget: state.gameTarget
  };
}