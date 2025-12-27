import type { GameState, Play } from '../types';
import { isEmptyBid } from '../types';
import { BID_TYPES } from '../constants';
import { getDominoPoints } from './dominoes';

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
 * Checks if target marks have been reached (any team reached target).
 * Works on raw marks data, not GameState.
 *
 * For GameState-based checks, use isGameComplete() from state.ts.
 */
export function isTargetReached(teamMarks: [number, number], gameTarget: number): boolean {
  return teamMarks[0] >= gameTarget || teamMarks[1] >= gameTarget;
}

/**
 * Gets the winning team (0 or 1) from raw marks, or null if target not reached.
 * Works on raw marks data, not GameState.
 *
 * For GameState-based checks, use getWinningTeam() from state.ts.
 */
export function getWinnerFromMarks(teamMarks: [number, number], gameTarget: number): number | null {
  if (!isTargetReached(teamMarks, gameTarget)) return null;

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
  const winningTeam = getWinnerFromMarks(state.teamMarks, state.gameTarget);

  return {
    winningTeam,
    finalMarks: state.teamMarks,
    handsPlayed: Math.max(...state.teamMarks),
    gameTarget: state.gameTarget
  };
}