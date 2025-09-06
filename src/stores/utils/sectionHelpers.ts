import type { GameState, StateTransition } from '../../game/types';
import { createInitialState, getNextStates } from '../../game';
import { deepClone } from './deepUtils';
import { getAttempts, recordWin } from './oneHandStats';

// Helper: determine if "we" (team 0; players 0 and 2) won the hand
// Returns true if we definitively won, false if definitively not, or null if undetermined
export function didWeWinHandAtScoring(state: GameState): boolean | null {
  // Only meaningful at scoring or game end
  if (state.phase !== 'scoring' && state.phase !== 'game_end') return null;

  // Prefer explicit winner if present at game_end
  const winner = (state as { winner?: number }).winner;
  if (winner !== undefined && winner !== -1) {
    return winner === 0;
  }

  // During scoring (before game_end), outcome may not be finalized yet
  // In that case, return null and let caller avoid Challenge-specific UI
  return null;
}

// Determine winner team at scoring or game_end and return scores
export function getWinnerTeamAndScores(state: GameState): { winnerTeam: 0 | 1; us: number; them: number } | null {
  if (state.phase !== 'scoring' && state.phase !== 'game_end') return null;
  const us = state.teamScores[0] || 0;
  const them = state.teamScores[1] || 0;
  // If game_end has explicit winner, trust it
  const explicit = (state as { winner?: number }).winner;
  if (state.phase === 'game_end' && explicit !== undefined && explicit !== -1) {
    return { winnerTeam: explicit as 0 | 1, us, them };
  }
  // Otherwise compute based on bid and tricks
  const wb = state.winningBidder;
  if (wb === -1) return null;
  const bid = state.currentBid;
  const bidPlayer = state.players[wb];
  if (!bid || !bidPlayer) return null;
  const biddingTeam = bidPlayer.teamId as 0 | 1;
  const defendingTeam = biddingTeam === 0 ? 1 : 0;
  const biddingScore = biddingTeam === 0 ? us : them;
  const defendingScore = biddingTeam === 0 ? them : us;
  const trickWinnerTeamCounts = { 0: 0, 1: 0 } as Record<0 | 1, number>;
  for (const trick of state.tricks) {
    if (trick.winner !== undefined) {
      const w = state.players[trick.winner];
      if (w) trickWinnerTeamCounts[w.teamId as 0 | 1]++;
    }
  }
  switch (bid.type) {
    case 'points': {
      const bidValue = bid.value || 0;
      const winnerTeam = biddingScore >= bidValue ? biddingTeam : defendingTeam;
      return { winnerTeam, us, them };
    }
    case 'marks': {
      const winnerTeam = defendingScore > 0 ? defendingTeam : biddingTeam;
      return { winnerTeam, us, them };
    }
    case 'nello': {
      const biddingTricks = trickWinnerTeamCounts[biddingTeam];
      const winnerTeam = biddingTricks > 0 ? defendingTeam : biddingTeam;
      return { winnerTeam, us, them };
    }
    case 'splash':
    case 'plunge': {
      const defendingTricks = trickWinnerTeamCounts[defendingTeam];
      const winnerTeam = defendingTricks > 0 ? defendingTeam : biddingTeam;
      return { winnerTeam, us, them };
    }
  }
  return null;
}

type OverlayPayload = {
  type: 'oneHand';
  phase: GameState['phase'];
  seed: number;
  canChallenge?: boolean;
  attemptsForWin?: number;
  attemptsCount?: number;
  weWon?: boolean;
  usScore?: number;
  themScore?: number;
};

export function buildOverlayPayload(state: GameState, initialState: GameState): OverlayPayload {
  const seed = initialState.shuffleSeed;
  const winnerInfo = getWinnerTeamAndScores(state);
  const weWon = winnerInfo ? (winnerInfo.winnerTeam === 0) : undefined;
  const usScore = winnerInfo?.us;
  const themScore = winnerInfo?.them;
  
  const result: OverlayPayload = { 
    type: 'oneHand', 
    phase: state.phase, 
    seed
  };
  
  if (weWon === true) {
    result.canChallenge = true;
    result.attemptsForWin = recordWin(seed);
  } else if (weWon === false) {
    result.attemptsCount = getAttempts(seed);
  }
  
  if (weWon !== undefined) result.weWon = weWon;
  if (usScore !== undefined) result.usScore = usScore;
  if (themScore !== undefined) result.themScore = themScore;
  
  return result;
}

// Helper: deterministically prepare a playing-phase state for one-hand section
export async function prepareDeterministicHand(seed = 424242): Promise<GameState> {
  // Start from deterministic initial state with default player types
  let state = createInitialState({ shuffleSeed: seed, playerTypes: ['human', 'ai', 'ai', 'ai'] });
  // Progress through bidding (4 actions), prefer P0 bids 30 when it's P0's turn
  for (let i = 0; i < 4 && state.phase === 'bidding'; i++) {
    const ts = getNextStates(state);
    let pick: StateTransition | undefined;
    if (state.currentPlayer === 0) {
      pick = ts.find((t) => t.id === 'bid-30') || ts.find((t) => t.id.startsWith('bid-')) || ts.find((t) => t.id === 'pass');
    } else {
      pick = ts.find((t) => t.id.startsWith('bid-')) || ts.find((t) => t.id === 'pass');
    }
    if (!pick) break;
    state = pick.newState;
  }
  // If all-pass, redeal once
  if (state.bids.length === 4 && state.winningBidder === -1) {
    const ts = getNextStates(state);
    const redeal = ts.find((t) => t.id === 'redeal');
    if (redeal) state = redeal.newState;
    // Re-bid; again prefer P0 bid-30 when it's P0's turn
    for (let i = 0; i < 4 && state.phase === 'bidding'; i++) {
      const ts2 = getNextStates(state);
      let pick2: StateTransition | undefined;
      if (state.currentPlayer === 0) {
        pick2 = ts2.find((t) => t.id === 'bid-30') || ts2.find((t) => t.id.startsWith('bid-')) || ts2.find((t) => t.id === 'pass');
      } else {
        pick2 = ts2.find((t) => t.id.startsWith('bid-')) || ts2.find((t) => t.id === 'pass');
      }
      if (!pick2) break;
      state = pick2.newState;
    }
  }
  // Trump selection: pick first available trump
  if (state.phase === 'trump_selection') {
    const ts = getNextStates(state);
    const trump = ts.find((t) => t.id.startsWith('trump-'));
    if (trump) state = trump.newState;
  }
  return state;
}

// Build a deterministic action list to reach playing phase from a given initial state
export function buildActionsToPlayingFromState(initial: GameState): string[] {
  let state = deepClone(initial);
  const actions: string[] = [];
  // Follow: pass, bid-30, pass, pass regardless of dealer/currentPlayer order
  for (let i = 0; i < 4 && state.phase === 'bidding'; i++) {
    const ts = getNextStates(state);
    let pick: StateTransition | undefined;
    // If current player is human (P0), pass; otherwise bid-30 if possible
    if (state.currentPlayer === 0) {
      pick = ts.find((t) => t.id === 'pass') || ts.find((t) => t.id.startsWith('bid-'));
    } else {
      pick = ts.find((t) => t.id === 'bid-30') || ts.find((t) => t.id.startsWith('bid-')) || ts.find((t) => t.id === 'pass');
    }
    if (!pick) break;
    actions.push(pick.id);
    state = pick.newState;
  }
  // Include trump selection explicitly to ensure we reach playing deterministically
  if (state.phase === 'trump_selection') {
    const ts = getNextStates(state);
    const trump = ts.find((t) => t.id.startsWith('trump-'));
    if (trump) {
      actions.push(trump.id);
      state = trump.newState;
    }
  }
  return actions;
}