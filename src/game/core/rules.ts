import type { GameState, Bid, Domino, Trump, PlayedDomino, Player, SuitRanking } from '../types';
import { GAME_CONSTANTS, BID_TYPES } from '../constants';
import { countDoubles } from './dominoes';
import { calculateTrickWinner, calculateTrickPoints } from './scoring';

/**
 * Validates if a bid is legal in the current game state
 */
export function isValidBid(state: GameState, bid: Bid, playerHand?: Domino[]): boolean {
  // Check if bids array exists and player has already bid
  if (!state.bids) return false;
  const playerBids = state.bids.filter(b => b.player === bid.player);
  if (playerBids.length > 0) return false;
  
  // Check turn order only if we're in an active bidding phase
  if (state.phase === 'bidding' && state.currentPlayer !== bid.player) return false;
  
  // Pass is always valid if player hasn't bid yet
  if (bid.type === BID_TYPES.PASS) return true;
  
  const previousBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);
  
  // Opening bid constraints
  if (previousBids.length === 0) {
    return isValidOpeningBid(bid, playerHand, state.tournamentMode);
  }
  
  // All subsequent bids must be higher than current high bid
  const lastBid = previousBids[previousBids.length - 1];
  const lastBidValue = getBidComparisonValue(lastBid);
  const currentBidValue = getBidComparisonValue(bid);
  
  if (currentBidValue <= lastBidValue) return false;
  
  return isValidSubsequentBid(bid, lastBid, previousBids, playerHand, state.tournamentMode);
}

/**
 * Validates opening bids
 */
export function isValidOpeningBid(bid: Bid, playerHand?: Domino[], tournamentMode: boolean = true): boolean {
  switch (bid.type) {
    case BID_TYPES.POINTS:
      return bid.value !== undefined && 
             bid.value >= GAME_CONSTANTS.MIN_BID && 
             bid.value <= GAME_CONSTANTS.MAX_BID;
    
    case BID_TYPES.MARKS:
      // Tournament rules: maximum opening bid is 2 marks
      return bid.value !== undefined && bid.value >= 1 && bid.value <= 2;
    
    case BID_TYPES.NELLO:
      // Special contracts not allowed in tournament mode
      if (tournamentMode) return false;
      return bid.value !== undefined && bid.value >= 1;
    
    case BID_TYPES.SPLASH:
      // Special contracts not allowed in tournament mode
      if (tournamentMode) return false;
      return bid.value !== undefined && 
             bid.value >= 2 && 
             bid.value <= 3 && 
             playerHand !== undefined && 
             countDoubles(playerHand) >= 3;
    
    case BID_TYPES.PLUNGE:
      // Special contracts not allowed in tournament mode
      if (tournamentMode) return false;
      return bid.value !== undefined && 
             bid.value >= 4 && 
             playerHand !== undefined && 
             countDoubles(playerHand) >= 4;
    
    default:
      return false;
  }
}

/**
 * Validates subsequent bids
 */
function isValidSubsequentBid(
  bid: Bid, 
  lastBid: Bid, 
  previousBids: Bid[], 
  playerHand?: Domino[],
  tournamentMode: boolean = true
): boolean {
  switch (bid.type) {
    case BID_TYPES.POINTS:
      return bid.value !== undefined && 
             bid.value <= GAME_CONSTANTS.MAX_BID &&
             (lastBid.type !== BID_TYPES.POINTS || bid.value > lastBid.value!);
    
    case BID_TYPES.MARKS:
      return isValidMarkBid(bid, lastBid, previousBids);
    
    case BID_TYPES.NELLO:
      // Special contracts not allowed in tournament mode
      if (tournamentMode) return false;
      return bid.value !== undefined && bid.value >= 1;
    
    case BID_TYPES.SPLASH:
      // Special contracts not allowed in tournament mode
      if (tournamentMode) return false;
      return bid.value !== undefined && 
             bid.value >= 2 && 
             bid.value <= 3 && 
             playerHand !== undefined && 
             countDoubles(playerHand) >= 3;
    
    case BID_TYPES.PLUNGE:
      // Special contracts not allowed in tournament mode
      if (tournamentMode) return false;
      return bid.value !== undefined && 
             bid.value >= 4 && 
             playerHand !== undefined && 
             countDoubles(playerHand) >= 4;
    
    default:
      return false;
  }
}

/**
 * Validates mark bids with tournament progression rules
 */
function isValidMarkBid(bid: Bid, lastBid: Bid, previousBids: Bid[]): boolean {
  if (bid.value === undefined) return false;
  
  // After point bids, can bid 1 or 2 marks
  if (lastBid.type === BID_TYPES.POINTS) {
    return bid.value >= 1 && bid.value <= 2;
  }
  
  // Mark bid progression: 3+ marks can only be bid after 2 marks
  if (lastBid.type === BID_TYPES.MARKS) {
    // Include current lastBid in the check for 2 marks
    const allMarkBids = [...previousBids, lastBid];
    const hasTwoMarks = allMarkBids.some(b => b.type === BID_TYPES.MARKS && b.value === 2);
    
    // Can bid 2 marks if not already bid
    if (!hasTwoMarks && bid.value === 2) return true;
    
    // Can only bid 3+ marks after 2 marks, and only one additional mark
    if (hasTwoMarks && bid.value === lastBid.value! + 1) return true;
    
    return false;
  }
  
  return bid.value >= 1;
}

/**
 * Gets the comparison value for bid ordering
 */
export function getBidComparisonValue(bid: Bid): number {
  if (bid.value === undefined) return 0;
  
  switch (bid.type) {
    case BID_TYPES.POINTS:
      return bid.value;
    case BID_TYPES.MARKS:
    case BID_TYPES.NELLO:
    case BID_TYPES.SPLASH:
    case BID_TYPES.PLUNGE:
      return bid.value * 42;
    default:
      return 0;
  }
}

/**
 * Validates if a domino play is legal using suit analysis from game state
 */
export function isValidPlay(
  state: GameState,
  domino: Domino,
  playerId: number
): boolean {
  if (state.phase !== 'playing' || state.trump === null) return false;
  
  // Validate player bounds
  if (playerId < 0 || playerId >= state.players.length) return false;
  
  const player = state.players[playerId];
  if (!player || !player.hand.some(d => d.id === domino.id)) return false;
  
  // First play of trick is always legal
  if (state.currentTrick.length === 0) return true;
  
  // Use the currentSuit from state instead of computing it
  const leadSuit = state.currentSuit;
  if (leadSuit === null) return true; // No suit to follow
  
  // Use suit analysis to check if player can follow suit
  if (!player.suitAnalysis) return true; // If no analysis, allow any play
  
  // Handle doubles trump special case
  if (leadSuit === 7) {
    // When doubles are led (trump = 7), only doubles can follow
    const doubles = player.suitAnalysis.rank.doubles;
    if (doubles && doubles.length > 0) {
      return doubles.some(d => d.id === domino.id);
    }
    return true; // Can't follow doubles, any play is legal
  }
  
  // For regular suits (0-6), use the suit ranking
  const suitDominoes = player.suitAnalysis.rank[leadSuit as keyof Omit<SuitRanking, 'doubles' | 'trump'>];
  
  // If player has dominoes in the led suit, must play one of them
  if (suitDominoes && suitDominoes.length > 0) {
    return suitDominoes.some(d => d.id === domino.id);
  }
  
  // If player can't follow suit, any play is legal
  return true;
}

/**
 * Checks if player can follow the lead suit using suit analysis
 */
export function canFollowSuit(
  player: Player,
  leadSuit: number
): boolean {
  if (!player.suitAnalysis) return false;
  
  // Handle doubles trump special case
  if (leadSuit === 7) {
    return player.suitAnalysis.rank.doubles && player.suitAnalysis.rank.doubles.length > 0;
  }
  
  const suitDominoes = player.suitAnalysis.rank[leadSuit as keyof Omit<SuitRanking, 'doubles' | 'trump'>];
  return suitDominoes && suitDominoes.length > 0;
}

/**
 * Helper function to convert trump to numeric value
 */
function getTrumpNumber(trump: Trump): number | null {
  if (typeof trump === 'number') return trump;
  if (typeof trump === 'object' && 'suit' in trump) {
    if (typeof trump.suit === 'number') return trump.suit;
    const suitMap: Record<string, number | null> = {
      'blanks': 0, 'ones': 1, 'twos': 2, 'threes': 3, 
      'fours': 4, 'fives': 5, 'sixes': 6, 'no-trump': 8, 'doubles': 7
    };
    return suitMap[trump.suit] ?? 0;
  }
  return trump as number;
}

/**
 * Gets all valid plays for a player using suit analysis
 */
export function getValidPlays(
  state: GameState,
  playerId: number
): Domino[] {
  if (state.phase !== 'playing' || state.trump === null) return [];
  
  const player = state.players[playerId];
  if (!player) return [];
  if (!player.suitAnalysis) return [...player.hand];
  
  // First play of trick - all dominoes are valid
  if (state.currentTrick.length === 0) return [...player.hand];
  
  // Use the currentSuit from state instead of computing it
  const leadSuit = state.currentSuit;
  if (leadSuit === null) return [...player.hand]; // No suit to follow
  
  // Handle doubles trump special case
  if (leadSuit === 7) {
    const doubles = player.suitAnalysis.rank.doubles;
    if (doubles && doubles.length > 0) {
      return doubles;
    }
    return [...player.hand]; // Can't follow doubles, all plays valid
  }
  
  // Get dominoes that can follow the led suit from suit analysis
  const suitDominoes = player.suitAnalysis.rank[leadSuit as keyof Omit<SuitRanking, 'doubles' | 'trump'>];
  
  // If can follow suit, must follow suit
  if (suitDominoes && suitDominoes.length > 0) {
    return suitDominoes;
  }
  
  // If can't follow suit, all dominoes are valid
  return [...player.hand];
}

/**
 * Gets the winner of a trick (alias for calculateTrickWinner)
 */
export function getTrickWinner(trick: { player: number; domino: Domino }[], trump: Trump, leadSuit: number): number {
  return calculateTrickWinner(trick, trump, leadSuit);
}

/**
 * Gets the points in a trick (alias for calculateTrickPoints)
 */
export function getTrickPoints(trick: { player: number; domino: Domino }[]): number {
  return calculateTrickPoints(trick);
}

/**
 * Determines the winner of a trick (alternative interface)
 */
export function determineTrickWinner(trick: { player: number; domino: Domino }[] | PlayedDomino[], trump: Trump, leadSuit: number): number {
  return calculateTrickWinner(trick as PlayedDomino[], trump, leadSuit);
}

/**
 * Validates if a trump suit is valid
 */
export function isValidTrump(trump: { suit: number | string; followsSuit: boolean }): boolean {
  if (typeof trump.suit === 'number') {
    return trump.suit >= 0 && trump.suit <= 7; // Include doubles (7)
  }
  if (typeof trump.suit === 'string') {
    const validSuits = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes', 'doubles'];
    return validSuits.includes(trump.suit);
  }
  return false;
}

/**
 * Gets the numeric value of a trump suit
 */
export function getTrumpValue(trump: { suit: number | string; followsSuit: boolean }): number {
  if (typeof trump.suit === 'number' && trump.suit >= 0 && trump.suit <= 7) {
    // Differentiate between regular trump and follow-suit trump
    // Regular trump: 0-7, follow-suit trump: 10-17
    return trump.followsSuit ? trump.suit + 10 : trump.suit;
  }
  if (typeof trump.suit === 'string') {
    const suitMap: Record<string, number> = {
      'blanks': 0, 'ones': 1, 'twos': 2, 'threes': 3, 
      'fours': 4, 'fives': 5, 'sixes': 6, 'doubles': 7
    };
    const numericSuit = suitMap[trump.suit];
    if (numericSuit !== undefined) {
      return trump.followsSuit ? numericSuit + 10 : numericSuit;
    }
  }
  throw new Error(`Invalid trump suit: ${trump.suit}`);
}

/**
 * Gets the current suit being led in a trick for display purposes
 */
export function getCurrentSuit(state: GameState): string {
  if (state.currentSuit === null) {
    return 'None (no domino led)';
  }
  
  if (state.trump === null) {
    return 'None (no trump set)';
  }
  
  const leadSuit = state.currentSuit;
  
  const suitNames: Record<number, string> = {
    0: 'Blanks',
    1: 'Ones', 
    2: 'Twos',
    3: 'Threes',
    4: 'Fours',
    5: 'Fives',
    6: 'Sixes',
    7: 'Doubles (Trump)',
    8: 'No Trump'
  };
  
  const trumpSuit = getTrumpNumber(state.trump);
  
  // Special case: if lead suit equals trump suit, indicate it's trump
  if (leadSuit === trumpSuit && trumpSuit !== 7) {
    return `${suitNames[leadSuit]} (Trump)`;
  }
  
  return suitNames[leadSuit] || `Unknown (${leadSuit})`;
}