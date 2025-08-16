import type { GameState, Bid, Domino, TrumpSelection, PlayedDomino, Player, SuitRanking } from '../types';
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
function isValidMarkBid(bid: Bid, lastBid: Bid, _previousBids: Bid[]): boolean {
  if (bid.value === undefined) return false;
  
  // After point bids, can bid 1 or 2 marks
  if (lastBid.type === BID_TYPES.POINTS) {
    return bid.value >= 1 && bid.value <= 2;
  }
  
  // Mark bid progression: can only bid one more mark than the last mark bid
  if (lastBid.type === BID_TYPES.MARKS) {
    // Can always bid 2 marks (standard max opening bid rule)
    if (bid.value === 2) return true;
    
    // For 3+ marks, can only bid one more than the last mark bid if last bid was 2+
    if (bid.value >= 3 && lastBid.value! >= 2 && bid.value === lastBid.value! + 1) return true;
    
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
  if (state.phase !== 'playing' || state.trump.type === 'none') return false;
  
  // Validate player bounds
  if (playerId < 0 || playerId >= state.players.length) return false;
  
  const player = state.players[playerId];
  if (!player || !player.hand.some(d => d.id === domino.id)) return false;
  
  // First play of trick is always legal
  if (state.currentTrick.length === 0) return true;
  
  // Use the currentSuit from state instead of computing it
  const leadSuit = state.currentSuit;
  if (leadSuit === -1) return true; // No suit to follow
  
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
  
  // Check if trump is being led
  const trumpValue = getTrumpNumber(state.trump);
  if (trumpValue === leadSuit) {
    // Trump is led - must follow with trump if possible
    const trumpDominoes = player.suitAnalysis.rank.trump;
    if (trumpDominoes && trumpDominoes.length > 0) {
      return trumpDominoes.some(d => d.id === domino.id);
    }
    return true; // Can't follow trump, any play is legal
  }
  
  // For regular suits (0-6), check if player can follow
  const suitDominoes = player.suitAnalysis.rank[leadSuit as keyof Omit<SuitRanking, 'doubles' | 'trump'>];
  
  // Filter out trump dominoes - they can't follow non-trump suits
  const nonTrumpSuitDominoes = suitDominoes ? suitDominoes.filter(d => {
    // If trump is a regular suit, dominoes containing that suit are trump
    if (trumpValue !== null && trumpValue >= 0 && trumpValue <= 6) {
      return d.high !== trumpValue && d.low !== trumpValue;
    }
    // If doubles are trump, doubles can't follow regular suits
    if (trumpValue === 7) {
      return d.high !== d.low;
    }
    return true;
  }) : [];
  
  // If player has non-trump dominoes in the led suit, must play one of them
  if (nonTrumpSuitDominoes.length > 0) {
    return nonTrumpSuitDominoes.some(d => d.id === domino.id);
  }
  
  // If player can't follow suit (no non-trump dominoes in that suit), any play is legal
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
function getTrumpNumber(trump: TrumpSelection): number | null {
  switch (trump.type) {
    case 'none': return null;
    case 'suit': return trump.suit!;
    case 'doubles': return 7;
    case 'no-trump': return 8;
  }
}

/**
 * Gets all valid plays for a player using suit analysis
 */
export function getValidPlays(
  state: GameState,
  playerId: number
): Domino[] {
  if (state.phase !== 'playing' || state.trump.type === 'none') return [];
  
  const player = state.players[playerId];
  if (!player) return [];
  if (!player.suitAnalysis) return [...player.hand];
  
  // First play of trick - all dominoes are valid
  if (state.currentTrick.length === 0) return [...player.hand];
  
  // Use the currentSuit from state instead of computing it
  const leadSuit = state.currentSuit;
  if (leadSuit === -1) return [...player.hand]; // No suit to follow
  
  // Handle doubles trump special case
  if (leadSuit === 7) {
    const doubles = player.suitAnalysis.rank.doubles;
    if (doubles && doubles.length > 0) {
      return doubles;
    }
    return [...player.hand]; // Can't follow doubles, all plays valid
  }
  
  // Check if trump is being led
  const trumpValue = getTrumpNumber(state.trump);
  if (trumpValue === leadSuit) {
    // Trump is led - must follow with trump if possible
    const trumpDominoes = player.suitAnalysis.rank.trump;
    if (trumpDominoes && trumpDominoes.length > 0) {
      return trumpDominoes;
    }
    return [...player.hand]; // Can't follow trump, all plays valid
  }
  
  // Get dominoes that can follow the led suit from suit analysis
  const suitDominoes = player.suitAnalysis.rank[leadSuit as keyof Omit<SuitRanking, 'doubles' | 'trump'>];
  
  // Filter out trump dominoes - they can't follow non-trump suits
  const nonTrumpSuitDominoes = suitDominoes ? suitDominoes.filter(d => {
    // If trump is a regular suit, dominoes containing that suit are trump
    if (trumpValue !== null && trumpValue >= 0 && trumpValue <= 6) {
      return d.high !== trumpValue && d.low !== trumpValue;
    }
    // If doubles are trump, doubles can't follow regular suits
    if (trumpValue === 7) {
      return d.high !== d.low;
    }
    return true;
  }) : [];
  
  // If player has non-trump dominoes in the led suit, must play one of them
  if (nonTrumpSuitDominoes.length > 0) {
    return nonTrumpSuitDominoes;
  }
  
  // If player can't follow suit (no non-trump dominoes in that suit), all dominoes are valid
  return [...player.hand];
}

/**
 * Gets the winner of a trick (alias for calculateTrickWinner)
 */
export function getTrickWinner(trick: { player: number; domino: Domino }[], trump: TrumpSelection, leadSuit: number): number {
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
export function determineTrickWinner(trick: { player: number; domino: Domino }[] | PlayedDomino[], trump: TrumpSelection, leadSuit: number): number {
  return calculateTrickWinner(trick as PlayedDomino[], trump, leadSuit);
}

/**
 * Validates if a trump suit is valid
 */
export function isValidTrump(trump: TrumpSelection): boolean {
  if (trump.type === 'suit') {
    return trump.suit !== undefined && trump.suit >= 0 && trump.suit <= 6;
  }
  return trump.type === 'doubles' || trump.type === 'no-trump';
}

/**
 * Gets the numeric value of a trump suit
 */
export function getTrumpValue(trump: TrumpSelection): number {
  switch (trump.type) {
    case 'none': return -1;
    case 'suit': return trump.suit!;
    case 'doubles': return 7;
    case 'no-trump': return 8;
  }
}

/**
 * Gets the current suit being led in a trick for display purposes
 */
export function getCurrentSuit(state: GameState): string {
  if (state.currentSuit === -1) {
    return 'None (no domino led)';
  }
  
  if (state.trump.type === 'none') {
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