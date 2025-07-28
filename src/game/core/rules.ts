import type { GameState, Bid, Domino, Trump, PlayedDomino } from '../types';
import { GAME_CONSTANTS, BID_TYPES } from '../constants';
import { countDoubles } from './dominoes';
import { calculateTrickWinner, calculateTrickPoints } from './scoring';

/**
 * Validates if a bid is legal in the current game state
 * Overloaded to support legacy test signature: isValidBid(bid, currentBid, state)
 */
export function isValidBid(stateOrBid: GameState | Bid, bidOrCurrentBid: Bid | null, playerHandOrState?: Domino[] | GameState): boolean {
  // Handle legacy signature: isValidBid(bid, currentBid, state)
  // Check if first arg is Bid and third arg is GameState (second can be null)
  if ('type' in stateOrBid && 'player' in stateOrBid && 
      playerHandOrState && 'phase' in playerHandOrState) {
    const bid = stateOrBid as Bid;
    const currentBid = bidOrCurrentBid as Bid | null;
    const state = playerHandOrState as GameState;
    return isValidBidCoreLegacy(state, bid, currentBid);
  }
  
  // Handle new signature: isValidBid(state, bid, playerHand?)
  const state = stateOrBid as GameState;
  const bid = bidOrCurrentBid as Bid;
  const playerHand = playerHandOrState as Domino[] | undefined;
  return isValidBidCore(state, bid, playerHand);
}

/**
 * Legacy bid validation logic (without current player check)
 */
function isValidBidCoreLegacy(state: GameState, bid: Bid, currentBid: Bid | null, playerHand?: Domino[]): boolean {
  // Check if bids array exists and player has already bid
  if (state.bids) {
    const playerBids = state.bids.filter(b => b.player === bid.player);
    if (playerBids.length > 0) return false;
  }
  
  // Pass is always valid if player hasn't bid yet
  if (bid.type === BID_TYPES.PASS) return true;
  
  // If no current bid, this is an opening bid
  if (!currentBid) {
    return isValidOpeningBid(bid, playerHand, state.tournamentMode);
  }
  
  // Compare against current bid
  const lastBidValue = getBidComparisonValue(currentBid);
  const currentBidValue = getBidComparisonValue(bid);
  
  if (currentBidValue <= lastBidValue) return false;
  
  // Get all previous bids from state, or use current bid if no state.bids
  const previousBids = state.bids ? state.bids.filter(b => b.type !== BID_TYPES.PASS) : [currentBid];
  return isValidSubsequentBid(bid, currentBid, previousBids, playerHand);
}

/**
 * Core bid validation logic
 */
function isValidBidCore(state: GameState, bid: Bid, playerHand?: Domino[]): boolean {
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
  
  return isValidSubsequentBid(bid, lastBid, previousBids, playerHand);
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
  playerHand?: Domino[]
): boolean {
  switch (bid.type) {
    case BID_TYPES.POINTS:
      return bid.value !== undefined && 
             bid.value <= GAME_CONSTANTS.MAX_BID &&
             (lastBid.type !== BID_TYPES.POINTS || bid.value > lastBid.value!);
    
    case BID_TYPES.MARKS:
      return isValidMarkBid(bid, lastBid, previousBids);
    
    case BID_TYPES.NELLO:
      return bid.value !== undefined && bid.value >= 1;
    
    case BID_TYPES.SPLASH:
      return bid.value !== undefined && 
             bid.value >= 2 && 
             bid.value <= 3 && 
             playerHand !== undefined && 
             countDoubles(playerHand) >= 3;
    
    case BID_TYPES.PLUNGE:
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
 * Validates if a domino play is legal (overloaded for different call signatures)
 */
export function isValidPlay(
  dominoOrState: Domino | GameState, 
  handOrDomino?: Domino[] | Domino, 
  currentTrickOrPlayerId?: { player: number; domino: Domino }[] | number | Domino[], 
  trump?: Trump
): boolean {
  // Handle the simple signature: isValidPlay(domino, hand, currentTrick, trump)
  if (trump !== undefined && Array.isArray(handOrDomino) && Array.isArray(currentTrickOrPlayerId)) {
    const domino = dominoOrState as Domino;
    const hand = handOrDomino;
    const currentTrick = currentTrickOrPlayerId as { player: number; domino: Domino }[];
    
    // First play of trick is always legal
    if (currentTrick.length === 0) return true;
    
    const leadPlay = currentTrick[0];
    const leadSuit = getLeadSuit(leadPlay.domino, trump);
    
    // If following suit, always legal
    if (canDominoFollowLedSuit(domino, leadSuit, trump)) return true;
    
    // If can't follow suit, any play is legal
    return !canFollowSuit(hand, leadSuit, trump);
  }
  
  // Handle the test signature: isValidPlay(state, domino, hand)
  if (typeof dominoOrState === 'object' && dominoOrState !== null && 'phase' in dominoOrState && 
      typeof handOrDomino === 'object' && handOrDomino !== null && 'id' in handOrDomino && 
      Array.isArray(currentTrickOrPlayerId)) {
    const state = dominoOrState as GameState;
    const domino = handOrDomino as Domino;
    const hand = currentTrickOrPlayerId as Domino[];
    
    if (state.phase !== 'playing' || state.trump === null) return false;
    
    // Check if domino is in hand
    if (!hand.some(d => d.id === domino.id)) return false;
    
    // First play of trick is always legal
    if (state.currentTrick.length === 0) return true;
    
    const leadPlay = state.currentTrick[0];
    const leadSuit = getLeadSuit(leadPlay.domino, state.trump);
    
    // If following suit, always legal
    if (canDominoFollowLedSuit(domino, leadSuit, state.trump)) return true;
    
    // If can't follow suit, any play is legal
    return !canFollowSuit(hand, leadSuit, state.trump);
  }
  
  // Handle the GameState signature: isValidPlay(state, domino, playerId)
  const state = dominoOrState as GameState;
  const domino = handOrDomino as Domino;
  const playerId = currentTrickOrPlayerId as number;
  
  if (state.phase !== 'playing' || state.trump === null) return false;
  
  // Validate player bounds
  if (playerId < 0 || playerId >= state.players.length) return false;
  
  const player = state.players[playerId];
  if (!player || !player.hand.some(d => d.id === domino.id)) return false;
  
  // First play of trick is always legal
  if (state.currentTrick.length === 0) return true;
  
  const leadPlay = state.currentTrick[0];
  const leadSuit = getLeadSuit(leadPlay.domino, state.trump);
  
  // If following suit, always legal
  if (canDominoFollowLedSuit(domino, leadSuit, state.trump)) return true;
  
  // If can't follow suit, any play is legal
  return !canFollowSuit(player.hand, leadSuit, state.trump);
}

/**
 * Checks if player can follow the lead suit
 */
export function canFollowSuit(
  hand: Domino[], 
  leadSuit: number, 
  trump: Trump
): boolean {
  return hand.some(domino => canDominoFollowLedSuit(domino, leadSuit, trump));
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
 * Checks if a domino can follow the led suit
 */
function canDominoFollowLedSuit(domino: Domino, leadSuit: number, trump: Trump): boolean {
  const trumpSuit = getTrumpNumber(trump);
  
  // Handle doubles trump case (trump = 7)
  if (trumpSuit === 7) {
    // When doubles are trump, all doubles form the trump suit
    if (domino.high === domino.low) {
      // This double can follow if trump suit (7) was led
      return leadSuit === 7;
    }
    // Non-doubles can follow trump suit if they contain the trump number
    // But trump number 7 doesn't exist on dominoes, so non-doubles never follow doubles trump
    if (leadSuit === 7) {
      return false; // Non-double cannot follow doubles trump
    }
    // For non-trump leads, check if domino contains the led suit number
    return domino.high === leadSuit || domino.low === leadSuit;
  }
  
  // Standard tournament rules: doubles belong to their natural suit
  // A double can follow suit if its natural suit matches the led suit
  if (domino.high === domino.low) {
    // Double follows suit if its natural value matches led suit
    if (domino.high === leadSuit) {
      return true;
    }
    // Double can also be trump if trump is not doubles trump and not no-trump
    const isDoubleTrump = trumpSuit !== null && trumpSuit !== 8;
    if (isDoubleTrump && leadSuit === trumpSuit) {
      return true; // Trump double following trump suit
    }
    return false; // Double doesn't follow non-matching suit
  }
  
  // Check if this non-double domino is trump
  const isDominoTrump = trumpSuit !== null && (domino.high === trumpSuit || domino.low === trumpSuit);
  
  // If trump was led, then trump dominoes follow suit
  if (leadSuit === trumpSuit && isDominoTrump) {
    return true;
  }
  
  // If non-trump was led and domino is trump, it doesn't follow suit (it trumps)
  if (leadSuit !== trumpSuit && isDominoTrump) {
    return false;
  }
  
  // For non-trump dominoes, check if they contain the led suit number
  return domino.high === leadSuit || domino.low === leadSuit;
}

/**
 * Gets the suit that was led (for following suit purposes)
 * This is different from getDominoSuit - doubles lead their natural suit unless doubles are trump
 */
function getLeadSuit(domino: Domino, trump: Trump): number {
  const trumpSuit = getTrumpNumber(trump);
  
  // For doubles, check if doubles are trump
  if (domino.high === domino.low) {
    // When doubles are trump (trump = 7), all doubles lead trump suit
    if (trumpSuit === 7) {
      return 7;
    }
    // Otherwise, doubles lead their natural suit (the pip value)
    return domino.high;
  }
  
  // For non-doubles containing trump, if trump was led, return trump suit
  if (trumpSuit !== null && (domino.high === trumpSuit || domino.low === trumpSuit)) {
    return trumpSuit;
  }
  
  // Otherwise, use the higher value
  return Math.max(domino.high, domino.low);
}

/**
 * Gets all valid plays for a player
 */
export function getValidPlays(
  stateOrHand: GameState | Domino[], 
  handOrCurrentTrick?: Domino[] | { player: number; domino: Domino }[], 
  trump?: Trump
): Domino[] {
  // Handle overloaded signatures
  if (Array.isArray(stateOrHand) && stateOrHand.length > 0 && 'id' in stateOrHand[0]) {
    // Called as getValidPlays(hand, currentTrick, trump)
    const hand = stateOrHand as Domino[];
    const currentTrick = handOrCurrentTrick as { player: number; domino: Domino }[];
    return getValidPlaysCore(hand, currentTrick, trump!);
  } else {
    // Called as getValidPlays(state, hand)
    const state = stateOrHand as GameState;
    const hand = handOrCurrentTrick as Domino[];
    return getValidPlaysCore(hand, state.currentTrick, state.trump);
  }
}

function getValidPlaysCore(
  hand: Domino[], 
  currentTrick: { player: number; domino: Domino }[], 
  trump: Trump | null
): Domino[] {
  // First play of trick - all dominoes are valid
  if (currentTrick.length === 0) return [...hand];
  
  // If no trump is set, all plays are valid
  if (trump === null) return [...hand];
  
  const leadPlay = currentTrick[0];
  const leadSuit = getLeadSuit(leadPlay.domino, trump);
  
  // Get all dominoes that can follow the led suit (contain the led suit number and are not trump)
  const followSuitPlays = hand.filter(domino => 
    canDominoFollowLedSuit(domino, leadSuit, trump)
  );
  
  // If can follow suit, must follow suit
  if (followSuitPlays.length > 0) return followSuitPlays;
  
  // If can't follow suit, all dominoes are valid
  return [...hand];
}

/**
 * Gets the winner of a trick (alias for calculateTrickWinner)
 */
export function getTrickWinner(trick: { player: number; domino: Domino }[], trump: Trump): number {
  return calculateTrickWinner(trick, trump);
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
export function determineTrickWinner(trick: { player: number; domino: Domino }[] | PlayedDomino[], trump: Trump): number {
  return calculateTrickWinner(trick as PlayedDomino[], trump);
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