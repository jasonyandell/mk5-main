/**
 * Player selection utilities - explicit methods instead of modulus operations
 */

/**
 * Gets the next player in clockwise order (0 -> 1 -> 2 -> 3 -> 0)
 */
export function getNextPlayer(currentPlayer: number): number {
  switch (currentPlayer) {
    case 0: return 1;
    case 1: return 2;
    case 2: return 3;
    case 3: return 0;
    default: throw new Error(`Invalid player ID: ${currentPlayer}`);
  }
}

/**
 * Gets the player to the left of dealer (first bidder)
 */
export function getPlayerLeftOfDealer(dealer: number): number {
  return getNextPlayer(dealer);
}

/**
 * Gets the next dealer after current dealer
 */
export function getNextDealer(currentDealer: number): number {
  return getNextPlayer(currentDealer);
}

/**
 * Gets the player after the specified number of positions clockwise
 */
export function getPlayerAfter(startPlayer: number, positions: number): number {
  let player = startPlayer;
  for (let i = 0; i < positions; i++) {
    player = getNextPlayer(player);
  }
  return player;
}

/**
 * Gets the player N positions after the dealer (for dealing order)
 */
export function getPlayerAtDealPosition(dealer: number, position: number): number {
  return getPlayerAfter(dealer, position);
}

/**
 * Validates that a player ID is valid (0-3)
 */
export function isValidPlayerId(playerId: number): boolean {
  return playerId >= 0 && playerId <= 3;
}

/**
 * Gets all player IDs in clockwise order starting from specified player
 */
export function getPlayersInOrder(startPlayer: number): number[] {
  if (!isValidPlayerId(startPlayer)) {
    throw new Error(`Invalid start player ID: ${startPlayer}`);
  }
  
  const players: number[] = [];
  let current = startPlayer;
  
  for (let i = 0; i < 4; i++) {
    players.push(current);
    current = getNextPlayer(current);
  }
  
  return players;
}