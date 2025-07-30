import type { Domino } from '../types';

export interface DealerDrawResult {
  playerId: number;
  domino: Domino;
}

export interface DealerDeterminationResult {
  dealer: number | null;
  requiresRedraw: boolean;
  tiedPlayers?: number[];
}

/**
 * Creates a domino for the dealer draw process
 */
export function createDominoForDraw(high: number, low: number): Domino {
  if (high < 0 || high > 6 || low < 0 || low > 6) {
    throw new Error('Invalid domino values: must be between 0 and 6');
  }
  
  // Ensure high >= low for consistency
  const [sortedHigh, sortedLow] = high >= low ? [high, low] : [low, high];
  
  return {
    high: sortedHigh,
    low: sortedLow,
    id: `${sortedHigh}-${sortedLow}`,
  };
}

/**
 * Determines the first dealer based on domino draws
 * Each player draws one domino, highest total pip count becomes dealer
 * In case of tie, affected players must redraw
 */
export function determineDealerFromDraw(draws: DealerDrawResult[]): DealerDeterminationResult {
  // Validate exactly 4 players
  if (draws.length !== 4) {
    throw new Error('Exactly 4 players must draw for dealer determination');
  }

  // Validate player IDs are 0-3
  const playerIds = draws.map(d => d.playerId).sort();
  if (!playerIds.every((id, index) => id === index)) {
    throw new Error('Invalid player IDs: must be 0, 1, 2, 3');
  }

  // Validate domino values
  for (const draw of draws) {
    if (draw.domino.high < 0 || draw.domino.high > 6 || 
        draw.domino.low < 0 || draw.domino.low > 6) {
      throw new Error('Invalid domino values: must be between 0 and 6');
    }
  }

  // Calculate pip counts
  const pipCounts = draws.map(draw => ({
    playerId: draw.playerId,
    total: draw.domino.high + draw.domino.low
  }));

  // Find highest pip count
  const maxPips = Math.max(...pipCounts.map(p => p.total));
  
  // Find all players with highest pip count
  const playersWithMaxPips = pipCounts
    .filter(p => p.total === maxPips)
    .map(p => p.playerId);

  // Check for ties
  if (playersWithMaxPips.length > 1) {
    return {
      dealer: null,
      requiresRedraw: true,
      tiedPlayers: playersWithMaxPips.sort()
    };
  }

  // Single winner
  return {
    dealer: playersWithMaxPips[0],
    requiresRedraw: false
  };
}