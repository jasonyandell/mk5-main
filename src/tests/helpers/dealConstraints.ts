/**
 * Deal Constraints - Pure constraint satisfaction for generating domino deals
 *
 * A pure function with zero game dependencies. Given constraints about what
 * players should have in their hands, generates a valid deal that satisfies
 * all constraints (or throws a descriptive error if impossible).
 *
 * Per Stable Dependency Principle: once correct, never needs to change.
 *
 * @example Basic usage
 * ```typescript
 * // Player 0 must have 4 doubles
 * const hands = generateDealFromConstraints({
 *   players: { 0: { minDoubles: 4 } },
 *   fillSeed: 42
 * });
 * ```
 *
 * @example Complex scenario
 * ```typescript
 * const hands = generateDealFromConstraints({
 *   players: {
 *     0: { exactDominoes: ['6-6'], minDoubles: 3 },
 *     1: { voidInSuit: [6], maxDoubles: 1 }
 *   },
 *   fillSeed: 99999
 * });
 * ```
 */

// ============================================================================
// Types
// ============================================================================

/**
 * Simplified domino representation for constraint solving.
 * Uses string ID format "high-low" (e.g., "6-5").
 */
export interface ConstraintDomino {
  high: number;
  low: number;
  id: string;
}

/**
 * Constraints for a single player's hand.
 * All constraints are optional - unspecified constraints are not checked.
 */
export interface PlayerConstraint {
  /** Must have these exact dominoes in hand */
  exactDominoes?: string[];

  /** Minimum number of doubles required */
  minDoubles?: number;

  /** Maximum number of doubles allowed */
  maxDoubles?: number;

  /** Must have at least one domino containing each of these suits */
  mustHaveSuit?: number[];

  /** Must have zero dominoes containing these suits */
  voidInSuit?: number[];

  /** Minimum count per suit: { suit: minCount } */
  minSuitCount?: Partial<Record<number, number>>;

  /** Minimum total point value of hand */
  minPoints?: number;
}

/**
 * Constraints for the entire deal.
 */
export interface DealConstraints {
  /** Per-player constraints (0-3) */
  players?: Partial<Record<0 | 1 | 2 | 3, PlayerConstraint>>;

  /** Seed for deterministic filling of remaining slots */
  fillSeed?: number;
}

/**
 * Result of constraint generation - 4 hands of 7 dominoes each.
 */
export type DealResult = [ConstraintDomino[], ConstraintDomino[], ConstraintDomino[], ConstraintDomino[]];

// ============================================================================
// Constants
// ============================================================================

/** All 28 dominoes in a standard set */
const ALL_DOMINO_IDS: readonly string[] = [
  '0-0', '1-0', '2-0', '3-0', '4-0', '5-0', '6-0',
  '1-1', '2-1', '3-1', '4-1', '5-1', '6-1',
  '2-2', '3-2', '4-2', '5-2', '6-2',
  '3-3', '4-3', '5-3', '6-3',
  '4-4', '5-4', '6-4',
  '5-5', '6-5',
  '6-6'
] as const;

// Note: DOUBLE_IDS not currently used but kept for documentation
// const DOUBLE_IDS = ['0-0', '1-1', '2-2', '3-3', '4-4', '5-5', '6-6'] as const;

const HAND_SIZE = 7;

// ============================================================================
// Pure Utilities
// ============================================================================

/**
 * Creates a ConstraintDomino from string ID.
 */
export function parseDominoId(id: string): ConstraintDomino {
  const [a, b] = id.split('-').map(Number);
  if (a === undefined || b === undefined || isNaN(a) || isNaN(b)) {
    throw new Error(`Invalid domino ID: "${id}"`);
  }
  const high = Math.max(a, b);
  const low = Math.min(a, b);
  return { high, low, id: `${high}-${low}` };
}

/**
 * Normalizes a domino ID to canonical "high-low" format.
 */
function normalizeId(id: string): string {
  const d = parseDominoId(id);
  return d.id;
}

/**
 * Creates all 28 ConstraintDominoes.
 */
function createAllDominoes(): ConstraintDomino[] {
  return ALL_DOMINO_IDS.map(parseDominoId);
}

/**
 * Checks if a domino is a double.
 */
function isDouble(domino: ConstraintDomino): boolean {
  return domino.high === domino.low;
}

/**
 * Checks if a domino contains a specific suit (pip value).
 */
function hasSuit(domino: ConstraintDomino, suit: number): boolean {
  return domino.high === suit || domino.low === suit;
}

/**
 * Gets the point value of a domino.
 * Texas 42 scoring: 5-5=10, 6-4=10, 5-0/4-1/3-2=5 each.
 */
function getPoints(domino: ConstraintDomino): number {
  const sum = domino.high + domino.low;
  if (domino.high === 5 && domino.low === 5) return 10;
  if (domino.high === 6 && domino.low === 4) return 10;
  if (sum === 5) return 5; // 5-0, 4-1, 3-2
  return 0;
}

/**
 * Simple seeded random number generator (LCG).
 * Same algorithm as HandBuilder for consistency.
 */
function createSeededRng(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1664525 + 1013904223) % 2147483648;
    return state / 2147483648;
  };
}

/**
 * Fisher-Yates shuffle with seeded RNG.
 */
function shuffleWithSeed<T>(array: T[], seed: number): T[] {
  const rng = createSeededRng(seed);
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j]!, result[i]!];
  }
  return result;
}

// ============================================================================
// Constraint Validation
// ============================================================================

class ConstraintError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConstraintError';
  }
}

/**
 * Validates that constraints are internally consistent and potentially satisfiable.
 * Throws ConstraintError with descriptive message if impossible.
 */
function validateConstraints(constraints: DealConstraints): void {
  const { players = {} } = constraints;

  // Collect all exact dominoes across all players
  const allExact = new Set<string>();

  for (const [playerStr, playerConstraint] of Object.entries(players)) {
    const player = Number(playerStr) as 0 | 1 | 2 | 3;

    if (!playerConstraint) continue;

    const {
      exactDominoes = [],
      minDoubles,
      maxDoubles,
      mustHaveSuit = [],
      voidInSuit = [],
      minSuitCount = {},
      minPoints
    } = playerConstraint;

    // Validate exactDominoes are valid domino IDs
    for (const id of exactDominoes) {
      const normalized = normalizeId(id);
      if (!ALL_DOMINO_IDS.includes(normalized)) {
        throw new ConstraintError(`Player ${player}: Invalid domino ID "${id}"`);
      }
      if (allExact.has(normalized)) {
        throw new ConstraintError(`Domino "${normalized}" assigned to multiple players`);
      }
      allExact.add(normalized);
    }

    // Validate exactDominoes count
    if (exactDominoes.length > HAND_SIZE) {
      throw new ConstraintError(
        `Player ${player}: Cannot have ${exactDominoes.length} exact dominoes (max ${HAND_SIZE})`
      );
    }

    // Validate minDoubles/maxDoubles range
    if (minDoubles !== undefined && (minDoubles < 0 || minDoubles > 7)) {
      throw new ConstraintError(`Player ${player}: minDoubles must be 0-7, got ${minDoubles}`);
    }
    if (maxDoubles !== undefined && (maxDoubles < 0 || maxDoubles > 7)) {
      throw new ConstraintError(`Player ${player}: maxDoubles must be 0-7, got ${maxDoubles}`);
    }
    if (minDoubles !== undefined && maxDoubles !== undefined && minDoubles > maxDoubles) {
      throw new ConstraintError(
        `Player ${player}: minDoubles (${minDoubles}) > maxDoubles (${maxDoubles})`
      );
    }

    // Validate suit values
    for (const suit of mustHaveSuit) {
      if (suit < 0 || suit > 6) {
        throw new ConstraintError(`Player ${player}: Invalid suit ${suit} in mustHaveSuit`);
      }
    }
    for (const suit of voidInSuit) {
      if (suit < 0 || suit > 6) {
        throw new ConstraintError(`Player ${player}: Invalid suit ${suit} in voidInSuit`);
      }
    }

    // Check mustHaveSuit/voidInSuit conflict
    for (const suit of mustHaveSuit) {
      if (voidInSuit.includes(suit)) {
        throw new ConstraintError(
          `Player ${player}: Suit ${suit} in both mustHaveSuit and voidInSuit`
        );
      }
    }

    // Validate minSuitCount
    for (const [suitStr, count] of Object.entries(minSuitCount)) {
      const suit = Number(suitStr);
      if (suit < 0 || suit > 6) {
        throw new ConstraintError(`Player ${player}: Invalid suit ${suit} in minSuitCount`);
      }
      if (count! < 0 || count! > HAND_SIZE) {
        throw new ConstraintError(
          `Player ${player}: minSuitCount[${suit}] must be 0-7, got ${count}`
        );
      }
      if (voidInSuit.includes(suit) && count! > 0) {
        throw new ConstraintError(
          `Player ${player}: Cannot require minSuitCount[${suit}]=${count} with voidInSuit[${suit}]`
        );
      }
    }

    // Validate minPoints is achievable
    if (minPoints !== undefined) {
      const maxPossiblePoints = 42; // All count dominoes: 10+10+5+5+5+5+2=42
      if (minPoints > maxPossiblePoints) {
        throw new ConstraintError(
          `Player ${player}: minPoints (${minPoints}) exceeds maximum possible (${maxPossiblePoints})`
        );
      }
    }
  }
}

// ============================================================================
// Hand Validation
// ============================================================================

/**
 * Checks if a hand satisfies a player's constraints.
 * Returns null if satisfied, or error message if not.
 */
function checkHandConstraints(
  hand: ConstraintDomino[],
  constraint: PlayerConstraint,
  player: number
): string | null {
  const {
    exactDominoes = [],
    minDoubles,
    maxDoubles,
    mustHaveSuit = [],
    voidInSuit = [],
    minSuitCount = {},
    minPoints
  } = constraint;

  // Check exactDominoes are present
  const handIds = new Set(hand.map(d => d.id));
  for (const id of exactDominoes) {
    const normalized = normalizeId(id);
    if (!handIds.has(normalized)) {
      return `Player ${player}: Missing required domino "${normalized}"`;
    }
  }

  // Check doubles count
  const doublesCount = hand.filter(isDouble).length;
  if (minDoubles !== undefined && doublesCount < minDoubles) {
    return `Player ${player}: Has ${doublesCount} doubles, needs at least ${minDoubles}`;
  }
  if (maxDoubles !== undefined && doublesCount > maxDoubles) {
    return `Player ${player}: Has ${doublesCount} doubles, max allowed is ${maxDoubles}`;
  }

  // Check mustHaveSuit
  for (const suit of mustHaveSuit) {
    if (!hand.some(d => hasSuit(d, suit))) {
      return `Player ${player}: Missing required suit ${suit}`;
    }
  }

  // Check voidInSuit
  for (const suit of voidInSuit) {
    if (hand.some(d => hasSuit(d, suit))) {
      return `Player ${player}: Has domino with forbidden suit ${suit}`;
    }
  }

  // Check minSuitCount
  for (const [suitStr, minCount] of Object.entries(minSuitCount)) {
    const suit = Number(suitStr);
    const count = hand.filter(d => hasSuit(d, suit)).length;
    if (count < minCount!) {
      return `Player ${player}: Has ${count} dominoes with suit ${suit}, needs at least ${minCount}`;
    }
  }

  // Check minPoints
  if (minPoints !== undefined) {
    const totalPoints = hand.reduce((sum, d) => sum + getPoints(d), 0);
    if (totalPoints < minPoints) {
      return `Player ${player}: Has ${totalPoints} points, needs at least ${minPoints}`;
    }
  }

  return null;
}

// ============================================================================
// Main Algorithm
// ============================================================================

/**
 * Generates a deal satisfying all constraints.
 *
 * Algorithm:
 * 1. Validate constraints (detect impossibilities early)
 * 2. Create pool of all 28 dominoes
 * 3. Assign exactDominoes (remove from pool)
 * 4. Satisfy minDoubles by assigning doubles from pool
 * 5. Satisfy mustHaveSuit by assigning suit dominoes
 * 6. Respect voidInSuit when filling remaining slots
 * 7. Fill remaining with seeded shuffle of remaining pool
 * 8. Validate final hands satisfy all constraints
 *
 * @throws ConstraintError if constraints are impossible to satisfy
 */
export function generateDealFromConstraints(constraints: DealConstraints = {}): DealResult {
  // Step 1: Validate constraints
  validateConstraints(constraints);

  const { players = {}, fillSeed = 0 } = constraints;

  // Step 2: Create pool of all dominoes
  let pool = createAllDominoes();
  const poolSet = new Set(pool.map(d => d.id));

  // Initialize hands
  const hands: ConstraintDomino[][] = [[], [], [], []];

  // Helper to remove from pool
  function removeFromPool(id: string): ConstraintDomino | null {
    const normalized = normalizeId(id);
    if (!poolSet.has(normalized)) return null;
    poolSet.delete(normalized);
    const index = pool.findIndex(d => d.id === normalized);
    if (index === -1) return null;
    return pool.splice(index, 1)[0]!;
  }

  // Helper to add to hand
  function addToHand(player: number, domino: ConstraintDomino): void {
    if (hands[player]!.length >= HAND_SIZE) {
      throw new ConstraintError(`Player ${player}: Hand already has ${HAND_SIZE} dominoes`);
    }
    hands[player]!.push(domino);
  }

  // Step 3: Assign exactDominoes
  for (const [playerStr, constraint] of Object.entries(players)) {
    const player = Number(playerStr) as 0 | 1 | 2 | 3;
    if (!constraint?.exactDominoes) continue;

    for (const id of constraint.exactDominoes) {
      const domino = removeFromPool(id);
      if (!domino) {
        throw new ConstraintError(`Cannot assign domino "${id}" to player ${player} (already used)`);
      }
      addToHand(player, domino);
    }
  }

  // Step 4: Satisfy minDoubles
  for (const [playerStr, constraint] of Object.entries(players)) {
    const player = Number(playerStr) as 0 | 1 | 2 | 3;
    if (!constraint?.minDoubles) continue;

    const currentDoubles = hands[player]!.filter(isDouble).length;
    const needed = constraint.minDoubles - currentDoubles;

    if (needed <= 0) continue;

    // Find available doubles
    const availableDoubles = pool.filter(isDouble);

    // Filter by voidInSuit constraint
    const voidSuits = new Set(constraint.voidInSuit || []);
    const validDoubles = availableDoubles.filter(d => !voidSuits.has(d.high));

    if (validDoubles.length < needed) {
      throw new ConstraintError(
        `Player ${player}: Cannot satisfy minDoubles=${constraint.minDoubles}. ` +
        `Has ${currentDoubles}, need ${needed} more, only ${validDoubles.length} available.`
      );
    }

    // Prefer higher doubles (for plunge scenarios)
    validDoubles.sort((a, b) => b.high - a.high);

    for (let i = 0; i < needed; i++) {
      const domino = removeFromPool(validDoubles[i]!.id)!;
      addToHand(player, domino);
    }
  }

  // Step 5: Satisfy mustHaveSuit
  for (const [playerStr, constraint] of Object.entries(players)) {
    const player = Number(playerStr) as 0 | 1 | 2 | 3;
    if (!constraint?.mustHaveSuit) continue;

    for (const suit of constraint.mustHaveSuit) {
      // Check if already satisfied
      if (hands[player]!.some(d => hasSuit(d, suit))) continue;

      // Find domino with this suit
      const voidSuits = new Set(constraint.voidInSuit || []);
      const candidates = pool.filter(d =>
        hasSuit(d, suit) &&
        // Respect void constraints (check the other suit)
        (d.high === d.low || !voidSuits.has(d.high === suit ? d.low : d.high))
      );

      if (candidates.length === 0) {
        throw new ConstraintError(
          `Player ${player}: Cannot satisfy mustHaveSuit for suit ${suit} (none available)`
        );
      }

      const domino = removeFromPool(candidates[0]!.id)!;
      addToHand(player, domino);
    }
  }

  // Step 5b: Satisfy minSuitCount (after mustHaveSuit)
  for (const [playerStr, constraint] of Object.entries(players)) {
    const player = Number(playerStr) as 0 | 1 | 2 | 3;
    if (!constraint?.minSuitCount) continue;

    for (const [suitStr, minCount] of Object.entries(constraint.minSuitCount)) {
      const suit = Number(suitStr);
      const currentCount = hands[player]!.filter(d => hasSuit(d, suit)).length;
      const needed = minCount! - currentCount;

      if (needed <= 0) continue;

      const voidSuits = new Set(constraint.voidInSuit || []);
      const candidates = pool.filter(d =>
        hasSuit(d, suit) &&
        (d.high === d.low || !voidSuits.has(d.high === suit ? d.low : d.high))
      );

      if (candidates.length < needed) {
        throw new ConstraintError(
          `Player ${player}: Cannot satisfy minSuitCount[${suit}]=${minCount}. ` +
          `Has ${currentCount}, need ${needed} more, only ${candidates.length} available.`
        );
      }

      for (let i = 0; i < needed; i++) {
        const domino = removeFromPool(candidates[i]!.id)!;
        addToHand(player, domino);
      }
    }
  }

  // Step 5c: Satisfy minPoints
  for (const [playerStr, constraint] of Object.entries(players)) {
    const player = Number(playerStr) as 0 | 1 | 2 | 3;
    if (!constraint?.minPoints) continue;

    const currentPoints = hands[player]!.reduce((sum, d) => sum + getPoints(d), 0);
    if (currentPoints >= constraint.minPoints) continue;

    const voidSuits = new Set(constraint.voidInSuit || []);
    const maxDoubles = constraint.maxDoubles;

    // Keep adding high-point dominoes until we meet the requirement
    while (hands[player]!.length < HAND_SIZE) {
      const currentHandPoints = hands[player]!.reduce((sum, d) => sum + getPoints(d), 0);
      if (currentHandPoints >= constraint.minPoints) break;

      // Find point-bearing dominoes, prioritize highest points
      const candidates = pool
        .filter(d => {
          // Check void constraint
          if (voidSuits.has(d.high) || voidSuits.has(d.low)) return false;
          // Check maxDoubles constraint
          if (maxDoubles !== undefined && isDouble(d)) {
            const currentDoubles = hands[player]!.filter(isDouble).length;
            if (currentDoubles >= maxDoubles) return false;
          }
          return getPoints(d) > 0;
        })
        .sort((a, b) => getPoints(b) - getPoints(a));

      if (candidates.length === 0) {
        throw new ConstraintError(
          `Player ${player}: Cannot satisfy minPoints=${constraint.minPoints}. ` +
          `Has ${currentHandPoints} points, no more point dominoes available.`
        );
      }

      const domino = removeFromPool(candidates[0]!.id)!;
      addToHand(player, domino);
    }
  }

  // Step 6 & 7: Fill remaining slots respecting voidInSuit
  // Shuffle remaining pool deterministically
  pool = shuffleWithSeed(pool, fillSeed);

  for (let player = 0; player < 4; player++) {
    const constraint = players[player as 0 | 1 | 2 | 3];
    const voidSuits = new Set(constraint?.voidInSuit || []);
    const maxDoubles = constraint?.maxDoubles;

    while (hands[player]!.length < HAND_SIZE) {
      // Find valid domino from pool
      let found = false;

      for (let i = 0; i < pool.length; i++) {
        const domino = pool[i]!;

        // Check void constraint
        if (voidSuits.has(domino.high) || voidSuits.has(domino.low)) {
          continue;
        }

        // Check maxDoubles constraint
        if (maxDoubles !== undefined && isDouble(domino)) {
          const currentDoubles = hands[player]!.filter(isDouble).length;
          if (currentDoubles >= maxDoubles) {
            continue;
          }
        }

        // Use this domino
        pool.splice(i, 1);
        poolSet.delete(domino.id);
        addToHand(player, domino);
        found = true;
        break;
      }

      if (!found) {
        throw new ConstraintError(
          `Player ${player}: Cannot fill remaining slots. ` +
          `Has ${hands[player]!.length}/${HAND_SIZE} dominoes, ` +
          `${pool.length} in pool but none valid.`
        );
      }
    }
  }

  // Step 8: Validate final hands
  for (const [playerStr, constraint] of Object.entries(players)) {
    const player = Number(playerStr) as 0 | 1 | 2 | 3;
    if (!constraint) continue;

    const error = checkHandConstraints(hands[player]!, constraint, player);
    if (error) {
      throw new ConstraintError(`Final validation failed: ${error}`);
    }
  }

  // Verify all 28 dominoes distributed
  const totalDominoes = hands.reduce((sum, h) => sum + h.length, 0);
  if (totalDominoes !== 28) {
    throw new ConstraintError(
      `Internal error: Distributed ${totalDominoes} dominoes, expected 28`
    );
  }

  return hands as DealResult;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/**
 * Creates a deal where a specific player has N doubles.
 * Useful for plunge/splash testing.
 */
export function dealWithDoubles(
  player: 0 | 1 | 2 | 3,
  minDoubles: number,
  fillSeed: number = 0
): DealResult {
  return generateDealFromConstraints({
    players: { [player]: { minDoubles } },
    fillSeed
  });
}

/**
 * Creates a deal where a specific player has exact dominoes.
 */
export function dealWithExactHand(
  player: 0 | 1 | 2 | 3,
  dominoes: string[],
  fillSeed: number = 0
): DealResult {
  return generateDealFromConstraints({
    players: { [player]: { exactDominoes: dominoes } },
    fillSeed
  });
}

/**
 * Creates a deal where a player is void in specific suits.
 */
export function dealWithVoid(
  player: 0 | 1 | 2 | 3,
  voidSuits: number[],
  fillSeed: number = 0
): DealResult {
  return generateDealFromConstraints({
    players: { [player]: { voidInSuit: voidSuits } },
    fillSeed
  });
}
