/**
 * CFR Tractability Metrics for Texas 42
 *
 * Measures key metrics to determine if Counterfactual Regret Minimization (CFR)
 * is tractable for Texas 42:
 *
 * 1. Branching factor distribution - Legal action counts at decision points
 * 2. Unique information sets - (hand, public history, trick state) tuples
 * 3. Canonical information sets - Same with non-trump suits permuted to canonical order
 * 4. State revisitation rate - How often same info set appears across games
 *
 * Key insight: < 1M canonical info sets → solve directly, > 10M → need abstraction
 */

import type { GameState, Domino, Trick, TrumpSelection, RegularSuit } from '../types';
import type { ValidAction } from '../../multiplayer/types';
import { getTrumpSuit, isDoublesTrump, isNoTrump } from '../core/dominoes';
import { HeadlessRoom } from '../../server/HeadlessRoom';
import { RandomAIStrategy } from './strategies';

// ============================================================================
// Types
// ============================================================================

export interface InformationSetHash {
  /** Full hash without suit canonicalization */
  raw: string;
  /** Hash with non-trump suits permuted to canonical order */
  canonical: string;
  /** Count-centric abstraction hash */
  countCentric: string;
}

export interface CFRMetrics {
  /** All observed legal action counts at decision points */
  branchingFactors: number[];

  /** Unique raw information set hashes */
  uniqueInfoSets: Set<string>;

  /** Unique canonical (suit-isomorphic) information set hashes */
  canonicalInfoSets: Set<string>;

  /** Unique count-centric abstraction hashes */
  countCentricInfoSets: Set<string>;

  /** Count of occurrences per canonical info set (for revisitation rate) */
  infoSetCounts: Map<string, number>;

  /** Count of occurrences per count-centric info set */
  countCentricCounts: Map<string, number>;

  /** Total decision points observed */
  totalDecisionPoints: number;
}

export interface MetricsReport {
  gamesSimulated: number;
  totalDecisionPoints: number;

  branchingFactor: {
    min: number;
    max: number;
    mean: number;
    median: number;
    distribution: Record<number, number>;
  };

  uniqueInfoSets: number;
  canonicalInfoSets: number;
  countCentricInfoSets: number;
  compressionRatio: number;
  countCentricCompressionRatio: number;

  revisitationRate: number;
  singletonRate: number;
  countCentricRevisitationRate: number;
  countCentricSingletonRate: number;
}

// ============================================================================
// Suit Canonicalization
// ============================================================================

/**
 * Build a permutation map for canonicalizing non-trump suits.
 *
 * Trump suit (if any) maps to itself.
 * Other suits are assigned canonical IDs (0-5) based on:
 *   1. Count in hand (descending)
 *   2. Original pip value (ascending) for ties
 *
 * This collapses suit-isomorphic states.
 */
export function buildCanonicalSuitMap(
  hand: Domino[],
  trump: TrumpSelection
): Map<RegularSuit, RegularSuit> {
  const trumpSuit = getTrumpSuit(trump);
  const isDoublesMode = isDoublesTrump(trumpSuit);
  const isNoTrumpMode = isNoTrump(trumpSuit);

  // Count non-trump suits in hand
  const suitCounts = new Map<RegularSuit, number>();
  for (let suit = 0; suit <= 6; suit++) {
    suitCounts.set(suit as RegularSuit, 0);
  }

  for (const domino of hand) {
    // In doubles-as-trump mode, doubles don't belong to regular suits
    if (isDoublesMode && domino.high === domino.low) {
      continue;
    }

    // Count each pip as belonging to its suit (unless it's trump)
    for (const pip of [domino.high, domino.low]) {
      if (!isNoTrumpMode && pip === trumpSuit) {
        continue; // Skip trump pip
      }
      const current = suitCounts.get(pip as RegularSuit) || 0;
      suitCounts.set(pip as RegularSuit, current + 1);
    }
  }

  // Sort non-trump suits by (count DESC, pip ASC)
  const nonTrumpSuits: RegularSuit[] = [];
  for (let suit = 0; suit <= 6; suit++) {
    if (suit !== trumpSuit) {
      nonTrumpSuits.push(suit as RegularSuit);
    }
  }

  nonTrumpSuits.sort((a, b) => {
    const countA = suitCounts.get(a) || 0;
    const countB = suitCounts.get(b) || 0;
    if (countB !== countA) {
      return countB - countA; // Descending by count
    }
    return a - b; // Ascending by pip value
  });

  // Build mapping: each non-trump suit maps to its canonical position
  const mapping = new Map<RegularSuit, RegularSuit>();

  // Trump suit maps to itself (or use 6 as "trump slot" for regular trump)
  if (trumpSuit !== null && trumpSuit >= 0 && trumpSuit <= 6) {
    mapping.set(trumpSuit as RegularSuit, trumpSuit as RegularSuit);
  }

  // Assign canonical IDs to non-trump suits
  // We use the sorted order as the canonical mapping
  let canonicalId = 0;
  for (const suit of nonTrumpSuits) {
    // Skip the trump slot
    if (canonicalId === trumpSuit) {
      canonicalId++;
    }
    mapping.set(suit, canonicalId as RegularSuit);
    canonicalId++;
    if (canonicalId === trumpSuit) {
      canonicalId++;
    }
  }

  return mapping;
}

/**
 * Apply canonical suit mapping to a domino ID.
 * Returns the canonicalized domino ID.
 */
function canonicalizeDominoId(
  domino: Domino,
  suitMap: Map<RegularSuit, RegularSuit>,
  trump: TrumpSelection
): string {
  const trumpSuit = getTrumpSuit(trump);
  const isDoublesMode = isDoublesTrump(trumpSuit);

  // Doubles stay as-is in doubles mode (they're trump)
  if (isDoublesMode && domino.high === domino.low) {
    return `${domino.high}-${domino.low}`;
  }

  // Map each pip through the suit permutation
  const newHigh = suitMap.get(domino.high as RegularSuit) ?? domino.high;
  const newLow = suitMap.get(domino.low as RegularSuit) ?? domino.low;

  // Normalize so high >= low
  const hi = Math.max(newHigh, newLow);
  const lo = Math.min(newHigh, newLow);

  return `${hi}-${lo}`;
}

// ============================================================================
// Information Set Hashing
// ============================================================================

/**
 * Compute the information set hash for a player at a decision point.
 *
 * The information set captures everything the player knows:
 * - Their current hand
 * - Trump selection
 * - All completed tricks (public)
 * - Current trick state (public)
 * - Position in current trick
 *
 * Returns both raw and canonical (suit-isomorphic) hashes.
 */
export function computeInformationSetHash(
  state: GameState,
  playerIndex: number
): InformationSetHash {
  const player = state.players[playerIndex];
  if (!player) {
    throw new Error(`Invalid player index: ${playerIndex}`);
  }

  const hand = player.hand;
  const trump = state.trump;

  // Build components for raw hash
  const rawComponents: string[] = [];

  // 1. Trump type and suit
  rawComponents.push(`T:${trump.type}`);
  if (trump.suit !== undefined) {
    rawComponents.push(`TS:${trump.suit}`);
  }

  // 2. Player's hand (sorted for consistency)
  const sortedHand = [...hand].sort((a, b) => String(a.id).localeCompare(String(b.id)));
  rawComponents.push(`H:${sortedHand.map(d => d.id).join(',')}`);

  // 3. Completed tricks (order matters - it's the history)
  for (let i = 0; i < state.tricks.length; i++) {
    const trick = state.tricks[i];
    if (trick) {
      const trickStr = encodeTrick(trick);
      rawComponents.push(`T${i}:${trickStr}`);
    }
  }

  // 4. Current trick
  if (state.currentTrick.length > 0) {
    const currentStr = state.currentTrick
      .map(p => `${p.player}:${p.domino.id}`)
      .join(',');
    rawComponents.push(`CT:${currentStr}`);
  }

  // 5. Current suit (if led)
  if (state.currentSuit >= 0) {
    rawComponents.push(`CS:${state.currentSuit}`);
  }

  // 6. Position in trick (how many plays before us)
  rawComponents.push(`POS:${state.currentTrick.length}`);

  const rawHash = rawComponents.join('|');

  // Build canonical hash with suit permutation
  const suitMap = buildCanonicalSuitMap(hand, trump);
  const canonicalComponents: string[] = [];

  // 1. Trump type (suit is implicit in canonical form)
  canonicalComponents.push(`T:${trump.type}`);

  // 2. Canonicalized hand
  const canonicalHand = sortedHand
    .map(d => canonicalizeDominoId(d, suitMap, trump))
    .sort();
  canonicalComponents.push(`H:${canonicalHand.join(',')}`);

  // 3. Canonicalized completed tricks
  for (let i = 0; i < state.tricks.length; i++) {
    const trick = state.tricks[i];
    if (trick) {
      const trickStr = encodeTrickCanonical(trick, suitMap, trump);
      canonicalComponents.push(`T${i}:${trickStr}`);
    }
  }

  // 4. Canonicalized current trick
  if (state.currentTrick.length > 0) {
    const currentStr = state.currentTrick
      .map(p => `${p.player}:${canonicalizeDominoId(p.domino, suitMap, trump)}`)
      .join(',');
    canonicalComponents.push(`CT:${currentStr}`);
  }

  // 5. Canonicalized current suit
  if (state.currentSuit >= 0 && state.currentSuit <= 6) {
    const canonicalSuit = suitMap.get(state.currentSuit as RegularSuit) ?? state.currentSuit;
    canonicalComponents.push(`CS:${canonicalSuit}`);
  } else if (state.currentSuit === 7) {
    // Doubles suit stays as 7
    canonicalComponents.push(`CS:7`);
  }

  // 6. Position (same as raw)
  canonicalComponents.push(`POS:${state.currentTrick.length}`);

  const canonicalHash = canonicalComponents.join('|');

  // =========================================================================
  // COUNT-CENTRIC ABSTRACTION
  // The 5 count dominoes (5-0, 5-5, 6-4, 3-2, 4-1) = 35 points total
  // Everything else exists to control when count can be safely played
  // =========================================================================
  const countCentricHash = computeCountCentricHash(state, playerIndex);

  return { raw: rawHash, canonical: canonicalHash, countCentric: countCentricHash };
}

// ============================================================================
// Count-Centric Abstraction
// ============================================================================

/**
 * The 5 count dominoes that contain all 35 points in Texas 42.
 * These are the "atoms" of strategy - everything else is scaffolding.
 */
const COUNT_DOMINO_IDS = new Set(['5-0', '5-5', '6-4', '3-2', '4-1']);
const COUNT_DOMINO_POINTS: Record<string, number> = {
  '5-0': 5,
  '5-5': 10,
  '6-4': 10,
  '3-2': 5,
  '4-1': 5
};

/**
 * Check if a domino is a count domino.
 */
function isCountDomino(domino: Domino): boolean {
  return COUNT_DOMINO_IDS.has(String(domino.id));
}

/**
 * Compute count-centric abstraction hash.
 *
 * This hash captures the strategically relevant features:
 * - Which count dominoes are in my hand
 * - Which count dominoes have been captured by each team
 * - Trump control state (who leads, trump remaining)
 * - Position in trick
 *
 * Non-count domino distinctions (4-3 vs 4-2) collapse when irrelevant.
 */
export function computeCountCentricHash(state: GameState, playerIndex: number): string {
  const player = state.players[playerIndex];
  if (!player) {
    return 'INVALID';
  }

  const myTeam = playerIndex % 2; // 0 or 1
  const hand = player.hand;
  const trump = state.trump;

  // 1. Count dominoes in my hand (sorted for consistency)
  const countInHand = hand
    .filter(isCountDomino)
    .map(d => String(d.id))
    .sort()
    .join(',');

  // 2. Count captured by each team (from completed tricks)
  let countCapturedUs = 0;
  let countCapturedThem = 0;

  for (const trick of state.tricks) {
    if (!trick || trick.winner === undefined) continue;
    const winnerTeam = trick.winner % 2;
    for (const play of trick.plays) {
      if (isCountDomino(play.domino)) {
        const points = COUNT_DOMINO_POINTS[String(play.domino.id)] ?? 0;
        if (winnerTeam === myTeam) {
          countCapturedUs += points;
        } else {
          countCapturedThem += points;
        }
      }
    }
  }

  // 3. Count in current trick (not yet captured)
  let countInCurrentTrick = 0;
  for (const play of state.currentTrick) {
    if (isCountDomino(play.domino)) {
      countInCurrentTrick += COUNT_DOMINO_POINTS[String(play.domino.id)] ?? 0;
    }
  }

  // 4. Who leads (control state) - simplified to: am I leading or following?
  const iLead = state.currentTrick.length === 0;

  // 5. Trump strength estimate - how many trump do I have?
  const myTrumpCount = countTrumpInHand(hand, trump);

  // 6. Trick number (game progress)
  const trickNum = state.tricks.length;

  // 7. Position in trick
  const posInTrick = state.currentTrick.length;

  // 8. Non-count hand size (control cards remaining)
  const nonCountInHand = hand.filter(d => !isCountDomino(d)).length;

  // Build hash - these features determine strategic decisions
  const components = [
    `C:${countInHand || 'none'}`,      // which count I hold
    `US:${countCapturedUs}`,            // points we've secured
    `THEM:${countCapturedThem}`,        // points they've secured
    `POT:${countInCurrentTrick}`,       // points at stake in current trick
    `LEAD:${iLead ? 1 : 0}`,            // control state
    `TR:${myTrumpCount}`,               // trump control
    `TK:${trickNum}`,                   // game progress
    `POS:${posInTrick}`,                // position in trick
    `NC:${nonCountInHand}`,             // control cards remaining
    `TT:${trump.type}`                  // trump type matters for strategy
  ];

  return components.join('|');
}

/**
 * Count trump cards in hand.
 */
function countTrumpInHand(hand: Domino[], trump: TrumpSelection): number {
  if (trump.type === 'no-trump') {
    return 0;
  }

  if (trump.type === 'doubles') {
    // In doubles mode, doubles are trump
    return hand.filter(d => d.high === d.low).length;
  }

  // Regular trump suit
  const trumpSuit = trump.suit;
  if (trumpSuit === undefined) return 0;

  return hand.filter(d => d.high === trumpSuit || d.low === trumpSuit).length;
}

/**
 * Encode a completed trick as a string.
 */
function encodeTrick(trick: Trick): string {
  const plays = trick.plays.map(p => `${p.player}:${p.domino.id}`).join(',');
  return `${plays}|W:${trick.winner}|L:${trick.ledSuit ?? -1}`;
}

/**
 * Encode a completed trick with canonical suit mapping.
 */
function encodeTrickCanonical(
  trick: Trick,
  suitMap: Map<RegularSuit, RegularSuit>,
  trump: TrumpSelection
): string {
  const plays = trick.plays
    .map(p => `${p.player}:${canonicalizeDominoId(p.domino, suitMap, trump)}`)
    .join(',');

  let ledSuit = trick.ledSuit ?? -1;
  if (ledSuit >= 0 && ledSuit <= 6) {
    ledSuit = suitMap.get(ledSuit as RegularSuit) ?? ledSuit;
  }

  return `${plays}|W:${trick.winner}|L:${ledSuit}`;
}

// ============================================================================
// Metrics Collection
// ============================================================================

/**
 * Create empty metrics container.
 */
export function createEmptyMetrics(): CFRMetrics {
  return {
    branchingFactors: [],
    uniqueInfoSets: new Set(),
    canonicalInfoSets: new Set(),
    countCentricInfoSets: new Set(),
    infoSetCounts: new Map(),
    countCentricCounts: new Map(),
    totalDecisionPoints: 0
  };
}

/**
 * Record a decision point in the metrics.
 */
export function collectDecisionPoint(
  state: GameState,
  legalActions: ValidAction[],
  playerIndex: number,
  metrics: CFRMetrics
): void {
  // Only collect during playing phase
  if (state.phase !== 'playing') {
    return;
  }

  // Record branching factor
  metrics.branchingFactors.push(legalActions.length);

  // Compute information set hash
  const hash = computeInformationSetHash(state, playerIndex);

  // Add to unique sets
  metrics.uniqueInfoSets.add(hash.raw);
  metrics.canonicalInfoSets.add(hash.canonical);
  metrics.countCentricInfoSets.add(hash.countCentric);

  // Update revisitation counts
  const count = metrics.infoSetCounts.get(hash.canonical) || 0;
  metrics.infoSetCounts.set(hash.canonical, count + 1);

  const countCentricCount = metrics.countCentricCounts.get(hash.countCentric) || 0;
  metrics.countCentricCounts.set(hash.countCentric, countCentricCount + 1);

  metrics.totalDecisionPoints++;
}

/**
 * Run metrics collection over N simulated games.
 */
export async function runMetricsCollection(
  numGames: number,
  baseSeed: number = 12345,
  onProgress?: (gameNum: number, numGames: number) => void
): Promise<CFRMetrics> {
  const metrics = createEmptyMetrics();

  // Use random AI for speed - we just need to observe decision points,
  // not make smart decisions. This avoids Monte Carlo overhead.
  const randomAI = new RandomAIStrategy();

  for (let gameNum = 0; gameNum < numGames; gameNum++) {
    const seed = baseSeed + gameNum * 1000000;

    // Report progress
    if (onProgress && gameNum % 10 === 0) {
      onProgress(gameNum, numGames);
    }

    // Create HeadlessRoom for this game
    const room = new HeadlessRoom(
      {
        playerTypes: ['ai', 'ai', 'ai', 'ai'],
        shuffleSeed: seed
      },
      seed
    );

    let actionsExecuted = 0;
    const maxActions = 5000;

    // Run game until completion
    while (room.getState().phase !== 'game_end' && actionsExecuted < maxActions) {
      const state = room.getState();

      // Get all valid actions
      const actionsMap = room.getAllActions();
      const allActions: ValidAction[] = Object.values(actionsMap).flat();

      if (allActions.length === 0) {
        break;
      }

      // Check for auto-execute actions
      const autoAction = allActions.find(va =>
        va.action.type === 'complete-trick' ||
        va.action.type === 'score-hand' ||
        va.action.autoExecute === true
      );

      let selectedAction: ValidAction | undefined;
      const currentPlayer = state.currentPlayer;

      if (autoAction) {
        selectedAction = autoAction;
      } else {
        // Get player-specific actions
        const playerActions = room.getValidActions(currentPlayer);

        if (playerActions.length === 0) {
          break;
        }

        // COLLECT METRICS: Only during playing phase, before AI decision
        if (state.phase === 'playing') {
          collectDecisionPoint(state, playerActions, currentPlayer, metrics);
        }

        // Select action using RANDOM AI (fast!) for metrics collection
        selectedAction = randomAI.chooseAction(state, playerActions);
      }

      if (!selectedAction) {
        break;
      }

      // Execute action
      const executingPlayer = 'player' in selectedAction.action
        ? selectedAction.action.player
        : currentPlayer;

      try {
        room.executeAction(executingPlayer, selectedAction.action);
        actionsExecuted++;
      } catch {
        break;
      }
    }

    // Yield periodically
    if (gameNum % 100 === 0) {
      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  return metrics;
}

// ============================================================================
// Report Generation
// ============================================================================

/**
 * Generate a summary report from collected metrics.
 */
export function generateReport(metrics: CFRMetrics, gamesSimulated: number): MetricsReport {
  const {
    branchingFactors,
    uniqueInfoSets,
    canonicalInfoSets,
    countCentricInfoSets,
    infoSetCounts,
    countCentricCounts,
    totalDecisionPoints
  } = metrics;

  // Branching factor stats
  const sortedBF = [...branchingFactors].sort((a, b) => a - b);
  const bfMin = sortedBF[0] || 0;
  const bfMax = sortedBF[sortedBF.length - 1] || 0;
  const bfMean = sortedBF.length > 0
    ? sortedBF.reduce((a, b) => a + b, 0) / sortedBF.length
    : 0;
  const bfMedian = sortedBF.length > 0
    ? sortedBF[Math.floor(sortedBF.length / 2)] || 0
    : 0;

  // Branching factor distribution
  const bfDist: Record<number, number> = {};
  for (const bf of branchingFactors) {
    bfDist[bf] = (bfDist[bf] || 0) + 1;
  }

  // Compression ratio (canonical vs raw)
  const compressionRatio = canonicalInfoSets.size > 0
    ? uniqueInfoSets.size / canonicalInfoSets.size
    : 1;

  // Count-centric compression ratio (vs canonical)
  const countCentricCompressionRatio = countCentricInfoSets.size > 0
    ? canonicalInfoSets.size / countCentricInfoSets.size
    : 1;

  // Revisitation stats for canonical
  let totalVisits = 0;
  let singletons = 0;
  for (const count of infoSetCounts.values()) {
    totalVisits += count;
    if (count === 1) {
      singletons++;
    }
  }

  const revisitationRate = infoSetCounts.size > 0
    ? totalVisits / infoSetCounts.size
    : 1;
  const singletonRate = infoSetCounts.size > 0
    ? singletons / infoSetCounts.size
    : 1;

  // Revisitation stats for count-centric
  let ccTotalVisits = 0;
  let ccSingletons = 0;
  for (const count of countCentricCounts.values()) {
    ccTotalVisits += count;
    if (count === 1) {
      ccSingletons++;
    }
  }

  const countCentricRevisitationRate = countCentricCounts.size > 0
    ? ccTotalVisits / countCentricCounts.size
    : 1;
  const countCentricSingletonRate = countCentricCounts.size > 0
    ? ccSingletons / countCentricCounts.size
    : 1;

  return {
    gamesSimulated,
    totalDecisionPoints,

    branchingFactor: {
      min: bfMin,
      max: bfMax,
      mean: Math.round(bfMean * 100) / 100,
      median: bfMedian,
      distribution: bfDist
    },

    uniqueInfoSets: uniqueInfoSets.size,
    canonicalInfoSets: canonicalInfoSets.size,
    countCentricInfoSets: countCentricInfoSets.size,
    compressionRatio: Math.round(compressionRatio * 100) / 100,
    countCentricCompressionRatio: Math.round(countCentricCompressionRatio * 100) / 100,

    revisitationRate: Math.round(revisitationRate * 100) / 100,
    singletonRate: Math.round(singletonRate * 10000) / 100, // As percentage
    countCentricRevisitationRate: Math.round(countCentricRevisitationRate * 100) / 100,
    countCentricSingletonRate: Math.round(countCentricSingletonRate * 10000) / 100 // As percentage
  };
}
