/**
 * Monte Carlo Evaluator for Intermediate AI
 *
 * Evaluates candidate plays by simulating many games forward
 * with sampled opponent hands and beginner AI rollout.
 *
 * Key insight: We evaluate TEAM outcome, not individual trick wins.
 * Partnership dynamics emerge naturally from the simulation.
 */

import type { GameState, Domino, TrumpSelection } from '../types';
import { NO_LEAD_SUIT } from '../types';
import type { ValidAction } from '../../multiplayer/types';
import type { HandConstraints } from './constraint-tracker';
import type { SampledHands } from './hand-sampler';
import { sampleOpponentHands, sampleBiddingHands, createSeededRng, countValidConfigurations, enumerateAllConfigurations } from './hand-sampler';
import { getExpectedHandSizes, buildCanFollowCache } from './constraint-tracker';

/** Threshold for switching from sampling to enumeration */
const ENUMERATION_THRESHOLD = 100;
import { executeAction, type ExecuteActionOptions } from '../core/actions';
import type { ExecutionContext } from '../types/execution';
import { createDominoes } from '../core/dominoes';
import { determineBestTrump } from './hand-strength';
import { minimaxEvaluate, createTerminalState } from './minimax';

/** Reusable options object to avoid allocation per call */
const SIMULATION_OPTIONS: ExecuteActionOptions = { skipHistory: true };

/**
 * Configuration for Monte Carlo evaluation
 */
export interface MonteCarloConfig {
  /** Number of simulations for bidding decisions (default: 5) */
  biddingSimulations: number;

  /** Number of simulations for play decisions (default: 10) */
  playingSimulations: number;

  /** Random seed for reproducibility (optional) */
  seed?: number;
}

/**
 * Result of evaluating a single action
 */
export interface ActionEvaluation {
  /** The action that was evaluated */
  action: ValidAction;

  /** Average team points across simulations */
  avgTeamPoints: number;

  /** Win rate (fraction of simulations where team made their bid) */
  winRate: number;

  /** Number of simulations run */
  simulationCount: number;
}

/**
 * Result of evaluating a bid action
 */
export interface BidEvaluation {
  /** The bid action that was evaluated */
  action: ValidAction;

  /** Numeric bid value (30-42) */
  bidValue: number;

  /** Fraction of simulations where bid was made */
  makeBidRate: number;

  /** Average team points when bid was made */
  avgPointsWhenMade: number;

  /** Average team points overall (including failures) */
  avgPointsOverall: number;

  /** Number of simulations run */
  simulationCount: number;
}

// All 28 dominoes (cached for bidding simulations)
const ALL_DOMINOES = createDominoes();

/**
 * Evaluate all candidate bid actions using Monte Carlo simulation.
 *
 * For each candidate bid:
 * 1. Sample opponent hands (no constraints during bidding)
 * 2. Determine best trump for bidder's hand
 * 3. Rollout full hand to completion
 * 4. Check if bidder's team made the bid
 *
 * @param state Current game state (bidding phase)
 * @param bidActions Array of valid bid actions to evaluate
 * @param myPlayerIndex The AI player's index (0-3)
 * @param ctx Execution context with composed rules
 * @param config Monte Carlo configuration
 * @returns Evaluations for each action, sorted by makeBidRate descending
 */
export function evaluateBidActions(
  state: GameState,
  bidActions: ValidAction[],
  myPlayerIndex: number,
  ctx: ExecutionContext,
  config: MonteCarloConfig
): BidEvaluation[] {
  const rng = config.seed !== undefined
    ? createSeededRng(config.seed)
    : { random: () => Math.random() };

  const myTeam = myPlayerIndex % 2;
  const myHand = state.players[myPlayerIndex]?.hand ?? [];

  // Determine best trump once (same hand, same trump choice)
  const bestTrump = determineBestTrump(myHand);

  const evaluations: BidEvaluation[] = [];

  for (const bidAction of bidActions) {
    // Extract bid value from action
    const bidValue = getBidValue(bidAction);
    if (bidValue === 0) {
      // Pass action - skip evaluation (will be handled separately)
      continue;
    }

    let totalPoints = 0;
    let pointsWhenMade = 0;
    let madeCount = 0;

    for (let sim = 0; sim < config.biddingSimulations; sim++) {
      // Sample opponent hands
      const sampledHands = sampleBiddingHands(ALL_DOMINOES, myHand, myPlayerIndex, rng);

      // Create state ready for play (skip bidding/trump selection)
      // We're asking: "if I win this bid, can I make it?"
      const simState = createPlayReadyState(
        state,
        sampledHands,
        myPlayerIndex,
        myHand,
        bestTrump,
        bidValue
      );

      // Rollout to hand completion
      const finalState = rolloutToHandEnd(simState, ctx);

      // Record outcome
      const teamPoints = finalState.teamScores[myTeam] ?? 0;
      totalPoints += teamPoints;

      if (teamPoints >= bidValue) {
        madeCount++;
        pointsWhenMade += teamPoints;
      }
    }

    evaluations.push({
      action: bidAction,
      bidValue,
      makeBidRate: madeCount / config.biddingSimulations,
      avgPointsWhenMade: madeCount > 0 ? pointsWhenMade / madeCount : 0,
      avgPointsOverall: totalPoints / config.biddingSimulations,
      simulationCount: config.biddingSimulations
    });
  }

  // Sort by makeBidRate descending
  evaluations.sort((a, b) => b.makeBidRate - a.makeBidRate);

  return evaluations;
}

/**
 * Extract numeric bid value from a bid action.
 * Returns 0 for pass actions.
 */
function getBidValue(action: ValidAction): number {
  if (action.action.type === 'pass') {
    return 0;
  }
  if (action.action.type === 'bid') {
    if (action.action.bid === 'marks') {
      return (action.action.value ?? 1) * 42;
    }
    return action.action.value ?? 30;
  }
  return 0;
}

/**
 * Create a state ready for play simulation.
 *
 * Sets up the state as if bidding and trump selection are complete:
 * - Phase is 'playing'
 * - Bidder is set as winning bidder with their bid value
 * - Trump is selected
 * - All hands are in place
 * - Bidder leads (currentPlayer = bidder)
 */
function createPlayReadyState(
  state: GameState,
  sampledHands: SampledHands,
  bidderIndex: number,
  bidderHand: Domino[],
  trump: TrumpSelection,
  bidValue: number
): GameState {
  const newPlayers = state.players.map((player, index) => {
    let hand: Domino[];
    if (index === bidderIndex) {
      hand = bidderHand;
    } else {
      hand = sampledHands.get(index) ?? player.hand;
    }

    return {
      ...player,
      hand
    };
  });

  return {
    ...state,
    players: newPlayers,
    // Override playerTypes to all AI so consensus layer auto-executes
    // (without this, simulations with human players would stall on agree-trick)
    playerTypes: ['ai', 'ai', 'ai', 'ai'] as const,
    phase: 'playing' as const,
    trump,
    winningBidder: bidderIndex,
    currentBid: { type: 'points' as const, value: bidValue, player: bidderIndex },
    currentPlayer: bidderIndex, // Bidder leads first trick
    currentTrick: [],
    tricks: [],
    teamScores: [0, 0],
    currentSuit: NO_LEAD_SUIT
  };
}

/**
 * Evaluate all candidate play actions using Monte Carlo simulation.
 *
 * For each candidate action:
 * 1. Sample opponent hands (respecting constraints)
 * 2. Apply the candidate action
 * 3. Rollout to hand completion using beginner AI
 * 4. Record team outcome
 *
 * Optimization: In late game when few configurations exist, enumerate all
 * possible opponent hands rather than sampling. This gives exact expected
 * value instead of Monte Carlo approximation.
 *
 * @param state Current game state (with hidden opponent hands)
 * @param playActions Array of valid play actions to evaluate
 * @param myPlayerIndex The AI player's index (0-3)
 * @param constraints Constraints on opponent hands
 * @param ctx Execution context with composed rules
 * @param config Monte Carlo configuration
 * @returns Evaluations for each action, sorted by avgTeamPoints descending
 */
export function evaluatePlayActions(
  state: GameState,
  playActions: ValidAction[],
  myPlayerIndex: number,
  constraints: HandConstraints,
  ctx: ExecutionContext,
  config: MonteCarloConfig
): ActionEvaluation[] {
  const myTeam = myPlayerIndex % 2; // 0 or 1 (players 0,2 vs 1,3)
  const expectedSizes = getExpectedHandSizes(state);

  // Build canFollow cache once (trump is fixed for entire evaluation)
  const canFollowCache = buildCanFollowCache(state, ctx.rules);

  // Check if we should enumerate rather than sample
  // Count configurations with early termination at threshold
  const configCount = countValidConfigurations(
    constraints,
    expectedSizes,
    state,
    ctx.rules,
    ENUMERATION_THRESHOLD,
    canFollowCache
  );

  // Use enumeration for small configuration counts
  if (configCount <= ENUMERATION_THRESHOLD) {
    return evaluateWithEnumeration(
      state,
      playActions,
      myPlayerIndex,
      myTeam,
      constraints,
      expectedSizes,
      ctx,
      canFollowCache
    );
  }

  // Fall back to Monte Carlo sampling
  return evaluateWithSampling(
    state,
    playActions,
    myPlayerIndex,
    myTeam,
    constraints,
    expectedSizes,
    ctx,
    config,
    canFollowCache
  );
}

/**
 * Evaluate actions by enumerating all possible opponent hand configurations.
 *
 * Used in endgame when the number of valid configurations is small.
 * Gives exact expected value rather than Monte Carlo approximation.
 */
function evaluateWithEnumeration(
  state: GameState,
  playActions: ValidAction[],
  myPlayerIndex: number,
  myTeam: number,
  constraints: HandConstraints,
  expectedSizes: [number, number, number, number],
  ctx: ExecutionContext,
  canFollowCache: ReturnType<typeof buildCanFollowCache>
): ActionEvaluation[] {
  // Enumerate all valid configurations
  const allConfigurations = enumerateAllConfigurations(
    constraints,
    expectedSizes,
    state,
    ctx.rules,
    ENUMERATION_THRESHOLD,
    canFollowCache
  );

  // If no configurations found (shouldn't happen), fall back to empty result
  if (allConfigurations.length === 0) {
    return playActions.map(action => ({
      action,
      avgTeamPoints: 0,
      winRate: 0,
      simulationCount: 0
    }));
  }

  const evaluations: ActionEvaluation[] = [];

  for (const action of playActions) {
    let totalTeamPoints = 0;
    let wins = 0;

    // Evaluate against every possible configuration
    for (const sampledHands of allConfigurations) {
      // Create state with injected opponent hands
      const fullState = injectSampledHands(state, sampledHands, myPlayerIndex);

      // Apply the candidate action
      let simState = executeAction(fullState, action.action, ctx.rules, SIMULATION_OPTIONS);

      // Rollout to hand completion
      simState = rolloutToHandEnd(simState, ctx);

      // Record outcome
      const teamPoints = simState.teamScores[myTeam] ?? 0;
      totalTeamPoints += teamPoints;

      // Check if team made their bid
      const bidValue = getBidTargetForTeam(state, myTeam);
      if (teamPoints >= bidValue) {
        wins++;
      }
    }

    evaluations.push({
      action,
      avgTeamPoints: totalTeamPoints / allConfigurations.length,
      winRate: wins / allConfigurations.length,
      simulationCount: allConfigurations.length
    });
  }

  // Sort by average team points (descending)
  evaluations.sort((a, b) => b.avgTeamPoints - a.avgTeamPoints);

  return evaluations;
}

/**
 * Evaluate actions using Monte Carlo sampling.
 *
 * Used when the number of possible configurations is large.
 */
function evaluateWithSampling(
  state: GameState,
  playActions: ValidAction[],
  myPlayerIndex: number,
  myTeam: number,
  constraints: HandConstraints,
  expectedSizes: [number, number, number, number],
  ctx: ExecutionContext,
  config: MonteCarloConfig,
  canFollowCache: ReturnType<typeof buildCanFollowCache>
): ActionEvaluation[] {
  const rng = config.seed !== undefined
    ? createSeededRng(config.seed)
    : { random: () => Math.random() };

  const evaluations: ActionEvaluation[] = [];

  for (const action of playActions) {
    let totalTeamPoints = 0;
    let wins = 0;

    for (let sim = 0; sim < config.playingSimulations; sim++) {
      // Sample opponent hands
      let sampledHands: SampledHands;
      try {
        sampledHands = sampleOpponentHands(
          constraints,
          expectedSizes,
          state,
          ctx.rules,
          rng,
          canFollowCache
        );
      } catch (e) {
        // Add debug info to the error
        const debugInfo = {
          phase: state.phase,
          tricksPlayed: state.tricks.length,
          currentTrickSize: state.currentTrick.length,
          myPlayerIndex,
          expectedSizes,
          playedCount: constraints.played.size,
          myHandSize: constraints.myHand.size,
          voidSuits: Object.fromEntries(
            Array.from(constraints.voidInSuit.entries()).map(([k, v]) => [k, Array.from(v)])
          )
        };
        throw new Error(
          `${e instanceof Error ? e.message : e}\n` +
          `Debug info: ${JSON.stringify(debugInfo, null, 2)}`
        );
      }

      // Create state with injected opponent hands
      const fullState = injectSampledHands(state, sampledHands, myPlayerIndex);

      // Apply the candidate action
      let simState = executeAction(fullState, action.action, ctx.rules, SIMULATION_OPTIONS);

      // Rollout to hand completion
      simState = rolloutToHandEnd(simState, ctx);

      // Record outcome
      const teamPoints = simState.teamScores[myTeam] ?? 0;
      totalTeamPoints += teamPoints;

      // Check if team made their bid (simplified win check)
      const bidValue = getBidTargetForTeam(state, myTeam);
      if (teamPoints >= bidValue) {
        wins++;
      }
    }

    evaluations.push({
      action,
      avgTeamPoints: totalTeamPoints / config.playingSimulations,
      winRate: wins / config.playingSimulations,
      simulationCount: config.playingSimulations
    });
  }

  // Sort by average team points (descending)
  evaluations.sort((a, b) => b.avgTeamPoints - a.avgTeamPoints);

  return evaluations;
}

/**
 * Create a copy of the state with sampled opponent hands injected.
 *
 * The AI's own hand stays unchanged. Opponent hands are replaced
 * with the sampled distributions.
 */
function injectSampledHands(
  state: GameState,
  sampledHands: SampledHands,
  myPlayerIndex: number
): GameState {
  const newPlayers = state.players.map((player, index) => {
    if (index === myPlayerIndex) {
      // Keep AI's own hand
      return player;
    }

    const sampledHand = sampledHands.get(index);
    if (!sampledHand) {
      // Shouldn't happen, but keep original if missing
      return player;
    }

    return {
      ...player,
      hand: sampledHand
    };
  });

  return {
    ...state,
    players: newPlayers,
    // Override playerTypes to all AI so consensus layer auto-executes
    // (without this, simulations with human players would stall on agree-trick)
    playerTypes: ['ai', 'ai', 'ai', 'ai'] as const
  };
}

/**
 * Evaluate hand to completion using minimax.
 *
 * Replaces heuristic rollout with game-theoretic optimal search.
 * This ensures the AI finds "fighting lines" even when losing,
 * rather than giving up and dumping count.
 *
 * @param initialState State with all hands visible (sampled)
 * @param ctx Execution context with composed rules
 * @returns Terminal state representing hand completion
 */
function rolloutToHandEnd(
  initialState: GameState,
  ctx: ExecutionContext
): GameState {
  // Use minimax to find optimal outcome
  const result = minimaxEvaluate(initialState, ctx);

  // Return terminal state with computed scores
  return createTerminalState(initialState, result);
}

/**
 * Get the bid target for a team.
 *
 * Returns the number of points the team needs to make their bid.
 * For non-bidding team, returns 0 (any points are good).
 */
function getBidTargetForTeam(state: GameState, teamIndex: number): number {
  if (state.winningBidder === -1) {
    return 0;
  }

  const bidderTeam = state.winningBidder % 2;

  if (teamIndex !== bidderTeam) {
    // Defending team - any points count against bidder
    // Return high number so they "win" by setting the bid
    return 1; // Even 1 point is a "win" for defense (set the bid)
  }

  // Bidding team - need to make the bid
  const currentBid = state.currentBid;
  if (currentBid.type === 'points' && currentBid.value !== undefined) {
    return currentBid.value;
  }

  // For marks bids and special contracts, need all 42
  return 42;
}

/**
 * Quick evaluation: returns the best action without full evaluation data.
 * Use this when you just need the result, not the analysis.
 */
export function selectBestPlay(
  state: GameState,
  playActions: ValidAction[],
  myPlayerIndex: number,
  constraints: HandConstraints,
  ctx: ExecutionContext,
  config: MonteCarloConfig
): ValidAction | null {
  if (playActions.length === 0) {
    return null;
  }

  if (playActions.length === 1) {
    // Only one option - no need to simulate
    return playActions[0] ?? null;
  }

  const evaluations = evaluatePlayActions(
    state,
    playActions,
    myPlayerIndex,
    constraints,
    ctx,
    config
  );

  return evaluations[0]?.action ?? null;
}

// ============================================================================
// Trump-First Bidding Evaluation
// ============================================================================

/**
 * Result of evaluating a single trump option.
 */
export interface TrumpEvaluation {
  /** The trump selection */
  trump: TrumpSelection;

  /** Average points when this trump is selected */
  avgPoints: number;

  /** Point distribution percentiles */
  distribution: {
    /** Probability of making >= 30 points */
    p30: number;
    /** Probability of making >= 34 points */
    p34: number;
    /** Probability of making all 42 points */
    p42: number;
  };

  /** Expected value considering bid levels */
  expectedValue: number;

  /** Number of simulations run */
  simulationCount: number;

  /** Is this a special contract (nello, sevens) with all-or-nothing outcome? */
  isAllOrNothing: boolean;
}

/**
 * Recommended bid based on trump evaluations.
 */
export interface BidRecommendation {
  /** The recommended bid action */
  action: ValidAction;

  /** The trump to select if bid is won */
  trump: TrumpSelection;

  /** Expected value of this bid */
  expectedValue: number;

  /** Confidence level (0-1) */
  confidence: number;

  /** Should we pass instead? */
  shouldPass: boolean;
}

/**
 * Get all valid trump selections for the current state.
 *
 * This discovers trump options from the layer system by simulating
 * a state where bidding is won and checking getValidActions.
 *
 * @param state Current game state (bidding phase)
 * @param myPlayerIndex The bidder's player index
 * @param ctx Execution context with composed layers
 * @returns Array of valid trump selections
 */
export function discoverTrumpOptions(
  state: GameState,
  myPlayerIndex: number,
  ctx: ExecutionContext
): TrumpSelection[] {
  // Create a hypothetical state where we've won with a marks bid
  // (marks bids unlock nello/sevens trump options)
  const trumpSelectionState: GameState = {
    ...state,
    phase: 'trump_selection' as const,
    winningBidder: myPlayerIndex,
    currentPlayer: myPlayerIndex,
    currentBid: { type: 'marks' as const, value: 1, player: myPlayerIndex }
  };

  // Get all valid actions for trump selection phase
  const actions = ctx.getValidActions(trumpSelectionState);

  // Extract trump selections
  const trumpOptions: TrumpSelection[] = [];
  const seen = new Set<string>();

  for (const action of actions) {
    if (action.type === 'select-trump' && action.trump) {
      const key = action.trump.type + (action.trump.suit ?? '');
      if (!seen.has(key)) {
        seen.add(key);
        trumpOptions.push(action.trump);
      }
    }
  }

  return trumpOptions;
}

/**
 * Evaluate all trump options for a hand.
 *
 * For EACH trump option, runs N simulations to hand completion and tracks:
 * - Average team points
 * - Distribution: P(≥30), P(≥34), P(≥42)
 *
 * This is more efficient than the old approach which ran per-bid-level.
 *
 * @param state Current game state (bidding phase)
 * @param myPlayerIndex The bidder's player index
 * @param ctx Execution context with composed layers
 * @param config Monte Carlo configuration
 * @returns Map of trump selection to evaluation results
 */
export function evaluateTrumpOptions(
  state: GameState,
  myPlayerIndex: number,
  ctx: ExecutionContext,
  config: MonteCarloConfig
): Map<string, TrumpEvaluation> {
  const rng = config.seed !== undefined
    ? createSeededRng(config.seed)
    : { random: () => Math.random() };

  const myHand = state.players[myPlayerIndex]?.hand ?? [];
  const myTeam = myPlayerIndex % 2;

  // Discover all trump options from layers
  const trumpOptions = discoverTrumpOptions(state, myPlayerIndex, ctx);

  const evaluations = new Map<string, TrumpEvaluation>();

  for (const trump of trumpOptions) {
    const trumpKey = trump.type + (trump.suit ?? '');

    // Check if this is an all-or-nothing contract
    const isAllOrNothing = trump.type === 'nello' || trump.type === 'sevens';

    // Track outcomes
    const outcomes: number[] = [];
    let successCount = 0;

    for (let sim = 0; sim < config.biddingSimulations; sim++) {
      // Sample opponent hands
      const sampledHands = sampleBiddingHands(ALL_DOMINOES, myHand, myPlayerIndex, rng);

      // Create state ready for play with this trump
      const simState = createPlayReadyState(
        state,
        sampledHands,
        myPlayerIndex,
        myHand,
        trump,
        42 // Use max bid for simulation to get accurate point distribution
      );

      // Rollout to hand completion
      const finalState = rolloutToHandEnd(simState, ctx);

      // Record outcome
      const teamPoints = finalState.teamScores[myTeam] ?? 0;
      outcomes.push(teamPoints);

      // For all-or-nothing contracts, track success rate differently
      if (isAllOrNothing) {
        // Nello: success = 0 tricks taken by bidding team (checked by checkHandOutcome)
        // Sevens: success = all tricks won by bidding team
        // The teamScores reflect whether we succeeded
        if (trump.type === 'nello') {
          // In nello, we want to LOSE all tricks, so our score should be 0
          // But the way scoring works, we get the marks if we succeed
          // The final state's teamScores won't directly show this
          // Instead, check if we took any tricks (we shouldn't)
          successCount += teamPoints === 0 ? 1 : 0;
        } else {
          // For sevens, we need all 42 points
          successCount += teamPoints === 42 ? 1 : 0;
        }
      }
    }

    // Calculate statistics
    const avgPoints = outcomes.reduce((a, b) => a + b, 0) / outcomes.length;
    const p30 = outcomes.filter(p => p >= 30).length / outcomes.length;
    const p34 = outcomes.filter(p => p >= 34).length / outcomes.length;
    const p42 = outcomes.filter(p => p >= 42).length / outcomes.length;

    // Calculate expected value
    let expectedValue: number;
    if (isAllOrNothing) {
      // All-or-nothing: EV = P(success) * 42 + P(failure) * 0
      // (Actually the stakes are in marks, but we normalize to points)
      const successRate = successCount / config.biddingSimulations;
      expectedValue = successRate * 42;
    } else {
      // Standard trump: EV based on average points
      expectedValue = avgPoints;
    }

    evaluations.set(trumpKey, {
      trump,
      avgPoints,
      distribution: { p30, p34, p42 },
      expectedValue,
      simulationCount: config.biddingSimulations,
      isAllOrNothing
    });
  }

  return evaluations;
}

/**
 * Risk tolerance levels for bid decisions.
 */
export type RiskTolerance = 'conservative' | 'balanced' | 'aggressive';

/**
 * Decide what to bid based on trump evaluations.
 *
 * Takes the EV table from evaluateTrumpOptions and applies risk preferences
 * to determine the optimal bid.
 *
 * @param trumpEvaluations Results from evaluateTrumpOptions
 * @param validBidActions Available bid actions
 * @param riskTolerance How aggressive to be
 * @returns Recommended bid and associated trump
 */
export function decideBid(
  trumpEvaluations: Map<string, TrumpEvaluation>,
  validBidActions: ValidAction[],
  riskTolerance: RiskTolerance = 'balanced'
): BidRecommendation {
  // Find the best trump option
  let bestTrump: TrumpEvaluation | null = null;
  let bestEV = -Infinity;

  for (const evaluation of trumpEvaluations.values()) {
    const ev = evaluation.expectedValue;
    if (ev > bestEV) {
      bestEV = ev;
      bestTrump = evaluation;
    }
  }

  // If no trump evaluations, pass
  if (!bestTrump) {
    const passAction = validBidActions.find(a => a.action.type === 'pass');
    return {
      action: passAction ?? validBidActions[0]!,
      trump: { type: 'not-selected' },
      expectedValue: 0,
      confidence: 0,
      shouldPass: true
    };
  }

  // Determine bid thresholds based on risk tolerance
  let bidThreshold: number;
  let minMakeRate: number;

  switch (riskTolerance) {
    case 'conservative':
      bidThreshold = 34;  // Only bid if we expect 34+
      minMakeRate = 0.65; // Need 65% chance of making
      break;
    case 'aggressive':
      bidThreshold = 28;  // Bid with 28+ expected
      minMakeRate = 0.40; // 40% is enough
      break;
    case 'balanced':
    default:
      bidThreshold = 30;  // Standard 30-point threshold
      minMakeRate = 0.50; // 50% chance
      break;
  }

  // Find the highest bid we can make with acceptable risk
  const { distribution, isAllOrNothing, avgPoints } = bestTrump;

  // For all-or-nothing contracts, use p42 as the make rate
  // For standard contracts, use the appropriate threshold probability
  let recommendedBidValue = 0;
  let makeRate = 0;

  if (isAllOrNothing) {
    // All-or-nothing: either we bid marks or we don't
    makeRate = distribution.p42;
    if (makeRate >= minMakeRate) {
      recommendedBidValue = 42; // Will become a marks bid
    }
  } else {
    // Standard contracts: find highest safe bid
    if (distribution.p42 >= minMakeRate) {
      recommendedBidValue = 42;
      makeRate = distribution.p42;
    } else if (distribution.p34 >= minMakeRate) {
      recommendedBidValue = 34;
      makeRate = distribution.p34;
    } else if (distribution.p30 >= minMakeRate) {
      recommendedBidValue = 30;
      makeRate = distribution.p30;
    }

    // Also check if expected value meets threshold
    if (avgPoints < bidThreshold) {
      recommendedBidValue = 0; // Pass
    }
  }

  // Find the matching bid action
  let bidAction: ValidAction | undefined;

  if (recommendedBidValue >= 42) {
    // Look for marks bid first (for special contracts)
    bidAction = validBidActions.find(a =>
      a.action.type === 'bid' &&
      'bid' in a.action &&
      a.action.bid === 'marks'
    );
    // Fallback to 42 points bid
    if (!bidAction) {
      bidAction = validBidActions.find(a =>
        a.action.type === 'bid' &&
        'bid' in a.action &&
        a.action.bid === 'points' &&
        a.action.value === 42
      );
    }
  } else if (recommendedBidValue > 0) {
    // Find the highest valid points bid at or below recommendedBidValue
    const pointsBids = validBidActions
      .filter(a =>
        a.action.type === 'bid' &&
        'bid' in a.action &&
        a.action.bid === 'points' &&
        (a.action.value ?? 0) <= recommendedBidValue
      )
      .sort((a, b) => {
        const aVal = 'value' in a.action ? (a.action.value ?? 0) : 0;
        const bVal = 'value' in b.action ? (b.action.value ?? 0) : 0;
        return bVal - aVal;
      });
    bidAction = pointsBids[0];
  }

  // If no suitable bid found, pass
  if (!bidAction || recommendedBidValue === 0) {
    const passAction = validBidActions.find(a => a.action.type === 'pass');
    return {
      action: passAction ?? validBidActions[0]!,
      trump: bestTrump.trump,
      expectedValue: bestTrump.expectedValue,
      confidence: makeRate,
      shouldPass: true
    };
  }

  return {
    action: bidAction,
    trump: bestTrump.trump,
    expectedValue: bestTrump.expectedValue,
    confidence: makeRate,
    shouldPass: false
  };
}

/**
 * Complete bidding decision using trump-first evaluation.
 *
 * This is the new recommended entry point for AI bidding decisions.
 * It discovers all trump options, evaluates each one, and decides
 * the optimal bid based on the EV table.
 *
 * @param state Current game state (bidding phase)
 * @param validActions All valid actions including bids and pass
 * @param myPlayerIndex The bidder's player index
 * @param ctx Execution context with composed layers
 * @param config Monte Carlo configuration
 * @param riskTolerance How aggressive to bid
 * @returns The recommended bid action
 */
export function selectBestBid(
  state: GameState,
  validActions: ValidAction[],
  myPlayerIndex: number,
  ctx: ExecutionContext,
  config: MonteCarloConfig,
  riskTolerance: RiskTolerance = 'balanced'
): ValidAction {
  // Find pass action (always available during bidding)
  const passAction = validActions.find(va => va.action.type === 'pass');

  // Get bid actions (exclude pass)
  const bidActions = validActions.filter(va => va.action.type === 'bid');

  // If no bid actions available, must pass
  if (bidActions.length === 0) {
    if (passAction) return passAction;
    const fallback = validActions[0];
    if (!fallback) {
      throw new Error('selectBestBid: No valid actions available');
    }
    return fallback;
  }

  // Evaluate all trump options
  const trumpEvaluations = evaluateTrumpOptions(
    state,
    myPlayerIndex,
    ctx,
    config
  );

  // Decide on the best bid
  const recommendation = decideBid(
    trumpEvaluations,
    validActions,
    riskTolerance
  );

  return recommendation.action;
}
