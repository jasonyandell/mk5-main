/**
 * Monte Carlo Evaluator for Intermediate AI
 *
 * Evaluates candidate plays by simulating many games forward
 * with sampled opponent hands and beginner AI rollout.
 *
 * Key insight: We evaluate TEAM outcome, not individual trick wins.
 * Partnership dynamics emerge naturally from the simulation.
 */

import type { GameState, GameAction, Domino, TrumpSelection } from '../types';
import { NO_LEAD_SUIT } from '../types';
import type { ValidAction } from '../../multiplayer/types';
import type { HandConstraints } from './constraint-tracker';
import type { SampledHands } from './hand-sampler';
import { sampleOpponentHands, sampleBiddingHands, createSeededRng } from './hand-sampler';
import { getExpectedHandSizes } from './constraint-tracker';
import { executeAction } from '../core/actions';
import type { ExecutionContext } from '../types/execution';
import { analyzeSuits } from '../core/suit-analysis';
import { createDominoes } from '../core/dominoes';
import { determineBestTrump } from './hand-strength';
import { getRolloutStrategy } from './rollout-strategy';

/**
 * Configuration for Monte Carlo evaluation
 */
export interface MonteCarloConfig {
  /** Number of simulations per candidate action (default: 50) */
  simulations: number;

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

    for (let sim = 0; sim < config.simulations; sim++) {
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
      makeBidRate: madeCount / config.simulations,
      avgPointsWhenMade: madeCount > 0 ? pointsWhenMade / madeCount : 0,
      avgPointsOverall: totalPoints / config.simulations,
      simulationCount: config.simulations
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

    const suitAnalysis = analyzeSuits(hand, trump);

    return {
      ...player,
      hand,
      suitAnalysis
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
  const rng = config.seed !== undefined
    ? createSeededRng(config.seed)
    : { random: () => Math.random() };

  const myTeam = myPlayerIndex % 2; // 0 or 1 (players 0,2 vs 1,3)
  const expectedSizes = getExpectedHandSizes(state);

  const evaluations: ActionEvaluation[] = [];

  for (const action of playActions) {
    let totalTeamPoints = 0;
    let wins = 0;

    for (let sim = 0; sim < config.simulations; sim++) {
      // Sample opponent hands
      let sampledHands: SampledHands;
      try {
        sampledHands = sampleOpponentHands(
          constraints,
          expectedSizes,
          state,
          ctx.rules,
          rng
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
      let simState = executeAction(fullState, action.action, ctx.rules);

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
      avgTeamPoints: totalTeamPoints / config.simulations,
      winRate: wins / config.simulations,
      simulationCount: config.simulations
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

    // Recompute suit analysis for the new hand
    const suitAnalysis = analyzeSuits(sampledHand, state.trump);

    return {
      ...player,
      hand: sampledHand,
      suitAnalysis
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
 * Rollout from current state to hand completion using beginner AI.
 *
 * Simulates all players making moves until either:
 * - Hand is complete (all tricks played or early termination)
 * - Max iterations reached (safety limit)
 */
function rolloutToHandEnd(
  initialState: GameState,
  ctx: ExecutionContext,
  maxIterations: number = 200
): GameState {
  let state = initialState;
  let iterations = 0;

  while (iterations < maxIterations) {
    // Check if hand is complete
    if (isHandComplete(state)) {
      break;
    }

    // Get valid actions
    const actions = ctx.getValidActions(state);

    if (actions.length === 0) {
      // No actions available - hand should be complete
      break;
    }

    // Check for auto-execute actions first
    const autoAction = actions.find(a => a.autoExecute === true);
    if (autoAction) {
      state = executeAction(state, autoAction, ctx.rules);
      iterations++;
      continue;
    }

    // Convert to ValidAction format for AI selector
    const validActions = actions.map(action => actionToValidAction(action, state));

    // Use beginner AI DIRECTLY to select action
    // IMPORTANT: Do NOT use selectAIAction as it would use the default strategy,
    // which might be intermediate, causing infinite recursion
    const currentPlayer = state.currentPlayer;

    // Filter to current player's actions
    const myActions = validActions.filter(va => {
      if (!('player' in va.action)) return true;
      return va.action.player === currentPlayer;
    });

    if (myActions.length === 0) {
      break;
    }

    const selected = getRolloutStrategy().chooseAction(state, myActions);

    if (!selected) {
      // AI couldn't select - shouldn't happen with valid actions
      // Take first action as fallback
      const fallback = validActions[0];
      if (fallback) {
        state = executeAction(state, fallback.action, ctx.rules);
      }
      break;
    }

    state = executeAction(state, selected.action, ctx.rules);
    iterations++;
  }

  return state;
}

/**
 * Check if the current hand is complete.
 *
 * We only check for 'scoring' phase because:
 * 1. All game modes reach 'scoring' before transitioning elsewhere
 * 2. The rollout loop checks this BEFORE executing actions, so we stop
 *    at 'scoring' before any consensus/post-scoring transitions happen
 * 3. This avoids leaking layer-specific phase knowledge (like 'one-hand-complete')
 */
function isHandComplete(state: GameState): boolean {
  return state.phase === 'scoring';
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
 * Convert a GameAction to ValidAction format.
 */
function actionToValidAction(action: GameAction, state: GameState): ValidAction {
  return {
    action,
    label: getActionLabel(action, state),
    group: getActionGroup(action)
  };
}

/**
 * Generate a label for an action (simplified version).
 */
function getActionLabel(action: GameAction, state: GameState): string {
  switch (action.type) {
    case 'play': {
      const player = state.players[action.player];
      const domino = player?.hand.find(d => String(d.id) === action.dominoId);
      if (domino) {
        return `${domino.high}-${domino.low}`;
      }
      return action.dominoId;
    }
    case 'bid':
      return `Bid ${action.value}`;
    case 'pass':
      return 'Pass';
    case 'select-trump':
      return `Trump: ${action.trump.type}`;
    case 'complete-trick':
      return 'Complete Trick';
    case 'score-hand':
      return 'Score Hand';
    default:
      return action.type;
  }
}

/**
 * Get action group for UI organization.
 */
function getActionGroup(action: GameAction): string {
  switch (action.type) {
    case 'play':
      return 'plays';
    case 'bid':
    case 'pass':
      return 'bids';
    case 'select-trump':
      return 'trump';
    default:
      return 'other';
  }
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
