#!/usr/bin/env npx tsx
/**
 * Evaluate a game state using TypeScript minimax for cross-validation.
 *
 * Usage: npx tsx scripts/minimax-eval.ts <seed> <decl_id>
 *
 * Output: JSON with team0Points for comparison with Python solver.
 * The Python value should equal 2 * team0Points - 42.
 *
 * NOTE: This uses a modified minimax that plays all 7 tricks without
 * early termination based on bid outcome. This matches the Python solver
 * which doesn't model bid contracts.
 */

import { dealDominoesWithSeed } from '../src/game/core/dominoes';
import { StateBuilder } from '../src/tests/helpers/stateBuilder';
import { createExecutionContext, type ExecutionContext } from '../src/game/types/execution';
import { executeAction, type ExecuteActionOptions } from '../src/game/core/actions';
import type { GameState, GameAction, TrumpSelection } from '../src/game/types';

/** Reusable options object to avoid allocation per call */
const SIMULATION_OPTIONS: ExecuteActionOptions = { skipHistory: true };

/**
 * Simple minimax for cross-validation.
 * Plays all 7 tricks without early termination.
 */
function minimaxFullGame(
  state: GameState,
  ctx: ExecutionContext,
): { team0Points: number; nodesExplored: number } {
  let nodesExplored = 0;

  function search(currentState: GameState, alpha: number, beta: number): number {
    nodesExplored++;

    // Terminal: all 7 tricks complete
    if (currentState.tricks.length === 7 && currentState.currentTrick.length === 0) {
      return currentState.teamScores[0];
    }

    // Get valid actions
    const actions = ctx.getValidActions(currentState);
    if (actions.length === 0) {
      return currentState.teamScores[0];
    }

    // Handle auto-execute actions (complete-trick, etc.)
    const autoAction = actions.find(a => a.autoExecute === true);
    if (autoAction) {
      const nextState = executeAction(currentState, autoAction, ctx.rules, SIMULATION_OPTIONS);
      return search(nextState, alpha, beta);
    }

    // Get play actions
    const playActions = actions.filter(a => a.type === 'play');
    if (playActions.length === 0) {
      return currentState.teamScores[0];
    }

    // Determine if current player is maximizing (Team 0) or minimizing (Team 1)
    const currentTeam = currentState.currentPlayer % 2;
    const isMaximizing = currentTeam === 0;

    if (isMaximizing) {
      let maxEval = -Infinity;
      for (const action of playActions) {
        const nextState = executeAction(currentState, action, ctx.rules, SIMULATION_OPTIONS);
        const evalScore = search(nextState, alpha, beta);
        maxEval = Math.max(maxEval, evalScore);
        alpha = Math.max(alpha, evalScore);
        if (beta <= alpha) break;
      }
      return maxEval;
    } else {
      let minEval = Infinity;
      for (const action of playActions) {
        const nextState = executeAction(currentState, action, ctx.rules, SIMULATION_OPTIONS);
        const evalScore = search(nextState, alpha, beta);
        minEval = Math.min(minEval, evalScore);
        beta = Math.min(beta, evalScore);
        if (beta <= alpha) break;
      }
      return minEval;
    }
  }

  const team0Points = search(state, -Infinity, Infinity);
  return { team0Points, nodesExplored };
}

const TRUMP_NAMES = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes'];

function main() {
  const args = process.argv.slice(2);

  if (args.length !== 2) {
    console.error('Usage: npx tsx scripts/minimax-eval.ts <seed> <decl_id>');
    process.exit(1);
  }

  const seed = parseInt(args[0]!, 10);
  const declId = parseInt(args[1]!, 10);

  if (isNaN(seed) || isNaN(declId) || declId < 0 || declId > 6) {
    console.error('Invalid arguments: seed must be integer, decl_id must be 0-6');
    process.exit(1);
  }

  // Deal hands
  const hands = dealDominoesWithSeed(seed);

  // Create trump selection for pip trump
  const trump: TrumpSelection = {
    type: 'suit',
    suit: declId as 0 | 1 | 2 | 3 | 4 | 5 | 6,
  };

  // Build state with hands set and in playing phase
  const state = StateBuilder.inPlayingPhase(trump)
    .withHands(hands)
    .withConfig({ playerTypes: ['ai', 'ai', 'ai', 'ai'], layers: ['base', 'speed'] })
    .build();

  // Create simulation context (AI players, no consensus layer)
  const ctx = createExecutionContext({
    playerTypes: ['ai', 'ai', 'ai', 'ai'],
    layers: ['base', 'speed'],
  });

  // Run minimax evaluation (full game, no early termination)
  const result = minimaxFullGame(state, ctx);

  // Output result
  // Python expects value = 2 * team0Points - 42
  const team1Points = 42 - result.team0Points;
  const pythonValue = 2 * result.team0Points - 42;

  console.log(JSON.stringify({
    seed,
    declId,
    trump: TRUMP_NAMES[declId],
    team0Points: result.team0Points,
    team1Points,
    nodesExplored: result.nodesExplored,
    pythonValue,
  }));
}

main();
