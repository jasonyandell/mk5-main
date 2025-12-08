#!/usr/bin/env npx tsx
/**
 * Time one full tree traversal to assess CFR viability.
 *
 * Unlike metrics collection which samples one path per game,
 * this explores ALL possible play sequences from a single deal.
 *
 * Uses pure functions (executeAction + getValidActions) for backtracking -
 * no mutable state, just recursive traversal with immutable state objects.
 *
 * Usage:
 *   npx tsx scripts/tree-traversal-timing.ts [seed] [maxNodes]
 *
 * Examples:
 *   npx tsx scripts/tree-traversal-timing.ts              # Default seed, 10M node limit
 *   npx tsx scripts/tree-traversal-timing.ts 12345 1000000  # Custom seed, 1M limit
 */

import type { GameState } from '../src/game/types';
import type { ExecutionContext } from '../src/game/types/execution';
import { executeAction } from '../src/game/core/actions';
import { StateBuilder } from '../src/tests/helpers/stateBuilder';
import { ACES } from '../src/game/types';
import { baseLayer, composeRules, composeGetValidActions } from '../src/game/layers';
import { generateStructuralActions } from '../src/game/layers/base';

interface TraversalStats {
  nodes: number;
  leaves: number;
  maxDepth: number;
  depthDistribution: Map<number, number>;
  bailedEarly: boolean;
}

const MAX_NODES_DEFAULT = 10_000_000; // 10M node limit

/**
 * Recursively traverse all branches of the game tree.
 *
 * Pure functional approach: state is immutable, each branch gets
 * its own state copy via executeAction. No mutable Room needed.
 */
function traverseTree(
  state: GameState,
  ctx: ExecutionContext,
  depth: number,
  stats: TraversalStats,
  maxNodes: number,
  startTime: number
): void {
  // Check node limit
  if (stats.nodes >= maxNodes) {
    stats.bailedEarly = true;
    return;
  }

  // Progress reporting every 100K nodes
  if (stats.nodes > 0 && stats.nodes % 100000 === 0) {
    const elapsed = (Date.now() - startTime) / 1000;
    const rate = stats.nodes / elapsed;
    process.stdout.write(`\r  Nodes: ${stats.nodes.toLocaleString()} (${rate.toFixed(0)}/sec) depth=${depth}`);
  }

  // Track max depth
  if (depth > stats.maxDepth) {
    stats.maxDepth = depth;
  }

  // Track depth distribution
  const depthCount = stats.depthDistribution.get(depth) || 0;
  stats.depthDistribution.set(depth, depthCount + 1);

  // Check if terminal (hand complete)
  if (state.phase === 'scoring' || state.phase === 'game_end') {
    stats.leaves++;
    return;
  }

  // Get all valid actions using pure function
  const actions = ctx.getValidActions(state);

  if (actions.length === 0) {
    stats.leaves++;
    return;
  }

  stats.nodes++;

  // Handle auto-execute actions (complete-trick, score-hand)
  // These don't branch - they're deterministic game flow
  const autoAction = actions.find(a =>
    a.type === 'complete-trick' ||
    a.type === 'score-hand'
  );

  if (autoAction) {
    // Auto-execute doesn't branch - just follow it
    const newState = executeAction(state, autoAction, ctx.rules);
    traverseTree(newState, ctx, depth + 1, stats, maxNodes, startTime);
    return;
  }

  // Filter to current player's actions (play actions)
  const currentPlayer = state.currentPlayer;
  const playerActions = actions.filter(a => {
    if (a.type !== 'play') return false;
    return (a as { player: number }).player === currentPlayer;
  });

  if (playerActions.length === 0) {
    stats.leaves++;
    return;
  }

  // Traverse ALL branches - pure functional, no state mutation
  for (const action of playerActions) {
    if (stats.bailedEarly) return;

    // executeAction returns new state, original unchanged
    const newState = executeAction(state, action, ctx.rules);
    traverseTree(newState, ctx, depth + 1, stats, maxNodes, startTime);
    // No restore needed - state is immutable!
  }
}

async function main() {
  const seed = parseInt(process.argv[2] || '12345', 10);
  const maxNodes = parseInt(process.argv[3] || String(MAX_NODES_DEFAULT), 10);

  console.log('═'.repeat(70));
  console.log('FULL TREE TRAVERSAL TIMING');
  console.log('═'.repeat(70));
  console.log(`Deal seed: ${seed}`);
  console.log(`Node limit: ${maxNodes.toLocaleString()}`);
  console.log('');

  // Use StateBuilder to create a playing phase state with seeded hands
  console.log('Creating playing phase state...');
  const state = StateBuilder
    .inPlayingPhase({ type: 'suit', suit: ACES })
    .withSeed(seed)
    .withCurrentPlayer(1) // Player left of dealer leads
    .build();

  console.log(`Trump: ${state.trump.type}${state.trump.suit !== undefined ? ` (suit ${state.trump.suit})` : ''}`);
  console.log(`Bidder: Player ${state.winningBidder}`);
  console.log(`Leader: Player ${state.currentPlayer}`);
  console.log('');

  // Create ExecutionContext for pure traversal - NO consensus layer
  // Consensus adds agree-* actions which aren't decision points for CFR
  // We want pure base execution: play, complete-trick, score-hand
  const layers = [baseLayer];
  const rules = composeRules(layers);
  const base = (s: GameState) => generateStructuralActions(s, rules);
  const getValidActions = composeGetValidActions(layers, base);

  const ctx: ExecutionContext = Object.freeze({
    layers: Object.freeze(layers),
    rules,
    getValidActions
  });

  const stats: TraversalStats = {
    nodes: 0,
    leaves: 0,
    maxDepth: 0,
    depthDistribution: new Map(),
    bailedEarly: false
  };

  console.log('Starting traversal...');
  const startTime = Date.now();

  traverseTree(state, ctx, 0, stats, maxNodes, startTime);

  const elapsed = (Date.now() - startTime) / 1000;

  // Clear progress line
  process.stdout.write('\r' + ' '.repeat(80) + '\r');

  console.log('─'.repeat(70));
  console.log('RESULTS');
  console.log('─'.repeat(70));
  console.log(`Traversal time: ${elapsed.toFixed(2)} seconds`);
  console.log(`Total nodes visited: ${stats.nodes.toLocaleString()}`);
  console.log(`Leaves (terminal states): ${stats.leaves.toLocaleString()}`);
  console.log(`Max depth reached: ${stats.maxDepth}`);
  console.log(`Nodes per second: ${(stats.nodes / elapsed).toFixed(0).toLocaleString()}`);

  if (stats.bailedEarly) {
    console.log('');
    console.log(`⚠️  BAILED EARLY: Hit ${maxNodes.toLocaleString()} node limit`);
    console.log(`   Tree is larger than limit - MCCFR definitely needed`);
  }

  console.log('');
  console.log('DEPTH DISTRIBUTION (sampled):');
  const sortedDepths = [...stats.depthDistribution.entries()].sort((a, b) => a[0] - b[0]);
  for (const [depth, count] of sortedDepths.slice(0, 20)) {
    console.log(`  Depth ${depth.toString().padStart(2)}: ${count.toLocaleString().padStart(10)} nodes`);
  }
  if (sortedDepths.length > 20) {
    console.log(`  ... and ${sortedDepths.length - 20} more depth levels`);
  }

  console.log('');
  console.log('═'.repeat(70));
  console.log('VERDICT');
  console.log('═'.repeat(70));

  if (stats.bailedEarly) {
    console.log('');
    console.log('  ⚠️  MCCFR IS THE PATH');
    console.log(`     Tree exceeds ${maxNodes.toLocaleString()} nodes from just ONE deal`);
    console.log('     Full CFR enumeration is infeasible');
    console.log('');
    console.log('  Recommendation: Use Monte Carlo CFR (MCCFR) with:');
    console.log('    - Count-centric abstraction (~37K buckets)');
    console.log('    - External sampling or outcome sampling');
  } else if (stats.nodes < 100000) {
    console.log('');
    console.log('  ✓  FULL CFR VIABLE');
    console.log(`     Only ${stats.nodes.toLocaleString()} nodes per deal`);
    console.log('     Direct enumeration is tractable');
  } else if (stats.nodes < 1000000) {
    console.log('');
    console.log('  ⚡ FULL CFR POSSIBLE BUT TIGHT');
    console.log(`     ${stats.nodes.toLocaleString()} nodes per deal`);
    console.log('     May need abstraction for efficiency');
  } else {
    console.log('');
    console.log('  ⚠️  MCCFR RECOMMENDED');
    console.log(`     ${stats.nodes.toLocaleString()} nodes per deal is substantial`);
    console.log('     MCCFR with sampling will be more efficient');
  }

  console.log('');
}

main().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
