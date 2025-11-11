/**
 * URL Replay System - Deterministic game replay from encoded URLs.
 *
 * Replays games from URLs containing:
 * - Game configuration (rulesets, transformers, player types)
 * - Shuffle seed
 * - Action history (compressed IDs)
 *
 * Uses HeadlessRoom for execution (single composition point) and
 * action-resolution for translating compact IDs to GameActions.
 */

import { decodeGameUrl } from '../core/url-compression';
import { resolveActionIds } from '../core/action-resolution';
import { HeadlessRoom } from '../../server/HeadlessRoom';
import type { GameState } from '../types';
import type { GameConfig } from '../types/config';
import type { GameAction } from '../types';

export interface ReplayOptions {
  /** Stop replay at specific action index */
  stopAt?: number;
  /** Enable verbose logging */
  verbose?: boolean;
  /** Show actions in range [start, end] */
  actionRangeStart?: number;
  actionRangeEnd?: number;
  /** Show trick winners and points */
  showTricks?: boolean;
  /** Focus on specific hand number */
  focusHand?: number;
  /** Compact one-line output format */
  compact?: boolean;
}

export interface ReplayResult {
  /** Final game state */
  state: GameState;
  /** Game configuration used */
  config: GameConfig;
  /** Resolved actions executed */
  actions: GameAction[];
  /** Number of actions executed */
  actionCount: number;
  /** Errors encountered during replay */
  errors?: string[];
}

/**
 * Replay game actions from a URL containing encoded state.
 *
 * Decodes URL to extract config + action IDs, resolves IDs to GameActions,
 * then replays through HeadlessRoom.
 *
 * @param url - URL or query string with encoded game state
 * @param options - Replay options for debugging/logging
 * @returns Replay result with final state, config, and actions
 *
 * @example
 * ```typescript
 * const result = replayFromUrl('?s=abc&at=t&rs=n&a=CAAS');
 * console.log(result.state.teamScores); // [2, 0]
 * console.log(result.config.enabledRuleSets); // ['nello']
 * ```
 */
export function replayFromUrl(url: string, options: ReplayOptions = {}): ReplayResult {
  // Extract query string from URL
  const queryString = url.includes('?') ? url.split('?')[1] : url;
  const decoded = decodeGameUrl(queryString || '');

  // Build GameConfig from decoded URL data
  const config: GameConfig = {
    playerTypes: decoded.playerTypes,
    shuffleSeed: decoded.seed
  };

  // Only set optional properties if they have values
  if (decoded.theme !== 'business') {
    config.theme = decoded.theme;
  }

  if (Object.keys(decoded.colorOverrides).length > 0) {
    config.colorOverrides = decoded.colorOverrides;
  }

  if (decoded.actionTransformers) {
    config.actionTransformers = decoded.actionTransformers;
  }

  if (decoded.enabledRuleSets) {
    config.enabledRuleSets = decoded.enabledRuleSets;
  }

  return replayActions(decoded.seed, decoded.actions, config, options);
}

/**
 * Replay a sequence of action IDs with a given configuration.
 *
 * This is the core replay function that:
 * 1. Resolves action IDs to GameActions using the config
 * 2. Creates HeadlessRoom with full config (composition point)
 * 3. Replays actions with optional logging/filtering
 *
 * @param seed - Shuffle seed for deterministic replay
 * @param actionIds - Array of action IDs (e.g., ["C", "A", "A", "A"])
 * @param config - Game configuration (rulesets, transformers, etc.)
 * @param options - Replay options for debugging/logging
 * @returns Replay result with final state and metadata
 */
export function replayActions(
  seed: number,
  actionIds: string[],
  config: GameConfig,
  options: ReplayOptions = {}
): ReplayResult {
  const errors: string[] = [];

  // Handle focus hand by finding action range
  let startIdx = 0;
  let endIdx = actionIds.length;

  if (options.focusHand) {
    let currentHand = 1;
    let foundStart = false;

    for (let i = 0; i < actionIds.length; i++) {
      const actionId = actionIds[i];
      if (actionId === 'score-hand') {
        currentHand++;
        if (currentHand === options.focusHand) {
          startIdx = i + 1;
          foundStart = true;
        } else if (foundStart && currentHand > options.focusHand) {
          endIdx = i;
          break;
        }
      }
    }
  }

  // Apply stopAt limit
  if (options.stopAt !== undefined) {
    endIdx = Math.min(endIdx, options.stopAt);
  }

  // Slice action IDs to replay range
  const idsToReplay = actionIds.slice(startIdx, endIdx);

  // Resolve action IDs to GameActions using the config
  let resolvedActions: GameAction[];
  try {
    resolvedActions = resolveActionIds(idsToReplay, config, seed);
  } catch (error) {
    errors.push(`Action resolution failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    // Return early with empty state
    return {
      state: new HeadlessRoom(config, seed).getState(),
      config,
      actions: [],
      actionCount: 0,
      errors
    };
  }

  // Create HeadlessRoom with full config
  const room = new HeadlessRoom(config, seed);

  // Replay actions with optional logging
  let actionCount = 0;

  for (let i = 0; i < resolvedActions.length; i++) {
    const action = resolvedActions[i];
    const globalIndex = startIdx + i; // Original index in full action list
    const actionId = idsToReplay[i];

    if (!action || !actionId) continue;

    const prevState = room.getState();
    const prevPhase = prevState.phase;
    const prevScores = [...prevState.teamScores];

    // Execute action
    try {
      room.replayActions([action]);
      actionCount++;
    } catch (error) {
      const errorMsg = `Action ${globalIndex}: Execution failed for "${actionId}": ${error instanceof Error ? error.message : 'Unknown error'}`;
      errors.push(errorMsg);

      if (shouldShowAction(globalIndex, options)) {
        console.error(errorMsg);
      }
      break;
    }

    const newState = room.getState();

    // Logging based on options
    if (shouldShowAction(globalIndex, options)) {
      if (options.compact) {
        // Compact one-line format
        const scoreChange = (prevScores[0] !== newState.teamScores[0] || prevScores[1] !== newState.teamScores[1])
          ? ` [${prevScores}] → [${newState.teamScores}]`
          : '';
        const phaseChange = prevPhase !== newState.phase ? ` ${newState.phase.toUpperCase()}` : '';
        console.log(`${globalIndex}: ${actionId}${scoreChange}${phaseChange}`);
      } else if (!options.showTricks) {
        console.log(`Action ${globalIndex}: ${actionId} (Phase: ${prevPhase} → ${newState.phase})`);
      }
    }

    // Show tricks if requested
    if (options.showTricks && actionId === 'complete-trick') {
      const trickNum = newState.tricks.length;
      const trick = newState.tricks[trickNum - 1];
      if (trick) {
        const winner = trick.winner !== undefined ? newState.players[trick.winner] : undefined;
        if (winner) {
          console.log(`Trick ${trickNum}: Won by P${trick.winner} (team ${winner.teamId}) → ${trick.points + 1} points`);
        }
      }
    }
  }

  const result: ReplayResult = {
    state: room.getState(),
    config,
    actions: resolvedActions,
    actionCount
  };

  if (errors.length > 0) {
    result.errors = errors;
  }

  return result;
}

/**
 * Replay actions up to a specific action index.
 * Convenience wrapper for replayActions with stopAt option.
 *
 * @param seed - Shuffle seed
 * @param actionIds - Array of action IDs
 * @param config - Game configuration
 * @param stopAt - Action index to stop at
 * @returns Final game state at stopAt index
 */
export function replayToAction(
  seed: number,
  actionIds: string[],
  config: GameConfig,
  stopAt: number
): GameState {
  const result = replayActions(seed, actionIds, config, { stopAt });
  return result.state;
}

/**
 * Helper to determine if an action should be logged based on options.
 */
function shouldShowAction(index: number, options: ReplayOptions): boolean {
  if (options.verbose) return true;

  if (options.actionRangeStart !== undefined) {
    const end = options.actionRangeEnd ?? index;
    return index >= options.actionRangeStart && index <= end;
  }

  return false;
}
