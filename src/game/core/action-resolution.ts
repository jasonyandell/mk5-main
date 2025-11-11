/**
 * Pure action ID resolution utilities for URL replay.
 *
 * Provides translation between compact URL action IDs (e.g., "C" = "bid-30")
 * and structured GameAction objects using HeadlessRoom for composition.
 *
 * Design:
 * - Pure function API (stateless from caller perspective)
 * - Uses HeadlessRoom internally to ensure single composition point
 * - Matches action IDs against valid actions at each state
 * - Throws descriptive errors if ID not found
 */

import { HeadlessRoom } from '../../server/HeadlessRoom';
import { actionToId } from './actions';
import type { GameConfig } from '../types/config';
import type { GameAction } from '../types';

/**
 * Resolve action IDs to GameActions by replaying through HeadlessRoom.
 *
 * This function creates a temporary HeadlessRoom with the given config and seed,
 * then for each action ID:
 * 1. Gets all valid actions at current state
 * 2. Converts each to an ID using actionToId()
 * 3. Matches the requested ID
 * 4. Executes the matched action
 * 5. Continues to next ID
 *
 * @param actionIds - Array of action IDs (e.g., ["C", "A", "A", "A"])
 * @param config - Game configuration (with rulesets, transformers, etc.)
 * @param seed - Shuffle seed for deterministic replay
 * @returns Array of resolved GameActions
 * @throws Error if any ID cannot be matched to a valid action
 *
 * @example
 * ```typescript
 * const config: GameConfig = {
 *   playerTypes: ['human', 'ai', 'ai', 'ai'],
 *   enabledRuleSets: ['nello', 'plunge'],
 *   actionTransformers: [{ type: 'tournament' }]
 * };
 * const actions = resolveActionIds(['C', 'A', 'A', 'A'], config, 12345);
 * // Returns [{ type: 'bid', ... }, { type: 'pass' }, ...]
 * ```
 */
export function resolveActionIds(
  actionIds: string[],
  config: GameConfig,
  seed: number
): GameAction[] {
  // Create temporary HeadlessRoom with full config (includes composition)
  const room = new HeadlessRoom(config, seed);
  const resolvedActions: GameAction[] = [];

  for (let i = 0; i < actionIds.length; i++) {
    const targetId = actionIds[i];
    if (!targetId) {
      throw new Error(`Action ID at index ${i} is undefined`);
    }

    // Get all valid actions at current state
    const allActions = room.getAllActions();

    // Find matching action across all players
    let matchedAction: GameAction | undefined;
    let matchedPlayerIndex: number | undefined;

    for (const [playerId, validActions] of Object.entries(allActions)) {
      for (const validAction of validActions) {
        const id = actionToId(validAction.action);
        if (id === targetId) {
          matchedAction = validAction.action;
          // Extract player index from playerId (format: "player-N" or "__unfiltered__")
          // The '__unfiltered__' key comes from buildActionsMap when actions exist
          // outside the normal player-indexed structure. Default to 0 if no index found.
          matchedPlayerIndex = parseInt(playerId.split('-')[1] ?? '0', 10);
          break;
        }
      }
      if (matchedAction) break;
    }

    if (!matchedAction || matchedPlayerIndex === undefined) {
      // Build helpful error message
      const state = room.getState();
      const availableIds = Object.values(allActions)
        .flat()
        .map(va => actionToId(va.action))
        .join(', ');

      throw new Error(
        `Cannot resolve action ID "${targetId}" at index ${i}. ` +
        `Phase: ${state.phase}. ` +
        `Available IDs: [${availableIds}]. ` +
        `This may indicate a URL from a different game mode or corrupted replay.`
      );
    }

    // Execute the matched action
    room.executeAction(matchedPlayerIndex, matchedAction);

    // Store the resolved action
    resolvedActions.push(matchedAction);
  }

  return resolvedActions;
}

/**
 * Resolve a single action ID at a specific game state.
 * Useful for one-off action resolution without full replay.
 *
 * @param actionId - Single action ID to resolve
 * @param config - Game configuration
 * @param seed - Shuffle seed
 * @param previousActions - Actions to replay before resolving this ID
 * @returns Resolved GameAction
 * @throws Error if ID cannot be matched
 */
export function resolveActionId(
  actionId: string,
  config: GameConfig,
  seed: number,
  previousActions: GameAction[] = []
): GameAction {
  const room = new HeadlessRoom(config, seed);

  // Replay previous actions to reach desired state
  if (previousActions.length > 0) {
    room.replayActions(previousActions);
  }

  // Find matching action
  const allActions = room.getAllActions();

  for (const validActions of Object.values(allActions)) {
    for (const validAction of validActions) {
      const id = actionToId(validAction.action);
      if (id === actionId) {
        return validAction.action;
      }
    }
  }

  // Not found
  const availableIds = Object.values(allActions)
    .flat()
    .map(va => actionToId(va.action))
    .join(', ');

  throw new Error(
    `Cannot resolve action ID "${actionId}". ` +
    `Available IDs: [${availableIds}]`
  );
}
