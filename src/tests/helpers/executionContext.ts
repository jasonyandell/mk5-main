/**
 * Test helper for creating ExecutionContext instances.
 *
 * Provides convenient factory functions for tests that need ExecutionContext
 * without duplicating setup code.
 */

import { createExecutionContext, type ExecutionContext } from '../../game/types/execution';
import type { GameConfig } from '../../game/types/config';

/**
 * Create a default ExecutionContext for testing (standard 42, no special rules).
 */
export function createTestContext(config?: Partial<GameConfig>): ExecutionContext {
  return createExecutionContext({
    playerTypes: ['human', 'human', 'human', 'human'],
    ...config
  });
}

/**
 * Create ExecutionContext with specific rulesets enabled.
 */
export function createTestContextWithRuleSets(ruleSetNames: string[]): ExecutionContext {
  return createExecutionContext({
    playerTypes: ['human', 'human', 'human', 'human'],
    enabledRuleSets: ruleSetNames
  });
}

/**
 * Create ExecutionContext with AI players.
 */
export function createAITestContext(config?: Partial<GameConfig>): ExecutionContext {
  return createExecutionContext({
    playerTypes: ['ai', 'ai', 'ai', 'ai'],
    ...config
  });
}
