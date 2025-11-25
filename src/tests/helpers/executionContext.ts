/**
 * Test helper for creating ExecutionContext instances.
 *
 * Provides convenient factory functions for tests that need ExecutionContext
 * without duplicating setup code.
 *
 * ## When to Use These Helpers
 *
 * **UNIT TESTS ONLY** - Use these helpers when testing composition behavior:
 * - Testing Layer composition (e.g., base + nello + plunge)
 * - Testing individual rule methods in isolation
 *
 * **DO NOT USE** for integration tests:
 * - For full game flows, use HeadlessRoom instead
 * - For multi-action scenarios, use HeadlessRoom instead
 * - For state transition verification, use HeadlessRoom instead
 *
 * See docs/TESTING_PATTERNS.md for detailed guidance.
 */

import { createExecutionContext, type ExecutionContext } from '../../game/types/execution';
import type { GameConfig } from '../../game/types/config';

/**
 * Create a default ExecutionContext for testing (standard 42, no special rules).
 *
 * @internal
 * @testOnly - Only use in unit tests that need to test composition behavior directly.
 * For integration tests, prefer HeadlessRoom.
 *
 * @example
 * ```ts
 * // GOOD: Testing Layer composition
 * const ctx = createTestContext({ layers: ['base', 'nello'] });
 * const rules = ctx.rules;
 * expect(rules.getTrumpSelector(state, bid)).toBe(expectedPlayer);
 * ```
 *
 * @example
 * ```ts
 * // BAD: Testing full game flow (use HeadlessRoom instead)
 * const ctx = createTestContext();
 * let state = createInitialState();
 * state = executeAction(state, action1, ctx.rules);
 * state = executeAction(state, action2, ctx.rules);
 * ```
 */
export function createTestContext(config?: Partial<GameConfig>): ExecutionContext {
  return createExecutionContext({
    playerTypes: ['human', 'human', 'human', 'human'],
    ...config
  });
}

/**
 * Create ExecutionContext with specific layers enabled.
 *
 * @internal
 * @testOnly - Only use in unit tests that need to test composition behavior directly.
 * For integration tests, prefer HeadlessRoom.
 *
 * @example
 * ```ts
 * // GOOD: Testing specific layer combinations
 * const ctx = createTestContextWithLayers(['base', 'nello', 'plunge']);
 * const rules = ctx.rules;
 * // Test composition behavior...
 * ```
 */
export function createTestContextWithLayers(layerNames: string[]): ExecutionContext {
  return createExecutionContext({
    playerTypes: ['human', 'human', 'human', 'human'],
    layers: layerNames
  });
}

/**
 * Create ExecutionContext with AI players.
 *
 * @internal
 * @testOnly - Only use in unit tests that need to test composition behavior directly.
 * For integration tests, prefer HeadlessRoom.
 *
 * @example
 * ```ts
 * // GOOD: Testing AI-specific rule behavior
 * const ctx = createAITestContext();
 * const rules = ctx.rules;
 * // Test AI-specific composition...
 * ```
 */
export function createAITestContext(config?: Partial<GameConfig>): ExecutionContext {
  return createExecutionContext({
    playerTypes: ['ai', 'ai', 'ai', 'ai'],
    ...config
  });
}
