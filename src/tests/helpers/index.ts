/**
 * Test helpers index - exports all test utilities
 *
 * Use StateBuilder and HandBuilder for state/hand creation in tests.
 * Use TestLayer for isolated layer unit testing.
 * Use consensus helpers for integration tests that simulate full game flows.
 */

// StateBuilder and helper builders (PRIMARY API)
export { StateBuilder, DominoBuilder, HandBuilder } from './stateBuilder';

// TestLayer for isolated layer testing
export {
  createTestLayer,
  createSentinelLayer,
  PASSTHROUGH_SENTINEL,
  PLAYER_SENTINEL,
  type TestLayerOverrides
} from './TestLayer';

// Consensus helpers for integration tests
export {
  processSequentialConsensus,
  processCompleteTrick,
  processHandScoring
} from './consensusHelpers';

// Execution context for tests
export { createTestContext } from './executionContext';
