/**
 * Test helpers index - exports all test utilities
 *
 * Use StateBuilder and HandBuilder for state/hand creation in tests.
 * Use consensus helpers for integration tests that simulate full game flows.
 */

// StateBuilder and helper builders (PRIMARY API)
export { StateBuilder, DominoBuilder, HandBuilder } from './stateBuilder';

// Consensus helpers for integration tests
export {
  processSequentialConsensus,
  processCompleteTrick,
  processHandScoring
} from './consensusHelpers';

// Execution context for tests
export { createTestContext } from './executionContext';
