/**
 * Core game configuration types shared across engine, server, and protocol.
 * Lives under src/game to avoid upward dependencies from core modules.
 */

import type { Domino } from '../types';

/**
 * Serializable action transformer configuration
 */
export interface ActionTransformerConfig {
  type: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config?: Record<string, any>;
}

/**
 * Override default dealing behavior.
 *
 * Use cases:
 * - Test scenarios with known hands
 * - Teaching scenarios (share URLs with specific deals)
 * - Bug reproduction (exact game state replay)
 * - Challenge scenarios (can you win with this hand?)
 */
export interface DealOverrides {
  /**
   * Override initial hand distribution with specific hands.
   *
   * When specified:
   * - Overrides seed-based shuffling
   * - Must be exactly 4 arrays of 7 dominoes (28 unique total)
   * - Deterministic and replayable
   * - Serialized to URL state for sharing
   *
   * Priority: dealOverrides.initialHands > shuffleSeed > random
   *
   * Validation: Throws InvalidDealOverrideError if duplicates, wrong count, or invalid dominoes.
   */
  initialHands?: Domino[][];

  // Future extensibility:
  // aiSeed?: number;        // Separate AI randomness from deal randomness
  // delays?: number[];      // Artificial delays for animations
}

/**
 * Error thrown when dealOverrides.initialHands is invalid.
 */
export class InvalidDealOverrideError extends Error {
  details?: unknown;

  constructor(reason: string, details?: unknown) {
    super(`Invalid dealOverrides.initialHands: ${reason}`);
    this.name = 'InvalidDealOverrideError';
    this.details = details;
  }
}

/**
 * Game configuration for creating new games
 */
export interface GameConfig {
  /** Player control types */
  playerTypes: ('human' | 'ai')[];

  /** Composable action transformers (left-to-right pipeline) */
  actionTransformers?: ActionTransformerConfig[];

  /** Enabled rule sets (e.g., ['nello', 'plunge']) */
  enabledRuleSets?: string[];

  /** Random seed for deterministic games */
  shuffleSeed?: number;

  /**
   * Override default dealing behavior.
   *
   * Valid for both testing AND production (URL sharing, teaching, challenges).
   *
   * Priority: dealOverrides.initialHands > shuffleSeed > random
   */
  dealOverrides?: DealOverrides;

  /** Theme configuration */
  theme?: string;

  /** Color overrides for theme */
  colorOverrides?: Record<string, string>;

  /** AI difficulty levels (optional) */
  aiDifficulty?: ('beginner' | 'intermediate' | 'expert')[];

  /** Time limits (optional) */
  timeLimits?: {
    perAction?: number; // ms
    perHand?: number; // ms
  };
}

export type { GameConfig as DefaultGameConfig };
