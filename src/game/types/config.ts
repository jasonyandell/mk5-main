/**
 * Core game configuration types shared across engine, server, and protocol.
 * Lives under src/game to avoid upward dependencies from core modules.
 */

/**
 * Serializable action transformer configuration
 */
export interface ActionTransformerConfig {
  type: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config?: Record<string, any>;
}

/**
 * Legacy single action transformer descriptor (kept for compatibility with older configs)
 */
export interface GameActionTransformer {
  type: 'standard' | 'one-hand' | 'tournament';

  config?: {
    // One-hand mode options
    targetHand?: number;
    maxAttempts?: number;
    originalSeed?: number;

    // Tournament mode options
    bestOf?: number;

    // Future action transformers can extend this bag of parameters
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: any;
  };
}

/**
 * Game configuration for creating new games
 */
export interface GameConfig {
  /** Player control types */
  playerTypes: ('human' | 'ai')[];

  /** Single action transformer (deprecated) */
  variant?: GameActionTransformer;

  /** Composable action transformers */
  variants?: ActionTransformerConfig[];

  /** Enabled rule sets (e.g., ['nello', 'plunge']) */
  enabledRuleSets?: string[];

  /** Random seed for deterministic games */
  shuffleSeed?: number;

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
