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
