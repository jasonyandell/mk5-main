/**
 * Core game configuration types shared across engine, server, and protocol.
 * Lives under src/game to avoid upward dependencies from core modules.
 */

/**
 * Serializable variant configuration
 */
export interface VariantConfig {
  type: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  config?: Record<string, any>;
}

/**
 * Legacy single-variant descriptor (kept for compatibility with older configs)
 */
export interface GameVariant {
  type: 'standard' | 'one-hand' | 'tournament';

  config?: {
    // One-hand mode options
    targetHand?: number;
    maxAttempts?: number;
    originalSeed?: number;

    // Tournament mode options
    bestOf?: number;

    // Future variants can extend this bag of parameters
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

  /** Single variant (deprecated) */
  variant?: GameVariant;

  /** Composable variants */
  variants?: VariantConfig[];

  /** Enabled rule layers (e.g., ['nello', 'plunge']) */
  enabledLayers?: string[];

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
