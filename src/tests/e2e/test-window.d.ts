// Type definitions for window object in e2e tests
export interface TestWindow {
  setAISpeedProfile?: (profile: string) => void;
  playFirstAction?: () => void;
  // Deprecated - for skipped section tests only
  getSectionOverlay?: () => { type?: string; config?: unknown; weWon?: boolean; [key: string]: unknown };
  gameActions?: {
    startSection?: (config: unknown) => void;
    resumeSection?: () => void;
    leaveSection?: () => void;
    [key: string]: unknown;
  };
  getGameState?: () => {
    phase: string;
    currentTrick?: unknown[];
    currentPlayer?: number;
    shuffleSeed?: number;
    actionHistory?: unknown[];
    actions?: unknown[];
    history?: unknown[];
    consensus?: {
      completeTrick?: Set<number>;
      scoreHand?: Set<number>;
    };
    [key: string]: unknown;
  };
  quickplayActions?: {
    toggle?: () => void;
    togglePlayer?: (player: number) => void;
    [key: string]: unknown;
  };
  getQuickplayState?: () => {
    enabled: boolean;
    aiPlayers: Set<number>;
  };
  getNextStates?: (state: unknown) => Array<{ id: string; action: unknown }>;
  SEED_FINDER_CONFIG?: {
    MAX_SEEDS_TO_TRY: number;
    GAMES_PER_SEED: number;
    SEARCH_TIMEOUT_MS: number;
    PROGRESS_REPORT_INTERVAL: number;
    MAX_ACTIONS_PER_GAME: number;
    TARGET_WIN_RATE_MIN: number;
    TARGET_WIN_RATE_MAX: number;
    FALLBACK_SEED: number;
  };
  seedFinderStore?: {
    confirmSeed: () => void;
    setSeedFound: (seed: number, winRate: number) => void;
    [key: string]: unknown;
  };
  [key: string]: unknown;
}