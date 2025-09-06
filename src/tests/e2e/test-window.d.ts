// Type definitions for window object in e2e tests
export interface TestWindow {
  setAISpeedProfile?: (profile: string) => void;
  playFirstAction?: () => void;
  getSectionOverlay?: () => { type?: string; config?: unknown };
  gameActions?: {
    startSection?: (config: unknown) => void;
    resumeSection?: () => void;
    continueOneHand?: () => void;
    requestNewHand?: () => void;
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
  [key: string]: unknown;
}