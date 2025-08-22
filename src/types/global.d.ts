import type { GameState } from '../game/types';
import type { Writable } from 'svelte/store';

declare global {
  interface Window {
    getGameState: () => GameState;
    gameActions: {
      executeAction: (action: import('../game/types').GameAction) => void;
      loadFromURL: () => void;
      startNewGame: (seed?: number) => void;
      resetToQuickplay: () => void;
    };
    quickplayActions: {
      skipToHand: (handNumber: number) => void;
      skipByActions: (actionCount: number) => void;
      playFirstDomino: () => void;
    };
    quickplayState: Writable<{
      isEnabled: boolean;
      targetHand: number;
      targetActionCount: number;
    }>;
    getQuickplayState: () => {
      isEnabled: boolean;
      targetHand: number;
      targetActionCount: number;
    };
    controllerManager?: {
      skipAIDelays: () => void;
    };
    checkSkipCalled?: () => boolean;
  }
}

export {};