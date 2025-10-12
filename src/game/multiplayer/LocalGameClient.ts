import type { GameClient } from './GameClient';
import type { GameState, GameAction } from '../types';
import type { MultiplayerGameState, PlayerSession, Result } from './types';
import { ok, err } from './types';
import { authorizeAndExecute, getValidActionsForPlayer } from './authorization';
import { selectAIAction } from '../core/ai-scheduler';
import { getNextStates } from '../core/gameEngine';

/**
 * LocalGameClient - In-memory implementation of GameClient.
 *
 * Runs in the same process as the UI. Perfect for:
 * - Single-player with AI
 * - Local multiplayer (hot-seat)
 * - Offline mode
 * - Testing
 *
 * AI is managed in-process via existing ai-scheduler.ts functions.
 */
export class LocalGameClient implements GameClient {
  private mpState: MultiplayerGameState;
  private listeners: Set<(state: MultiplayerGameState) => void> = new Set();
  private aiScheduler: NodeJS.Timeout | null = null;

  constructor(initialState: GameState, playerTypes: ('human' | 'ai')[] = ['human', 'ai', 'ai', 'ai']) {
    // Create initial sessions
    const sessions: PlayerSession[] = playerTypes.map((type, i) => ({
      playerId: i,
      sessionId: `local-${i}`,
      type
    }));

    this.mpState = {
      state: initialState,
      sessions
    };

    // Start AI scheduler
    this.startAIScheduler();
  }

  getState(): MultiplayerGameState {
    return this.mpState;
  }

  async requestAction(playerId: number, action: GameAction): Promise<Result<void>> {
    const request = {
      playerId,
      action,
      sessionId: `local-${playerId}`
    };

    const result = authorizeAndExecute(this.mpState, request);

    if (!result.ok) {
      return err(result.error);
    }

    // Update state and notify listeners
    this.mpState = result.value;
    this.notifyListeners();

    return ok(undefined);
  }

  subscribe(listener: (state: MultiplayerGameState) => void): () => void {
    // Call immediately with current state
    listener(this.mpState);

    // Add to listeners
    this.listeners.add(listener);

    // Return unsubscribe function
    return () => {
      this.listeners.delete(listener);
    };
  }

  async setPlayerControl(playerId: number, type: 'human' | 'ai'): Promise<void> {
    // Update session
    const sessions = this.mpState.sessions.map(session =>
      session.playerId === playerId
        ? { ...session, type }
        : session
    );

    // Update state
    const newState = {
      ...this.mpState.state,
      playerTypes: sessions.map(s => s.type) as ('human' | 'ai')[]
    };

    this.mpState = {
      state: newState,
      sessions
    };

    this.notifyListeners();
  }

  destroy(): void {
    // Stop AI scheduler
    if (this.aiScheduler) {
      clearInterval(this.aiScheduler);
      this.aiScheduler = null;
    }

    // Clear listeners
    this.listeners.clear();
  }

  /**
   * Private: Notify all subscribers of state change
   */
  private notifyListeners(): void {
    for (const listener of this.listeners) {
      listener(this.mpState);
    }
  }

  /**
   * Private: AI scheduler runs periodically to check if AI should act.
   * This replaces the old game loop from gameStore.ts.
   */
  private startAIScheduler(): void {
    // Run every 200ms (5 times per second)
    this.aiScheduler = setInterval(() => {
      this.tickAI();
    }, 200);
  }

  /**
   * Private: Check if AI should act and execute action if so.
   */
  private tickAI(): void {
    const { state, sessions } = this.mpState;

    // Don't process if game is over
    if (state.phase === 'game_end') {
      return;
    }

    // Check if current player is AI
    const currentSession = sessions[state.currentPlayer];
    if (!currentSession || currentSession.type !== 'ai') {
      return;
    }

    // Get valid actions for AI
    const validActions = getValidActionsForPlayer(state, state.currentPlayer);

    if (validActions.length === 0) {
      return;
    }

    // Get transitions (for selectAIAction compatibility)
    const allTransitions = getNextStates(state);
    const aiTransitions = allTransitions.filter(t => {
      const action = t.action;
      if (!('player' in action)) return true;
      return action.player === state.currentPlayer;
    });

    // Ask AI to choose action
    const choice = selectAIAction(state, state.currentPlayer, aiTransitions);

    if (!choice) {
      return;
    }

    // Execute the AI's chosen action
    this.requestAction(state.currentPlayer, choice.action).catch(error => {
      console.error('AI action failed:', error);
    });
  }
}
