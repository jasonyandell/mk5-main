/**
 * AIClient - Protocol-speaking AI player.
 *
 * Unlike the old system where AI was baked into LocalGameClient,
 * this AI client speaks the same protocol as human clients.
 *
 * Design principles:
 * - No special privileges - uses public protocol
 * - Subscribes to STATE_UPDATE like any client
 * - Sends EXECUTE_ACTION like any client
 * - Can be killed/spawned dynamically
 */

import type {
  ServerMessage,
  GameView,
  ValidAction
} from '../../shared/multiplayer/protocol';
import type { IGameAdapter } from '../../server/adapters/IGameAdapter';
import { selectAIAction } from '../core/ai-scheduler';
import { getNextStates } from '../core/gameEngine';

/**
 * AI difficulty levels
 */
export type AIDifficulty = 'beginner' | 'intermediate' | 'expert';

/**
 * AIClient - Autonomous AI player
 */
export class AIClient {
  private gameId: string;
  private playerIndex: number;
  private playerId: string;
  private adapter: IGameAdapter;
  private difficulty: AIDifficulty;
  private unsubscribe: (() => void) | undefined;
  private thinkingTimer: ReturnType<typeof setTimeout> | undefined;
  private destroyed = false;

  // AI timing configuration
  private readonly timing = {
    beginner: { min: 800, max: 2000 },
    intermediate: { min: 500, max: 1500 },
    expert: { min: 200, max: 800 }
  };

  constructor(
    gameId: string,
    playerIndex: number,
    adapter: IGameAdapter,
    playerId: string,
    difficulty: AIDifficulty = 'beginner'
  ) {
    this.gameId = gameId;
    this.playerIndex = playerIndex;
    this.playerId = playerId;
    this.adapter = adapter;
    this.difficulty = difficulty;
  }

  /**
   * Start the AI client
   */
  start(): void {
    if (this.destroyed) return;

    // Subscribe to game updates
    this.unsubscribe = this.adapter.subscribe((message) => {
      this.handleServerMessage(message);
    });

    // Send subscribe message
    this.adapter.send({
      type: 'SUBSCRIBE',
      gameId: this.gameId,
      clientId: this.playerId
    });
  }

  /**
   * Stop and clean up the AI client
   */
  destroy(): void {
    if (this.destroyed) return;

    this.destroyed = true;

    // Clear any pending timers
    if (this.thinkingTimer) {
      clearTimeout(this.thinkingTimer);
      this.thinkingTimer = undefined;
    }

    // Unsubscribe from messages
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = undefined;
    }

    // Send unsubscribe message
    if (this.adapter.isConnected()) {
      this.adapter.send({
        type: 'UNSUBSCRIBE',
        gameId: this.gameId,
        clientId: this.playerId
      });
    }
  }

  /**
   * Private: Handle server messages
   */
  private handleServerMessage(message: ServerMessage): void {
    if (this.destroyed) return;

    // Only care about our game
    if ('gameId' in message && message.gameId !== this.gameId) {
      return;
    }

    switch (message.type) {
      case 'STATE_UPDATE':
        this.handleStateUpdate(message.view);
        break;

      case 'GAME_CREATED':
        if (message.gameId === this.gameId) {
          this.handleStateUpdate(message.view);
        }
        break;

      // Ignore other message types
      default:
        break;
    }
  }

  /**
   * Private: Handle state update
   */
  private handleStateUpdate(view: GameView): void {
    if (this.destroyed) return;

    const { state, validActions } = view;

    // Check if it's our turn
    if (state.currentPlayer !== this.playerIndex) {
      return;
    }

    // Filter to only our actions
    const myActions = validActions.filter(va => {
      const action = va.action;
      // Neutral actions (no player field) are available to everyone
      if (!('player' in action)) {
        return true;
      }
      // Actions with player field are only for that player
      return action.player === this.playerIndex;
    });

    if (myActions.length === 0) {
      return;
    }

    // Immediately execute consensus actions
    const consensusAction = myActions.find(va =>
      va.action.type === 'agree-score-hand' ||
      va.action.type === 'agree-complete-trick'
    );

    if (consensusAction) {
      this.executeAction(consensusAction);
      return;
    }

    // For other actions, think first
    this.thinkAndAct(view, myActions);
  }

  /**
   * Private: Think and execute action
   */
  private thinkAndAct(view: GameView, validActions: ValidAction[]): void {
    if (this.destroyed) return;

    // Clear any existing timer
    if (this.thinkingTimer) {
      clearTimeout(this.thinkingTimer);
    }

    // Calculate thinking time
    const timings = this.timing[this.difficulty];
    const thinkTime = timings.min + Math.random() * (timings.max - timings.min);

    // Set timer to execute action
    this.thinkingTimer = setTimeout(() => {
      if (this.destroyed) return;

      this.selectAndExecuteAction(view, validActions);
    }, thinkTime);
  }

  /**
   * Private: Select best action and execute
   */
  private selectAndExecuteAction(view: GameView, validActions: ValidAction[]): void {
    if (this.destroyed) return;

    const { state } = view;

    // Convert ValidActions to StateTransitions for AI selector
    // (This is a compatibility layer - future AI could work directly with ValidActions)
    const allTransitions = getNextStates(state);
    const myTransitions = allTransitions.filter(t => {
      const action = t.action;
      // Find matching valid action
      return validActions.some(va => {
        if (va.action.type !== action.type) return false;

        // Match player if present
        if ('player' in va.action && 'player' in action) {
          if (va.action.player !== action.player) return false;
        }

        // Match other fields based on type
        switch (action.type) {
          case 'bid':
            return 'bid' in va.action &&
                   va.action.bid === action.bid &&
                   va.action.value === action.value;
          case 'select-trump':
            return 'trump' in va.action &&
                   JSON.stringify(va.action.trump) === JSON.stringify(action.trump);
          case 'play':
            return 'dominoId' in va.action && 'dominoId' in action &&
                   va.action.dominoId === action.dominoId;
          default:
            return true;
        }
      });
    });

    // Let AI strategy select best action
    const choice = selectAIAction(state, this.playerIndex, myTransitions);

    if (!choice) {
      // Fallback: pick first valid action
      const firstAction = validActions[0];
      if (firstAction) {
        this.executeAction(firstAction);
      }
      return;
    }

    // Find matching ValidAction for the choice
    const validAction = validActions.find(va => {
      const action = choice.action;
      if (va.action.type !== action.type) return false;

      // Match fields based on type
      switch (action.type) {
        case 'bid':
          return 'bid' in va.action &&
                 va.action.bid === action.bid &&
                 va.action.value === action.value;
        case 'select-trump':
          return 'trump' in va.action &&
                 JSON.stringify(va.action.trump) === JSON.stringify(action.trump);
        case 'play':
          return 'dominoId' in va.action && 'dominoId' in action &&
                 va.action.dominoId === action.dominoId;
        default:
          return true;
      }
    });

    if (validAction) {
      this.executeAction(validAction);
    } else {
      // Fallback: pick first action
      const firstAction = validActions[0];
      if (firstAction) {
        this.executeAction(firstAction);
      }
    }
  }

  /**
   * Private: Execute an action
   */
  private executeAction(validAction: ValidAction): void {
    if (this.destroyed) return;

    // Send EXECUTE_ACTION message
    this.adapter.send({
      type: 'EXECUTE_ACTION',
      gameId: this.gameId,
      playerId: this.playerId,
      action: validAction.action
    }).catch(error => {
      console.error(`AI ${this.playerId} action failed:`, error);
    });
  }

  /**
   * Get AI client info
   */
  getInfo() {
    return {
      gameId: this.gameId,
      playerId: this.playerId,
      playerIndex: this.playerIndex,
      difficulty: this.difficulty,
      active: !this.destroyed
    };
  }
}