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
import type { Connection } from '../../server/transports/Transport';
import { selectAIAction } from '../ai/actionSelector';

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
  private connection: Connection;
  private difficulty: AIDifficulty;
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
    connection: Connection,
    playerId: string,
    difficulty: AIDifficulty = 'beginner'
  ) {
    this.gameId = gameId;
    this.playerIndex = playerIndex;
    this.playerId = playerId;
    this.connection = connection;
    this.difficulty = difficulty;
  }

  /**
   * Start the AI client
   */
  start(): void {
    if (this.destroyed) return;

    // Subscribe to game updates
    this.connection.onMessage((message) => {
      this.handleServerMessage(message);
    });

    // Send subscribe message
    this.connection.send({
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

    // Send unsubscribe message before disconnecting
    this.connection.send({
      type: 'UNSUBSCRIBE',
      gameId: this.gameId,
      clientId: this.playerId
    });

    // Disconnect
    this.connection.disconnect();
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

    // Let AI strategy select best action directly from valid actions
    const choice = selectAIAction(state, this.playerIndex, validActions);

    if (choice) {
      this.executeAction(choice);
    } else {
      // Fallback: pick first valid action
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
    this.connection.send({
      type: 'EXECUTE_ACTION',
      gameId: this.gameId,
      playerId: this.playerId,
      action: validAction.action
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
