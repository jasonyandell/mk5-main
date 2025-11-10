/**
 * High-level test helper for e2e tests.
 *
 * This helper provides a clean DSL for testing game UI without:
 * - Direct window object access
 * - page.evaluate() calls for game state
 * - Coupling to game implementation details
 *
 * Instead, it:
 * - Uses connections (Mock or Spy) to control game state
 * - Tests only UI behavior and DOM interactions
 * - Provides high-level assertions
 *
 * @example
 * ```typescript
 * const helper = await TestGameHelper.createWithMockState(page, biddingPhaseView);
 * await helper.assertPhase('bidding');
 * await helper.clickAction('Bid 30');
 * await helper.assertPhase('bidding'); // Still bidding, waiting for others
 * ```
 */

import type { Page } from '@playwright/test';
import type { GameView } from '../../../shared/multiplayer/protocol';
import type { Connection } from '../../../server/transports/Transport';
import { MockConnection } from '../../adapters/MockAdapter';
import { SpyConnection } from '../../adapters/SpyAdapter';

export class TestGameHelper {
  private constructor(
    private page: Page,
    private connection: Connection
  ) {}

  /**
   * Create helper with mock connection (for fast UI-only tests).
   *
   * @param page - Playwright page
   * @param states - Pre-configured state sequence
   * @param options - Mock connection options
   */
  static async createWithMockState(
    page: Page,
    states: GameView | GameView[],
    options?: {
      autoAdvance?: boolean;
      simulateLatency?: number;
    }
  ): Promise<TestGameHelper> {
    const stateArray = Array.isArray(states) ? states : [states];
    const connection = new MockConnection(stateArray, options);
    const helper = new TestGameHelper(page, connection);
    await helper.initialize();
    return helper;
  }

  /**
   * Create helper with spy connection (for protocol verification tests).
   *
   * @param page - Playwright page
   * @param wrappedConnection - Connection to wrap (usually a real connection)
   */
  static async createWithSpy(
    page: Page,
    wrappedConnection?: Connection
  ): Promise<TestGameHelper> {
    // This method is used internally by tests - provide a wrapped real connection if not provided
    if (!wrappedConnection) {
      throw new Error(
        'createWithSpy() requires a wrappedConnection. Use createWithRealGame() instead for a simple real game test.'
      );
    }
    const connection = new SpyConnection(wrappedConnection);
    const helper = new TestGameHelper(page, connection);
    await helper.initialize();
    return helper;
  }

  /**
   * Create helper with real game (for integration tests).
   *
   * @param page - Playwright page
   * @param realConnection - A real connection implementation that runs actual game logic
   */
  static async createWithRealGame(page: Page, realConnection: Connection): Promise<TestGameHelper> {
    const helper = new TestGameHelper(page, realConnection);
    await helper.initialize();
    return helper;
  }

  /**
   * Initialize the helper by exposing connection to page context.
   */
  private async initialize(): Promise<void> {
    // Expose connection to page context for client to use
    await this.page.exposeFunction('__testConnection_send', (message: unknown) => {
      this.connection.send(message as import('../../../shared/multiplayer/protocol').ClientMessage);
    });

    await this.page.exposeFunction('__testConnection_onMessage', (handlerId: string) => {
      this.connection.onMessage((message) => {
        void this.page.evaluate(
          ({ handlerId, message }) => {
            const handlers = (window as { __testConnection_handlers?: Record<string, (msg: unknown) => void> }).__testConnection_handlers;
            handlers?.[handlerId]?.(message);
          },
          { handlerId, message }
        );
      });
    });

    // Navigate to app with test flag
    await this.page.goto('/?testMode=true');
    await this.waitForGameReady();
  }

  // ============================================================================
  // Navigation & Setup
  // ============================================================================

  /**
   * Wait for game to be ready (app loaded, client initialized).
   */
  async waitForGameReady(): Promise<void> {
    await this.page.waitForSelector('[data-testid="app-container"]', { timeout: 10000 });
    await this.page.waitForTimeout(100); // Brief stabilization
  }

  /**
   * Reload page and wait for ready.
   */
  async reload(): Promise<void> {
    await this.page.reload();
    await this.waitForGameReady();
  }

  // ============================================================================
  // State Assertions
  // ============================================================================

  /**
   * Assert current game phase.
   */
  async assertPhase(expectedPhase: string): Promise<void> {
    const phase = await this.page.getAttribute('[data-testid="app-container"]', 'data-phase');
    if (phase !== expectedPhase) {
      throw new Error(`Expected phase "${expectedPhase}" but got "${phase}"`);
    }
  }

  /**
   * Get current phase from DOM.
   */
  async getCurrentPhase(): Promise<string | null> {
    return this.page.getAttribute('[data-testid="app-container"]', 'data-phase');
  }

  /**
   * Assert element is visible.
   */
  async assertVisible(selector: string): Promise<void> {
    const isVisible = await this.page.isVisible(selector);
    if (!isVisible) {
      throw new Error(`Expected element "${selector}" to be visible`);
    }
  }

  /**
   * Assert element is hidden.
   */
  async assertHidden(selector: string): Promise<void> {
    const isVisible = await this.page.isVisible(selector);
    if (isVisible) {
      throw new Error(`Expected element "${selector}" to be hidden`);
    }
  }

  /**
   * Assert element contains text.
   */
  async assertText(selector: string, expectedText: string): Promise<void> {
    const text = await this.page.textContent(selector);
    if (!text?.includes(expectedText)) {
      throw new Error(`Expected element "${selector}" to contain "${expectedText}" but got "${text}"`);
    }
  }

  // ============================================================================
  // User Actions
  // ============================================================================

  /**
   * Click action button by label.
   */
  async clickAction(label: string): Promise<void> {
    await this.page.click(`button:has-text("${label}")`);
    await this.page.waitForTimeout(50); // Brief debounce
  }

  /**
   * Click element by selector.
   */
  async click(selector: string): Promise<void> {
    await this.page.click(selector);
    await this.page.waitForTimeout(50);
  }

  /**
   * Click element by test ID.
   */
  async clickTestId(testId: string): Promise<void> {
    await this.page.click(`[data-testid="${testId}"]`);
    await this.page.waitForTimeout(50);
  }

  /**
   * Fill input field.
   */
  async fill(selector: string, value: string): Promise<void> {
    await this.page.fill(selector, value);
  }

  /**
   * Press key.
   */
  async press(key: string): Promise<void> {
    await this.page.keyboard.press(key);
  }

  // ============================================================================
  // Game-Specific Actions
  // ============================================================================

  /**
   * Click a bid button (by value or type).
   */
  async bid(value: number | 'pass' | 'nello'): Promise<void> {
    if (value === 'pass') {
      await this.clickAction('Pass');
    } else if (value === 'nello') {
      await this.clickAction('Nello');
    } else {
      await this.clickAction(`Bid ${value}`);
    }
  }

  /**
   * Select trump suit.
   */
  async selectTrump(suit: string): Promise<void> {
    await this.clickAction(`Trump: ${suit}`);
  }

  /**
   * Play a domino.
   */
  async playDomino(dominoId: string): Promise<void> {
    await this.page.click(`[data-domino-id="${dominoId}"]`);
  }

  /**
   * Click "Complete Trick" button.
   */
  async completeTrick(): Promise<void> {
    await this.clickTestId('complete-trick-button');
  }

  /**
   * Click "Score Hand" button.
   */
  async scoreHand(): Promise<void> {
    await this.clickTestId('score-hand-button');
  }

  // ============================================================================
  // Connection Control (for MockConnection)
  // ============================================================================

  /**
   * Manually advance to next state (when autoAdvance is false).
   */
  advanceState(): void {
    if (this.connection instanceof MockConnection) {
      this.connection.advanceState();
    } else {
      throw new Error('advanceState() only works with MockConnection');
    }
  }

  /**
   * Jump to specific state index.
   */
  setState(index: number): void {
    if (this.connection instanceof MockConnection) {
      this.connection.setState(index);
    } else {
      throw new Error('setState() only works with MockConnection');
    }
  }

  /**
   * Simulate server error.
   */
  simulateError(code: string, message: string): void {
    if (this.connection instanceof MockConnection) {
      this.connection.simulateError(code, message);
    } else {
      throw new Error('simulateError() only works with MockConnection');
    }
  }

  // ============================================================================
  // Protocol Verification (for SpyConnection)
  // ============================================================================

  /**
   * Get all messages sent to server.
   */
  getSentMessages(): import('../../../shared/multiplayer/protocol').ClientMessage[] {
    if (this.connection instanceof SpyConnection || this.connection instanceof MockConnection) {
      return this.connection.getSentMessages();
    }
    throw new Error('getSentMessages() only works with SpyConnection or MockConnection');
  }

  /**
   * Get all messages received from server.
   */
  getReceivedMessages(): import('../../../shared/multiplayer/protocol').ServerMessage[] {
    if (this.connection instanceof SpyConnection || this.connection instanceof MockConnection) {
      return this.connection.getReceivedMessages();
    }
    throw new Error('getReceivedMessages() only works with SpyConnection or MockConnection');
  }

  /**
   * Assert a message was sent.
   */
  assertMessageSent(type: string): void {
    const messages = this.getSentMessages();
    const found = messages.some(msg => msg.type === type);
    if (!found) {
      throw new Error(`Expected message type "${type}" to be sent but it wasn't. Sent: ${messages.map(m => m.type).join(', ')}`);
    }
  }

  /**
   * Clear message history.
   */
  clearMessageHistory(): void {
    if (this.connection instanceof SpyConnection || this.connection instanceof MockConnection) {
      this.connection.clearMessageHistory();
    }
  }

  // ============================================================================
  // Utilities
  // ============================================================================

  /**
   * Take screenshot.
   */
  async screenshot(path: string): Promise<void> {
    await this.page.screenshot({ path });
  }

  /**
   * Get page for advanced operations.
   */
  getPage(): Page {
    return this.page;
  }

  /**
   * Get connection for advanced operations.
   */
  getConnection(): Connection {
    return this.connection;
  }

  /**
   * Wait for selector.
   */
  async waitFor(selector: string, options?: { timeout?: number }): Promise<void> {
    await this.page.waitForSelector(selector, options);
  }

  /**
   * Check if element exists.
   */
  async exists(selector: string): Promise<boolean> {
    return (await this.page.$(selector)) !== null;
  }

  /**
   * Get element count.
   */
  async count(selector: string): Promise<number> {
    return this.page.locator(selector).count();
  }

  /**
   * Clean up resources.
   */
  async cleanup(): Promise<void> {
    this.connection.disconnect();
  }
}
