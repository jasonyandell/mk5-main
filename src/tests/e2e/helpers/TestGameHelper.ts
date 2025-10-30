/**
 * High-level test helper for e2e tests.
 *
 * This helper provides a clean DSL for testing game UI without:
 * - Direct window object access
 * - page.evaluate() calls for game state
 * - Coupling to game implementation details
 *
 * Instead, it:
 * - Uses adapters (Mock or Spy) to control game state
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
import type { IGameAdapter } from '../../../shared/multiplayer/protocol';
import { MockAdapter } from '../../adapters/MockAdapter';
import { SpyAdapter } from '../../adapters/SpyAdapter';

export class TestGameHelper {
  private constructor(
    private page: Page,
    private adapter: IGameAdapter
  ) {}

  /**
   * Create helper with mock adapter (for fast UI-only tests).
   *
   * @param page - Playwright page
   * @param states - Pre-configured state sequence
   * @param options - Mock adapter options
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
    const adapter = new MockAdapter(stateArray, options);
    const helper = new TestGameHelper(page, adapter);
    await helper.initialize();
    return helper;
  }

  /**
   * Create helper with spy adapter (for protocol verification tests).
   *
   * @param page - Playwright page
   * @param wrappedAdapter - Adapter to wrap (usually a real adapter)
   */
  static async createWithSpy(
    page: Page,
    wrappedAdapter?: IGameAdapter
  ): Promise<TestGameHelper> {
    // This method is used internally by tests - provide a wrapped real adapter if not provided
    if (!wrappedAdapter) {
      throw new Error(
        'createWithSpy() requires a wrappedAdapter. Use createWithRealGame() instead for a simple real game test.'
      );
    }
    const adapter = new SpyAdapter(wrappedAdapter);
    const helper = new TestGameHelper(page, adapter);
    await helper.initialize();
    return helper;
  }

  /**
   * Create helper with real game (for integration tests).
   *
   * @param page - Playwright page
   * @param realAdapter - A real adapter implementation that runs actual game logic
   */
  static async createWithRealGame(page: Page, realAdapter: IGameAdapter): Promise<TestGameHelper> {
    const helper = new TestGameHelper(page, realAdapter);
    await helper.initialize();
    return helper;
  }

  /**
   * Initialize the helper by exposing adapter to page context.
   */
  private async initialize(): Promise<void> {
    // Expose adapter to page context for client to use
    await this.page.exposeFunction('__testAdapter_send', async (message: unknown) => {
      await this.adapter.send(message as import('../../../shared/multiplayer/protocol').ClientMessage);
    });

    await this.page.exposeFunction('__testAdapter_subscribe', (handlerId: string) => {
      const unsubscribe = this.adapter.subscribe((message) => {
        this.page.evaluate(
          ({ handlerId, message }) => {
            const handlers = (window as { __testAdapter_handlers?: Record<string, (msg: unknown) => void> }).__testAdapter_handlers;
            handlers?.[handlerId]?.(message);
          },
          { handlerId, message }
        );
      });

      // Store unsubscribe function
      type AdapterWithUnsubscribers = typeof this.adapter & { __unsubscribers?: Record<string, () => void> };
      const adapterWithUnsubscribers = this.adapter as AdapterWithUnsubscribers;
      adapterWithUnsubscribers.__unsubscribers = adapterWithUnsubscribers.__unsubscribers || {};
      adapterWithUnsubscribers.__unsubscribers[handlerId] = unsubscribe;
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
  // Adapter Control (for MockAdapter)
  // ============================================================================

  /**
   * Manually advance to next state (when autoAdvance is false).
   */
  advanceState(): void {
    if (this.adapter instanceof MockAdapter) {
      this.adapter.advanceState();
    } else {
      throw new Error('advanceState() only works with MockAdapter');
    }
  }

  /**
   * Jump to specific state index.
   */
  setState(index: number): void {
    if (this.adapter instanceof MockAdapter) {
      this.adapter.setState(index);
    } else {
      throw new Error('setState() only works with MockAdapter');
    }
  }

  /**
   * Simulate server error.
   */
  simulateError(code: string, message: string): void {
    if (this.adapter instanceof MockAdapter) {
      this.adapter.simulateError(code, message);
    } else {
      throw new Error('simulateError() only works with MockAdapter');
    }
  }

  // ============================================================================
  // Protocol Verification (for SpyAdapter)
  // ============================================================================

  /**
   * Get all messages sent to server.
   */
  getSentMessages(): import('../../../shared/multiplayer/protocol').ClientMessage[] {
    if (this.adapter instanceof SpyAdapter || this.adapter instanceof MockAdapter) {
      return this.adapter.getSentMessages();
    }
    throw new Error('getSentMessages() only works with SpyAdapter or MockAdapter');
  }

  /**
   * Get all messages received from server.
   */
  getReceivedMessages(): import('../../../shared/multiplayer/protocol').ServerMessage[] {
    if (this.adapter instanceof SpyAdapter || this.adapter instanceof MockAdapter) {
      return this.adapter.getReceivedMessages();
    }
    throw new Error('getReceivedMessages() only works with SpyAdapter or MockAdapter');
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
    if (this.adapter instanceof SpyAdapter || this.adapter instanceof MockAdapter) {
      this.adapter.clearMessageHistory();
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
   * Get adapter for advanced operations.
   */
  getAdapter(): IGameAdapter {
    return this.adapter;
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
    this.adapter.destroy();
  }
}
