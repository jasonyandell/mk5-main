/* eslint-disable @typescript-eslint/no-explicit-any */
// Reason: page.waitForFunction() and page.evaluate() run in browser context where TypeScript cannot
// verify window properties. The use of 'any' for window casts is architectural, not a shortcut.

import { expect } from '@playwright/test';
import type { Page, Locator } from '@playwright/test';
import { encodeURLData } from '../../../game/core/url-compression';
import type { URLData } from '../../../game/core/url-compression';

/**
 * Simplified, deterministic Playwright helper for Texas 42 E2E testing
 * Uses Playwright's built-in auto-waiting and deterministic patterns
 */
interface ActionInfo {
  index: number;
  type: string;
  value?: string | number | undefined;
  id: string;
}

interface GameMetrics {
  scores: [number, number];
  trump: string;
}

interface GotoOptions {
  disableUrlUpdates?: boolean;
}

interface QuickplayState {
  enabled: boolean;
  aiPlayers: Set<number>;
}

interface QuickplayActions {
  toggle(): void;
  togglePlayer(playerId: number): void;
}

interface GameWindow {
  quickplayActions?: QuickplayActions;
  getQuickplayState?: () => QuickplayState;
}

export class PlaywrightGameHelper {
  private page: Page;
  // Timeout strategy for different operation types
  static TIMEOUTS = {
    quick: 1000,     // For immediate DOM checks
    normal: 5000,    // For standard operations
    slow: 10000,     // For page loads and complex operations
    test: 10000      // Overall test timeout
  };

  // Centralized selector definitions using data attributes and roles
  static SELECTORS = {
    app: '.app-container',
    appHeader: '.app-header',
    phase: '[data-phase]',
    actionButton: '.action-button, .tap-indicator',
    domino: {
      playable: '.domino-wrapper[data-playable="true"] button[data-testid^="domino-"]',
      any: 'button[data-testid^="domino-"]',
      byId: (id: string) => `[data-testid="domino-${id}"]`
    },
    score: {
      display: '.score-display',
      us: '.score-card.us .score-value',
      them: '.score-card.them .score-value'
    },
    trump: '.info-badge.trump .info-value',
    trick: {
      table: '.trick-table',
      tappable: '.trick-table.tappable',
      spot: '.trick-spot',
      played: '.trick-spot .played-domino'
    },
    nav: (item: string) => `[data-testid="nav-${item}"]`,
    bid: (value: number, isMarks: boolean) => `[data-testid="bid-${value}${isMarks ? '-marks' : ''}"]`,
    pass: '[data-testid="pass"]',
    trumpSelect: (suit: string) => `[data-testid="trump-${suit}"]`,
    flash: '.flash-message',
    debug: {
      panel: '.debug-panel',
      button: '.debug-btn',
      closeButton: '.close-button',
      historyTab: '.tab',
      historyItem: '.history-item',
      timeTravelButton: '.time-travel-button'
    },
    playingArea: '.playing-area',
    tapIndicator: '.tap-indicator',
    tapText: '.tap-text',
    turnPlayer: '.turn-player'
  };

  constructor(page: Page) {
    this.page = page;
    // Set default timeout for all operations (5 seconds for reliability)
    this.page.setDefaultTimeout(PlaywrightGameHelper.TIMEOUTS.slow);
  }

  /**
   * Navigate to game with deterministic seed and wait for ready state
   */
  async goto(seed = 12345, options: GotoOptions = {}): Promise<void> {
    const urlData: URLData = {
      v: 1 as const,
      s: { s: seed },
      a: []
    };
    const encoded = encodeURLData(urlData);
    
    // Add testMode for deterministic testing (disables AI controllers)
    const url = `/?d=${encoded}&testMode=true`;
    
    // Navigate and wait for network idle for deterministic loading
    await this.page.goto(url, { 
      waitUntil: 'networkidle',
      timeout: PlaywrightGameHelper.TIMEOUTS.slow 
    });
    
    // Wait for game to be fully ready
    await this.waitForGameReady();
    
    // Disable animations and AI in a single evaluation
    await this.page.evaluate((opts: GotoOptions) => {
      // Disable all animations for deterministic testing
      const style = document.createElement('style');
      style.textContent = `
        *, *::before, *::after {
          animation-duration: 0s !important;
          animation-delay: 0s !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
        }
      `;
      document.head.appendChild(style);
      
      // Disable AI for all players
      const gameWindow = window as GameWindow;
      if (gameWindow.quickplayActions && gameWindow.getQuickplayState) {
        const state = gameWindow.getQuickplayState();
        if (state.enabled) {
          gameWindow.quickplayActions.toggle();
        }
        for (const playerId of state.aiPlayers) {
          gameWindow.quickplayActions.togglePlayer(playerId);
        }
      }
      
      // Disable URL updates unless explicitly enabled
      if (opts?.disableUrlUpdates !== false) {
        window.history.pushState = () => {};
        window.history.replaceState = () => {};
      }
    }, options);
  }

  /**
   * Wait for game to be ready using deterministic conditions
   */
  async waitForGameReady(): Promise<void> {
    // Wait for app container and phase to be set
    await this.page.waitForSelector(PlaywrightGameHelper.SELECTORS.app);
    
    // Wait for phase attribute to be set and game state to be stable
    await this.page.waitForFunction(
      () => {
        const container = document.querySelector('.app-container');
        const phase = container?.getAttribute('data-phase');
        const gameState = (window as any).getGameState?.();
        
        // Check that we have both a phase and a game state
        // and that the game is not in a transitional state
        return !!(phase && gameState && !gameState.isProcessing);
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.normal }
    );
  }
  
  /**
   * Wait for navigation state to be restored after browser navigation
   * Simply waits for game state to be ready after navigation
   */
  async waitForNavigationRestore(): Promise<void> {
    // Wait for game state to exist and be stable
    await this.page.waitForFunction(
      () => {
        const gameState = (window as any).getGameState?.();
        return !!gameState && !('isProcessing' in gameState && gameState.isProcessing);
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.normal }
    );
    
    // Then wait for game to be ready
    await this.waitForGameReady();
  }

  /**
   * Get current game phase directly from data attribute
   */
  async getCurrentPhase(): Promise<string> {
    const container = this.page.locator(PlaywrightGameHelper.SELECTORS.app);
    return await container.getAttribute('data-phase') || 'unknown';
  }

  /**
   * Wait for specific phase change
   */
  async waitForPhase(expectedPhase: string, timeout = PlaywrightGameHelper.TIMEOUTS.normal): Promise<void> {
    await this.page.waitForFunction(
      (phase: string) => document.querySelector('.app-container')?.getAttribute('data-phase') === phase,
      expectedPhase,
      { timeout }
    );
  }

  /**
   * Get current player from UI
   */
  async getCurrentPlayer(): Promise<string> {
    const turnPlayer = this.page.locator(PlaywrightGameHelper.SELECTORS.turnPlayer);
    await turnPlayer.waitFor({ state: 'visible', timeout: PlaywrightGameHelper.TIMEOUTS.quick });
    const text = await turnPlayer.textContent();
    return text ? text.trim() : 'P0';
  }

  /**
   * Unified action selection method
   */
  async selectAction(criteria: number | string | { type: string; value?: string | number }): Promise<void> {
    // Wait for any action to be available
    await this.page.waitForSelector(PlaywrightGameHelper.SELECTORS.actionButton);
    
    let locator;
    
    if (typeof criteria === 'number') {
      // Select by index
      const buttons = await this.page.locator(PlaywrightGameHelper.SELECTORS.actionButton).all();
      if (criteria >= 0 && criteria < buttons.length) {
        locator = buttons[criteria];
      }
    } else if (typeof criteria === 'string') {
      // Select by test id or type
      locator = this.page.locator(`[data-testid="${criteria}"]`).or(
        this.page.locator(PlaywrightGameHelper.SELECTORS.actionButton).filter({ hasText: criteria })
      );
    } else if (criteria.type) {
      // Select by action type and optional value
      const actions = await this.getAvailableActions();
      const action = actions.find(a => 
        a.type === criteria.type && 
        (criteria.value === undefined || a.value == criteria.value)
      );
      if (action) {
        return this.selectAction(action.index);
      }
    }
    
    if (!locator) {
      throw new Error(`No action found matching criteria: ${JSON.stringify(criteria)}`);
    }
    
    // Click and wait for state change
    const preClickPhase = await this.getCurrentPhase();
    await locator.click();
    
    // Wait for any state change (phase or action availability)
    try {
      await this.page.waitForFunction(
        ({ prevPhase }: { prevPhase: string }) => {
          const currentPhase = document.querySelector('.app-container')?.getAttribute('data-phase');
          const hasActions = document.querySelectorAll('.action-button, .tap-indicator').length > 0;
          return currentPhase !== prevPhase || hasActions;
        },
        { prevPhase: preClickPhase },
        { timeout: PlaywrightGameHelper.TIMEOUTS.quick }
      );
    } catch {
      // State might have already changed, continue
    }
  }

  /**
   * Get all available actions in current state
   */
  async getAvailableActions(): Promise<ActionInfo[]> {
    // First ensure there are action buttons or we're in a stable state
    await this.page.waitForFunction(
      () => {
        const buttons = document.querySelectorAll('.action-button, .tap-indicator');
        // Either we have buttons, or we're in a phase that doesn't need them
        const phase = document.querySelector('.app-container')?.getAttribute('data-phase');
        return buttons.length > 0 || phase === 'game_over' || phase === 'waiting';
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.quick }
    );
    
    const buttons = await this.page.locator(PlaywrightGameHelper.SELECTORS.actionButton).all();
    const actions = [];
    
    for (let i = 0; i < buttons.length; i++) {
      const button = buttons[i];
      if (!button) continue;
      const testId = await button.getAttribute('data-testid');
      const text = await button.textContent();
      
      // Parse action type and value from testId or text
      let type = 'unknown';
      let value = undefined;
      
      if (testId) {
        if (testId === 'pass') {
          type = 'pass';
        } else if (testId.startsWith('bid-')) {
          const parts = testId.split('-');
          type = parts[2] === 'marks' ? 'bid_marks' : 'bid_points';
          value = parseInt(parts[1] || '0');
        } else if (testId.startsWith('trump-')) {
          type = 'trump_selection';
          value = testId.substring(6);
        } else if (testId.startsWith('domino-')) {
          type = 'play_domino';
          value = testId.substring(7);
        } else {
          // Map common action IDs
          const actionMap: Record<string, string> = {
            'complete-trick': 'complete_trick',
            'score-hand': 'score_hand',
            'next-trick': 'next_trick',
            'redeal': 'redeal',
            'start-hand': 'start_hand'
          };
          const mapped = actionMap[testId as keyof typeof actionMap];
          type = mapped || testId;
        }
      } else if (text) {
        // Fallback to text-based detection
        const textLower = text.toLowerCase();
        if (textLower.includes('complete')) type = 'complete_trick';
        else if (textLower.includes('score')) type = 'score_hand';
        else if (textLower.includes('next')) type = 'next_trick';
        else if (textLower.includes('deal')) type = 'redeal';
      }
      
      actions.push({ index: i, type, value, id: testId || `action-${i}` });
    }
    
    return actions;
  }

  /**
   * Perform bid action
   */
  async bid(value: number, isMarks = false): Promise<void> {
    const selector = PlaywrightGameHelper.SELECTORS.bid(value, isMarks);
    await this.page.locator(selector).click();
  }

  /**
   * Pass bid
   */
  async pass(): Promise<void> {
    await this.page.locator(PlaywrightGameHelper.SELECTORS.pass).click();
  }

  /**
   * Set trump suit
   */
  async setTrump(suit: string): Promise<void> {
    // Normalize suit name
    const suitMap: Record<string, string> = {
      '0s': 'blanks', '1s': 'ones', '2s': 'twos', '3s': 'threes',
      '4s': 'fours', '5s': 'fives', '6s': 'sixes', 'doubles': 'doubles'
    };
    const normalizedSuit = (suitMap[suit] || suit).toLowerCase();
    
    // Ensure we're in trump selection phase
    const currentPhase = await this.getCurrentPhase();
    if (currentPhase !== 'trump_selection') {
      throw new Error(`Cannot set trump: not in trump_selection phase (current: ${currentPhase})`);
    }
    
    // Wait for trump buttons to be available
    const trumpButton = this.page.locator(`[data-testid="trump-${normalizedSuit}"]`);
    await trumpButton.waitFor({ state: 'visible', timeout: PlaywrightGameHelper.TIMEOUTS.normal });
    
    // Click via JavaScript to bypass any overlay issues
    await trumpButton.evaluate((el: Element) => (el as HTMLElement).click());
    
    // Wait for phase to change from trump_selection
    await this.page.waitForFunction(
      () => {
        const phase = document.querySelector('.app-container')?.getAttribute('data-phase');
        return phase && phase !== 'trump_selection';
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.normal }
    );
  }

  /**
   * Play any available domino
   */
  async playAnyDomino(): Promise<void> {
    // Dismiss any flash messages first
    const flash = this.page.locator(PlaywrightGameHelper.SELECTORS.flash);
    if (await flash.isVisible()) {
      await flash.click();
    }
    
    // Click first playable domino using auto-waiting
    const playable = this.page.locator(PlaywrightGameHelper.SELECTORS.domino.playable)
      .or(this.page.locator(PlaywrightGameHelper.SELECTORS.domino.any + ':not([disabled])'))
      .first();
    
    // Try to find and click a playable domino with a short timeout
    try {
      await playable.waitFor({ state: 'visible', timeout: 500 });
      await playable.click({ force: true });
    } catch {
      // If no playable dominoes visible, try any domino
      const anyDomino = this.page.locator(PlaywrightGameHelper.SELECTORS.domino.any).first();
      const count = await anyDomino.count();
      if (count > 0) {
        await anyDomino.click({ force: true });
      } else {
        throw new Error('No dominoes found to play');
      }
    }
    
    // Wait for any state changes after playing domino
    // Use waitForFunction to detect when game processes the move
    await this.page.waitForFunction(
      () => {
        const state = (window as any).getGameState?.();
        return state && !(state as any).isProcessing;
      },
      { timeout: PlaywrightGameHelper.TIMEOUTS.quick }
    );
  }

  /**
   * Play specific domino
   */
  async playDomino(dominoId: string): Promise<void> {
    await this.page.locator(PlaywrightGameHelper.SELECTORS.domino.byId(dominoId)).click();
  }

  /**
   * Get game metrics (scores, marks, trump, etc.)
   */
  async getGameMetrics(): Promise<GameMetrics> {
    const [usScore, themScore, trump] = await Promise.all([
      this.page.locator(PlaywrightGameHelper.SELECTORS.score.us).textContent(),
      this.page.locator(PlaywrightGameHelper.SELECTORS.score.them).textContent(),
      this.page.locator(PlaywrightGameHelper.SELECTORS.trump).textContent()
    ]);
    
    return {
      scores: [
        parseInt(usScore?.match(/\d+/)?.[0] || '0'),
        parseInt(themScore?.match(/\d+/)?.[0] || '0')
      ],
      trump: trump || ''
    };
  }

  /**
   * Get player's hand
   */
  async getPlayerHand(playerIndex = 0): Promise<string[]> {
    if (playerIndex !== 0) return []; // Only support current player
    
    const dominoes = await this.page.locator(PlaywrightGameHelper.SELECTORS.domino.any).all();
    const hand = [];
    
    for (const domino of dominoes) {
      const testId = await domino.getAttribute('data-testid');
      const match = testId?.match(/domino-(\d)-(\d)/);
      if (match) {
        hand.push(`${match[1]}-${match[2]}`);
      }
    }
    
    return hand;
  }

  /**
   * Get current trick
   */
  async getCurrentTrick(): Promise<string[]> {
    return await this.page.locator(PlaywrightGameHelper.SELECTORS.trick.played).allTextContents();
  }

  /**
   * Perform common game actions
   */
  async performGameAction(action: string): Promise<void> {
    const actionMap: Record<string, string | (() => Promise<void>)> = {
      'complete': 'complete-trick',
      'score': 'score-hand',
      'next': 'next-trick',
      'redeal': 'redeal',
      'new': async (): Promise<void> => { await this.page.reload(); }
    };
    
    const mapped = actionMap[action];
    if (typeof mapped === 'function') {
      await mapped();
    } else {
      await this.selectAction(mapped || action);
    }
  }

  /**
   * Complete a full trick (4 plays)
   */
  async playFullTrick(): Promise<void> {
    for (let i = 0; i < 4; i++) {
      await this.playAnyDomino();
    }
  }

  /**
   * Complete bidding round
   */
  async completeBidding(bids: Array<{ type: string; value?: number }>): Promise<void> {
    for (const bid of bids) {
      await this.page.waitForSelector(PlaywrightGameHelper.SELECTORS.actionButton);
      
      if (bid.type === 'pass') {
        await this.pass();
      } else if (bid.type === 'points') {
        await this.bid(bid.value || 30, false);
      } else if (bid.type === 'marks') {
        await this.bid(bid.value || 1, true);
      }
      
      // Wait for next player's turn
      await this.page.waitForFunction(
        () => document.querySelectorAll('.action-button, .tap-indicator').length > 0,
        null,
        { timeout: PlaywrightGameHelper.TIMEOUTS.quick }
      );
    }
  }

  /**
   * Load game state from URL
   */
  async loadStateWithActions(seed: number, actions: string[]): Promise<void> {
    const urlData: URLData = {
      v: 1 as const,
      s: { s: seed },
      a: actions.map((id: string) => ({ i: id }))
    };
    const encoded = encodeURLData(urlData);
    
    await this.page.goto(`/?d=${encoded}&testMode=true`, { 
      waitUntil: 'networkidle',
      timeout: PlaywrightGameHelper.TIMEOUTS.slow 
    });
    await this.waitForGameReady();
  }

  /**
   * Navigate tabs - DEPRECATED
   * The game automatically shows the correct view based on game state
   * This method is kept only for backward compatibility but should not be used
   */
  async navigateTo(tab: string): Promise<void> {
    // Do nothing - the game handles tab switching automatically
    console.warn(`navigateTo('${tab}') called but is deprecated - game handles tab switching automatically`);
  }

  /**
   * Debug panel operations
   */
  async openDebugPanel(): Promise<void> {
    // Click the debug button in the header
    await this.page.locator(PlaywrightGameHelper.SELECTORS.debug.button).click();
    await expect(this.page.locator(PlaywrightGameHelper.SELECTORS.debug.panel)).toBeVisible();
  }

  async closeDebugPanel(): Promise<void> {
    await this.page.locator(PlaywrightGameHelper.SELECTORS.debug.closeButton).click();
    await expect(this.page.locator(PlaywrightGameHelper.SELECTORS.debug.panel)).not.toBeVisible();
  }

  /**
   * Get current URL
   */
  async getCurrentURL(): Promise<string> {
    return this.page.url();
  }

  /**
   * Check if URL has snapshot
   */
  async hasSnapshotInURL(): Promise<boolean> {
    const url = await this.getCurrentURL();
    return url.includes('d=');
  }

  /**
   * Enable AI for other players
   */
  async enableAIForOtherPlayers(): Promise<void> {
    await this.page.evaluate(() => {
      const gameWindow = window as GameWindow;
      if (gameWindow.quickplayActions && gameWindow.getQuickplayState) {
        const state = gameWindow.getQuickplayState();
        for (let i = 1; i <= 3; i++) {
          if (!state.aiPlayers.has(i)) {
            gameWindow.quickplayActions.togglePlayer(i);
          }
        }
        if (!state.enabled) {
          gameWindow.quickplayActions.toggle();
        }
      }
    });
  }

  /**
   * Wait for AI to complete its move(s)
   * Detects when AI has finished by monitoring action count or current player change
   */
  async waitForAIMove(expectedActionCount?: number): Promise<void> {
    const initialPlayer = await this.page.evaluate(() => {
      const state = (window as any).getGameState?.();
      return state?.currentPlayer;
    });
    
    const initialActionCount = await this.page.evaluate(() => {
      const url = new URL(window.location.href);
      const d = url.searchParams.get('d');
      if (!d) return 0;
      try {
        // Decode base64 and parse JSON to count actions
        const decoded = JSON.parse(atob(d));
        return decoded.a?.length || 0;
      } catch {
        return 0;
      }
    });

    // Wait for either action count increase or player change (AI finished)
    await this.page.waitForFunction(
      ({ initialCount, initialP, expected }: { initialCount: number; initialP: number | undefined; expected: number | undefined }) => {
        const url = new URL(window.location.href);
        const d = url.searchParams.get('d');
        const state = (window as any).getGameState?.();
        const currentPlayer = state?.currentPlayer;
        
        // Check if it's back to human player (player 0)
        if (currentPlayer === 0 && initialP !== 0) {
          return true;
        }
        
        if (!d) return false;
        try {
          const decoded = JSON.parse(atob(d));
          const currentCount = decoded.a?.length || 0;
          // If we have an expected count, wait for that, otherwise just wait for any increase
          return expected ? currentCount >= expected : currentCount > initialCount;
        } catch {
          return false;
        }
      },
      { initialCount: initialActionCount, initialP: initialPlayer, expected: expectedActionCount },
      { 
        timeout: PlaywrightGameHelper.TIMEOUTS.normal,
        polling: 50 
      }
    );

    // Also ensure game state is stable
    await this.waitForGameReady();
  }

  /**
   * Perform complete game with strategy
   */
  async performCompleteGame(strategy = 'random'): Promise<void> {
    while (true) {
      const phase = await this.getCurrentPhase();
      if (phase.includes('game_end') || phase.includes('complete')) break;
      
      const actions = await this.getAvailableActions();
      if (actions.length === 0) break;
      
      let index = 0;
      switch (strategy) {
        case 'aggressive':
          index = actions.length - 1;
          break;
        case 'conservative':
          index = actions.findIndex(a => a.type === 'pass') || 0;
          break;
        default:
          index = Math.floor(Math.random() * actions.length);
      }
      
      await this.selectAction(index);
    }
  }

  /**
   * Simplified locator access
   */
  locator(selector: string): Locator {
    // Map old selectors to new ones for compatibility
    const mappings: Record<string, string> = {
      '.trick-horizontal': '.trick-table',
      '.trick-position': '.trick-spot',
      '.proceed-action-button': '.tap-indicator',
      '[data-testid="complete-trick"]': '.tap-indicator',
      '.history-row': '.history-item',
      '.trump-display': PlaywrightGameHelper.SELECTORS.trump
    };
    
    return this.page.locator(mappings[selector] || selector);
  }

  /**
   * Get page object for direct access
   */
  getPage(): Page {
    return this.page;
  }

  /**
   * Get locators for common elements
   */
  getLocators() {
    return {
      app: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.app),
      appHeader: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.appHeader),
      playingArea: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.playingArea),
      trickTable: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.trick.table),
      trickTableTappable: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.trick.tappable),
      tapIndicator: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.tapIndicator),
      tapText: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.tapText),
      nav: (item: string): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.nav(item)),
      flash: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.flash),
      debug: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.panel),
      debugButton: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.button),
      debugCloseButton: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.closeButton),
      historyTab: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.historyTab).filter({ hasText: 'History' }),
      historyItem: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.historyItem),
      timeTravelButton: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.debug.timeTravelButton),
      scoreDisplay: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.score.display),
      scoreUs: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.score.us),
      scoreThem: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.score.them),
      trump: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.trump),
      turnPlayer: (): Locator => this.page.locator(PlaywrightGameHelper.SELECTORS.turnPlayer)
    };
  }
}

// Export singleton pattern for backward compatibility
export const playwrightHelper = {
  loadState: async (page: Page, state: { shuffleSeed: number }) => {
    const helper = new PlaywrightGameHelper(page);
    return helper.loadStateWithActions(state.shuffleSeed, []);
  },
  clickAction: async (page: Page, actionId: string | number) => {
    const helper = new PlaywrightGameHelper(page);
    return helper.selectAction(actionId);
  },
  getAvailableActions: async (page: Page) => {
    const helper = new PlaywrightGameHelper(page);
    return helper.getAvailableActions();
  }
};