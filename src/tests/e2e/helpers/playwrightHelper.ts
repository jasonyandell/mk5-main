import { Page, expect } from '@playwright/test';

export class PlaywrightHelper {
  constructor(private page: Page) {}

  async navigateToDebugUI() {
    await this.page.goto('/debug');
    await this.page.waitForLoadState('networkidle');
  }

  async clickButton(text: string) {
    await this.page.click(`button:has-text("${text}")`);
  }

  async getGameState() {
    return await this.page.evaluate(() => {
      // Access game state from window or DOM
      return {};
    });
  }

  async waitForStateChange(timeout = 1000) {
    await this.page.waitForTimeout(timeout);
  }

  async verifyState(expectedState: any) {
    const actualState = await this.getGameState();
    expect(actualState).toEqual(expectedState);
  }
}