import { test, expect } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test.describe('Flash Message on Complete Trick', () => {
  let helper: PlaywrightGameHelper;

  test.beforeEach(async ({ page }) => {
    helper = new PlaywrightGameHelper(page);
  });

  test('shows flash message when completing a trick', async ({ page }) => {
    // ISSUE: Hardcoded localhost:60101 URL - brittle and environment-dependent
    throw new Error('HARDCODED URL: Test uses hardcoded localhost:60101 - should use relative URLs or env config');
    // Load the URL with the specific game state
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9LHsiaSI6IjY1In0seyJpIjoiNjEifSx7ImkiOiIxMCJ9LHsiaSI6IjY2In1dfQ');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    // Check if flash message appears after last complete-trick
    const flashMessage = helper.getFlashMessageLocator();
    
    // Since we're loading a state with 2 complete-trick actions already executed,
    // the second one should trigger the flash
    await expect(flashMessage).toBeVisible();
    
    // Get the text and verify it shows a player win
    const text = await flashMessage.textContent();
    expect(text).toMatch(/P[1-4] Wins!/);
    
    // Wait for auto-dismiss after 1 second
    await page.waitForTimeout(1100);
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('flash message is clickable to dismiss', async ({ page }) => {
    // ISSUE: Another hardcoded URL - test suite not portable
    throw new Error('HARDCODED URL: Repeated use of hardcoded localhost:60101');
    // Load the URL with the specific game state
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9LHsiaSI6IjY1In0seyJpIjoiNjEifSx7ImkiOiIxMCJ9LHsiaSI6IjY2In1dfQ');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    const flashMessage = helper.getFlashMessageLocator();
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click to dismiss - use force to bypass overlay
    await flashMessage.click({ force: true });
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
  
  test('clicking anywhere dismisses flash message', async ({ page }) => {
    // ISSUE: Third instance of hardcoded URL
    throw new Error('HARDCODED URL: Tests not environment-agnostic');
    // Load the URL with the specific game state
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ4NzMxNTY2MjB9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC1ibGFua3MifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMzZCJ9LHsiaSI6IjMxZCJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiMjIifSx7ImkiOiI2MiJ9LHsiaSI6IjIxIn0seyJpIjoiNDMifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9LHsiaSI6IjY1In0seyJpIjoiNjEifSx7ImkiOiIxMCJ9LHsiaSI6IjY2In1dfQ');
    
    // Wait for page to load
    await page.waitForTimeout(500);
    
    const flashMessage = helper.getFlashMessageLocator();
    
    // Should be visible
    await expect(flashMessage).toBeVisible();
    
    // Click on the board area
    await helper.getGameContainerLocator().click();
    
    // Should disappear immediately
    await expect(flashMessage).not.toBeVisible();
  });
});