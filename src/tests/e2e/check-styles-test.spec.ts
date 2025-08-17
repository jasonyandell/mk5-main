import { test } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test('check computed styles of trick container', async ({ page }) => {
  const helper = new PlaywrightGameHelper(page);
  await helper.goto();
  
  // Navigate to Play tab to see the trick table
  await page.locator('[data-testid="nav-game"]').click();
  
  // Wait for the trick area to be visible (mapped to .trick-table)
  await helper.waitForSelector('.trick-horizontal', { timeout: 5000 });
  
  // Get computed styles
  const display = await helper.locator('.trick-horizontal').evaluate(el => {
    const styles = window.getComputedStyle(el);
    return {
      display: styles.display,
      flexDirection: styles.flexDirection,
      gridTemplateColumns: styles.gridTemplateColumns,
      gridTemplateRows: styles.gridTemplateRows,
      justifyContent: styles.justifyContent,
      gap: styles.gap
    };
  });
  
  console.log('Computed styles for .trick-horizontal:', display);
  
  // Check positions of trick elements (mapped to .trick-spot)
  const positions = await helper.locator('.trick-position').evaluateAll(elements => {
    return elements.map(el => {
      const rect = el.getBoundingClientRect();
      return {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        playerData: el.getAttribute('data-player')
      };
    });
  });
  
  console.log('Trick positions:', positions);
  
  // Check if there's any CSS that might be overriding
  const allStyles = await helper.locator('.trick-horizontal').evaluate(() => {
    // Get all stylesheets
    const sheets = Array.from(document.styleSheets);
    const rules: Array<{ selector: string; style: string }> = [];
    
    sheets.forEach(sheet => {
      try {
        const sheetRules = Array.from(sheet.cssRules || []);
        sheetRules.forEach(rule => {
          if (rule instanceof CSSStyleRule) {
            if (rule.selectorText && (
              rule.selectorText.includes('trick-horizontal') || 
              rule.selectorText.includes('trick-grid') ||
              rule.selectorText.includes('trick-position')
            )) {
              rules.push({
                selector: rule.selectorText,
                style: rule.style.cssText
              });
            }
          }
        });
      } catch {
        // Some stylesheets might be cross-origin
      }
    });
    
    return rules;
  });
  
  console.log('All CSS rules for trick elements:', allStyles);
});