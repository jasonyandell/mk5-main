import { test } from '@playwright/test';

test('check computed styles of trick container', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('header', { timeout: 5000 });
  
  // Wait for the trick area to be visible
  await page.waitForSelector('.trick-horizontal', { timeout: 5000 });
  
  // Get computed styles
  const display = await page.locator('.trick-horizontal').evaluate(el => {
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
  
  // Check positions of trick elements
  const positions = await page.locator('.trick-position').evaluateAll(elements => {
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
  const allStyles = await page.locator('.trick-horizontal').evaluate(el => {
    // Get all stylesheets
    const sheets = Array.from(document.styleSheets);
    const rules = [];
    
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
      } catch (e) {
        // Some stylesheets might be cross-origin
      }
    });
    
    return rules;
  });
  
  console.log('All CSS rules for trick elements:', allStyles);
});