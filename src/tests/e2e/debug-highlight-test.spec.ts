import { test } from '@playwright/test';

test('debug highlighting issue', async ({ page }) => {
  await page.goto('/');
  await page.waitForSelector('.game-progress', { timeout: 5000 });
  
  // Get all dominoes and their classes
  const dominoInfo = await page.locator('.domino').evaluateAll(elements => 
    elements.map(el => ({
      title: el.getAttribute('title'),
      classes: el.className,
      hasHighlightPrimary: el.classList.contains('highlight-primary'),
      hasHighlightSecondary: el.classList.contains('highlight-secondary')
    }))
  );
  
  console.log('Initial domino states:', dominoInfo);
  
  // Find a specific domino to hover
  const domino33 = page.locator('[data-testid="domino-3-3"]');
  if (await domino33.count() > 0) {
    const box = await domino33.boundingBox();
    if (box) {
      await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await page.waitForTimeout(500);
      
      // Check classes again
      const afterHover = await page.locator('.domino').evaluateAll(elements => 
        elements.map(el => ({
          title: el.getAttribute('title'),
          classes: el.className,
          hasHighlightPrimary: el.classList.contains('highlight-primary'),
          hasHighlightSecondary: el.classList.contains('highlight-secondary')
        }))
      );
      
      console.log('\nAfter hovering 3-3:', afterHover.filter(d => 
        d.hasHighlightPrimary || d.hasHighlightSecondary
      ));
      
      // Check the shouldHighlight function is being called
      const highlightedCount = afterHover.filter(d => 
        d.hasHighlightPrimary || d.hasHighlightSecondary
      ).length;
      
      console.log('Total highlighted:', highlightedCount);
    }
  }
});