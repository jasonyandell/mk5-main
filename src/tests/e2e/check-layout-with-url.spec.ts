import { test } from '@playwright/test';

test('check layout with provided URL', async ({ page }) => {
  // Use the provided URL with game state
  await page.goto('/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQzMzIxNDYzNTh9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiIzNSJ9LHsiaSI6InAifSx7ImkiOiJwIn0seyJpIjoidHJ1bXAtYmxhbmtzIn0seyJpIjoiNjMifSx7ImkiOiI2MCJ9LHsiaSI6IjYyIn0seyJpIjoiNjQifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9XX0');
  
  await page.waitForSelector('.current-trick-area', { timeout: 5000 });
  
  // Take screenshot
  await page.screenshot({ path: 'trick-with-dominoes.png', fullPage: true });
  
  // Check the actual CSS being applied
  const trickContainer = page.locator('.trick-horizontal');
  const computedStyles = await trickContainer.evaluate(el => {
    const styles = window.getComputedStyle(el);
    return {
      display: styles.display,
      flexDirection: styles.flexDirection,
      justifyContent: styles.justifyContent,
      gap: styles.gap,
      className: el.className,
      innerHTML: el.innerHTML.substring(0, 200)
    };
  });
  
  console.log('Trick container styles:', computedStyles);
  
  // Get positions of all dominoes
  const positions = await page.locator('.trick-position').evaluateAll(elements => {
    return elements.map((el, index) => {
      const rect = el.getBoundingClientRect();
      return {
        index,
        player: el.getAttribute('data-player'),
        x: Math.round(rect.x),
        y: Math.round(rect.y),
        text: el.textContent?.trim()
      };
    });
  });
  
  console.log('Domino positions:', positions);
  
  // Check if positions are in a line (same Y) or grid (different Y)
  const uniqueYPositions = [...new Set(positions.map(p => p.y))];
  console.log('Unique Y positions:', uniqueYPositions);
  console.log('Layout is:', uniqueYPositions.length === 1 ? 'HORIZONTAL' : 'GRID (2x2)');
});