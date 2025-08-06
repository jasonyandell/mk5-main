import { test, expect } from '@playwright/test';

test.describe('Domino Highlighting System', () => {
  test.beforeEach(async ({ page }) => {
    // Set a reasonable timeout
    page.setDefaultTimeout(5000);
  });

  test('highlighting works in bidding phase', async ({ page }) => {
    // Load bidding phase
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ0Mjg2MjcwOTR9LCJhIjpbeyJpIjoicCJ9LHsiaSI6IjMwIn0seyJpIjoicCJ9XX0');
    
    // Wait for the action panel to be visible (bidding happens in ActionPanel)
    await page.waitForSelector('.action-panel', { timeout: 5000 });
    
    // Find a non-double domino (5-3) in the ActionPanel's hand display
    const domino53 = await page.locator('.action-panel [data-testid="domino-5-3"]').first();
    await expect(domino53).toBeVisible();
    
    // Get the bounding box to click on specific halves
    const box = await domino53.boundingBox();
    if (!box) throw new Error('Could not get domino bounding box');
    
    // Click on top half (should highlight 5s)
    console.log('Clicking top half of 5-3 domino...');
    await page.mouse.click(box.x + box.width / 2, box.y + box.height * 0.25);
    
    // Check if any dominoes have highlight classes
    const highlightedDominoes = await page.locator('.domino.highlight-primary, .domino.highlight-secondary').count();
    console.log(`Found ${highlightedDominoes} highlighted dominoes`);
    
    // Check if the suit badge appears
    const suitBadge = page.locator('.suit-highlight-badge');
    const badgeVisible = await suitBadge.isVisible().catch(() => false);
    console.log(`Suit badge visible: ${badgeVisible}`);
    if (badgeVisible) {
      const badgeText = await suitBadge.textContent();
      console.log(`Badge text: ${badgeText}`);
    }
    
    // Check specific dominoes that should be highlighted (5-3 and 5-0)
    const domino50 = page.locator('[data-testid="domino-5-0"]').first();
    const has50Highlight = await domino50.evaluate(el => {
      const classes = el.className;
      console.log('5-0 classes:', classes);
      return classes.includes('highlight-primary') || classes.includes('highlight-secondary');
    });
    console.log(`5-0 highlighted: ${has50Highlight}`);
    
    // Click again to turn off
    await page.mouse.click(box.x + box.width / 2, box.y + box.height * 0.25);
    await page.waitForTimeout(100);
    
    const highlightedAfterOff = await page.locator('.domino.highlight-primary, .domino.highlight-secondary').count();
    console.log(`Highlighted dominoes after turning off: ${highlightedAfterOff}`);
    
    // Test results
    if (highlightedDominoes > 0) {
      console.log('✅ Highlighting is working!');
    } else {
      console.log('❌ No highlighting detected');
      
      // Debug: Check if suittap event is being fired
      await page.evaluate(() => {
        const dominoes = document.querySelectorAll('.domino');
        console.log(`Found ${dominoes.length} dominoes on page`);
        
        // Try to manually trigger a click and see what happens
        const firstDomino = dominoes[0] as HTMLElement;
        if (firstDomino) {
          console.log('Manually clicking first domino...');
          firstDomino.click();
        }
      });
    }
  });

  test('highlighting works in playing phase with trick history', async ({ page }) => {
    // Load playing phase with tricks
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ0MjYzOTc5NzV9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiJwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJ0cnVtcC10aHJlZXMifSx7ImkiOiIzM2QifSx7ImkiOiI0MyJ9LHsiaSI6IjMxZCJ9LHsiaSI6IjEwIn0seyJpIjoiY29tcGxldGUtdHJpY2sifSx7ImkiOiIzMmQifSx7ImkiOiI2MyJ9LHsiaSI6IjMwZCJ9LHsiaSI6IjYwIn0seyJpIjoiY29tcGxldGUtdHJpY2sifSx7ImkiOiI2NSJ9LHsiaSI6IjYxIn0seyJpIjoiNTIifSx7ImkiOiI2MiJ9LHsiaSI6ImNvbXBsZXRlLXRyaWNrIn0seyJpIjoiNjYifSx7ImkiOiI2NCJ9LHsiaSI6IjIyIn0seyJpIjoiNDBkIn0seyJpIjoiY29tcGxldGUtdHJpY2sifSx7ImkiOiJzaCJ9LHsiaSI6IjMwIn0seyJpIjoicCJ9LHsiaSI6InAifSx7ImkiOiJwIn0seyJpIjoidHJ1bXAtYmxhbmtzIn0seyJpIjoiMTAifSx7ImkiOiIwMCJ9LHsiaSI6IjYzIn0seyJpIjoiMjAifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9LHsiaSI6IjExIn0seyJpIjoiNDFkIn0seyJpIjoiNDBkIn0seyJpIjoiMjEifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9LHsiaSI6IjYwIn0seyJpIjoiNjYifSx7ImkiOiI1MCJ9XX0');
    
    // Wait for the play area
    await page.waitForSelector('.playing-area', { timeout: 5000 });
    
    // Expand trick history if available
    const trickCounter = page.locator('.trick-counter.expandable');
    if (await trickCounter.isVisible()) {
      console.log('Expanding trick history...');
      await trickCounter.click();
      await page.waitForTimeout(300);
      
      // Check if history is visible
      const historyVisible = await page.locator('.trick-history').isVisible();
      console.log(`Trick history visible: ${historyVisible}`);
      
      if (historyVisible) {
        // Try clicking a domino in history
        const historyDomino = await page.locator('.trick-history .domino').first();
        if (await historyDomino.isVisible()) {
          console.log('Clicking domino in trick history...');
          await historyDomino.click();
          await page.waitForTimeout(100);
          
          // Check for highlights
          const highlighted = await page.locator('.domino.highlight-primary, .domino.highlight-secondary').count();
          console.log(`Highlighted dominoes in history: ${highlighted}`);
        }
      }
    }
    
    // Check hand dominoes
    const handDominoes = await page.locator('.hand-dominoes .domino').count();
    console.log(`Hand dominoes found: ${handDominoes}`);
    
    if (handDominoes > 0) {
      // Click on a non-playable domino to test highlighting
      const firstHandDomino = page.locator('.hand-dominoes .domino').first();
      await firstHandDomino.click();
      await page.waitForTimeout(100);
      
      const highlighted = await page.locator('.domino.highlight-primary, .domino.highlight-secondary').count();
      console.log(`Highlighted after clicking hand domino: ${highlighted}`);
    }
  });

  test('check if suittap events are being dispatched', async ({ page }) => {
    // Load bidding phase
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ0Mjg2MjcwOTR9LCJhIjpbeyJpIjoicCJ9LHsiaSI6IjMwIn0seyJpIjoicCJ9XX0');
    
    await page.waitForSelector('.hand-display', { timeout: 5000 });
    
    // Inject event listener to catch suittap events
    const events = await page.evaluate(() => {
      const capturedEvents: any[] = [];
      
      // Add listeners to all dominoes
      document.querySelectorAll('.domino').forEach((domino, index) => {
        console.log(`Adding listeners to domino ${index}`);
        
        // Listen for click events
        domino.addEventListener('click', (e) => {
          console.log(`Click event on domino ${index}`, e);
          capturedEvents.push({ type: 'click', index, target: (e.target as HTMLElement).className });
        });
        
        // Listen for custom events (suittap)
        domino.addEventListener('suittap', (e) => {
          console.log(`Suittap event on domino ${index}`, e);
          capturedEvents.push({ type: 'suittap', index, detail: (e as CustomEvent).detail });
        });
      });
      
      // Click the first domino
      const firstDomino = document.querySelector('.domino') as HTMLElement;
      if (firstDomino) {
        console.log('Clicking first domino programmatically...');
        firstDomino.click();
      }
      
      return capturedEvents;
    });
    
    console.log('Captured events:', events);
    
    // Now click via Playwright and check again
    const domino = page.locator('.domino').first();
    await domino.click();
    
    // Check what events were captured
    const eventsAfterClick = await page.evaluate(() => {
      return (window as any).capturedEvents || [];
    });
    
    console.log('Events after Playwright click:', eventsAfterClick);
  });
});