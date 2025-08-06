import { test, expect } from '@playwright/test';

test.describe('Debug Highlighting', () => {
  test('check console for errors and events', async ({ page }) => {
    // Capture console messages
    const consoleMessages: string[] = [];
    page.on('console', msg => {
      consoleMessages.push(`[${msg.type()}] ${msg.text()}`);
    });
    
    // Load bidding phase
    await page.goto('http://localhost:60101/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQ0Mjg2MjcwOTR9LCJhIjpbeyJpIjoicCJ9LHsiaSI6IjMwIn0seyJpIjoicCJ9XX0');
    
    await page.waitForSelector('.hand-display', { timeout: 5000 });
    
    // Check what dominoes exist
    const dominoes = await page.locator('.domino').evaluateAll(els => 
      els.map(el => ({
        testId: el.getAttribute('data-testid'),
        classes: el.className,
        disabled: el.hasAttribute('disabled')
      }))
    );
    
    console.log('Found dominoes:', dominoes);
    
    // Try clicking the first domino and see what happens
    await page.evaluate(() => {
      console.log('=== Starting debug ===');
      const firstDomino = document.querySelector('.domino') as HTMLElement;
      if (firstDomino) {
        console.log('Found domino:', firstDomino);
        
        // Add debug listeners
        firstDomino.addEventListener('click', (e) => {
          console.log('Native click event:', e);
        });
        
        firstDomino.addEventListener('suittap', (e) => {
          console.log('Suittap event received:', (e as CustomEvent).detail);
        });
        
        // Check if Svelte component exists
        const svelteComponent = (firstDomino as any).__svelte;
        console.log('Svelte component attached:', !!svelteComponent);
        
        // Try clicking
        console.log('Clicking domino...');
        firstDomino.click();
      }
    });
    
    // Wait a bit for events to fire
    await page.waitForTimeout(500);
    
    // Print all console messages
    console.log('\n=== Console messages ===');
    consoleMessages.forEach(msg => console.log(msg));
    
    // Check if any highlights were applied
    const highlightedCount = await page.locator('.highlight-primary, .highlight-secondary').count();
    console.log(`Highlighted dominoes: ${highlightedCount}`);
    
    // Check the highlight state in the component
    const highlightState = await page.evaluate(() => {
      // Try to access Svelte stores
      const actionPanel = document.querySelector('.action-panel');
      console.log('Action panel found:', !!actionPanel);
      
      // Check if any Svelte internals are accessible
      const svelteData = (actionPanel as any)?.__svelte;
      console.log('Action panel Svelte data:', !!svelteData);
      
      return {
        actionPanelExists: !!actionPanel,
        hasSvelteData: !!svelteData
      };
    });
    
    console.log('Highlight state:', highlightState);
  });
});