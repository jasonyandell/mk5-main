import { test } from '@playwright/test';

test('take final screenshot', async ({ page }) => {
  // Use the provided URL with game state
  await page.goto('/?d=eyJ2IjoxLCJzIjp7InMiOjE3NTQzMzIxNDYzNTh9LCJhIjpbeyJpIjoiMzAifSx7ImkiOiIzNSJ9LHsiaSI6InAifSx7ImkiOiJwIn0seyJpIjoidHJ1bXAtYmxhbmtzIn0seyJpIjoiNjMifSx7ImkiOiI2MCJ9LHsiaSI6IjYyIn0seyJpIjoiNjQifSx7ImkiOiJjb21wbGV0ZS10cmljayJ9XX0');
  
  await page.waitForSelector('.current-trick-area', { timeout: 5000 });
  
  // Take screenshot
  await page.screenshot({ path: 'final-layout.png', fullPage: true });
  
  console.log('Screenshot saved as final-layout.png');
});