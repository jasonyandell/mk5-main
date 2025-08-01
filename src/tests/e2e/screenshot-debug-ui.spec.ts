import { test } from '@playwright/test';
import { PlaywrightGameHelper } from './helpers/playwrightHelper';

test('test all domino glyphs', async ({ page }) => {
  test.setTimeout(10000);
  
  const htmlContent = `
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="UTF-8">
      <style>
        body {
          font-family: "Noto Sans Symbols", "Noto Sans Symbols 2", "Segoe UI Symbol", "Segoe UI Emoji", "Apple Color Emoji", "Symbola", sans-serif;
          padding: 20px;
        }
        .domino-grid {
          display: grid;
          grid-template-columns: repeat(7, 1fr);
          gap: 10px;
        }
        .domino {
          text-align: center;
          border: 1px solid #ccc;
          padding: 10px;
        }
        .glyph {
          font-size: 48px;
          margin: 10px 0;
        }
        .label {
          font-size: 14px;
        }
      </style>
    </head>
    <body>
      <h1>Domino Glyph Test</h1>
      <div class="domino-grid">
        <div class="domino"><div class="label">0-0</div><div class="glyph">🀰</div></div>
        <div class="domino"><div class="label">0-1</div><div class="glyph">🀱</div></div>
        <div class="domino"><div class="label">0-2</div><div class="glyph">🀲</div></div>
        <div class="domino"><div class="label">0-3</div><div class="glyph">🀳</div></div>
        <div class="domino"><div class="label">0-4</div><div class="glyph">🀴</div></div>
        <div class="domino"><div class="label">0-5</div><div class="glyph">🀵</div></div>
        <div class="domino"><div class="label">0-6</div><div class="glyph">🀶</div></div>
        <div class="domino"><div class="label">1-1</div><div class="glyph">🀷</div></div>
        <div class="domino"><div class="label">1-2</div><div class="glyph">🀸</div></div>
        <div class="domino"><div class="label">1-3</div><div class="glyph">🀹</div></div>
        <div class="domino"><div class="label">1-4</div><div class="glyph">🀺</div></div>
        <div class="domino"><div class="label">1-5</div><div class="glyph">🀻</div></div>
        <div class="domino"><div class="label">1-6</div><div class="glyph">🀼</div></div>
        <div class="domino"><div class="label">2-2</div><div class="glyph">🀽</div></div>
        <div class="domino"><div class="label">2-3</div><div class="glyph">🀾</div></div>
        <div class="domino"><div class="label">2-4</div><div class="glyph">🀿</div></div>
        <div class="domino"><div class="label">2-5</div><div class="glyph">🁀</div></div>
        <div class="domino"><div class="label">2-6</div><div class="glyph">🁁</div></div>
        <div class="domino"><div class="label">3-3</div><div class="glyph">🁂</div></div>
        <div class="domino"><div class="label">3-4</div><div class="glyph">🁃</div></div>
        <div class="domino"><div class="label">3-5</div><div class="glyph">🁄</div></div>
        <div class="domino"><div class="label">3-6</div><div class="glyph">🁅</div></div>
        <div class="domino"><div class="label">4-4</div><div class="glyph">🁆</div></div>
        <div class="domino"><div class="label">4-5</div><div class="glyph">🁇</div></div>
        <div class="domino"><div class="label">4-6</div><div class="glyph">🁈</div></div>
        <div class="domino"><div class="label">5-5</div><div class="glyph">🁉</div></div>
        <div class="domino"><div class="label">5-6</div><div class="glyph">🁊</div></div>
        <div class="domino"><div class="label">6-6</div><div class="glyph">🁋</div></div>
      </div>
    </body>
    </html>
  `;
  
  await page.setContent(htmlContent);
  await page.waitForTimeout(1000);
  
  // Take screenshot
  await page.screenshot({ 
    path: 'all-dominoes-test.png',
    fullPage: true 
  });
  
  console.log('Screenshot saved: all-dominoes-test.png');
});