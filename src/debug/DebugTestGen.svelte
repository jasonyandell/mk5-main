<script lang="ts">
  import { gameState, actionHistory } from '../stores/gameStore';
  
  let testName = '';
  let testGenerated = false;
  let testCode = '';
  
  function generateTest() {
    const state = $gameState;
    const history = $actionHistory;
    
    // Generate test code
    const testImports = `import { test, expect } from '@playwright/test';
import { playwrightHelper } from '../helpers/playwrightHelper';`;
    
    const testSetup = `test('${testName || 'Generated test case'}', async ({ page }) => {
  const helper = new playwrightHelper(page);
  await helper.loadDebugMode();`;
    
    // Generate action steps
    const actionSteps = history.map(action => {
      if (action.id.startsWith('bid-')) {
        return `  await helper.makeBid('${action.id}');`;
      } else if (action.id.startsWith('set-trump-')) {
        return `  await helper.selectTrump('${action.id}');`;
      } else if (action.id.startsWith('play-domino-')) {
        return `  await helper.playDomino('${action.id}');`;
      } else {
        return `  await helper.clickAction('${action.id}');`;
      }
    }).join('\n');
    
    // Generate assertions
    const assertions = `
  // Verify final state
  await expect(page.getByTestId('phase')).toHaveText('${state.phase}');
  await expect(page.getByTestId('current-player')).toHaveText('${state.currentPlayer}');
  await expect(page.getByTestId('team-0-score')).toHaveText('${state.teamScores[0]}');
  await expect(page.getByTestId('team-1-score')).toHaveText('${state.teamScores[1]}');`;
    
    const testEnd = `
});`;
    
    testCode = [
      testImports,
      '',
      testSetup,
      actionSteps,
      assertions,
      testEnd
    ].join('\n');
    
    testGenerated = true;
  }
  
  function copyTest() {
    navigator.clipboard.writeText(testCode).then(() => {
      alert('Test code copied to clipboard!');
    });
  }
  
  function downloadTest() {
    const blob = new Blob([testCode], { type: 'text/typescript' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${testName.replace(/\s+/g, '-').toLowerCase() || 'generated-test'}.spec.ts`;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  function reset() {
    testName = '';
    testGenerated = false;
    testCode = '';
  }
</script>

<div class="test-gen-panel">
  <h3>🧪 Test Generator</h3>
  
  {#if !testGenerated}
    <div class="test-form">
      <label for="test-name">Test Name:</label>
      <input 
        id="test-name"
        type="text"
        bind:value={testName}
        placeholder="e.g., should handle complex bidding scenario"
      />
      
      <div class="test-info">
        <h4>Test will include:</h4>
        <ul>
          <li>✓ {$actionHistory.length} actions from current game</li>
          <li>✓ State setup from initial conditions</li>
          <li>✓ Assertions for final state</li>
          <li>✓ Proper Playwright helper usage</li>
        </ul>
      </div>
      
      <button 
        class="generate-btn"
        on:click={generateTest}
        disabled={$actionHistory.length === 0}
        data-testid="generate-test"
      >
        Generate Test Code
      </button>
      
      {#if $actionHistory.length === 0}
        <p class="warning">Play some actions first to generate a meaningful test!</p>
      {/if}
    </div>
  {:else}
    <div class="test-result">
      <p class="success">✓ Test code generated successfully!</p>
      
      <div class="test-actions">
        <button on:click={copyTest} data-testid="copy-test">
          📋 Copy Code
        </button>
        <button on:click={downloadTest} data-testid="download-test">
          💾 Download .spec.ts
        </button>
        <button on:click={reset} data-testid="new-test">
          🔄 New Test
        </button>
      </div>
      
      <div class="code-preview">
        <h5>Generated Test Code:</h5>
        <pre><code>{testCode}</code></pre>
      </div>
    </div>
  {/if}
</div>

<style>
  .test-gen-panel {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
  }
  
  .test-gen-panel h3 {
    margin: 0 0 20px 0;
    color: #00BCD4;
    font-size: 18px;
  }
  
  .test-form {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  label {
    font-size: 14px;
    color: #aaa;
  }
  
  input {
    background: #333;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 10px;
    color: #fff;
    font-size: 14px;
  }
  
  input:focus {
    outline: none;
    border-color: #00BCD4;
  }
  
  .test-info {
    background: #333;
    border-radius: 6px;
    padding: 15px;
  }
  
  .test-info h4 {
    margin: 0 0 10px 0;
    color: #888;
    font-size: 14px;
  }
  
  .test-info ul {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  
  .test-info li {
    padding: 5px 0;
    font-size: 13px;
    color: #4CAF50;
  }
  
  .generate-btn {
    padding: 12px 24px;
    background: #00BCD4;
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .generate-btn:hover:not(:disabled) {
    background: #00ACC1;
  }
  
  .generate-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  
  .warning {
    color: #FFA726;
    font-size: 14px;
    margin: 0;
    text-align: center;
  }
  
  .test-result {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .success {
    color: #4CAF50;
    font-size: 16px;
    margin: 0;
  }
  
  .test-actions {
    display: flex;
    gap: 10px;
  }
  
  .test-actions button {
    flex: 1;
    padding: 10px;
    background: #333;
    border: 1px solid #555;
    border-radius: 6px;
    color: #fff;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 14px;
  }
  
  .test-actions button:hover {
    background: #444;
    border-color: #666;
  }
  
  .code-preview {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 15px;
    max-height: 400px;
    overflow: auto;
  }
  
  .code-preview h5 {
    margin: 0 0 10px 0;
    color: #888;
    font-size: 12px;
  }
  
  .code-preview pre {
    margin: 0;
  }
  
  .code-preview code {
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 12px;
    color: #e0e0e0;
    line-height: 1.5;
  }
  
  ::-webkit-scrollbar {
    width: 8px;
    height: 8px;
  }
  
  ::-webkit-scrollbar-track {
    background: #1a1a1a;
    border-radius: 4px;
  }
  
  ::-webkit-scrollbar-thumb {
    background: #555;
    border-radius: 4px;
  }
  
  ::-webkit-scrollbar-thumb:hover {
    background: #666;
  }
</style>