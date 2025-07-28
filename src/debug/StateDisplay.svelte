<script lang="ts">
  import type { GameState } from '../game/types';
  
  interface Props {
    state: GameState;
  }
  
  let { state }: Props = $props();
  
  function getStateJson() {
    return JSON.stringify(state, null, 2);
  }
  
  function copyToClipboard() {
    navigator.clipboard.writeText(getStateJson());
  }
  
  function downloadState() {
    const blob = new Blob([getStateJson()], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `texas42-state-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
</script>

<div class="state-display">
  <div class="state-header">
    <h3>Current Game State</h3>
    <div class="state-actions">
      <button onclick={copyToClipboard}>Copy</button>
      <button onclick={downloadState}>Download</button>
    </div>
  </div>
  
  <div class="state-summary">
    <div class="summary-item">
      <span class="label">Phase:</span>
      <span class="value">{state.phase}</span>
    </div>
    <div class="summary-item">
      <span class="label">Current Player:</span>
      <span class="value">Player {state.currentPlayer + 1}</span>
    </div>
    <div class="summary-item">
      <span class="label">Dealer:</span>
      <span class="value">Player {state.dealer + 1}</span>
    </div>
    {#if state.trump !== null}
      <div class="summary-item">
        <span class="label">Trump:</span>
        <span class="value">{['Blanks', 'Ones', 'Twos', 'Threes', 'Fours', 'Fives', 'Sixes'][state.trump]}</span>
      </div>
    {/if}
    <div class="summary-item">
      <span class="label">Team Scores:</span>
      <span class="value">{state.teamScores[0]} - {state.teamScores[1]}</span>
    </div>
    <div class="summary-item">
      <span class="label">Team Marks:</span>
      <span class="value">{state.teamMarks[0]} - {state.teamMarks[1]}</span>
    </div>
  </div>
  
  <div class="state-json">
    <h4>Full State JSON</h4>
    <pre><code>{getStateJson()}</code></pre>
  </div>
</div>

<style>
  .state-display {
    margin-bottom: 24px;
  }
  
  .state-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }
  
  .state-header h3 {
    margin: 0;
    color: #333;
  }
  
  .state-actions {
    display: flex;
    gap: 8px;
  }
  
  .state-actions button {
    padding: 6px 12px;
    border: 1px solid #ddd;
    border-radius: 4px;
    background: white;
    cursor: pointer;
    font-size: 12px;
  }
  
  .state-actions button:hover {
    background: #f5f5f5;
  }
  
  .state-summary {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 16px;
  }
  
  .summary-item {
    display: flex;
    justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid #e9ecef;
  }
  
  .summary-item:last-child {
    border-bottom: none;
  }
  
  .summary-item .label {
    font-weight: 500;
    color: #495057;
  }
  
  .summary-item .value {
    color: #212529;
    font-family: monospace;
  }
  
  .state-json {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 6px;
    padding: 16px;
  }
  
  .state-json h4 {
    margin: 0 0 12px 0;
    color: #495057;
    font-size: 14px;
  }
  
  .state-json pre {
    margin: 0;
    max-height: 300px;
    overflow: auto;
    font-size: 11px;
    line-height: 1.4;
  }
  
  .state-json code {
    color: #212529;
  }
</style>