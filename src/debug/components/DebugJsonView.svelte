<script lang="ts">
  import type { GameState } from '../../game/types';
  
  interface Props {
    gameState: GameState;
  }
  
  let { gameState }: Props = $props();
  
  let collapsed = $state(true);
  let selectedSection = $state<string>('state');
  
  const prettyJson = $derived.by(() => {
    try {
      return JSON.stringify(gameState, null, 2);
    } catch (e) {
      return 'Error serializing state';
    }
  });
  
  const stateSummary = $derived.by(() => {
    const summary = {
      phase: gameState.phase,
      currentPlayer: gameState.currentPlayer,
      dealer: gameState.dealer,
      trump: gameState.trump,
      bidsCount: gameState.bids.length,
      tricksCount: gameState.tricks.length,
      currentTrickSize: gameState.currentTrick.length,
      teamScores: gameState.teamScores,
      teamMarks: gameState.teamMarks,
      handsizes: gameState.players.map(p => p.hand.length)
    };
    return JSON.stringify(summary, null, 2);
  });
  
  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text).then(() => {
      // Could show a toast notification here
    }).catch(() => {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    });
  }
</script>

<div class="debug-json-view">
  <div class="json-header">
    <h3>State Inspection</h3>
    <div class="header-controls">
      <select bind:value={selectedSection} class="section-select">
        <option value="summary">Summary</option>
        <option value="state">Full State</option>
      </select>
      <button 
        class="toggle-btn"
        onclick={() => collapsed = !collapsed}
      >
        {collapsed ? 'Expand' : 'Collapse'}
      </button>
    </div>
  </div>
  
  {#if !collapsed}
    <div class="json-content">
      <div class="json-controls">
        <button 
          class="copy-btn"
          onclick={() => copyToClipboard(selectedSection === 'summary' ? stateSummary : prettyJson)}
        >
          Copy JSON
        </button>
      </div>
      
      <div class="json-display">
        <pre class="json-text">{selectedSection === 'summary' ? stateSummary : prettyJson}</pre>
      </div>
    </div>
  {:else}
    <div class="json-collapsed">
      <div class="summary-items">
        <div class="summary-item">
          <span class="key">Phase:</span>
          <span class="value">{gameState.phase}</span>
        </div>
        <div class="summary-item">
          <span class="key">Turn:</span>
          <span class="value">P{gameState.currentPlayer}</span>
        </div>
        <div class="summary-item">
          <span class="key">Bids:</span>
          <span class="value">{gameState.bids.length}/4</span>
        </div>
        <div class="summary-item">
          <span class="key">Tricks:</span>
          <span class="value">{gameState.tricks.length}/7</span>
        </div>
        <div class="summary-item">
          <span class="key">Scores:</span>
          <span class="value">{gameState.teamScores[0]}-{gameState.teamScores[1]}</span>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .debug-json-view {
    background: white;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    font-size: 11px;
    height: 100%;
    display: flex;
    flex-direction: column;
  }
  
  .json-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px;
    border-bottom: 1px solid #e9ecef;
    background: #f8f9fa;
  }
  
  .json-header h3 {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: #212529;
  }
  
  .header-controls {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  
  .section-select {
    font-size: 10px;
    padding: 2px 4px;
    border: 1px solid #ced4da;
    border-radius: 2px;
  }
  
  .toggle-btn {
    font-size: 10px;
    padding: 4px 8px;
    background: #6c757d;
    color: white;
    border: none;
    border-radius: 2px;
    cursor: pointer;
  }
  
  .toggle-btn:hover {
    background: #5a6268;
  }
  
  .json-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  
  .json-controls {
    padding: 8px 12px;
    border-bottom: 1px solid #e9ecef;
    background: #f8f9fa;
  }
  
  .copy-btn {
    font-size: 10px;
    padding: 2px 6px;
    background: #28a745;
    color: white;
    border: none;
    border-radius: 2px;
    cursor: pointer;
  }
  
  .copy-btn:hover {
    background: #218838;
  }
  
  .json-display {
    flex: 1;
    overflow: auto;
    padding: 8px;
  }
  
  .json-text {
    font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
    font-size: 9px;
    line-height: 1.3;
    margin: 0;
    color: #495057;
    white-space: pre-wrap;
    word-break: break-all;
  }
  
  .json-collapsed {
    padding: 12px;
    flex: 1;
  }
  
  .summary-items {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  
  .summary-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  
  .key {
    color: #6c757d;
    font-weight: 500;
  }
  
  .value {
    color: #212529;
    font-weight: 600;
    font-family: monospace;
  }
</style>