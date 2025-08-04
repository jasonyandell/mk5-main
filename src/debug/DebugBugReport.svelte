<script lang="ts">
  import { gameState, actionHistory, stateValidationError } from '../stores/gameStore';
  
  let description = '';
  let reportGenerated = false;
  let reportText = '';
  
  function generateBugReport() {
    const state = $gameState;
    const history = $actionHistory;
    const error = $stateValidationError;
    
    const report = {
      timestamp: new Date().toISOString(),
      description: description,
      error: error || null,
      state: {
        phase: state.phase,
        currentPlayer: state.currentPlayer,
        dealer: state.dealer,
        winningBidder: state.winningBidder,
        trump: state.trump,
        teamScores: state.teamScores,
        teamMarks: state.teamMarks,
        tricksPlayed: state.tricks.length,
        currentTrickPlays: state.currentTrick.length
      },
      actionHistory: history.map(h => ({
        id: h.id,
        label: h.label
      })),
      fullState: state
    };
    
    reportText = JSON.stringify(report, null, 2);
    reportGenerated = true;
  }
  
  function copyReport() {
    navigator.clipboard.writeText(reportText).then(() => {
      alert('Bug report copied to clipboard!');
    });
  }
  
  function downloadReport() {
    const blob = new Blob([reportText], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `bug-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }
  
  function reset() {
    description = '';
    reportGenerated = false;
    reportText = '';
  }
</script>

<div class="bug-report-panel">
  <h3>🐛 Bug Report</h3>
  
  {#if !reportGenerated}
    <div class="report-form">
      <label for="bug-description">Describe the issue:</label>
      <textarea 
        id="bug-description"
        bind:value={description}
        placeholder="What went wrong? What were you trying to do?"
        rows="4"
      ></textarea>
      
      <div class="current-state-info">
        <h4>Current State Summary</h4>
        <ul>
          <li>Phase: <span>{$gameState.phase}</span></li>
          <li>Current Player: <span>P{$gameState.currentPlayer}</span></li>
          <li>Actions Taken: <span>{$actionHistory.length}</span></li>
          <li>Tricks Played: <span>{$gameState.tricks.length}</span></li>
          {#if $stateValidationError}
            <li class="error">Validation Error: <span>YES</span></li>
          {/if}
        </ul>
      </div>
      
      <button 
        class="generate-btn"
        on:click={generateBugReport}
        data-testid="generate-bug-report"
      >
        Generate Bug Report
      </button>
    </div>
  {:else}
    <div class="report-result">
      <p class="success">✓ Bug report generated successfully!</p>
      
      <div class="report-actions">
        <button on:click={copyReport} data-testid="copy-bug-report">
          📋 Copy to Clipboard
        </button>
        <button on:click={downloadReport} data-testid="download-bug-report">
          💾 Download JSON
        </button>
        <button on:click={reset} data-testid="new-bug-report">
          🔄 New Report
        </button>
      </div>
      
      <div class="report-preview">
        <h5>Report Preview (first 500 chars):</h5>
        <pre>{reportText.substring(0, 500)}...</pre>
      </div>
    </div>
  {/if}
</div>

<style>
  .bug-report-panel {
    background: #2a2a2a;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 20px;
  }
  
  .bug-report-panel h3 {
    margin: 0 0 20px 0;
    color: #F44336;
    font-size: 18px;
  }
  
  .report-form {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  label {
    font-size: 14px;
    color: #aaa;
  }
  
  textarea {
    background: #333;
    border: 1px solid #555;
    border-radius: 6px;
    padding: 10px;
    color: #fff;
    font-size: 14px;
    resize: vertical;
  }
  
  textarea:focus {
    outline: none;
    border-color: #F44336;
  }
  
  .current-state-info {
    background: #333;
    border-radius: 6px;
    padding: 15px;
  }
  
  .current-state-info h4 {
    margin: 0 0 10px 0;
    color: #888;
    font-size: 14px;
  }
  
  .current-state-info ul {
    margin: 0;
    padding: 0;
    list-style: none;
  }
  
  .current-state-info li {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    font-size: 13px;
    color: #aaa;
  }
  
  .current-state-info li span {
    color: #fff;
    font-weight: 500;
  }
  
  .current-state-info li.error {
    color: #F44336;
  }
  
  .current-state-info li.error span {
    color: #F44336;
  }
  
  .generate-btn {
    padding: 12px 24px;
    background: #F44336;
    border: none;
    border-radius: 6px;
    color: white;
    font-size: 16px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
  }
  
  .generate-btn:hover {
    background: #D32F2F;
  }
  
  .report-result {
    display: flex;
    flex-direction: column;
    gap: 20px;
  }
  
  .success {
    color: #4CAF50;
    font-size: 16px;
    margin: 0;
  }
  
  .report-actions {
    display: flex;
    gap: 10px;
  }
  
  .report-actions button {
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
  
  .report-actions button:hover {
    background: #444;
    border-color: #666;
  }
  
  .report-preview {
    background: #1a1a1a;
    border: 1px solid #333;
    border-radius: 6px;
    padding: 15px;
  }
  
  .report-preview h5 {
    margin: 0 0 10px 0;
    color: #888;
    font-size: 12px;
  }
  
  .report-preview pre {
    margin: 0;
    font-family: monospace;
    font-size: 12px;
    color: #aaa;
    white-space: pre-wrap;
    word-break: break-all;
  }
</style>