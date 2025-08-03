<script lang="ts">
  import type { GameState } from '../../game/types';
  
  interface Props {
    gameState: GameState;
    generateBugReport: () => string;
    showBugPanel: boolean;
  }
  
  let { gameState, generateBugReport, showBugPanel = $bindable() }: Props = $props();
  
  let generatedTest = $state('');
  
  function copyBugReportToClipboard() {
    const testCode = generateBugReport();
    generatedTest = testCode;
    showBugPanel = true;
  }
  
  function copyTestCodeToClipboard() {
    navigator.clipboard
      .writeText(generatedTest)
      .then(() => {
        // Could show a brief visual feedback here
      })
      .catch((err) => {
        console.error('Failed to copy test code to clipboard:', err);
      });
  }
  
  function copyStateUrl() {
    navigator.clipboard
      .writeText(stateUrl)
      .then(() => {
        // Could show a brief visual feedback here
      })
      .catch((err) => {
        console.error('Failed to copy state URL to clipboard:', err);
      });
  }
  
  function toggleBugPanel() {
    showBugPanel = !showBugPanel;
  }
  
  const stateUrl = $derived.by(() => {
    if (typeof window !== 'undefined') {
      try {
        const stateParam = encodeURIComponent(JSON.stringify(gameState));
        return `${window.location.origin}${window.location.pathname}?state=${stateParam}`;
      } catch (e) {
        console.error('Error generating state URL:', e);
        return `${window.location.origin}${window.location.pathname}`;
      }
    }
    return '';
  });
</script>

<div class="bug-report-panel">
  <div class="bug-controls">
    <button class="bug-button" onclick={copyBugReportToClipboard} data-testid="bug-report-button">
      🐛 Generate Bug Report Test
    </button>
    <button onclick={toggleBugPanel} data-testid="toggle-bug-panel" class="toggle-button">
      {showBugPanel ? 'Hide' : 'Show'} Bug Panel
    </button>
  </div>

  {#if showBugPanel}
    <div class="bug-content" data-testid="bug-content">
      <div class="section-header">
        <h3>Generated Test Code</h3>
        {#if generatedTest}
          <button class="copy-button" onclick={copyTestCodeToClipboard} data-testid="copy-test-code">
            📋 Copy Test
          </button>
        {/if}
      </div>
      <textarea
        readonly
        value={generatedTest}
        placeholder="Click 'Generate Bug Report Test' to create a test with the current state"
        data-testid="generated-test-code"
        class="test-code-area"
      ></textarea>

      <div class="bug-info">
        <div class="section-header">
          <p><strong>Current State URL:</strong></p>
          <button class="copy-button" onclick={copyStateUrl} data-testid="copy-state-url">
            📋 Copy URL
          </button>
        </div>
        <input readonly value={stateUrl} data-testid="state-url" class="state-url-input" />

        <div class="state-summary">
          <p><strong>Quick State Summary:</strong></p>
          <ul>
            <li>Phase: {gameState.phase}</li>
            <li>Current Player: P{gameState.currentPlayer}</li>
            <li>Dealer: P{gameState.dealer}</li>
            <li>Team Marks: {gameState.teamMarks[0]} - {gameState.teamMarks[1]}</li>
            <li>Team Scores: {gameState.teamScores[0]} - {gameState.teamScores[1]}</li>
            <li>Bids: {gameState.bids.length}</li>
            <li>Tricks Completed: {gameState.tricks.length}/7</li>
            <li>Current Trick: {gameState.currentTrick.length}/4</li>
            {#if gameState.currentBid}
              <li>Current Bid: {gameState.currentBid.value} by P{gameState.currentBid.player}</li>
            {/if}
            {#if gameState.trump && gameState.trump.type !== 'none'}
              <li>Trump: {
                gameState.trump.type === 'suit' ? 
                  ['0s', '1s', '2s', '3s', '4s', '5s', '6s'][gameState.trump.suit!] :
                  gameState.trump.type === 'doubles' ? 'Doubles' :
                  gameState.trump.type === 'no-trump' ? 'Follow-me' : 'Unknown'
              }</li>
            {/if}
            {#if gameState.winningBidder !== -1}
              <li>Winning Bidder: P{gameState.winningBidder}</li>
            {/if}
          </ul>
        </div>

      </div>
    </div>
  {/if}
</div>

<style>
  .bug-report-panel {
    border: 2px solid #dc3545;
    margin-bottom: 8px;
    background: #1a0a0a;
    border-radius: 4px;
  }

  .bug-controls {
    display: flex;
    gap: 8px;
    padding: 8px;
    background: #2a1a1a;
  }

  .bug-button {
    background: #dc3545;
    color: #fff;
    border: 1px solid #dc3545;
    font-weight: bold;
    padding: 6px 12px;
    cursor: pointer;
    font-family: monospace;
    font-size: 12px;
    border-radius: 3px;
    transition: background-color 0.1s, color 0.1s;
  }

  .bug-button:hover {
    background: #fff;
    color: #dc3545;
  }

  .toggle-button {
    background: #000;
    color: #dc3545;
    border: 1px solid #dc3545;
    padding: 6px 12px;
    cursor: pointer;
    font-family: monospace;
    font-size: 12px;
    border-radius: 3px;
    transition: background-color 0.1s, color 0.1s;
  }

  .toggle-button:hover {
    background: #dc3545;
    color: #000;
  }

  .bug-content {
    padding: 8px;
    border-top: 1px solid #dc3545;
  }

  .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
  }
  
  .bug-content h3 {
    color: #dc3545;
    margin: 0;
    font-size: 14px;
  }
  
  .copy-button {
    background: #28a745;
    color: #fff;
    border: 1px solid #28a745;
    padding: 4px 8px;
    cursor: pointer;
    font-family: monospace;
    font-size: 10px;
    border-radius: 3px;
    transition: background-color 0.1s;
  }
  
  .copy-button:hover {
    background: #218838;
  }

  .test-code-area {
    width: 100%;
    height: 200px;
    background: #000;
    color: #fff;
    border: 1px solid #dc3545;
    padding: 8px;
    font-family: monospace;
    font-size: 10px;
    resize: vertical;
    margin-bottom: 8px;
    border-radius: 3px;
  }

  .bug-info {
    margin-top: 8px;
  }

  .bug-info p {
    color: #dc3545;
    margin: 0;
    font-weight: bold;
    font-size: 12px;
  }

  .state-url-input {
    width: 100%;
    background: #000;
    color: #fff;
    border: 1px solid #dc3545;
    padding: 4px;
    font-family: monospace;
    font-size: 10px;
    margin-bottom: 8px;
    border-radius: 3px;
  }

  .state-summary {
    margin-top: 8px;
    padding: 8px;
    border: 1px solid #dc3545;
    background: #111;
    border-radius: 3px;
  }

  .state-summary ul {
    margin: 4px 0;
    padding-left: 16px;
    color: #fff;
  }

  .state-summary li {
    margin: 2px 0;
    font-size: 11px;
  }

</style>