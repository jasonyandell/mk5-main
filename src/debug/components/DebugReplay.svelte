<script lang="ts">
  import { actionHistory, initialState, stateValidationError, gameState } from '../../stores/gameStore';
  import { getNextStates } from '../../game/core/actions';
  
  let expanded = $state(false);
  
  const phaseEmojis: Record<string, string> = {
    setup: '🎲',
    bidding: '💬',
    trump_selection: '🎯',
    playing: '🃏',
    scoring: '📊',
    game_end: '🏁'
  };
  
  function getPhaseFromAction(action: any, index: number): string {
    // For now, we'll infer phase from action type
    // This could be improved by tracking phase in the action itself
    if (action.id.startsWith('bid-') || action.id === 'pass' || action.id === 'redeal') return 'bidding';
    if (action.id.startsWith('trump-')) return 'trump_selection';
    if (action.id.startsWith('play-') || action.id === 'complete-trick') return 'playing';
    if (action.id === 'score-hand') return 'scoring';
    return 'playing'; // default
  }
  
  function handleEventClick(index: number) {
    // Time travel to the state at this point in history
    // index 0 is initial state, index 1 is after first action, etc.
    
    // Start from initial state
    let currentState = JSON.parse(JSON.stringify($initialState));
    
    // Apply actions up to the selected index
    for (let i = 0; i < index && i < $actionHistory.length; i++) {
      const action = $actionHistory[i];
      currentState = action.newState;
    }
    
    // Update the game state
    gameState.set(currentState);
    
    // Trim action history to this point
    actionHistory.set($actionHistory.slice(0, index));
  }
</script>

<div class="replay-container" data-testid={$actionHistory.length > 0 ? "debug-snapshot" : undefined}>
  <div class="replay-header">
    <h3>Action History</h3>
    <div class="snapshot-info">
      <span data-testid="snapshot-reason">All actions from initial state to current state</span>
      <span data-testid="snapshot-action-count">Actions: {$actionHistory.length}</span>
    </div>
  </div>
  
  {#if $stateValidationError}
    <div class="validation-error">
      <div class="error-header">⚠️ State Validation Error</div>
      <textarea 
        class="error-text" 
        readonly 
        rows="10"
        onclick={(e) => e.currentTarget.select()}
      >{$stateValidationError}</textarea>
      <div class="error-hint">Click to select all text for copying</div>
    </div>
  {/if}
  
  <div class="history-controls">
    <button 
      class="toggle-btn" 
      onclick={() => expanded = !expanded}
    >
      {expanded ? 'Hide' : 'Show'} Actions
    </button>
    {#if $actionHistory.length > 0}
      <button 
        class="validate-btn" 
        data-testid="validate-sequence-button"
        onclick={() => console.log('Validate sequence')}
      >
        Validate Sequence
      </button>
    {/if}
  </div>
  
  {#if expanded}
    <div class="action-log">
      <div class="log-entry">
        <div class="log-line">
          <span class="event-number" onclick={() => handleEventClick(0)} title="Click to restore game to initial state">00</span>
          <span class="phase-emoji">{phaseEmojis[$initialState.phase] || '🎲'}</span>
          <span class="action-id">initial-state</span>
        </div>
        <div class="log-line-detail">
          <span class="action-text">Initial State</span>
        </div>
      </div>
      
      {#each $actionHistory as action, index}
        {@const phase = getPhaseFromAction(action, index)}
        <div class="log-entry">
          <div class="log-line">
            <span class="event-number" onclick={() => handleEventClick(index + 1)} title="Click to time travel to this point">{String(index + 1).padStart(2, '0')}</span>
            <span class="phase-emoji">{phaseEmojis[phase] || '🎲'}</span>
            <span class="action-id">{action.id}</span>
          </div>
          <div class="log-line-detail">
            <span class="action-text">{action.label}</span>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .replay-container {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 12px;
    margin: 8px 0;
  }

  .replay-header {
    margin-bottom: 12px;
  }

  .replay-header h3 {
    margin: 0;
    font-size: 14px;
    color: #495057;
  }

  .snapshot-info {
    font-size: 11px;
    color: #6c757d;
    margin-top: 4px;
  }

  .snapshot-info span {
    margin-right: 12px;
  }


  .validation-error {
    background: #f8d7da;
    border: 1px solid #f5c6cb;
    border-radius: 4px;
    padding: 12px;
    margin-bottom: 12px;
  }

  .error-header {
    color: #721c24;
    font-weight: bold;
    font-size: 13px;
    margin-bottom: 8px;
  }

  .error-text {
    width: 100%;
    font-family: monospace;
    font-size: 11px;
    line-height: 1.4;
    background: white;
    border: 1px solid #f5c6cb;
    border-radius: 3px;
    padding: 8px;
    resize: vertical;
    cursor: text;
  }

  .error-hint {
    font-size: 10px;
    color: #721c24;
    margin-top: 4px;
    font-style: italic;
  }

  .history-controls {
    margin-bottom: 12px;
    display: flex;
    gap: 8px;
  }

  .toggle-btn,
  .validate-btn {
    background: #007bff;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 3px;
    font-size: 12px;
    cursor: pointer;
  }

  .toggle-btn:hover,
  .validate-btn:hover {
    background: #0056b3;
  }
  
  .validate-btn {
    background: #28a745;
  }
  
  .validate-btn:hover {
    background: #218838;
  }

  .action-log {
    background: #fafafa;
    border: 1px solid #e0e0e0;
    font-family: 'Courier New', 'Consolas', monospace;
    font-size: 9px;
    padding: 8px;
    max-height: 400px;
    overflow-y: auto;
    line-height: 1.6;
  }

  .action-log::-webkit-scrollbar {
    width: 10px;
  }

  .action-log::-webkit-scrollbar-track {
    background: #f0f0f0;
  }

  .action-log::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 5px;
  }

  .action-log::-webkit-scrollbar-thumb:hover {
    background: #666;
  }

  .log-entry {
    padding: 1px 4px;
    border-radius: 3px;
    transition: background-color 0.15s ease;
  }

  .log-entry:hover {
    background-color: #f0f0f0;
  }

  .log-line {
    display: flex;
    gap: 2px;
    align-items: baseline;
  }

  .log-line-detail {
    padding-left: 4px;
    color: #333;
  }

  .event-number {
    color: #0066cc;
    font-weight: bold;
    cursor: pointer;
    text-decoration: underline;
    min-width: 16px;
  }

  .event-number:hover {
    color: #0052a3;
    background: #e6f2ff;
  }

  .phase-emoji {
    font-size: 10px;
    min-width: 14px;
  }

  .action-id {
    color: #666;
  }

  .action-text {
    color: #333;
  }
</style>