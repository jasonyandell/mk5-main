<script lang="ts">
  import { debugSnapshot } from '../../stores/gameStore';
  import { getNextStates } from '../../game';
  import type { StateTransition } from '../../game/types';

  let replayState = $state<'idle' | 'replaying' | 'error'>('idle');
  let currentStep = $state(0);
  let errorMessage = $state('');
  let validationResults = $state<Array<{step: number, action: any, valid: boolean, error?: string}>>([]);

  function validateActionSequence() {
    if (!$debugSnapshot) {
      errorMessage = 'No debug snapshot available';
      replayState = 'error';
      return;
    }

    replayState = 'replaying';
    validationResults = [];
    errorMessage = '';
    
    let currentState = $debugSnapshot.baseState;
    
    for (let i = 0; i < $debugSnapshot.actions.length; i++) {
      const action = $debugSnapshot.actions[i];
      
      // Get available actions for current state
      const availableActions = getNextStates(currentState);
      
      // Check if the action is valid
      const validAction = availableActions.find(a => a.id === action.id);
      
      if (!validAction) {
        const currentPlayerName = `Player ${currentState.currentPlayer + 1}`;
        const phaseInfo = currentState.phase === 'playing' ? 
          ` (Trump: ${currentState.trump === null ? 'None' : currentState.trump === 7 ? 'Doubles' : currentState.trump}, Current trick: ${currentState.currentTrick.length}/4)` : 
          ` (Phase: ${currentState.phase})`;
        
        validationResults.push({
          step: i + 1,
          action: { id: action.id, label: action.label },
          valid: false,
          error: `${currentPlayerName} cannot perform action ${action.id} (${action.label})${phaseInfo}. Available actions: ${availableActions.map(a => a.id).join(', ')}`
        });
        replayState = 'error';
        errorMessage = `Invalid action at step ${i + 1}: ${currentPlayerName} cannot ${action.label.toLowerCase()}`;
        return;
      }
      
      validationResults.push({
        step: i + 1,
        action: { id: action.id, label: action.label },
        valid: true
      });
      
      // Update state for next iteration
      currentState = validAction.newState;
    }
    
    replayState = 'idle';
  }

  function resetValidation() {
    replayState = 'idle';
    currentStep = 0;
    errorMessage = '';
    validationResults = [];
  }
</script>

{#if $debugSnapshot}
  <div class="replay-container" data-testid="debug-snapshot">
    <div class="replay-header">
      <h3>Debug Replay Validation</h3>
      <div class="snapshot-info">
        <span class="reason" data-testid="snapshot-reason">From initial state</span>
        <span class="action-count" data-testid="snapshot-action-count">{$debugSnapshot.actions.length} actions</span>
      </div>
    </div>
    
    <div class="replay-controls">
      <button 
        class="validate-btn" 
        data-testid="validate-sequence-button"
        onclick={validateActionSequence}
        disabled={replayState === 'replaying'}
      >
        {replayState === 'replaying' ? 'Validating...' : 'Validate Action Sequence'}
      </button>
      
      {#if validationResults.length > 0}
        <button class="reset-btn" onclick={resetValidation}>
          Reset
        </button>
      {/if}
    </div>

    {#if errorMessage}
      <div class="error-message">
        ⚠️ {errorMessage}
      </div>
    {/if}

    {#if validationResults.length > 0}
      <div class="validation-results">
        <h4>Validation Results:</h4>
        <div class="results-list">
          {#each validationResults as result}
            <div class="result-item" class:valid={result.valid} class:invalid={!result.valid}>
              <div class="step-number">Step {result.step}</div>
              <div class="action-info">
                <span class="action-id">{result.action.id}</span>
                <span class="action-label">{result.action.label}</span>
              </div>
              <div class="status">
                {result.valid ? '✅' : '❌'}
              </div>
              {#if result.error}
                <div class="error-details" data-testid="validation-error">{result.error}</div>
              {/if}
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>
{:else}
  <div class="no-snapshot">
    <p>No debug snapshot available. Take some actions to generate one.</p>
  </div>
{/if}

<style>
  .replay-container {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 4px;
    padding: 12px;
    margin: 8px 0;
  }

  .replay-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .replay-header h3 {
    margin: 0;
    font-size: 14px;
    color: #495057;
  }

  .snapshot-info {
    display: flex;
    gap: 8px;
    font-size: 11px;
  }

  .reason {
    background: #007bff;
    color: white;
    padding: 2px 6px;
    border-radius: 2px;
  }

  .action-count {
    background: #6c757d;
    color: white;
    padding: 2px 6px;
    border-radius: 2px;
  }

  .replay-controls {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
  }

  .validate-btn {
    background: #28a745;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 3px;
    font-size: 12px;
    cursor: pointer;
  }

  .validate-btn:disabled {
    background: #6c757d;
    cursor: not-allowed;
  }

  .reset-btn {
    background: #6c757d;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 3px;
    font-size: 12px;
    cursor: pointer;
  }

  .error-message {
    background: #f8d7da;
    color: #721c24;
    padding: 8px;
    border-radius: 3px;
    font-size: 12px;
    margin-bottom: 12px;
  }

  .validation-results h4 {
    margin: 0 0 8px 0;
    font-size: 13px;
    color: #495057;
  }

  .results-list {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .result-item {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 8px;
    align-items: center;
    padding: 6px;
    border-radius: 3px;
    font-size: 11px;
  }

  .result-item.valid {
    background: #d4edda;
    border-left: 3px solid #28a745;
  }

  .result-item.invalid {
    background: #f8d7da;
    border-left: 3px solid #dc3545;
  }

  .step-number {
    font-weight: bold;
    color: #495057;
  }

  .action-info {
    display: flex;
    gap: 8px;
  }

  .action-id {
    font-family: monospace;
    background: rgba(0,0,0,0.1);
    padding: 1px 4px;
    border-radius: 2px;
  }

  .action-label {
    color: #6c757d;
  }

  .status {
    font-size: 14px;
  }

  .error-details {
    grid-column: 1 / -1;
    font-size: 10px;
    color: #721c24;
    background: rgba(220, 53, 69, 0.1);
    padding: 4px;
    border-radius: 2px;
    margin-top: 4px;
  }

  .no-snapshot {
    padding: 16px;
    text-align: center;
    color: #6c757d;
    font-size: 12px;
  }
</style>