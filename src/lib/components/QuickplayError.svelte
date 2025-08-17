<script lang="ts">
  import { quickplayErrorStore } from '../../stores/quickplayStore';
  
  $: error = $quickplayErrorStore;
  
  function clearError() {
    quickplayErrorStore.set(null);
  }
  
  function copyErrorDetails() {
    if (error) {
      const details = JSON.stringify({
        message: error.message,
        timestamp: error.timestamp,
        availableActions: error.availableActions,
        phase: error.state.phase,
        currentPlayer: error.state.currentPlayer,
        trickCount: error.state.tricks.length
      }, null, 2);
      
      navigator.clipboard.writeText(details);
    }
  }
</script>

{#if error}
  <div class="quickplay-error">
    <div class="error-header">
      <span class="error-icon">⚠️</span>
      <span class="error-title">Quickplay Error</span>
      <button class="error-close" on:click={clearError}>×</button>
    </div>
    <div class="error-body">
      <p class="error-message">{error.message}</p>
      <div class="error-details">
        <span>Phase: {error.state.phase}</span>
        <span>Player: {error.state.currentPlayer}</span>
        <span>Actions: {error.availableActions.length}</span>
      </div>
      <button class="copy-button" on:click={copyErrorDetails}>
        Copy Error Details
      </button>
    </div>
  </div>
{/if}

<style>
  .quickplay-error {
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--error-bg, #fee);
    border: 2px solid var(--error-border, #f88);
    border-radius: 8px;
    padding: 0;
    max-width: 400px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 1000;
  }
  
  .error-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: var(--error-header-bg, #fdd);
    border-bottom: 1px solid var(--error-border, #f88);
    border-radius: 6px 6px 0 0;
  }
  
  .error-icon {
    font-size: 1.2em;
  }
  
  .error-title {
    flex: 1;
    font-weight: 600;
    color: var(--error-text, #c00);
  }
  
  .error-close {
    background: none;
    border: none;
    font-size: 1.5em;
    cursor: pointer;
    padding: 0;
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--error-text, #c00);
    opacity: 0.7;
  }
  
  .error-close:hover {
    opacity: 1;
  }
  
  .error-body {
    padding: 16px;
  }
  
  .error-message {
    margin: 0 0 12px 0;
    color: var(--error-text, #c00);
    font-weight: 500;
  }
  
  .error-details {
    display: flex;
    gap: 16px;
    font-size: 0.9em;
    color: var(--error-detail-text, #666);
    margin-bottom: 12px;
  }
  
  .copy-button {
    background: var(--error-button-bg, #f88);
    color: white;
    border: none;
    padding: 8px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9em;
  }
  
  .copy-button:hover {
    background: var(--error-button-hover, #e77);
  }
</style>