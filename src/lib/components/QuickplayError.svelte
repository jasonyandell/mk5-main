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
  <div class="fixed top-5 right-5 bg-error/10 border-2 border-error rounded-lg p-0 max-w-[400px] shadow-lg z-[1000]">
    <div class="flex items-center gap-2 px-4 py-3 bg-error/20 border-b border-error rounded-t-md">
      <span class="text-xl">⚠️</span>
      <span class="flex-1 font-semibold text-error">Quickplay Error</span>
      <button class="btn btn-ghost btn-sm btn-circle text-error" on:click={clearError}>×</button>
    </div>
    <div class="p-4">
      <p class="mb-3 text-error font-medium">{error.message}</p>
      <div class="flex gap-4 text-sm text-base-content/60 mb-3">
        <span>Phase: {error.state.phase}</span>
        <span>Player: {error.state.currentPlayer}</span>
        <span>Actions: {error.availableActions.length}</span>
      </div>
      <button class="btn btn-error btn-sm" on:click={copyErrorDetails}>
        Copy Error Details
      </button>
    </div>
  </div>
{/if}