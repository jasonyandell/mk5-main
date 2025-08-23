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
  <div class="fixed top-5 right-5 bg-red-50 border-2 border-red-300 rounded-lg p-0 max-w-[400px] shadow-lg z-[1000]">
    <div class="flex items-center gap-2 px-4 py-3 bg-red-100 border-b border-red-300 rounded-t-md">
      <span class="text-xl">⚠️</span>
      <span class="flex-1 font-semibold text-red-700">Quickplay Error</span>
      <button class="bg-transparent border-none text-2xl cursor-pointer p-0 w-6 h-6 flex items-center justify-center text-red-700 opacity-70 hover:opacity-100" on:click={clearError}>×</button>
    </div>
    <div class="p-4">
      <p class="mb-3 text-red-700 font-medium">{error.message}</p>
      <div class="flex gap-4 text-sm text-gray-600 mb-3">
        <span>Phase: {error.state.phase}</span>
        <span>Player: {error.state.currentPlayer}</span>
        <span>Actions: {error.availableActions.length}</span>
      </div>
      <button class="bg-red-400 text-white border-none px-3 py-2 rounded cursor-pointer text-sm hover:bg-red-500" on:click={copyErrorDetails}>
        Copy Error Details
      </button>
    </div>
  </div>
{/if}