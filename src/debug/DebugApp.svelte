<script lang="ts">
  import { gameState } from '../ui/stores/gameStore';
  import StateDisplay from './StateDisplay.svelte';
  import TestGenerator from './TestGenerator.svelte';
  
  let showDebug = false;
  
  function toggleDebug() {
    showDebug = !showDebug;
  }
</script>

{#if showDebug}
  <div class="debug-overlay">
    <div class="debug-panel">
      <div class="debug-header">
        <h2>Debug Panel</h2>
        <button class="close-btn" onclick={toggleDebug}>×</button>
      </div>
      
      <div class="debug-content">
        <StateDisplay state={$gameState} />
        <TestGenerator />
      </div>
    </div>
  </div>
{/if}

<button class="debug-toggle" class:active={showDebug} onclick={toggleDebug} data-testid="debug-toggle">
  🐛
</button>

<style>
  .debug-toggle {
    position: fixed;
    top: 20px;
    right: 20px;
    width: 50px;
    height: 50px;
    border-radius: 50%;
    background: #ff6b6b;
    color: white;
    font-size: 20px;
    border: none;
    cursor: pointer;
    z-index: 1001;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
  }
  
  .debug-toggle:hover {
    transform: scale(1.1);
    background: #ff5252;
  }
  
  .debug-toggle.active {
    background: #4caf50;
  }
  
  .debug-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.8);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .debug-panel {
    background: white;
    border-radius: 8px;
    width: 90%;
    max-width: 800px;
    height: 80%;
    display: flex;
    flex-direction: column;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
  }
  
  .debug-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 20px;
    border-bottom: 1px solid #eee;
  }
  
  .debug-header h2 {
    margin: 0;
    color: #333;
  }
  
  .close-btn {
    background: none;
    border: none;
    font-size: 24px;
    cursor: pointer;
    color: #666;
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .close-btn:hover {
    color: #333;
    background: #f5f5f5;
    border-radius: 50%;
  }
  
  .debug-content {
    flex: 1;
    overflow: auto;
    padding: 20px;
  }
</style>