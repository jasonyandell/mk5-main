<script lang="ts">
  import { gameActions } from '../../stores/gameStore';
  import { quickplayActions, quickplayState } from '../../stores/quickplayStore';
  
  export let onOpenDebug: () => void;
  
  function reportBug() {
    const bugReport = {
      timestamp: new Date().toISOString(),
      // In a real app, this would capture more context
    };
    console.log('Bug Report:', bugReport);
    alert('Bug report copied to console');
  }
  
  function handleUndo() {
    gameActions.undo();
  }
  
  function toggleAutoPlay() {
    quickplayActions.toggle();
  }
</script>

<div class="quick-access-toolbar">
  <button 
    class="toolbar-button"
    on:click={reportBug}
    title="Report Bug"
    data-testid="quick-report-bug"
  >
    🐛
  </button>
  
  <div class="game-indicator" class:active={$quickplayState.enabled}>
    <span class="indicator-dot"></span>
    <span class="sr-only">
      {$quickplayState.enabled ? 'AI Active' : 'AI Paused'}
    </span>
  </div>
  
  <button 
    class="toolbar-button"
    on:click={handleUndo}
    title="Undo Last Action (Ctrl+Z)"
    data-testid="quick-undo"
  >
    ↩️
  </button>
  
  <button 
    class="toolbar-button"
    on:click={toggleAutoPlay}
    title="Toggle Auto-play"
    data-testid="quick-autoplay"
  >
    🤖
  </button>
  
  <div class="toolbar-separator"></div>
  
  <button 
    class="toolbar-button primary"
    on:click={onOpenDebug}
    title="Open Debug Panel (Ctrl+Shift+D)"
    data-testid="quick-open-debug"
  >
    🔧
  </button>
</div>

<style>
  .quick-access-toolbar {
    position: fixed;
    bottom: 20px;
    right: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 100;
  }
  
  .toolbar-button {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    font-size: 18px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .toolbar-button:hover {
    background-color: #e5e7eb;
    transform: translateY(-1px);
  }
  
  .toolbar-button.primary {
    background-color: #3b82f6;
    color: white;
    border-color: #3b82f6;
  }
  
  .toolbar-button.primary:hover {
    background-color: #2563eb;
  }
  
  .game-indicator {
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .indicator-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background-color: #9ca3af;
    transition: background-color 0.2s ease;
  }
  
  .game-indicator.active .indicator-dot {
    background-color: #22c55e;
    animation: pulse 1s infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
  
  .toolbar-separator {
    width: 1px;
    height: 24px;
    background-color: #e5e7eb;
    margin: 0 4px;
  }
  
  @media (max-width: 768px) {
    .quick-access-toolbar {
      bottom: 10px;
      right: 10px;
      padding: 6px;
      gap: 6px;
    }
    
    .toolbar-button {
      width: 32px;
      height: 32px;
      font-size: 16px;
    }
  }
</style>