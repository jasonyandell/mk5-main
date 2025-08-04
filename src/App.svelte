<script lang="ts">
  import { onMount } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import GameProgress from './lib/components/GameProgress.svelte';
  import PlayingArea from './lib/components/PlayingArea.svelte';
  import ActionPanel from './lib/components/ActionPanel.svelte';
  import DebugPanel from './lib/components/DebugPanel.svelte';
  import QuickAccessToolbar from './lib/components/QuickAccessToolbar.svelte';
  import { gameActions } from './stores/gameStore';

  let showDebugPanel = false;

  // Handle keyboard shortcuts
  function handleKeydown(e: KeyboardEvent) {
    if (e.ctrlKey && e.shiftKey && e.key === 'D') {
      e.preventDefault();
      showDebugPanel = !showDebugPanel;
    } else if (e.key === 'Escape' && showDebugPanel) {
      showDebugPanel = false;
    } else if (e.ctrlKey && e.key === 'z' && !showDebugPanel) {
      e.preventDefault();
      gameActions.undo();
    }
  }

  onMount(() => {
    // Try to load from URL on mount
    gameActions.loadFromURL();
  });
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="app-container">
  <Header />
  
  <div class="main-content">
    <aside class="left-panel">
      <GameProgress />
    </aside>
    
    <main class="center-panel">
      <PlayingArea />
    </main>
    
    <aside class="right-panel">
      <ActionPanel />
    </aside>
  </div>
  
  <footer class="app-footer">
    <span>Debug Mode • Ctrl+Shift+D for advanced debugging • v3.0</span>
  </footer>
</div>

{#if showDebugPanel}
  <DebugPanel on:close={() => showDebugPanel = false} />
{:else}
  <QuickAccessToolbar onOpenDebug={() => showDebugPanel = true} />
{/if}

<style>
  .app-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background-color: #ffffff;
    color: #002868;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }

  .main-content {
    flex: 1;
    display: flex;
    overflow: hidden;
  }

  .left-panel {
    width: 300px;
    background-color: #f9fafb;
    border-right: 1px solid #e5e7eb;
    overflow-y: auto;
  }

  .center-panel {
    flex: 1;
    padding: 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }

  .right-panel {
    width: 200px;
    background-color: #f9fafb;
    border-left: 1px solid #e5e7eb;
    overflow-y: auto;
  }

  .app-footer {
    padding: 8px 20px;
    background-color: #f3f4f6;
    border-top: 1px solid #e5e7eb;
    text-align: center;
    font-size: 12px;
    color: #6b7280;
  }

  /* Mobile responsive */
  @media (max-width: 768px) {
    .main-content {
      flex-direction: column;
    }

    .left-panel,
    .right-panel {
      width: 100%;
      border: none;
      border-bottom: 1px solid #e5e7eb;
    }
  }
</style>