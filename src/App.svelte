<script lang="ts">
  import { onMount } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import GameProgress from './lib/components/GameProgress.svelte';
  import PlayingArea from './lib/components/PlayingArea.svelte';
  import ActionPanel from './lib/components/ActionPanel.svelte';
  import DebugPanel from './lib/components/DebugPanel.svelte';
  import { gameActions } from './stores/gameStore';

  let showDebugPanel = false;
  let activeView: 'game' | 'actions' = 'game';

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
  
  <main class="game-container">
    {#if activeView === 'game'}
      <PlayingArea />
    {:else}
      <ActionPanel />
    {/if}
  </main>
  
  <nav class="bottom-nav">
    <button 
      class="nav-button"
      class:active={activeView === 'game'}
      on:click={() => activeView = 'game'}
    >
      <span class="nav-icon">🎯</span>
      <span class="nav-label">Play</span>
    </button>
    <button 
      class="nav-button"
      class:active={activeView === 'actions'}
      on:click={() => activeView = 'actions'}
    >
      <span class="nav-icon">🎲</span>
      <span class="nav-label">Actions</span>
    </button>
    <button 
      class="nav-button debug"
      on:click={() => showDebugPanel = true}
    >
      <span class="nav-icon">🔧</span>
      <span class="nav-label">Debug</span>
    </button>
  </nav>
</div>

{#if showDebugPanel}
  <DebugPanel on:close={() => showDebugPanel = false} />
{/if}

<style>
  .app-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    height: 100dvh; /* Dynamic viewport height for mobile */
    background-color: #ffffff;
    color: #002868;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    overflow: hidden;
  }

  .game-container {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    -webkit-overflow-scrolling: touch; /* Smooth scrolling on iOS */
    padding-bottom: env(safe-area-inset-bottom); /* Account for iPhone notch */
  }

  .bottom-nav {
    display: flex;
    background-color: #ffffff;
    border-top: 1px solid #e5e7eb;
    padding-bottom: env(safe-area-inset-bottom); /* Safe area for iPhone */
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
  }

  .nav-button {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 8px;
    border: none;
    background: none;
    color: #6b7280;
    cursor: pointer;
    transition: all 0.2s;
    -webkit-tap-highlight-color: transparent; /* Remove tap highlight on mobile */
    min-height: 44px; /* Touch target size */
  }

  .nav-button:active {
    transform: scale(0.95);
  }

  .nav-button.active {
    color: #002868;
  }

  .nav-button.active .nav-icon {
    transform: scale(1.1);
  }

  .nav-button.debug {
    color: #8b5cf6;
  }

  .nav-icon {
    font-size: 24px;
    transition: transform 0.2s;
  }

  .nav-label {
    font-size: 11px;
    font-weight: 500;
  }

</style>