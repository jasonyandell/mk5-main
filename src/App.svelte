<script lang="ts">
  import { onMount } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import GameProgress from './lib/components/GameProgress.svelte';
  import PlayingArea from './lib/components/PlayingArea.svelte';
  import ActionPanel from './lib/components/ActionPanel.svelte';
  import DebugPanel from './lib/components/DebugPanel.svelte';
  import { gameActions, gamePhase } from './stores/gameStore';
  import { fly, fade } from 'svelte/transition';

  let showDebugPanel = false;
  let activeView: 'game' | 'actions' = 'game';
  let touchStartY = 0;
  let touchStartTime = 0;

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

  // Handle swipe gestures
  function handleTouchStart(e: TouchEvent) {
    touchStartY = e.touches[0].clientY;
    touchStartTime = Date.now();
  }

  function handleTouchEnd(e: TouchEvent) {
    const touchEndY = e.changedTouches[0].clientY;
    const touchDuration = Date.now() - touchStartTime;
    const swipeDistance = touchStartY - touchEndY;

    // Quick swipe up opens debug panel
    if (swipeDistance > 100 && touchDuration < 300) {
      showDebugPanel = true;
    }
  }

  onMount(() => {
    // Try to load from URL on mount
    gameActions.loadFromURL();
  });
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="app-container" on:touchstart={handleTouchStart} on:touchend={handleTouchEnd}>
  <Header />
  
  <main class="game-container" class:no-scroll={activeView === 'actions'}>
    {#if activeView === 'game'}
      <div transition:fade={{ duration: 200 }}>
        <PlayingArea />
      </div>
    {:else}
      <div transition:fade={{ duration: 200 }} class="action-panel-wrapper">
        <ActionPanel />
      </div>
    {/if}
  </main>
  
  <nav class="bottom-nav">
    <div class="nav-indicator" style="transform: translateX({activeView === 'game' ? '0' : '50%'})"></div>
    <button 
      class="nav-button"
      class:active={activeView === 'game'}
      on:click={() => activeView = 'game'}
      data-testid="nav-game"
    >
      <span class="nav-icon">🎯</span>
      <span class="nav-label">Play</span>
    </button>
    <button 
      class="nav-button"
      class:active={activeView === 'actions'}
      on:click={() => activeView = 'actions'}
      data-testid="nav-actions"
    >
      <span class="nav-icon">🎲</span>
      <span class="nav-label">Actions</span>
    </button>
    <button 
      class="nav-button debug"
      on:click={() => showDebugPanel = true}
      data-testid="nav-debug"
    >
      <span class="nav-icon">🔧</span>
      <span class="nav-label">Debug</span>
    </button>
  </nav>
</div>

{#if showDebugPanel}
  <div transition:fly={{ y: 200, duration: 300 }}>
    <DebugPanel on:close={() => showDebugPanel = false} />
  </div>
{/if}

<style>
  .app-container {
    display: flex;
    flex-direction: column;
    height: 100vh;
    height: 100dvh; /* Dynamic viewport height for mobile */
    background: linear-gradient(to bottom, #f8fafc, #e2e8f0);
    color: #1e293b;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    overflow: hidden;
  }

  .game-container {
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    -webkit-overflow-scrolling: touch; /* Smooth scrolling on iOS */
    padding-bottom: env(safe-area-inset-bottom); /* Account for iPhone notch */
    position: relative;
  }
  
  .action-panel-wrapper {
    height: 100%;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  
  .game-container.no-scroll {
    overflow: hidden;
  }

  .bottom-nav {
    position: relative;
    display: flex;
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-top: 1px solid rgba(229, 231, 235, 0.5);
    padding-bottom: env(safe-area-inset-bottom); /* Safe area for iPhone */
    box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.05);
  }

  .nav-indicator {
    position: absolute;
    bottom: 0;
    left: 0;
    width: 50%;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    border-radius: 3px 3px 0 0;
  }

  .nav-button {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 10px 8px;
    border: none;
    background: none;
    color: #64748b;
    cursor: pointer;
    transition: all 0.2s;
    -webkit-tap-highlight-color: transparent; /* Remove tap highlight on mobile */
    min-height: 48px; /* Touch target size */
    position: relative;
    z-index: 1;
  }

  .nav-button:active {
    transform: scale(0.92);
  }

  .nav-button.active {
    color: #3b82f6;
  }

  .nav-button.active .nav-icon {
    transform: translateY(-2px) scale(1.15);
  }

  .nav-button.debug {
    color: #8b5cf6;
  }

  .nav-icon {
    font-size: 22px;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }

  .nav-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.025em;
  }

</style>