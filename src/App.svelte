<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import PlayingArea from './lib/components/PlayingArea.svelte';
  import ActionPanel from './lib/components/ActionPanel.svelte';
  import DebugPanel from './lib/components/DebugPanel.svelte';
  import QuickplayError from './lib/components/QuickplayError.svelte';
  import { gameActions, gamePhase, gameState, availableActions } from './stores/gameStore';
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
    if (e.touches.length > 0 && e.touches[0]) {
      touchStartY = e.touches[0].clientY;
      touchStartTime = Date.now();
    }
  }

  function handleTouchEnd(e: TouchEvent) {
    if (e.changedTouches.length > 0 && e.changedTouches[0]) {
      const touchEndY = e.changedTouches[0].clientY;
      const touchDuration = Date.now() - touchStartTime;
      const swipeDistance = touchStartY - touchEndY;

      // Quick swipe up opens debug panel
      if (swipeDistance > 100 && touchDuration < 300) {
        showDebugPanel = true;
      }
    }
  }

  onMount(() => {
    // Try to load from URL on mount
    gameActions.loadFromURL();
  });
  
  // Smart panel switching based on game phase
  // This handles both URL loading and normal game flow
  $: {
    // Automatically switch to appropriate panel based on game phase
    // This works for both:
    // 1. Loading from a URL with any game state
    // 2. Natural game progression
    
    if ($gamePhase === 'bidding' || $gamePhase === 'trump_selection') {
      // These phases need the Actions panel for decision making
      activeView = 'actions';
    } else if ($gamePhase === 'playing' || $gamePhase === 'setup') {
      // Playing phase and setup should show the game board
      activeView = 'game';
    } else if ($gamePhase === 'scoring') {
      // Scoring phase: stay on game view to see the "Score hand" button
      activeView = 'game';
    } else if ($gamePhase === 'game_end') {
      // Game end: could show either, let's show game board with final state
      activeView = 'game';
    }
  }
</script>

<svelte:window on:keydown={handleKeydown} />

<div class="app-container" role="application" data-phase={$gameState.phase} on:touchstart={handleTouchStart} on:touchend={handleTouchEnd}>
  <Header on:openDebug={() => showDebugPanel = true} />
  
  <main class="game-container" class:no-scroll={activeView === 'actions'}>
    {#if activeView === 'game'}
      <div transition:fade={{ duration: 200 }}>
        <PlayingArea on:switchToActions={() => activeView = 'actions'} />
      </div>
    {:else}
      <div transition:fade={{ duration: 200 }} class="action-panel-wrapper">
        <ActionPanel onswitchToPlay={() => activeView = 'game'} />
      </div>
    {/if}
  </main>
</div>

{#if showDebugPanel}
  <div transition:fly={{ y: 200, duration: 300 }}>
    <DebugPanel on:close={() => showDebugPanel = false} />
  </div>
{/if}

<QuickplayError />

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
</style>