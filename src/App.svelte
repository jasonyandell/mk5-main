<script lang="ts">
  import { onMount } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import PlayingArea from './lib/components/PlayingArea.svelte';
  import ActionPanel from './lib/components/ActionPanel.svelte';
  import DebugPanel from './lib/components/DebugPanel.svelte';
  import QuickplayError from './lib/components/QuickplayError.svelte';
  import { gameActions, gameState, startGameLoop, viewProjection } from './stores/gameStore';
  import { fly, fade } from 'svelte/transition';

  let showDebugPanel = $state(false);
  let activeView = $state<'game' | 'actions'>('game');
  let touchStartY = $state(0);
  let touchStartTime = $state(0);

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
    
    // Start the game loop for interactive play (not in test mode)
    const urlParams = new URLSearchParams(window.location.search);
    const testMode = urlParams.get('testMode') === 'true';
    if (!testMode) {
      startGameLoop();
    }
    
    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      document.documentElement.setAttribute('data-theme', savedTheme);
    }
  });
  
  // Smart panel switching based on game phase
  // This handles both URL loading and normal game flow
  $effect(() => {
    // Use the ViewProjection's computed activeView
    activeView = $viewProjection.ui.activeView;
  });
</script>

<svelte:window onkeydown={handleKeydown} />

<div 
  class="app-container flex flex-col h-screen bg-base-100 text-base-content font-sans overflow-hidden"
  style="height: 100dvh;"
  role="application" 
  data-phase={$gameState.phase} 
  ontouchstart={handleTouchStart} 
  ontouchend={handleTouchEnd}
>
  <Header on:openDebug={() => showDebugPanel = true} />
  
  <main class="flex-1 overflow-y-auto overflow-x-hidden touch-pan-y pb-safe relative {activeView === 'actions' ? 'overflow-hidden' : ''}">
    {#if activeView === 'game'}
      <div transition:fade={{ duration: 200 }}>
        <PlayingArea />
      </div>
    {:else}
      <div transition:fade={{ duration: 200 }} class="h-full flex flex-col overflow-hidden">
        <ActionPanel onswitchToPlay={() => activeView = 'game'} />
      </div>
    {/if}
  </main>
</div>

{#if showDebugPanel}
  <div transition:fly={{ y: 200, duration: 300 }}>
    <DebugPanel onclose={() => showDebugPanel = false} />
  </div>
{/if}

<QuickplayError />

