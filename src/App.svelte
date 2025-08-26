<script lang="ts">
  import { onMount } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import PlayingArea from './lib/components/PlayingArea.svelte';
  import ActionPanel from './lib/components/ActionPanel.svelte';
  import SettingsPanel from './lib/components/SettingsPanel.svelte';
  import QuickplayError from './lib/components/QuickplayError.svelte';
  import { gameActions, gameState, startGameLoop, viewProjection } from './stores/gameStore';
  import { fly, fade } from 'svelte/transition';

  let showSettingsPanel = $state(false);
  let activeView = $state<'game' | 'actions'>('game');

  // Handle keyboard shortcuts
  function handleKeydown(e: KeyboardEvent) {
    if (e.ctrlKey && e.shiftKey && e.key === 'D') {
      e.preventDefault();
      showSettingsPanel = !showSettingsPanel;
    } else if (e.key === 'Escape' && showSettingsPanel) {
      showSettingsPanel = false;
    } else if (e.ctrlKey && e.key === 'z' && !showSettingsPanel) {
      e.preventDefault();
      gameActions.undo();
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
    
    // Set default theme to coffee (no persistence)
    document.documentElement.setAttribute('data-theme', 'coffee');
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
>
  <Header on:openSettings={() => showSettingsPanel = true} />
  
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

{#if showSettingsPanel}
  <div transition:fly={{ y: 200, duration: 300 }}>
    <SettingsPanel onclose={() => showSettingsPanel = false} />
  </div>
{/if}

<QuickplayError />

