<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import PlayingArea from './lib/components/PlayingArea.svelte';
  import ActionPanel from './lib/components/ActionPanel.svelte';
  import DebugPanel from './lib/components/DebugPanel.svelte';
  import QuickplayError from './lib/components/QuickplayError.svelte';
  import { gameActions, gamePhase, gameState, availableActions } from './stores/gameStore';
  import { fly, fade } from 'svelte/transition';

  let showDebugPanel = $state(false);
  let activeView = $state<'game' | 'actions'>('game');
  let touchStartY = $state(0);
  let touchStartTime = $state(0);
  let currentTheme = $state('light');

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
    
    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      currentTheme = savedTheme;
      document.documentElement.setAttribute('data-theme', savedTheme);
    }
  });
  
  // Smart panel switching based on game phase
  // This handles both URL loading and normal game flow
  $effect(() => {
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
  });
</script>

<svelte:window onkeydown={handleKeydown} />

<div 
  class="flex flex-col h-screen bg-base-100 text-base-content font-sans overflow-hidden"
  style="height: 100dvh;"
  role="application" 
  data-phase={$gameState.phase} 
  ontouchstart={handleTouchStart} 
  ontouchend={handleTouchEnd}
>
  <Header onopenDebug={() => showDebugPanel = true} />
  
  <!-- Theme Switcher -->
  <div class="fixed bottom-4 left-4 z-50">
    <select 
      class="select select-bordered select-sm" 
      data-choose-theme
      bind:value={currentTheme}
      onchange={(e) => {
        const theme = e.currentTarget.value;
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
      }}>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
      <option value="cupcake">Cupcake</option>
      <option value="bumblebee">Bumblebee</option>
      <option value="emerald">Emerald</option>
      <option value="corporate">Corporate</option>
      <option value="retro">Retro</option>
      <option value="cyberpunk">Cyberpunk</option>
      <option value="valentine">Valentine</option>
      <option value="garden">Garden</option>
      <option value="forest">Forest</option>
      <option value="lofi">Lo-Fi</option>
      <option value="pastel">Pastel</option>
      <option value="wireframe">Wireframe</option>
      <option value="luxury">Luxury</option>
      <option value="dracula">Dracula</option>
      <option value="autumn">Autumn</option>
      <option value="business">Business</option>
      <option value="coffee">Coffee</option>
      <option value="winter">Winter</option>
    </select>
  </div>
  
  <main class="flex-1 overflow-y-auto overflow-x-hidden touch-pan-y pb-safe relative {activeView === 'actions' ? 'overflow-hidden' : ''}">
    {#if activeView === 'game'}
      <div transition:fade={{ duration: 200 }}>
        <PlayingArea onswitchToActions={() => activeView = 'actions'} />
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

