<script lang="ts">
  import { onMount } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import PlayingArea from './lib/components/PlayingAreaEvent.svelte';
  import ActionPanel from './lib/components/ActionPanelEvent.svelte';
  import DebugPanel from './lib/components/DebugPanelEvent.svelte';
  import QuickplayError from './lib/components/QuickplayError.svelte';
  import { eventGame } from './stores/eventGame';
  import { fly, fade } from 'svelte/transition';

  let showDebugPanel = $state(false);
  let activeView = $state<'game' | 'actions'>('game');
  let touchStartY = $state(0);
  let touchStartTime = $state(0);
  let currentTheme = $state('light');

  const gamePhase = eventGame.gamePhase;
  const currentPlayer = eventGame.currentPlayer;
  const playerTypes = eventGame.playerTypes;

  function handleKeydown(e: KeyboardEvent) {
    if (e.ctrlKey && e.shiftKey && e.key === 'D') {
      e.preventDefault();
      showDebugPanel = !showDebugPanel;
    } else if (e.key === 'Escape' && showDebugPanel) {
      showDebugPanel = false;
    }
  }

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

      if (swipeDistance > 100 && touchDuration < 300) {
        showDebugPanel = true;
      }
    }
  }

  onMount(() => {
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      currentTheme = savedTheme;
      document.documentElement.setAttribute('data-theme', savedTheme);
    }
  });
  
  $effect(() => {
    if ($gamePhase === 'bidding' || $gamePhase === 'trump_selection') {
      activeView = 'actions';
    } else if ($gamePhase === 'playing' || $gamePhase === 'setup') {
      activeView = 'game';
    } else if ($gamePhase === 'scoring') {
      activeView = 'game';
    } else if ($gamePhase === 'game_end') {
      activeView = 'game';
    }
  });
</script>

<svelte:window onkeydown={handleKeydown} />

<div 
  class="flex flex-col h-screen bg-base-100 text-base-content font-sans overflow-hidden"
  style="height: 100dvh;"
  role="application" 
  data-phase={$gamePhase} 
  ontouchstart={handleTouchStart} 
  ontouchend={handleTouchEnd}
>
  <Header onopenDebug={() => showDebugPanel = true} />
  
  <div class="fixed bottom-4 left-4 z-50">
    <label class="swap swap-rotate">
      <input 
        type="checkbox" 
        class="theme-controller" 
        checked={currentTheme === 'dark'}
        onchange={(e) => {
          const theme = e.currentTarget.checked ? 'dark' : 'light';
          currentTheme = theme;
          localStorage.setItem('theme', theme);
          document.documentElement.setAttribute('data-theme', theme);
        }}
      />
      <svg class="swap-off h-8 w-8 fill-current" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <path d="M5.64,17l-.71.71a1,1,0,0,0,0,1.41,1,1,0,0,0,1.41,0l.71-.71A1,1,0,0,0,5.64,17ZM5,12a1,1,0,0,0-1-1H3a1,1,0,0,0,0,2H4A1,1,0,0,0,5,12Zm7-7a1,1,0,0,0,1-1V3a1,1,0,0,0-2,0V4A1,1,0,0,0,12,5ZM5.64,7.05a1,1,0,0,0,.7.29,1,1,0,0,0,.71-.29,1,1,0,0,0,0-1.41l-.71-.71A1,1,0,0,0,4.93,6.34Zm12,.29a1,1,0,0,0,.7-.29l.71-.71a1,1,0,1,0-1.41-1.41L17,5.64a1,1,0,0,0,0,1.41A1,1,0,0,0,17.66,7.34ZM21,11H20a1,1,0,0,0,0,2h1a1,1,0,0,0,0-2Zm-9,8a1,1,0,0,0-1,1v1a1,1,0,0,0,2,0V20A1,1,0,0,0,12,19ZM18.36,17A1,1,0,0,0,17,18.36l.71.71a1,1,0,0,0,1.41,0,1,1,0,0,0,0-1.41ZM12,6.5A5.5,5.5,0,1,0,17.5,12,5.51,5.51,0,0,0,12,6.5Zm0,9A3.5,3.5,0,1,1,15.5,12,3.5,3.5,0,0,1,12,15.5Z"/>
      </svg>
      <svg class="swap-on h-8 w-8 fill-current" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
        <path d="M21.64,13a1,1,0,0,0-1.05-.14,8.05,8.05,0,0,1-3.37.73A8.15,8.15,0,0,1,9.08,5.49a8.59,8.59,0,0,1,.25-2A1,1,0,0,0,8,2.36,10.14,10.14,0,1,0,22,14.05,1,1,0,0,0,21.64,13Zm-9.5,6.69A8.14,8.14,0,0,1,7.08,5.22v.27A10.15,10.15,0,0,0,17.22,15.63a9.79,9.79,0,0,0,2.1-.22A8.11,8.11,0,0,1,12.14,19.73Z"/>
      </svg>
    </label>
  </div>

  <main class="flex-1 flex flex-col items-center justify-center overflow-hidden">
    <div class="w-full max-w-7xl h-full px-2 sm:px-4 py-2 sm:py-4">
      <div class="flex flex-col sm:flex-row gap-2 sm:gap-4 h-full">
        <div class="flex-1 min-h-0">
          {#if activeView === 'game'}
            <div transition:fade={{ duration: 150 }}>
              <PlayingArea />
            </div>
          {:else}
            <div transition:fade={{ duration: 150 }}>
              <ActionPanel />
            </div>
          {/if}
        </div>

        <div class="flex flex-row sm:flex-col gap-2 justify-center items-center sm:w-32">
          <button
            class="btn btn-primary btn-sm sm:btn-md"
            class:btn-outline={activeView === 'game'}
            onclick={() => activeView = 'game'}
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5 2a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V4a2 2 0 00-2-2H5zm0 2h10v12H5V4z" clip-rule="evenodd" />
            </svg>
            <span class="hidden sm:inline">Game</span>
          </button>
          
          <button
            class="btn btn-primary btn-sm sm:btn-md"
            class:btn-outline={activeView === 'actions'}
            onclick={() => activeView = 'actions'}
          >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd" />
            </svg>
            <span class="hidden sm:inline">Actions</span>
          </button>
        </div>
      </div>
    </div>
  </main>

  <QuickplayError />

  {#if showDebugPanel}
    <div
      class="fixed inset-0 bg-black bg-opacity-50 z-50"
      transition:fade={{ duration: 200 }}
      onclick={() => showDebugPanel = false}
    >
      <div
        class="absolute bottom-0 left-0 right-0 bg-base-100 rounded-t-xl shadow-2xl max-h-[80vh] overflow-hidden"
        transition:fly={{ y: 300, duration: 300 }}
        onclick={(e) => e.stopPropagation()}
      >
        <DebugPanel onclose={() => showDebugPanel = false} />
      </div>
    </div>
  {/if}
</div>