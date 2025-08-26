<script lang="ts">
  import type { Play } from '../../game/types';
  import Domino from './Domino.svelte';
  
  interface CompletedTrick {
    plays: Play[];
    winner?: number;
    points: number;
  }
  
  interface Props {
    completedTricks: CompletedTrick[];
    currentTrick: Play[];
    trickNumber: number;
    isOpen?: boolean;
    onToggle?: () => void;
    onStateChange?: (state: 'collapsed' | 'expanded') => void;
  }
  
  let { 
    completedTricks = [], 
    currentTrick = [],
    trickNumber = 1,
    onToggle,
    onStateChange
  }: Props = $props();
  
  let drawerState = $state<'collapsed' | 'expanded'>('collapsed');
  
  function toggleDrawer() {
    if (drawerState === 'collapsed') {
      drawerState = 'expanded';
      onStateChange?.('expanded');
    } else {
      drawerState = 'collapsed';
      onStateChange?.('collapsed');
    }
    onToggle?.();
  }
  
  // Get plays for a specific player in a trick
  function getPlayerPlay(plays: Play[], playerIndex: number): Play | null {
    return plays.find(p => p.player === playerIndex) || null;
  }
  
</script>

<!-- Mobile: Left Side Panel -->
<div class="lg:hidden fixed left-0 top-0 h-full z-40 flex pointer-events-none">
  <!-- Side Panel Container -->
  <div
    class="relative bg-base-100 shadow-2xl transition-transform duration-300 ease-out h-full flex flex-col pointer-events-auto"
    style="width: min(70vw, 280px); transform: translateX({drawerState === 'expanded' ? '0' : '-100%'});"
  >
    <!-- Header with close button -->
    <div class="flex items-center justify-between p-4 border-b border-base-300">
      <h3 class="text-sm font-bold">Tricks ({completedTricks.length}/7)</h3>
      <button
        class="btn btn-ghost btn-sm btn-circle"
        onclick={toggleDrawer}
        aria-label="Close tricks drawer"
      >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-5 h-5">
          <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
    </div>
    
    <!-- Panel Content -->
    <div class="flex-1 h-full overflow-y-auto">
      <div class="p-4">
        
        <!-- Column Headers -->
        <div class="grid grid-cols-5 gap-1 mb-2 text-xs font-bold text-base-content/70">
          <div></div>
          <div class="text-center">P0</div>
          <div class="text-center">P1</div>
          <div class="text-center">P2</div>
          <div class="text-center">P3</div>
        </div>
        
        <!-- Completed Tricks -->
        {#each completedTricks as trick, index}
          <div class="mb-2">
            <div class="grid grid-cols-5 gap-1 items-center">
              <div class="text-xs font-bold text-base-content/60">{index + 1}:</div>
              {#each [0, 1, 2, 3] as playerIndex}
                {@const play = getPlayerPlay(trick.plays, playerIndex)}
                <div class="flex justify-center">
                  {#if play}
                    <div 
                      class="transition-all {trick.winner === playerIndex ? 'scale-110 drop-shadow-md ring-2 ring-primary rounded' : ''}"
                    >
                      <Domino 
                        domino={play.domino} 
                        small={true} 
                        showPoints={false} 
                        clickable={false}
                      />
                    </div>
                  {:else}
                    <div class="w-10 h-14 border-2 border-dashed border-base-300 rounded"></div>
                  {/if}
                </div>
              {/each}
            </div>
            <div class="text-right text-xs text-base-content/70 mt-1">
              {#if trick.winner !== undefined}
                P{trick.winner} won ({trick.points} pts)
              {:else}
                ({trick.points} pts)
              {/if}
            </div>
          </div>
        {/each}
        
        <!-- Current Trick (if in progress) -->
        {#if currentTrick.length > 0}
          <div class="mb-2 opacity-70">
            <div class="grid grid-cols-5 gap-1 items-center">
              <div class="text-xs font-bold text-warning">{completedTricks.length + 1}:</div>
              {#each [0, 1, 2, 3] as playerIndex}
                {@const play = getPlayerPlay(currentTrick, playerIndex)}
                <div class="flex justify-center">
                  {#if play}
                    <Domino domino={play.domino} small={true} showPoints={false} clickable={false} />
                  {:else}
                    <div class="w-10 h-14 border-2 border-dashed border-base-300 rounded animate-pulse"></div>
                  {/if}
                </div>
              {/each}
            </div>
            <div class="text-right text-xs text-warning mt-1">
              In progress...
            </div>
          </div>
        {/if}
        
        <!-- Empty state -->
        {#if completedTricks.length === 0 && currentTrick.length === 0}
          <div class="text-center py-8 text-base-content/50">
            <div class="text-4xl mb-2">🎲</div>
            <div class="text-sm">No tricks played yet</div>
          </div>
        {/if}
      </div>
    </div>
    
    <!-- Vertical Tab (always visible) -->
    <button
      class="absolute -right-8 top-1/2 -translate-y-1/2 w-8 h-24 bg-base-200 rounded-r-lg shadow-lg flex items-center justify-center hover:bg-base-300 transition-colors pointer-events-auto"
      onclick={toggleDrawer}
      aria-expanded={drawerState !== 'collapsed'}
      aria-label="Toggle trick history"
      style="writing-mode: vertical-rl; text-orientation: mixed;"
    >
      <div class="flex items-center gap-1">
        <svg 
          class="w-4 h-4 transition-transform {drawerState === 'expanded' ? 'rotate-180' : ''}"
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
        </svg>
        <span class="text-xs font-medium">Tricks {trickNumber}/7</span>
      </div>
    </button>
  </div>
  
  <!-- Overlay (when expanded) -->
  {#if drawerState === 'expanded'}
    <button
      class="fixed inset-0 bg-black/20 -z-10"
      onclick={toggleDrawer}
      aria-label="Close drawer"
    ></button>
  {/if}
</div>

<!-- Desktop/Landscape: Side Drawer (can implement later) -->
<div class="hidden lg:block">
  <!-- Desktop implementation would go here -->
</div>

<style>
  /* Ensure smooth animations */
  /* svelte-ignore css_unused_selector */
  .drawer-transition {
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  }
  
  /* Remove iOS bounce scrolling in drawer */
  .overflow-y-auto {
    -webkit-overflow-scrolling: touch;
    overscroll-behavior: contain;
  }
</style>