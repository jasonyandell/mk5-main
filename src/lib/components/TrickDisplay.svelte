<script lang="ts">
  import type { Play } from '../../game/types';
  import Domino from './Domino.svelte';
  
  interface Props {
    trick: Play[];
    collapsed?: boolean;
    onToggle?: () => void;
  }
  
  let { trick = [], collapsed = false, onToggle }: Props = $props();
  
  let isCollapsed = $state(collapsed);
  
  $effect(() => {
    isCollapsed = collapsed;
  });
  
  function handleToggle() {
    isCollapsed = !isCollapsed;
    onToggle?.();
  }
</script>

<div class="trick-display-container">
  <!-- Mobile: Collapsible trick display -->
  <div class="lg:hidden">
    {#if trick.length > 0}
      <button
        class="btn btn-sm btn-ghost w-full flex items-center justify-between"
        onclick={handleToggle}
        aria-expanded={!isCollapsed}
        aria-label="Toggle trick display"
      >
        <span class="text-sm font-medium">Current Trick ({trick.length}/4)</span>
        <svg 
          class="w-4 h-4 transition-transform duration-200"
          class:rotate-180={!isCollapsed}
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24"
        >
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
        </svg>
      </button>
      
      <div 
        class="trick-content overflow-hidden transition-all duration-300"
        class:max-h-0={isCollapsed}
        class:max-h-96={!isCollapsed}
      >
        <div class="flex flex-wrap justify-center gap-2 p-3">
          {#each trick as play}
            <div class="flex flex-col items-center">
              <Domino domino={play.domino} size="small" />
              <span class="text-xs mt-1 badge badge-ghost badge-sm">P{play.player + 1}</span>
            </div>
          {/each}
        </div>
      </div>
    {:else}
      <div class="text-center py-2 text-sm text-base-content/60">
        No tricks played yet
      </div>
    {/if}
  </div>
  
  <!-- Desktop: Always visible trick display -->
  <div class="hidden lg:block">
    <div class="card bg-base-200">
      <div class="card-body p-4">
        <h3 class="card-title text-sm">Current Trick</h3>
        {#if trick.length > 0}
          <div class="grid grid-cols-2 gap-3">
            {#each trick as play}
              <div class="flex flex-col items-center">
                <Domino domino={play.domino} size="medium" />
                <span class="text-xs mt-1 badge badge-outline badge-sm">P{play.player + 1}</span>
              </div>
            {/each}
          </div>
        {:else}
          <div class="text-center py-4 text-sm text-base-content/60">
            Waiting for plays...
          </div>
        {/if}
      </div>
    </div>
  </div>
</div>

<style>
  .trick-display-container {
    @apply w-full;
  }
  
  .trick-content {
    @apply bg-base-200 rounded-lg;
  }
  
  /* Smooth transitions */
  .rotate-180 {
    transform: rotate(180deg);
  }
</style>