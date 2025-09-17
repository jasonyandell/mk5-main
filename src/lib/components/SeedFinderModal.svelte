<script lang="ts">
  import { seedFinderStore } from '../../stores/seedFinderStore';
  import { onMount, onDestroy } from 'svelte';
  import { TARGET_WIN_RATE_MIN, TARGET_WIN_RATE_MAX } from '../../game/core/seedFinder';

  const cancel = () => {
    seedFinderStore.cancel();
  };
  
  const confirmSeed = () => {
    seedFinderStore.confirmSeed();
  };
  
  const regenerate = () => {
    seedFinderStore.clearFoundSeed();
    seedFinderStore.startSearch();
    // Trigger regeneration in game store
    const event = new CustomEvent('regenerate-seed');
    window.dispatchEvent(event);
  };
  
  // Handle Escape key
  const handleKeydown = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && ($seedFinderStore.isSearching || $seedFinderStore.waitingForConfirmation)) {
      e.preventDefault();
      cancel();
    }
  };
  
  onMount(() => {
    window.addEventListener('keydown', handleKeydown);
  });
  
  onDestroy(() => {
    window.removeEventListener('keydown', handleKeydown);
  });
</script>

{#if $seedFinderStore.isSearching || $seedFinderStore.waitingForConfirmation}
  <div class="modal modal-open">
    <form method="dialog" class="modal-box max-w-[12rem] w-full">
      <div class="text-center">
        {#if $seedFinderStore.waitingForConfirmation}
          <h3 class="font-bold text-xl mb-2">Seed Found!</h3>
          <div class="stat bg-base-200 rounded-lg p-2 mb-3">
            <div class="stat-title text-xs">Seed {$seedFinderStore.foundSeed}</div>
            <div class="stat-value text-lg">
              {($seedFinderStore.foundWinRate * 100).toFixed(1)}% win rate
            </div>
          </div>
          <p class="text-xs opacity-70 mb-4">
            Random play wins {($seedFinderStore.foundWinRate * 100).toFixed(0)}% of games
          </p>
          <div class="flex flex-col gap-2">
            <button class="btn btn-primary w-full" onclick={confirmSeed}>
              Start Game
            </button>
            <button class="btn btn-ghost w-full" onclick={regenerate}>
              Regenerate
            </button>
          </div>
        {:else}
          <h3 class="font-bold text-xl mb-4">Finding Balanced Game</h3>
          
          <div class="flex justify-center mb-4">
            <span class="loading loading-spinner loading-lg"></span>
          </div>
          
          <div class="space-y-3">
            <div class="stat bg-base-200 rounded-lg p-3">
              <div class="stat-title text-xs">Seeds Tested</div>
              <div class="stat-value text-2xl">{$seedFinderStore.seedsTried}</div>
            </div>
            
            {#if $seedFinderStore.currentSeed > 0}
              <div class="stat bg-base-200 rounded-lg p-3">
                <div class="stat-title text-xs">Current Seed: {$seedFinderStore.currentSeed}</div>
                <div class="flex items-center justify-center gap-2">
                  <progress 
                    class="progress progress-primary w-full" 
                    value={$seedFinderStore.gamesPlayed} 
                    max={$seedFinderStore.totalGames}
                  ></progress>
                  <span class="text-sm font-mono whitespace-nowrap">
                    {$seedFinderStore.gamesPlayed}/{$seedFinderStore.totalGames}
                  </span>
                </div>
                {#if $seedFinderStore.gamesPlayed > 0}
                  <div class="text-xs mt-1 opacity-70">
                    Win rate: {($seedFinderStore.currentWinRate * 100).toFixed(1)}%
                  </div>
                {/if}
              </div>
            {/if}
            
            <div class="text-sm opacity-70">
              Looking for a seed where random play wins {(TARGET_WIN_RATE_MIN * 100).toFixed(0)}-{(TARGET_WIN_RATE_MAX * 100).toFixed(0)}% of games...
            </div>
          </div>
          
          <div class="modal-action justify-center mt-6">
            <button class="btn btn-ghost" onclick={cancel}>
              Cancel
            </button>
          </div>
        {/if}
      </div>
    </form>
  </div>
{/if}