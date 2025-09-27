<script lang="ts">
  import { seedFinderStore } from '../../stores/seedFinderStore';
  import { onMount, onDestroy } from 'svelte';
  import { writable } from 'svelte/store';
  import { TARGET_WIN_RATE_MIN, TARGET_WIN_RATE_MAX } from '../../game/core/seedFinder';

  // RAF-optimized store values
  const rafState = writable({
    isSearching: false,
    waitingForConfirmation: false,
    currentSeed: 0,
    seedsTried: 0,
    gamesPlayed: 0,
    totalGames: 100,
    currentWinRate: 0,
    bestSeed: null as number | null,
    bestWinRate: 1.0,
    foundSeed: null as number | null,
    foundWinRate: 0
  });

  let rafId: number | null = null;
  let pendingUpdate: any = null;
  let unsubscribe: (() => void) | null = null;

  const useBest = () => {
    seedFinderStore.useBest();
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
    if (e.key === 'Escape') {
      rafState.subscribe(state => {
        if (state.isSearching) {
          e.preventDefault();
          useBest();
        }
        // No ESC handling for confirmation modal - user must choose Start Game or Regenerate
      })();
    }
  };

  onMount(() => {
    window.addEventListener('keydown', handleKeydown);

    // Subscribe to store with RAF batching
    unsubscribe = seedFinderStore.subscribe(state => {
      pendingUpdate = state;

      if (!rafId) {
        rafId = requestAnimationFrame(() => {
          if (pendingUpdate) {
            rafState.set(pendingUpdate);
          }
          rafId = null;
        });
      }
    });
  });

  onDestroy(() => {
    window.removeEventListener('keydown', handleKeydown);
    if (rafId) {
      cancelAnimationFrame(rafId);
    }
    if (unsubscribe) {
      unsubscribe();
    }
  });
</script>

{#if $rafState.isSearching || $rafState.waitingForConfirmation}
  <div class="modal modal-open">
    <form method="dialog" class="modal-box max-w-[12rem] w-full">
      <div class="text-center">
        {#if $rafState.waitingForConfirmation}
          <h3 class="font-bold text-xl mb-2">Seed Found!</h3>
          <div class="stat bg-base-200 rounded-lg p-2 mb-3">
            <div class="stat-title text-xs">Seed {$rafState.foundSeed}</div>
            <div class="stat-value text-lg">
              {($rafState.foundWinRate * 100).toFixed(1)}% win rate
            </div>
          </div>
          <p class="text-xs opacity-70 mb-4">
            Random play wins {($rafState.foundWinRate * 100).toFixed(0)}% of games
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
              <div class="stat-value text-2xl">{$rafState.seedsTried}</div>
            </div>

            {#if $rafState.currentSeed > 0}
              <div class="stat bg-base-200 rounded-lg p-3">
                <div class="stat-title text-xs">Current Seed: {$rafState.currentSeed}</div>
                <div class="flex items-center justify-center gap-2">
                  <progress
                    class="progress progress-primary w-full"
                    value={$rafState.gamesPlayed}
                    max={$rafState.totalGames}
                  ></progress>
                  <span class="text-sm font-mono whitespace-nowrap">
                    {$rafState.gamesPlayed}/{$rafState.totalGames}
                  </span>
                </div>
                {#if $rafState.gamesPlayed > 0}
                  <div class="text-xs mt-1 opacity-70">
                    Win rate: {($rafState.currentWinRate * 100).toFixed(1)}%
                  </div>
                {/if}
              </div>
            {/if}

            {#if $rafState.bestSeed}
              <div class="stat bg-success/10 rounded-lg p-2">
                <div class="stat-title text-xs">Best Found: {$rafState.bestSeed}</div>
                <div class="stat-value text-sm">
                  {($rafState.bestWinRate * 100).toFixed(1)}% win rate
                </div>
              </div>
            {/if}
            
            <div class="text-sm opacity-70">
              Looking for a seed where random play wins {(TARGET_WIN_RATE_MIN * 100).toFixed(0)}-{(TARGET_WIN_RATE_MAX * 100).toFixed(0)}% of games...
            </div>
          </div>
          
          <div class="modal-action justify-center mt-6">
            <button
              class="btn btn-primary"
              onclick={useBest}
              disabled={!$rafState.bestSeed}
            >
              {$rafState.bestSeed ? `Use Best (${($rafState.bestWinRate * 100).toFixed(0)}%)` : 'Use Best'}
            </button>
          </div>
        {/if}
      </div>
    </form>
  </div>
{/if}