<script lang="ts">
  import { onMount } from 'svelte';
  import PerfectHandDisplay from './lib/components/PerfectHandDisplay.svelte';
  import Domino from './lib/components/Domino.svelte';
  import { parseDomino, computeExternalBeaters } from './lib/utils/dominoHelpers';
  import partitionsData from '../data/3hand-partitions.json';
  import type { Domino as DominoType } from './game/types';

  interface Partition {
    hands: Array<{
      dominoes: string[];
      trump: string;
      type: string;
    }>;
    leftover?: {
      dominoes: string[];
      bestTrump: string;
      externalBeaters: number;
    };
  }

  let partitions: Partition[] = $state([]);
  let currentPage = $state(0);
  let itemsPerPage = $state(5);

  function isTrump(domino: DominoType, trumpStr: string): boolean {
    if (trumpStr === 'no-trump') return false;
    if (trumpStr === 'doubles') return domino.high === domino.low;

    // For suit trumps, check if either pip matches the trump suit
    const suitMap: Record<string, number> = {
      'blanks': 0,
      'aces': 1,
      'deuces': 2,
      'tres': 3,
      'fours': 4,
      'fives': 5,
      'sixes': 6
    };
    const suit = suitMap[trumpStr];
    if (suit !== undefined) {
      return domino.high === suit || domino.low === suit;
    }
    return false;
  }

  function sortDominoes(dominoStrs: string[], trumpStr: string): string[] {
    return dominoStrs.slice().sort((a, b) => {
      const dominoA = parseDomino(a);
      const dominoB = parseDomino(b);

      const aIsTrump = isTrump(dominoA, trumpStr);
      const bIsTrump = isTrump(dominoB, trumpStr);

      if (aIsTrump && !bIsTrump) return -1;
      if (!aIsTrump && bIsTrump) return 1;

      // Within trumps or non-trumps, sort by high pip then low pip
      if (dominoA.high !== dominoB.high) return dominoB.high - dominoA.high;
      return dominoB.low - dominoA.low;
    });
  }

  onMount(() => {
    partitions = partitionsData.partitions as Partition[];

    // Always use business theme for this page
    document.documentElement.setAttribute('data-theme', 'business');
  });

  const totalPages = $derived(Math.ceil(partitions.length / itemsPerPage));
  const currentPartitions = $derived(
    partitions.slice(currentPage * itemsPerPage, (currentPage + 1) * itemsPerPage)
  );

  function nextPage() {
    if (currentPage < totalPages - 1) {
      currentPage++;
      window.scrollTo(0, 0);
    }
  }

  function prevPage() {
    if (currentPage > 0) {
      currentPage--;
      window.scrollTo(0, 0);
    }
  }

  // Touch gesture support for mobile
  let touchStartX = 0;
  let touchStartY = 0;

  function handleTouchStart(e: TouchEvent) {
    const touch = e.touches[0];
    if (touch) {
      touchStartX = touch.clientX;
      touchStartY = touch.clientY;
    }
  }

  function handleTouchEnd(e: TouchEvent) {
    if (!e.changedTouches.length || !e.changedTouches[0]) return;

    const touchEndX = e.changedTouches[0].clientX;
    const touchEndY = e.changedTouches[0].clientY;
    const swipeX = touchEndX - touchStartX;
    const swipeY = touchEndY - touchStartY;

    // Only handle horizontal swipes that are significant enough
    // and more horizontal than vertical
    if (Math.abs(swipeX) > 50 && Math.abs(swipeX) > Math.abs(swipeY)) {
      if (swipeX > 0) {
        // Swipe right - go to previous page
        prevPage();
      } else {
        // Swipe left - go to next page
        nextPage();
      }
    }
  }
</script>

<div class="min-h-screen bg-base-100">
  <header class="navbar bg-base-300 py-1 min-h-0 h-auto">
    <div class="flex-1">
      <h1 class="text-sm font-bold px-2">Near Perfect 42 Deals</h1>
    </div>
  </header>

  <main
    class="container mx-auto p-1 max-w-7xl"
    ontouchstart={handleTouchStart}
    ontouchend={handleTouchEnd}
  >
    {#if partitions.length === 0}
      <div class="flex items-center justify-center min-h-[50vh]">
        <div class="text-center">
          <div class="loading loading-spinner loading-lg"></div>
          <p class="mt-4">Loading perfect hands...</p>
        </div>
      </div>
    {:else}
      <div class="mb-1 flex justify-between items-center px-1">
        <p class="text-xs opacity-75">
          Showing {currentPage * itemsPerPage + 1}-{Math.min((currentPage + 1) * itemsPerPage, partitions.length)} of {partitions.length}
        </p>
        <div class="join">
          <button
            class="join-item btn btn-xs"
            onclick={prevPage}
            disabled={currentPage === 0}
          >
            «
          </button>
          <button class="join-item btn btn-xs">
            {currentPage + 1}/{totalPages}
          </button>
          <button
            class="join-item btn btn-xs"
            onclick={nextPage}
            disabled={currentPage >= totalPages - 1}
          >
            »
          </button>
        </div>
      </div>

      <div class="transition-opacity duration-200">
        {#each currentPartitions as partition, partitionIndex}
          <div class="mb-2 p-1 bg-base-200">
          <h2 class="text-xs font-bold mb-1">Deal {currentPage * itemsPerPage + partitionIndex + 1}</h2>
          <div class="grid grid-cols-1 gap-1">
            {#each partition.hands as hand, handIndex}
              <PerfectHandDisplay {hand} index={handIndex} />
            {/each}
          </div>

          {#if partition.leftover}
            {@const externalBeaters = computeExternalBeaters(partition.leftover.dominoes, partition.leftover.bestTrump)}
            {@const sortedLeftovers = sortDominoes(partition.leftover.dominoes, partition.leftover.bestTrump)}
            {@const sortedBeaters = sortDominoes(externalBeaters, partition.leftover.bestTrump)}
            <div class="mt-1 p-1 bg-base-300">
              <div class="flex items-center gap-1 mb-1">
                <span class="text-xs font-semibold">Leftover</span>
                {#if partition.leftover.bestTrump}
                  <span class="badge badge-xs badge-info">
                    {partition.leftover.bestTrump === 'no-trump' ? 'No Trump' :
                     partition.leftover.bestTrump.charAt(0).toUpperCase() + partition.leftover.bestTrump.slice(1)}
                  </span>
                {/if}
              </div>
              <div class="flex flex-wrap gap-0.5">
                {#each sortedLeftovers as dominoStr}
                  <Domino domino={parseDomino(dominoStr)} micro={true} showPoints={false} />
                {/each}
              </div>

              {#if externalBeaters.length > 0}
                <div class="mt-1">
                  <div class="text-xs font-semibold mb-0.5">
                    External Beaters ({externalBeaters.length}):
                  </div>
                  <div class="flex flex-wrap gap-0.5">
                    {#each sortedBeaters as dominoStr}
                      <Domino domino={parseDomino(dominoStr)} micro={true} showPoints={false} />
                    {/each}
                  </div>
                </div>
              {/if}
            </div>
          {/if}
        </div>
      {/each}
      </div>

      <div class="flex justify-center mt-2 mb-2">
        <div class="join">
          <button
            class="join-item btn btn-sm"
            onclick={prevPage}
            disabled={currentPage === 0}
          >
            « Prev
          </button>
          <button class="join-item btn btn-sm btn-active">
            {currentPage + 1}/{totalPages}
          </button>
          <button
            class="join-item btn btn-sm"
            onclick={nextPage}
            disabled={currentPage >= totalPages - 1}
          >
            Next »
          </button>
        </div>
      </div>
    {/if}
  </main>
</div>