<script lang="ts">
  import { onMount } from 'svelte';
  import PerfectHandDisplay from './lib/components/PerfectHandDisplay.svelte';
  import Domino from './lib/components/Domino.svelte';
  import { parseDomino } from './lib/utils/dominoHelpers';
  import partitionsData from '../data/3hand-partitions.json';

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
</script>

<div class="min-h-screen bg-base-100">
  <header class="navbar bg-base-300 py-1 min-h-0 h-auto">
    <div class="flex-1">
      <h1 class="text-sm font-bold px-2">Perfect Hands</h1>
    </div>
  </header>

  <main class="container mx-auto p-1 max-w-7xl">
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

      {#each currentPartitions as partition, partitionIndex}
        <div class="mb-2 p-1 bg-base-200">
          <h2 class="text-xs font-bold mb-1">Partition {currentPage * itemsPerPage + partitionIndex + 1}</h2>
          <div class="grid grid-cols-1 gap-1">
            {#each partition.hands as hand, handIndex}
              <PerfectHandDisplay {hand} index={handIndex} />
            {/each}
          </div>

          {#if partition.leftover}
            <div class="mt-1 p-1 bg-base-300">
              <div class="flex items-center gap-1 mb-1">
                <span class="text-xs font-semibold">Leftover:</span>
                {#if partition.leftover.bestTrump}
                  <span class="badge badge-xs badge-neutral">
                    Best: {partition.leftover.bestTrump === 'no-trump' ? 'No Trump' :
                           partition.leftover.bestTrump.charAt(0).toUpperCase() + partition.leftover.bestTrump.slice(1)}
                  </span>
                {/if}
                {#if partition.leftover.externalBeaters !== undefined}
                  <span class="badge badge-xs badge-error">
                    {partition.leftover.externalBeaters} External Beaters
                  </span>
                {/if}
              </div>
              <div class="flex flex-wrap gap-0.5">
                {#each partition.leftover.dominoes as dominoStr}
                  <Domino domino={parseDomino(dominoStr)} micro={true} showPoints={false} />
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {/each}

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