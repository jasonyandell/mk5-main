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

  let partitions: Partition[] = $state(partitionsData.partitions as Partition[]);
  let currentPage = $state(0);
  let scrollContainer = $state<HTMLDivElement>();
  let loadedPages = $state(15); // Start with 15 pages (5 sets × 3 hands) for better initial performance

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

  // Pre-compute expensive operations for all partitions
  interface ProcessedPartition extends Partition {
    processed?: {
      externalBeaters: string[];
      sortedLeftovers: string[];
      sortedBeaters: string[];
    };
  }

  // Interface for flattened hand pages
  interface HandPage {
    hand: {
      dominoes: string[];
      trump: string;
      type: string;
    };
    handIndex: number; // 0, 1, or 2
    setIndex: number; // Which partition set
    isLastHandInSet: boolean;
    leftover?: ProcessedPartition['leftover'];
    processed?: ProcessedPartition['processed'];
  }

  function processPartitions(): ProcessedPartition[] {
    return partitions.map(partition => {
      const processed: ProcessedPartition = { ...partition };
      if (partition.leftover) {
        const externalBeaters = computeExternalBeaters(partition.leftover.dominoes, partition.leftover.bestTrump);
        processed.processed = {
          externalBeaters,
          sortedLeftovers: sortDominoes(partition.leftover.dominoes, partition.leftover.bestTrump),
          sortedBeaters: sortDominoes(externalBeaters, partition.leftover.bestTrump)
        };
      }
      return processed;
    });
  }

  let processedPartitions = $state<ProcessedPartition[]>([]);
  let flattenedHands = $state<HandPage[]>([]);

  onMount(() => {
    // Always use business theme for this page
    document.documentElement.setAttribute('data-theme', 'business');
    // Pre-process all partitions once
    processedPartitions = processPartitions();

    // Flatten into individual hand pages
    flattenedHands = processedPartitions.flatMap((partition, setIndex) =>
      partition.hands.map((hand, handIndex) => ({
        hand,
        handIndex,
        setIndex,
        isLastHandInSet: handIndex === partition.hands.length - 1,
        leftover: partition.leftover,
        processed: partition.processed
      }))
    );
  });

  // Set up scroll listener when container becomes available
  $effect(() => {
    if (scrollContainer) {
      scrollContainer.addEventListener('scroll', handleScroll);
      return () => scrollContainer?.removeEventListener('scroll', handleScroll);
    }
    return undefined;
  });

  const totalPages = $derived(flattenedHands.length);

  // Render all loaded pages (lazy loading approach)
  const visiblePages = $derived(() => {
    const pagesToShow = Math.min(loadedPages, totalPages);
    return flattenedHands.slice(0, pagesToShow);
  });

  let scrollTimeout: ReturnType<typeof setTimeout>;

  function handleScroll() {
    if (!scrollContainer) return;

    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(() => {
      if (!scrollContainer) return;
      const scrollPosition = scrollContainer.scrollLeft;
      const containerWidth = scrollContainer.clientWidth;
      const scrollIndex = Math.round(scrollPosition / containerWidth);

      // Update current page based on scroll position
      if (scrollIndex >= 0 && scrollIndex !== currentPage) {
        currentPage = scrollIndex;
      }

      // Load more pages when approaching the end
      if (scrollIndex >= loadedPages - 5 && loadedPages < totalPages) {
        // Load 10 more pages at a time for smoother performance (10 hands)
        loadedPages = Math.min(loadedPages + 10, totalPages);
      }
    }, 50); // Reduced debounce for snappier response
  }

</script>

<div class="min-h-screen bg-base-100">
  <header class="navbar bg-base-300 py-1 min-h-0 h-auto">
    <div class="flex-1">
      <h1 class="text-sm font-bold px-2">"Three Perfect Hands" 42 Challenge</h1>
    </div>
  </header>

  <main class="relative overflow-hidden">
    {#if partitions.length === 0}
      <div class="flex items-center justify-center min-h-[50vh]">
        <div class="text-center">
          <div class="loading loading-spinner loading-lg"></div>
          <p class="mt-4">Loading perfect hands...</p>
        </div>
      </div>
    {:else}
      <!-- Horizontal scroll container -->
      <div
        bind:this={scrollContainer}
        class="flex overflow-x-auto snap-x snap-mandatory scrollbar-hide"
        style="-webkit-overflow-scrolling: touch;"
      >
        {#each visiblePages() as handPage, pageIndex}
          <div class="min-w-full snap-start px-2 py-2 flex flex-col items-center justify-center min-h-[80vh]">
            <div class="w-full max-w-md">
              <h2 class="text-sm font-bold mb-2 text-center">
                Set {handPage.setIndex + 1} - Hand {handPage.handIndex + 1}
              </h2>

              <PerfectHandDisplay hand={handPage.hand} index={handPage.handIndex} />

              {#if handPage.isLastHandInSet && handPage.leftover && handPage.processed}
                <div class="mt-3 p-2 bg-base-300 rounded-lg">
                  <div class="flex items-center gap-2 mb-2">
                    <span class="text-sm font-semibold">Leftover Dominoes</span>
                    {#if handPage.leftover.bestTrump}
                      <span class="badge badge-sm badge-info">
                        {handPage.leftover.bestTrump === 'no-trump' ? 'No Trump' :
                         handPage.leftover.bestTrump.charAt(0).toUpperCase() + handPage.leftover.bestTrump.slice(1)}
                      </span>
                    {/if}
                  </div>
                  <div class="flex flex-wrap gap-1 justify-center">
                    {#each handPage.processed.sortedLeftovers as dominoStr}
                      <Domino domino={parseDomino(dominoStr)} micro={true} showPoints={false} />
                    {/each}
                  </div>

                  {#if handPage.processed.externalBeaters.length > 0}
                    <div class="mt-2">
                      <div class="text-sm font-semibold mb-1">
                        Vulnerable to ({handPage.processed.externalBeaters.length} dominoes)
                      </div>
                      <div class="flex flex-wrap gap-1 justify-center">
                        {#each handPage.processed.sortedBeaters as dominoStr}
                          <Domino domino={parseDomino(dominoStr)} micro={true} showPoints={false} />
                        {/each}
                      </div>
                    </div>
                  {/if}
                </div>
              {/if}

              <div class="text-center mt-4 text-sm text-base-content/60">
                Page {pageIndex + 1} of {totalPages}
              </div>
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </main>
</div>

<style>
  /* Hide scrollbar for cleaner mobile experience */
  .scrollbar-hide {
    -ms-overflow-style: none;  /* IE and Edge */
    scrollbar-width: none;  /* Firefox */
  }
  .scrollbar-hide::-webkit-scrollbar {
    display: none;  /* Chrome, Safari and Opera */
  }
</style>