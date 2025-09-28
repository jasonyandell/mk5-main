<script lang="ts">
  import Domino from './Domino.svelte';
  import { parseDomino } from '../utils/dominoHelpers';
  import type { Domino as DominoType } from '../../game/types';

  interface Props {
    hand: {
      dominoes: string[];
      trump: string;
      type: string;
    };
    index: number;
  }

  let { hand, index }: Props = $props();

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

  const sortedDominoObjects = $derived(() => {
    const dominoes = hand.dominoes.map(d => parseDomino(d));

    // Sort so trumps come first
    return dominoes.sort((a, b) => {
      const aIsTrump = isTrump(a, hand.trump);
      const bIsTrump = isTrump(b, hand.trump);

      if (aIsTrump && !bIsTrump) return -1;
      if (!aIsTrump && bIsTrump) return 1;

      // Within trumps or non-trumps, sort by high pip then low pip
      if (a.high !== b.high) return b.high - a.high;
      return b.low - a.low;
    });
  });

  const trumpDisplay = $derived(
    hand.trump === 'no-trump' ? 'No Trump' :
    hand.trump.charAt(0).toUpperCase() + hand.trump.slice(1)
  );
</script>

<div class="bg-base-100 border border-base-300 p-1">
  <div class="flex items-center justify-between mb-0.5">
    <div class="flex items-center gap-1">
      <span class="text-xs font-semibold">Hand {index + 1}</span>
      <span class="badge badge-xs badge-info">
        {trumpDisplay}
      </span>
    </div>
  </div>

  <div class="flex flex-wrap gap-0.5">
    {#each sortedDominoObjects() as domino}
      <Domino
        {domino}
        micro={true}
        showPoints={false}
      />
    {/each}
  </div>
</div>