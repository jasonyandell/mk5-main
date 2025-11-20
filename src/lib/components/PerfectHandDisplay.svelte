<script lang="ts">
  import Domino from './Domino.svelte';
  import { parseDomino } from '../utils/dominoHelpers';
  import { sortDominoesForDisplay } from '../utils/domino-sort';

  interface Props {
    hand: {
      dominoes: string[];
      trump: string;
      type: string;
    };
    index: number;
  }

  let { hand, index }: Props = $props();

  const sortedDominoObjects = $derived(() => {
    const dominoes = hand.dominoes.map(d => parseDomino(d));
    return sortDominoesForDisplay(dominoes, hand.trump);
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