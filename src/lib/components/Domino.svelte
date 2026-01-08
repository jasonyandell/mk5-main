<script lang="ts">
  import type { Domino } from '../../game/types';
  import { createEventDispatcher } from 'svelte';

  interface Props {
    domino: Domino;
    playable?: boolean;
    clickable?: boolean;
    small?: boolean;
    tiny?: boolean;
    micro?: boolean;
    showPoints?: boolean;
    winner?: boolean;
    tooltip?: string;
  }

  let {
    domino,
    playable = false,
    clickable = false,
    small = false,
    tiny = false,
    micro = false,
    showPoints = true,
    winner = false,
    tooltip = ''
  }: Props = $props();

  const dispatch = createEventDispatcher();

  function handleClick() {
    if (!clickable) return;
    dispatch('click', domino);
  }

  // Pip patterns for domino faces
  const pipPatterns: { [key: number]: string[] } = {
    0: [],
    1: ['center'],
    2: ['top-left', 'bottom-right'],
    3: ['top-left', 'center', 'bottom-right'],
    4: ['top-left', 'top-right', 'bottom-left', 'bottom-right'],
    5: ['top-left', 'top-right', 'center', 'bottom-left', 'bottom-right'],
    6: ['top-left', 'top-right', 'middle-left', 'middle-right', 'bottom-left', 'bottom-right']
  };

  // Counting dominoes (10 or 5 points)
  const isCounter = $derived(domino.points && domino.points > 0);
  const pointValue = $derived(domino.points || 0);

  // Size classes
  const sizeClasses = $derived(
    micro ? 'w-7 h-11' :
    tiny ? 'w-9 h-14' :
    small ? 'w-11 h-[68px]' :
    'w-14 h-[88px]'
  );
  const pipSize = $derived(
    micro ? 'w-1 h-1' :
    tiny ? 'w-1.5 h-1.5' :
    small ? 'w-1.5 h-1.5' :
    'w-2 h-2'
  );
  
  // State classes - keeping white background but adding colored borders/rings
  const stateClasses = $derived(playable 
    ? 'border-success !border-4 -translate-y-1 scale-105 shadow-xl hover:shadow-2xl ring-2 ring-success/50' 
    : clickable
    ? 'hover:scale-105 hover:shadow-xl cursor-pointer'
    : 'cursor-default');

  // Get pip position classes
  function getPipPosition(position: string): string {
    switch(position) {
      case 'top-left': return 'top-[15%] left-[20%]';
      case 'top-right': return 'top-[15%] right-[20%]';
      case 'center': return 'top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2';
      case 'bottom-left': return 'bottom-[15%] left-[20%]';
      case 'bottom-right': return 'bottom-[15%] right-[20%]';
      case 'middle-left': return 'top-1/2 left-[20%] -translate-y-1/2';
      case 'middle-right': return 'top-1/2 right-[20%] -translate-y-1/2';
      default: return '';
    }
  }
</script>

<button
  class="relative bg-white {winner ? 'border-4 border-primary' : 'border-2 border-base-300'} rounded-md shadow-lg transition-all {sizeClasses} {stateClasses} p-0 overflow-visible min-h-touch"
  onclick={handleClick}
  disabled={!clickable}
  title={tooltip || domino.high + '-' + domino.low}
  data-testid="domino-{domino.high}-{domino.low}"
>
  <!-- Top half -->
  <div class="relative h-[45%] flex items-center justify-center">
    {#each pipPatterns[domino.high] || [] as position}
      <span class="absolute {pipSize} bg-black rounded-full {getPipPosition(position)}"></span>
    {/each}
  </div>
  
  <!-- Divider -->
  <div class="h-[1px] bg-black/30 mx-2"></div>
  
  <!-- Bottom half -->
  <div class="relative h-[45%] flex items-center justify-center">
    {#each pipPatterns[domino.low] || [] as position}
      <span class="absolute {pipSize} bg-black rounded-full {getPipPosition(position)}"></span>
    {/each}
  </div>
  
  <!-- Points badge for counting dominoes -->
  {#if isCounter && showPoints}
    <span class="absolute -top-2 -right-2 badge badge-warning badge-sm font-bold">
      {pointValue}
    </span>
  {/if}
</button>