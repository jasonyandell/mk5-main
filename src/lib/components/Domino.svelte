<script lang="ts">
  import type { Domino } from '../../game/types';

  export let domino: Domino;
  export let playable: boolean = false;
  export let clickable: boolean = false;
  export let small: boolean = false;
  export let tiny: boolean = false;
  export let showPoints: boolean = true;
  export let winner: boolean = false;
  export let highlight: 'primary' | 'secondary' | null = null;
  export let tooltip: string = '';

  function handleClick() {
    if (clickable) {
      dispatch('click', domino);
    }
  }

  import { createEventDispatcher } from 'svelte';
  const dispatch = createEventDispatcher();

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
  $: isCounter = domino.points && domino.points > 0;
  $: pointValue = domino.points || 0;
  
</script>

<button
  class="domino"
  class:playable
  class:clickable
  class:small
  class:tiny
  class:counter={isCounter}
  class:winner
  class:disabled={!playable && clickable}
  class:highlight-primary={highlight === 'primary'}
  class:highlight-secondary={highlight === 'secondary'}
  on:click={handleClick}
  on:mouseenter
  on:mousemove
  on:mouseleave
  disabled={!clickable || (!playable && clickable)}
  title={tooltip || domino.high + '-' + domino.low}
  data-testid="domino-{domino.high}-{domino.low}"
>
  <div class="domino-face">
    <div class="domino-half">
      {#each pipPatterns[domino.high] || [] as position}
        <span class="pip {position}"></span>
      {/each}
    </div>
    <div class="domino-divider"></div>
    <div class="domino-half">
      {#each pipPatterns[domino.low] || [] as position}
        <span class="pip {position}"></span>
      {/each}
    </div>
  </div>
  {#if isCounter && showPoints}
    <span class="point-badge">{pointValue}</span>
  {/if}
</button>

<style>
  .domino {
    position: relative;
    background-color: #f5f3f0;
    border: 2px solid #8b4513;
    border-radius: 8px;
    padding: 0;
    cursor: default;
    transition: all 0.2s ease;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .domino.tiny {
    width: 36px;
    height: 56px;
  }

  .domino.small:not(.tiny) {
    width: 45px;
    height: 72px;
  }

  .domino:not(.small):not(.tiny) {
    width: 60px;
    height: 94px;
  }

  .domino.clickable {
    cursor: pointer;
  }

  .domino.playable {
    background-color: #e8f5e8;
    border-color: #22c55e;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(34, 197, 94, 0.2);
  }

  .domino.clickable:hover:not(.disabled) {
    transform: scale(1.05) translateY(-4px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
  }

  .domino.disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .domino.winner {
    background-color: #ede9fe;
    border-color: #8b5cf6;
    box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.2);
  }

  .domino.highlight-primary {
    background-color: #fef3c7 !important;
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 3px rgba(245, 158, 11, 0.3) !important;
  }

  .domino.highlight-secondary {
    background-color: #dbeafe !important;
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.3) !important;
  }

  .domino-face {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 4px;
  }

  .domino-half {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .domino-divider {
    height: 2px;
    background-color: #8b4513;
    margin: 2px 4px;
  }

  .pip {
    position: absolute;
    width: 8px;
    height: 8px;
    background-color: #1a1a1a;
    border-radius: 50%;
  }

  .small .pip {
    width: 6px;
    height: 6px;
  }

  .tiny .pip {
    width: 4px;
    height: 4px;
  }

  .tiny .domino-divider {
    height: 1px;
    margin: 1px 2px;
  }

  /* Pip positions */
  .pip.center {
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
  }

  .pip.top-left {
    top: 20%;
    left: 25%;
  }

  .pip.top-right {
    top: 20%;
    right: 25%;
  }

  .pip.middle-left {
    top: 50%;
    left: 25%;
    transform: translateY(-50%);
  }

  .pip.middle-right {
    top: 50%;
    right: 25%;
    transform: translateY(-50%);
  }

  .pip.bottom-left {
    bottom: 20%;
    left: 25%;
  }

  .pip.bottom-right {
    bottom: 20%;
    right: 25%;
  }

  .point-badge {
    position: absolute;
    top: -8px;
    right: -8px;
    background-color: #f59e0b;
    color: white;
    font-size: 12px;
    font-weight: bold;
    padding: 2px 6px;
    border-radius: 12px;
    border: 2px solid white;
  }
</style>