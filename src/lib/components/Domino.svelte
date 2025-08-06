<script lang="ts">
  import type { Domino } from '../../game/types';

  export let domino: Domino;
  export let playable: boolean = false;
  export let clickable: boolean = false;
  export let small: boolean = false;
  export let tiny: boolean = false;
  export let showPoints: boolean = true;
  export let winner: boolean = false;
  export let tooltip: string = '';

  function handleClick() {
    if (!clickable) return;
    dispatch('click', domino);
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
  class:disabled={false}
  on:click={handleClick}
  on:mouseenter
  on:mousemove
  on:mouseleave
  disabled={!clickable}
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
    background: linear-gradient(145deg, #fdfcfb 0%, #f0ebe4 100%);
    border: 2px solid #8b7355;
    border-radius: 12px;
    padding: 0;
    cursor: default;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 
      0 2px 4px rgba(0, 0, 0, 0.1),
      0 1px 2px rgba(0, 0, 0, 0.06),
      inset 0 1px 1px rgba(255, 255, 255, 0.5);
    will-change: transform;
  }

  .domino.tiny {
    width: 36px;
    height: 56px;
    border-radius: 8px;
  }

  .domino.small:not(.tiny) {
    width: 45px;
    height: 72px;
    border-radius: 10px;
  }

  .domino:not(.small):not(.tiny) {
    width: 56px;
    height: 88px;
  }

  .domino.clickable {
    cursor: pointer;
  }

  .domino.clickable:active:not(.disabled) {
    transform: scale(0.95);
  }

  .domino.playable {
    background: linear-gradient(145deg, #e8f9e8 0%, #d4f1d4 100%);
    border-color: #22c55e;
    transform: translateY(-3px) scale(1.02);
    box-shadow: 
      0 8px 16px rgba(34, 197, 94, 0.25),
      0 0 0 3px rgba(34, 197, 94, 0.15),
      inset 0 1px 2px rgba(255, 255, 255, 0.7);
    animation: playablePulse 2s ease-in-out infinite;
  }

  @keyframes playablePulse {
    0%, 100% { transform: translateY(-3px) scale(1.02); }
    50% { transform: translateY(-5px) scale(1.04); }
  }

  .domino.clickable:hover:not(.disabled) {
    transform: translateY(-6px) scale(1.06) rotate(1deg);
    box-shadow: 
      0 12px 24px rgba(0, 0, 0, 0.15),
      0 0 0 4px rgba(139, 92, 246, 0.1);
  }

  .domino.disabled {
    opacity: 0.5;
    cursor: not-allowed;
    filter: grayscale(0.3);
  }

  .domino.winner {
    background: linear-gradient(145deg, #f3f0ff 0%, #e9e3ff 100%);
    border-color: #8b5cf6;
    box-shadow: 
      0 0 0 4px rgba(139, 92, 246, 0.3),
      0 8px 24px rgba(139, 92, 246, 0.2);
    animation: winnerGlow 1s ease-in-out;
  }

  @keyframes winnerGlow {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
  }


  .domino-face {
    display: flex;
    flex-direction: column;
    height: 100%;
    padding: 6px;
    position: relative;
  }

  .domino-face::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 10%;
    right: 10%;
    height: 1px;
    background: linear-gradient(90deg, 
      transparent 0%, 
      rgba(139, 69, 19, 0.2) 20%, 
      rgba(139, 69, 19, 0.4) 50%, 
      rgba(139, 69, 19, 0.2) 80%, 
      transparent 100%);
    transform: translateY(-50%);
  }

  .domino-half {
    flex: 1;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .domino-divider {
    height: 3px;
    background: linear-gradient(90deg, 
      transparent 0%, 
      #8b7355 10%, 
      #6d5a45 50%, 
      #8b7355 90%, 
      transparent 100%);
    margin: 3px 6px;
    border-radius: 2px;
    position: relative;
    z-index: 1;
  }

  .pip {
    position: absolute;
    width: 9px;
    height: 9px;
    background: radial-gradient(circle at 30% 30%, #333, #000);
    border-radius: 50%;
    box-shadow: 
      inset 0 1px 2px rgba(0, 0, 0, 0.5),
      0 1px 1px rgba(255, 255, 255, 0.1);
  }

  .small .pip {
    width: 7px;
    height: 7px;
  }

  .tiny .pip {
    width: 5px;
    height: 5px;
  }

  .tiny .domino-divider {
    height: 2px;
    margin: 2px 3px;
  }

  /* Pip positions - adjusted to prevent overlap */
  .pip.center {
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
  }

  .pip.top-left {
    top: 14%;
    left: 16%;
  }

  .pip.top-right {
    top: 14%;
    right: 16%;
  }

  .pip.middle-left {
    top: 50%;
    left: 16%;
    transform: translateY(-50%);
  }

  .pip.middle-right {
    top: 50%;
    right: 16%;
    transform: translateY(-50%);
  }

  .pip.bottom-left {
    bottom: 14%;
    left: 16%;
  }

  .pip.bottom-right {
    bottom: 14%;
    right: 16%;
  }

  .point-badge {
    position: absolute;
    top: -10px;
    right: -10px;
    background: linear-gradient(135deg, #fbbf24, #f59e0b);
    color: white;
    font-size: 11px;
    font-weight: 800;
    padding: 3px 7px;
    border-radius: 14px;
    border: 2px solid white;
    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.5);
    z-index: 2;
    animation: badgeBounce 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }

  @keyframes badgeBounce {
    0% { transform: scale(0); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
  }
</style>