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
  export let highlightColor: string | null = null;
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
  class:highlight-suit-0={highlightColor === 'suit-0'}
  class:highlight-suit-1={highlightColor === 'suit-1'}
  class:highlight-suit-2={highlightColor === 'suit-2'}
  class:highlight-suit-3={highlightColor === 'suit-3'}
  class:highlight-suit-4={highlightColor === 'suit-4'}
  class:highlight-suit-5={highlightColor === 'suit-5'}
  class:highlight-suit-6={highlightColor === 'suit-6'}
  class:highlight-doubles={highlightColor === 'doubles'}
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

  .domino.highlight-primary {
    background: linear-gradient(145deg, #fffbeb 0%, #fef3c7 100%) !important;
    border-color: #f59e0b !important;
    box-shadow: 
      0 0 0 4px rgba(245, 158, 11, 0.4),
      0 8px 16px rgba(245, 158, 11, 0.2) !important;
    animation: highlightPrimary 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    transform: translateY(-4px) scale(1.05);
  }

  @keyframes highlightPrimary {
    0% { 
      transform: scale(1) translateY(0);
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    50% {
      transform: scale(1.08) translateY(-6px);
    }
    100% { 
      transform: scale(1.05) translateY(-4px);
    }
  }

  .domino.highlight-secondary {
    background: linear-gradient(145deg, #eff6ff 0%, #dbeafe 100%) !important;
    border-color: #3b82f6 !important;
    box-shadow: 
      0 0 0 3px rgba(59, 130, 246, 0.3),
      0 6px 12px rgba(59, 130, 246, 0.15) !important;
    animation: highlightSecondary 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    transform: translateY(-2px) scale(1.03);
  }

  @keyframes highlightSecondary {
    0% { 
      transform: scale(1) translateY(0);
      opacity: 0.8;
    }
    50% {
      transform: scale(1.05) translateY(-3px);
      opacity: 1;
    }
    100% { 
      transform: scale(1.03) translateY(-2px);
      opacity: 1;
    }
  }

  /* Suit-specific highlights for bidding */
  .domino.highlight-suit-0 { /* Blanks - White/Gray */
    background: linear-gradient(145deg, #f9fafb 0%, #e5e7eb 100%) !important;
    border-color: #6b7280 !important;
    box-shadow: 
      0 0 0 3px rgba(107, 114, 128, 0.3),
      0 6px 12px rgba(107, 114, 128, 0.2) !important;
    transform: translateY(-3px) scale(1.04);
  }

  .domino.highlight-suit-1 { /* Ones - Red */
    background: linear-gradient(145deg, #fee2e2 0%, #fecaca 100%) !important;
    border-color: #ef4444 !important;
    box-shadow: 
      0 0 0 3px rgba(239, 68, 68, 0.3),
      0 6px 12px rgba(239, 68, 68, 0.2) !important;
    transform: translateY(-3px) scale(1.04);
  }

  .domino.highlight-suit-2 { /* Twos - Orange */
    background: linear-gradient(145deg, #fed7aa 0%, #fdba74 100%) !important;
    border-color: #f97316 !important;
    box-shadow: 
      0 0 0 3px rgba(249, 115, 22, 0.3),
      0 6px 12px rgba(249, 115, 22, 0.2) !important;
    transform: translateY(-3px) scale(1.04);
  }

  .domino.highlight-suit-3 { /* Threes - Yellow */
    background: linear-gradient(145deg, #fef3c7 0%, #fde68a 100%) !important;
    border-color: #f59e0b !important;
    box-shadow: 
      0 0 0 3px rgba(245, 158, 11, 0.3),
      0 6px 12px rgba(245, 158, 11, 0.2) !important;
    transform: translateY(-3px) scale(1.04);
  }

  .domino.highlight-suit-4 { /* Fours - Green */
    background: linear-gradient(145deg, #d9f99d 0%, #bef264 100%) !important;
    border-color: #84cc16 !important;
    box-shadow: 
      0 0 0 3px rgba(132, 204, 22, 0.3),
      0 6px 12px rgba(132, 204, 22, 0.2) !important;
    transform: translateY(-3px) scale(1.04);
  }

  .domino.highlight-suit-5 { /* Fives - Blue */
    background: linear-gradient(145deg, #dbeafe 0%, #bfdbfe 100%) !important;
    border-color: #3b82f6 !important;
    box-shadow: 
      0 0 0 3px rgba(59, 130, 246, 0.3),
      0 6px 12px rgba(59, 130, 246, 0.2) !important;
    transform: translateY(-3px) scale(1.04);
  }

  .domino.highlight-suit-6 { /* Sixes - Purple */
    background: linear-gradient(145deg, #e9d5ff 0%, #d8b4fe 100%) !important;
    border-color: #a855f7 !important;
    box-shadow: 
      0 0 0 3px rgba(168, 85, 247, 0.3),
      0 6px 12px rgba(168, 85, 247, 0.2) !important;
    transform: translateY(-3px) scale(1.04);
  }

  .domino.highlight-doubles { /* Doubles - Pink/Rose */
    background: linear-gradient(145deg, #fce7f3 0%, #fbcfe8 100%) !important;
    border-color: #ec4899 !important;
    box-shadow: 
      0 0 0 3px rgba(236, 72, 153, 0.3),
      0 6px 12px rgba(236, 72, 153, 0.2) !important;
    transform: translateY(-3px) scale(1.04);
  }

  .domino.highlight-suit-0,
  .domino.highlight-suit-1,
  .domino.highlight-suit-2,
  .domino.highlight-suit-3,
  .domino.highlight-suit-4,
  .domino.highlight-suit-5,
  .domino.highlight-suit-6,
  .domino.highlight-doubles {
    animation: suitHighlight 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
  }

  @keyframes suitHighlight {
    0% { 
      transform: scale(1) translateY(0);
    }
    50% {
      transform: scale(1.06) translateY(-5px);
    }
    100% { 
      transform: scale(1.04) translateY(-3px);
    }
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