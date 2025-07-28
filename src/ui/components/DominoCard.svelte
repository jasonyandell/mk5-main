<script lang="ts">
  import type { Domino } from '../../game/types';
  
  interface Props {
    domino: Domino;
    playable?: boolean;
    selected?: boolean;
    onClick?: () => void;
  }
  
  let { domino, playable = false, selected = false, onClick }: Props = $props();
  
  function handleClick() {
    if (playable && onClick) {
      onClick();
    }
  }
  
  function renderDots(value: number): string {
    // Simplified dot rendering - could be enhanced with actual dot patterns
    return '●'.repeat(value) || '○';
  }
</script>

{#if playable}
  <button 
    class="domino-card playable"
    class:selected
    onclick={handleClick}
  >
    <div class="domino-half top">
      <div class="dots">{renderDots(domino.high)}</div>
      <div class="value">{domino.high}</div>
    </div>
    <div class="domino-divider"></div>
    <div class="domino-half bottom">
      <div class="dots">{renderDots(domino.low)}</div>
      <div class="value">{domino.low}</div>
    </div>
  </button>
{:else}
  <div 
    class="domino-card"
    class:selected
  >
    <div class="domino-half top">
      <div class="dots">{renderDots(domino.high)}</div>
      <div class="value">{domino.high}</div>
    </div>
    <div class="domino-divider"></div>
    <div class="domino-half bottom">
      <div class="dots">{renderDots(domino.low)}</div>
      <div class="value">{domino.low}</div>
    </div>
  </div>
{/if}

<style>
  .domino-card {
    width: 60px;
    height: 100px;
    background: white;
    border: 2px solid #333;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    cursor: default;
    user-select: none;
    transition: all 0.2s ease;
    contain: layout style paint;
  }
  
  .domino-card.playable {
    cursor: pointer;
  }
  
  .domino-card.playable:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
  }
  
  .domino-card.selected {
    border-color: #4CAF50;
    box-shadow: 0 0 0 2px #4CAF50;
  }
  
  .domino-half {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    flex-direction: column;
    gap: 2px;
  }
  
  .domino-divider {
    height: 2px;
    background: #333;
    margin: 0 8px;
  }
  
  .dots {
    font-size: 8px;
    color: #666;
    line-height: 1;
  }
  
  .value {
    font-size: 12px;
    font-weight: bold;
    color: #333;
  }
  
  @media (prefers-reduced-motion: reduce) {
    .domino-card {
      transition: none;
    }
    
    .domino-card.playable:hover {
      transform: none;
    }
  }
</style>