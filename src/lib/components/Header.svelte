<script lang="ts">
  import { gamePhase, teamInfo, currentPlayer } from '../../stores/gameStore';
  import { quickplayState, quickplayActions } from '../../stores/quickplayStore';

  // Phase badge color mapping
  const phaseColors = {
    setup: 'bg-gray-500',
    bidding: 'bg-blue-500',
    trump_selection: 'bg-purple-500',
    playing: 'bg-green-500',
    scoring: 'bg-yellow-500',
    game_end: 'bg-red-500'
  };

  // Phase display names
  const phaseNames = {
    setup: 'Setup',
    bidding: 'Bidding',
    trump_selection: 'Trump Selection',
    playing: 'Playing',
    scoring: 'Scoring',
    game_end: 'Game End'
  };

  function toggleQuickPlay() {
    quickplayActions.toggle();
  }

  function stepQuickPlay() {
    quickplayActions.step();
  }

  // Track phase changes for animation
  let previousPhase = $gamePhase;
  let phaseKey = 0;
  $: if ($gamePhase !== previousPhase) {
    previousPhase = $gamePhase;
    phaseKey++;
  }

  // Track score changes for animation
  let previousScores = [...$teamInfo.marks];
  let scoreKeys = [0, 0];
  $: {
    if ($teamInfo.marks[0] !== previousScores[0]) {
      scoreKeys[0]++;
      previousScores[0] = $teamInfo.marks[0];
    }
    if ($teamInfo.marks[1] !== previousScores[1]) {
      scoreKeys[1]++;
      previousScores[1] = $teamInfo.marks[1];
    }
  }
</script>

<header class="app-header">
  <div class="header-top">
    <div class="scores-section">
      {#key scoreKeys[0]}
        <div class="score-box us">
          <span class="team-label">US</span>
          <span class="score score-update-roll">{$teamInfo.marks[0]}</span>
        </div>
      {/key}
      
      <div class="game-info">
        <h1>Texas 42</h1>
        {#key phaseKey}
          <div class="phase-badge {phaseColors[$gamePhase]} phase-badge-change">
            {phaseNames[$gamePhase]}
          </div>
        {/key}
      </div>
      
      {#key scoreKeys[1]}
        <div class="score-box them">
          <span class="team-label">THEM</span>
          <span class="score score-update-roll">{$teamInfo.marks[1]}</span>
        </div>
      {/key}
    </div>
  </div>
  
  <div class="header-bottom">
    <div class="turn-indicator">
      P{$currentPlayer.id}'s Turn
    </div>
    
    <div class="ai-controls">
      <button 
        class="ai-button"
        on:click={toggleQuickPlay}
      >
        {#if $quickplayState.enabled}
          ⏸️
        {:else}
          ▶️
        {/if}
      </button>
    </div>
  </div>
</header>

<style>
  .app-header {
    background-color: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
  }

  .header-top {
    padding: 8px 12px;
  }

  .scores-section {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }

  .score-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 12px;
    border-radius: 8px;
    min-width: 50px;
  }

  .score-box.us {
    background-color: #e0f2fe;
    color: #0369a1;
  }

  .score-box.them {
    background-color: #fee2e2;
    color: #dc2626;
  }

  .team-label {
    font-size: 11px;
    font-weight: 600;
    opacity: 0.8;
  }

  .score {
    font-size: 20px;
    font-weight: bold;
  }

  .game-info {
    text-align: center;
    flex: 1;
  }

  h1 {
    margin: 0;
    font-size: 18px;
    font-weight: bold;
    color: #002868;
  }

  .phase-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    color: white;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
  }

  .header-bottom {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background-color: #f9fafb;
    border-top: 1px solid #e5e7eb;
  }

  .turn-indicator {
    font-size: 14px;
    font-weight: 600;
    color: #374151;
  }

  .ai-controls {
    display: flex;
    gap: 8px;
  }

  .ai-button {
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 20px;
    cursor: pointer;
    font-size: 20px;
    transition: all 0.2s;
    -webkit-tap-highlight-color: transparent;
  }

  .ai-button:active {
    transform: scale(0.9);
  }

  /* Phase colors */
  .bg-gray-500 {
    background-color: #6b7280;
  }

  .bg-blue-500 {
    background-color: #3b82f6;
  }

  .bg-purple-500 {
    background-color: #8b5cf6;
  }

  .bg-green-500 {
    background-color: #22c55e;
  }

  .bg-yellow-500 {
    background-color: #f59e0b;
  }

  .bg-red-500 {
    background-color: #dc2626;
  }
</style>