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
  <div class="header-left">
    <h1>Texas 42</h1>
    {#key phaseKey}
      <span class="phase-badge {phaseColors[$gamePhase]} phase-badge-change">
        {phaseNames[$gamePhase]}
      </span>
    {/key}
  </div>

  <div class="header-center">
    <div class="ai-controls">
      <button 
        class="ai-button"
        on:click={toggleQuickPlay}
        title={$quickplayState.enabled ? 'Pause AI' : 'Play All'}
      >
        {#if $quickplayState.enabled}
          <span class="icon">⏸️</span> Pause
        {:else}
          <span class="icon">▶️</span> Play All
        {/if}
      </button>
      <button 
        class="ai-button"
        on:click={stepQuickPlay}
        title="Step one AI action"
        disabled={$quickplayState.enabled}
      >
        <span class="icon">⏭️</span> Step
      </button>
      {#if $quickplayState.enabled}
        <span class="ai-status">AI Playing</span>
      {/if}
    </div>
  </div>

  <div class="header-right">
    <div class="team-scores">
      {#key scoreKeys[0]}
        <span class="score us score-update-roll">US: {$teamInfo.marks[0]}</span>
      {/key}
      <span class="score-separator">•</span>
      {#key scoreKeys[1]}
        <span class="score them score-update-roll">THEM: {$teamInfo.marks[1]}</span>
      {/key}
    </div>
    <div class="turn-indicator">
      Turn: <span class="player-badge">P{$currentPlayer.id}</span>
    </div>
  </div>
</header>

<style>
  .app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background-color: #ffffff;
    border-bottom: 2px solid #e5e7eb;
    min-height: 60px;
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  h1 {
    margin: 0;
    font-size: 24px;
    font-weight: bold;
    color: #002868;
  }

  .phase-badge {
    padding: 4px 12px;
    border-radius: 16px;
    color: white;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .header-center {
    flex: 1;
    display: flex;
    justify-content: center;
  }

  .ai-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .ai-button {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    background-color: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
    transition: background-color 0.2s;
  }

  .ai-button:hover:not(:disabled) {
    background-color: #e5e7eb;
  }

  .ai-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .icon {
    font-size: 16px;
  }

  .ai-status {
    margin-left: 8px;
    font-size: 12px;
    color: #059669;
    font-weight: 500;
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 20px;
  }

  .team-scores {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 16px;
    font-weight: 600;
  }

  .score.us {
    color: #002868;
  }

  .score.them {
    color: #dc2626;
  }

  .score-separator {
    color: #9ca3af;
  }

  .turn-indicator {
    font-size: 14px;
    color: #4b5563;
  }

  .player-badge {
    padding: 2px 8px;
    background-color: #002868;
    color: white;
    border-radius: 4px;
    font-weight: 600;
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