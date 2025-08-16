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

  // Track phase changes for animation
  let previousPhase = $gamePhase;
  let phaseKey = 0;
  $: if ($gamePhase !== previousPhase) {
    previousPhase = $gamePhase;
    phaseKey++;
  }

  // Track score changes for animation
  let previousScores: [number, number] = $teamInfo.marks ? [...$teamInfo.marks] : [0, 0];
  let scoreKeys: [number, number] = [0, 0];
  $: {
    const marks = $teamInfo.marks;
    if (marks) {
      if (marks[0] !== previousScores[0]) {
        scoreKeys = [scoreKeys[0] + 1, scoreKeys[1]];
        previousScores = [marks[0], previousScores[1]];
      }
      if (marks[1] !== previousScores[1]) {
        scoreKeys = [scoreKeys[0], scoreKeys[1] + 1];
        previousScores = [previousScores[0], marks[1]];
      }
    }
  }
</script>

<header class="app-header">
  <div class="game-status">
    {#key phaseKey}
      <div class="phase-indicator phase-animate">
        <span class="phase-dot {phaseColors[$gamePhase]}"></span>
        <span class="phase-name">{phaseNames[$gamePhase]}</span>
      </div>
    {/key}
    
    <div class="turn-display">
      <span class="turn-label">Turn</span>
      <span class="turn-player">P{$currentPlayer?.id ?? 1}</span>
    </div>
  </div>
  
  <div class="score-display">
    {#key scoreKeys[0]}
      <div class="score-card us" class:winning={($teamInfo.marks?.[0] ?? 0) > ($teamInfo.marks?.[1] ?? 0)}>
        <div class="score-value score-animate">{$teamInfo.marks?.[0] ?? 0}</div>
        <div class="score-label">US</div>
        <div class="score-progress">
          <div class="progress-fill" style="width: {(($teamInfo.marks?.[0] ?? 0) / 7) * 100}%"></div>
        </div>
      </div>
    {/key}
    
    <div class="vs-separator">
      <svg viewBox="0 0 24 24" width="20" height="20">
        <path d="M12 2L4 7v10c0 4.42 3.16 8.55 8 9.94 4.84-1.39 8-5.52 8-9.94V7l-8-5z" fill="currentColor" opacity="0.2"/>
        <text x="12" y="16" text-anchor="middle" font-size="10" font-weight="bold" fill="currentColor">VS</text>
      </svg>
    </div>
    
    {#key scoreKeys[1]}
      <div class="score-card them" class:winning={($teamInfo.marks?.[1] ?? 0) > ($teamInfo.marks?.[0] ?? 0)}>
        <div class="score-value score-animate">{$teamInfo.marks?.[1] ?? 0}</div>
        <div class="score-label">THEM</div>
        <div class="score-progress">
          <div class="progress-fill" style="width: {(($teamInfo.marks?.[1] ?? 0) / 7) * 100}%"></div>
        </div>
      </div>
    {/key}
  </div>
  
  <button 
    class="ai-fab"
    class:playing={$quickplayState.enabled}
    on:click={toggleQuickPlay}
  >
    <span class="ai-icon">
      {#if $quickplayState.enabled}
        <svg viewBox="0 0 24 24" width="24" height="24">
          <rect x="6" y="6" width="4" height="12" fill="currentColor"/>
          <rect x="14" y="6" width="4" height="12" fill="currentColor"/>
        </svg>
      {:else}
        <svg viewBox="0 0 24 24" width="24" height="24">
          <path d="M8 5v14l11-7z" fill="currentColor"/>
        </svg>
      {/if}
    </span>
  </button>
</header>

<style>
  .app-header {
    position: relative;
    background: linear-gradient(to bottom, rgba(255, 255, 255, 0.98), rgba(255, 255, 255, 0.95));
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(229, 231, 235, 0.3);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
    padding: 12px 16px;
  }

  .game-status {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .phase-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .phase-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    box-shadow: 0 0 8px currentColor;
  }

  .phase-name {
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #64748b;
  }

  .turn-display {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: rgba(59, 130, 246, 0.1);
    border-radius: 20px;
  }

  .turn-label {
    font-size: 11px;
    color: #64748b;
    font-weight: 500;
  }

  .turn-player {
    font-size: 13px;
    font-weight: 700;
    color: #3b82f6;
  }

  .score-display {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
  }

  .score-card {
    flex: 1;
    max-width: 140px;
    padding: 16px;
    background: white;
    border-radius: 16px;
    box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06);
    border: 2px solid transparent;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
  }

  .score-card.us {
    background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
    border-color: #93c5fd;
  }

  .score-card.them {
    background: linear-gradient(135deg, #fee2e2 0%, #fff1f2 100%);
    border-color: #fca5a5;
  }

  .score-card.winning {
    transform: scale(1.05);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
  }

  .score-card.winning::before {
    content: '👑';
    position: absolute;
    top: -8px;
    right: -8px;
    font-size: 20px;
    transform: rotate(15deg);
  }

  .score-value {
    font-size: 32px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
  }

  .score-card.us .score-value {
    color: #1e40af;
  }

  .score-card.them .score-value {
    color: #dc2626;
  }

  .score-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.1em;
    opacity: 0.7;
  }

  .score-progress {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: rgba(0, 0, 0, 0.1);
  }

  .progress-fill {
    height: 100%;
    background: currentColor;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    opacity: 0.5;
  }

  .vs-separator {
    color: #94a3b8;
  }

  .ai-fab {
    position: absolute;
    top: 50%;
    right: 16px;
    transform: translateY(-50%);
    width: 48px;
    height: 48px;
    border-radius: 24px;
    background: linear-gradient(135deg, #3b82f6, #8b5cf6);
    color: white;
    border: none;
    box-shadow: 0 4px 16px rgba(59, 130, 246, 0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
    -webkit-tap-highlight-color: transparent;
  }

  .ai-fab:active {
    transform: translateY(-50%) scale(0.9);
  }

  .ai-fab.playing {
    background: linear-gradient(135deg, #ef4444, #f97316);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { box-shadow: 0 4px 16px rgba(239, 68, 68, 0.3); }
    50% { box-shadow: 0 4px 24px rgba(239, 68, 68, 0.5); }
  }

  .ai-icon {
    display: flex;
    align-items: center;
    justify-content: center;
  }

  /* Phase animations */
  .phase-animate {
    animation: phaseIn 0.5s cubic-bezier(0.4, 0, 0.2, 1);
  }

  @keyframes phaseIn {
    from {
      opacity: 0;
      transform: translateX(-20px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }

  /* Score animations */
  .score-animate {
    animation: scoreUpdate 0.6s cubic-bezier(0.4, 0, 0.2, 1);
  }

  @keyframes scoreUpdate {
    0% { transform: scale(1); }
    50% { transform: scale(1.3); }
    100% { transform: scale(1); }
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