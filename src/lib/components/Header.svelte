<script lang="ts">
  import { gamePhase, teamInfo, currentPlayer } from '../../stores/gameStore';
  import { quickplayState, quickplayActions } from '../../stores/quickplayStore';
  import { createEventDispatcher } from 'svelte';
  
  const dispatch = createEventDispatcher();

  // Phase badge color mapping (daisyUI classes)
  const phaseColors = {
    setup: 'badge-neutral',
    bidding: 'badge-info',
    trump_selection: 'badge-secondary',
    playing: 'badge-success',
    scoring: 'badge-warning',
    game_end: 'badge-error'
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
  let previousPhase = $state($gamePhase);
  let phaseKey = $state(0);
  $effect(() => {
    if ($gamePhase !== previousPhase) {
      previousPhase = $gamePhase;
      phaseKey++;
    }
  });

  // Track score changes for animation
  let previousScores = $state<[number, number]>($teamInfo.marks ? [...$teamInfo.marks] : [0, 0]);
  let scoreKeys = $state<[number, number]>([0, 0]);
  $effect(() => {
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
  });
</script>

<header class="bg-base-100 shadow-lg px-3 py-2 sm:px-4 sm:py-3 border-b border-base-300">
  <div class="flex justify-between items-center w-full mb-2 sm:mb-3">
    {#key phaseKey}
      <div class="badge {phaseColors[$gamePhase]} badge-lg gap-2 animate-phase-in">
        {phaseNames[$gamePhase]}
      </div>
    {/key}
    
    <div class="flex gap-1.5">
      <button 
        class="btn btn-circle btn-sm btn-outline btn-secondary"
        onclick={() => dispatch('openDebug')}
        title="Debug Panel"
        aria-label="Open debug panel"
      >
        🔧
      </button>
      <button 
        class="btn btn-circle btn-sm {$quickplayState.enabled ? 'btn-error animate-pulse' : 'btn-primary btn-outline'}"
        onclick={toggleQuickPlay}
        title={$quickplayState.enabled ? 'Pause AI' : 'Start AI'}
        aria-label={$quickplayState.enabled ? 'Pause AI play' : 'Start AI play'}
      >
        {$quickplayState.enabled ? '⏸' : '🤖'}
      </button>
    </div>
    
    <div class="badge badge-info badge-outline badge-lg">
      <span class="text-xs font-medium mr-1">Turn</span>
      <span class="font-bold">P{$currentPlayer?.id ?? 1}</span>
    </div>
  </div>
  
  <div class="flex items-center justify-center gap-3 sm:gap-4">
    {#key scoreKeys[0]}
      <div class="stat bg-base-200 rounded-box p-2 sm:p-3 min-w-0 flex-1 max-w-[140px] {($teamInfo.marks?.[0] ?? 0) > ($teamInfo.marks?.[1] ?? 0) ? 'scale-105 shadow-lg ring-2 ring-primary' : ''} relative">
        {#if ($teamInfo.marks?.[0] ?? 0) > ($teamInfo.marks?.[1] ?? 0)}
          <span class="absolute -top-2 -right-2 text-lg rotate-12">👑</span>
        {/if}
        <div class="stat-title text-xs">US</div>
        <div class="stat-value text-primary text-xl sm:text-2xl animate-score-bounce">{$teamInfo.marks?.[0] ?? 0}</div>
        <progress class="progress progress-primary w-full" value={(($teamInfo.marks?.[0] ?? 0) / 7) * 100} max="100"></progress>
      </div>
    {/key}
    
    <div class="divider divider-horizontal mx-1 sm:mx-2">VS</div>
    
    {#key scoreKeys[1]}
      <div class="stat bg-base-200 rounded-box p-2 sm:p-3 min-w-0 flex-1 max-w-[140px] {($teamInfo.marks?.[1] ?? 0) > ($teamInfo.marks?.[0] ?? 0) ? 'scale-105 shadow-lg ring-2 ring-error' : ''} relative">
        {#if ($teamInfo.marks?.[1] ?? 0) > ($teamInfo.marks?.[0] ?? 0)}
          <span class="absolute -top-2 -right-2 text-lg rotate-12">👑</span>
        {/if}
        <div class="stat-title text-xs">THEM</div>
        <div class="stat-value text-error text-xl sm:text-2xl animate-score-bounce">{$teamInfo.marks?.[1] ?? 0}</div>
        <progress class="progress progress-error w-full" value={(($teamInfo.marks?.[1] ?? 0) / 7) * 100} max="100"></progress>
      </div>
    {/key}
  </div>
</header>

<style>
  .tap-highlight-transparent {
    -webkit-tap-highlight-color: transparent;
  }
</style>