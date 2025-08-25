<script lang="ts">
  import { gamePhase, teamInfo, currentPlayer } from '../../stores/gameStore';
  import { quickplayState, quickplayActions } from '../../stores/quickplayStore';
  import { GAME_PHASES } from '../../game';
  import { createEventDispatcher } from 'svelte';
  
  const dispatch = createEventDispatcher();

  // Phase badge color mapping (daisyUI classes)
  const phaseColors = {
    [GAME_PHASES.SETUP]: 'badge-neutral',
    [GAME_PHASES.BIDDING]: 'badge-info',
    [GAME_PHASES.TRUMP_SELECTION]: 'badge-secondary',
    [GAME_PHASES.PLAYING]: 'badge-success',
    [GAME_PHASES.SCORING]: 'badge-warning',
    [GAME_PHASES.GAME_END]: 'badge-error'
  };

  // Phase display names
  const phaseNames = {
    [GAME_PHASES.SETUP]: 'Setup',
    [GAME_PHASES.BIDDING]: 'Bidding',
    [GAME_PHASES.TRUMP_SELECTION]: 'Trump Selection',
    [GAME_PHASES.PLAYING]: 'Playing',
    [GAME_PHASES.SCORING]: 'Scoring',
    [GAME_PHASES.GAME_END]: 'Game End'
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

<header class="bg-base-100 shadow-lg px-4 py-3 border-b border-base-300">
  <div class="flex justify-between items-center w-full mb-3">
    {#key phaseKey}
      <div class="badge {phaseColors[$gamePhase]} badge-lg gap-2 motion-safe:animate-phase-in">
        {phaseNames[$gamePhase]}
      </div>
    {/key}
    
    <div class="flex gap-2">
      <button 
        class="btn btn-circle btn-sm btn-outline btn-secondary min-h-[44px] min-w-[44px]"
        onclick={() => dispatch('openDebug')}
        title="Debug Panel"
        aria-label="Open debug panel"
      >
        🔧
      </button>
      <button 
        class="btn btn-circle btn-sm {$quickplayState.enabled ? 'btn-error motion-safe:animate-pulse' : 'btn-primary btn-outline'} min-h-[44px] min-w-[44px]"
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
  
  <div class="flex items-center justify-center gap-4">
    {#key scoreKeys[0]}
      <div class="stat bg-base-200 rounded-box p-3 min-w-0 flex-1 max-w-[140px] {($teamInfo.marks?.[0] ?? 0) > ($teamInfo.marks?.[1] ?? 0) ? 'scale-105 shadow-lg ring-2 ring-primary' : ''} relative">
        {#if ($teamInfo.marks?.[0] ?? 0) > ($teamInfo.marks?.[1] ?? 0)}
          <span class="absolute -top-2 -right-2 text-lg rotate-12">👑</span>
        {/if}
        <div class="stat-title text-xs">US</div>
        <div class="stat-value text-primary text-2xl motion-safe:animate-score-bounce">{$teamInfo.marks?.[0] ?? 0}</div>
        <progress class="progress progress-primary w-full" value={(($teamInfo.marks?.[0] ?? 0) / 7) * 100} max="100"></progress>
      </div>
    {/key}
    
    <div class="divider divider-horizontal mx-2">VS</div>
    
    {#key scoreKeys[1]}
      <div class="stat bg-base-200 rounded-box p-3 min-w-0 flex-1 max-w-[140px] {($teamInfo.marks?.[1] ?? 0) > ($teamInfo.marks?.[0] ?? 0) ? 'scale-105 shadow-lg ring-2 ring-error' : ''} relative">
        {#if ($teamInfo.marks?.[1] ?? 0) > ($teamInfo.marks?.[0] ?? 0)}
          <span class="absolute -top-2 -right-2 text-lg rotate-12">👑</span>
        {/if}
        <div class="stat-title text-xs">THEM</div>
        <div class="stat-value text-error text-2xl motion-safe:animate-score-bounce">{$teamInfo.marks?.[1] ?? 0}</div>
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