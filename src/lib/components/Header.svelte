<script lang="ts">
  import { viewProjection } from '../../stores/gameStore';
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
  let previousPhase = $state($viewProjection.phase);
  let phaseKey = $state(0);
  $effect(() => {
    if ($viewProjection.phase !== previousPhase) {
      previousPhase = $viewProjection.phase;
      phaseKey++;
    }
  });

</script>

<header class="app-header bg-base-100 shadow-lg px-4 py-3 border-b border-base-300" data-testid="app-header">
  <div class="flex justify-between items-center w-full mb-3">
    {#key phaseKey}
      <div class="badge {phaseColors[$viewProjection.phase]} badge-lg gap-2 motion-safe:animate-phase-in">
        {phaseNames[$viewProjection.phase]}
      </div>
    {/key}
    
    <div class="flex gap-2">
      <button 
        class="debug-btn btn btn-circle btn-sm btn-outline btn-secondary min-h-[44px] min-w-[44px]"
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
      <span class="turn-player font-bold">P{$viewProjection.currentPlayer}</span>
    </div>
  </div>
  
  <div class="score-display flex items-center justify-center gap-3">
    <div class="badge badge-lg badge-primary font-semibold">
      US: {$viewProjection.scoring.teamMarks[0]}
    </div>
    <span class="text-xs opacity-50">vs</span>
    <div class="badge badge-lg badge-secondary font-semibold">
      THEM: {$viewProjection.scoring.teamMarks[1]}
    </div>
  </div>
</header>

<style>
  .tap-highlight-transparent {
    -webkit-tap-highlight-color: transparent;
  }
</style>