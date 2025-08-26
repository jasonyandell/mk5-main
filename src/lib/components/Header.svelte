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

  // Phase display names - shorter for mobile
  const phaseNames = {
    [GAME_PHASES.SETUP]: 'Setup',
    [GAME_PHASES.BIDDING]: 'Bidding',
    [GAME_PHASES.TRUMP_SELECTION]: 'Trump',
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

  // Dropdown menu state
  let menuOpen = $state(false);
  
  function closeMenu() {
    menuOpen = false;
  }

</script>

<header class="app-header bg-base-100 border-b border-base-300" data-testid="app-header">
  <div class="flex justify-between items-center px-3 py-2">
    <!-- Phase indicator on left -->
    {#key phaseKey}
      <div class="badge {phaseColors[$viewProjection.phase]} badge-sm font-medium motion-safe:animate-phase-in">
        {phaseNames[$viewProjection.phase]}
      </div>
    {/key}
    
    <!-- Scores in center -->
    <div class="flex items-center gap-3 text-sm font-semibold">
      <span class="text-primary">US {$viewProjection.scoring.teamMarks[0]}</span>
      <span class="text-base-content/40">•</span>
      <span class="text-secondary">THEM {$viewProjection.scoring.teamMarks[1]}</span>
    </div>
    
    <!-- Menu dropdown on right -->
    <div class="dropdown dropdown-end">
      <button 
        tabindex="0"
        class="btn btn-ghost btn-circle btn-sm"
        onclick={() => menuOpen = !menuOpen}
        aria-label="Menu"
      >
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-5 h-5">
          <path stroke-linecap="round" stroke-linejoin="round" d="M12 6.75a.75.75 0 110-1.5.75.75 0 010 1.5zM12 12.75a.75.75 0 110-1.5.75.75 0 010 1.5zM12 18.75a.75.75 0 110-1.5.75.75 0 010 1.5z" />
        </svg>
      </button>
      {#if menuOpen}
        <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
        <ul 
          tabindex="0" 
          class="dropdown-content menu p-2 shadow-lg bg-base-100 rounded-box w-52 border border-base-300 z-50"
          onmouseleave={closeMenu}
        >
          <li>
            <button
              class="settings-btn flex items-center justify-between"
              onclick={() => {
                dispatch('openSettings');
                closeMenu();
              }}
            >
              <span>⚙️ Settings</span>
            </button>
          </li>
          <li>
            <button
              class="flex items-center justify-between"
              onclick={() => {
                toggleQuickPlay();
                closeMenu();
              }}
            >
              <span>{$quickplayState.enabled ? '✋ Manual Play' : '🤖 Auto Play'}</span>
              {#if $quickplayState.enabled}
                <span class="badge badge-success badge-xs animate-pulse">AUTO</span>
              {/if}
            </button>
          </li>
          <li class="text-base-content/50 text-xs mt-2">
            <div class="flex items-center gap-2">
              <span>Turn:</span>
              <span class="font-bold">P{$viewProjection.currentPlayer}</span>
            </div>
          </li>
        </ul>
      {/if}
    </div>
  </div>
</header>

<style>
  .tap-highlight-transparent {
    -webkit-tap-highlight-color: transparent;
  }
</style>