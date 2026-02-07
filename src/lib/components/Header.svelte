<script lang="ts">
  import { viewProjection, game, modes, availablePerspectives, currentPerspective } from '../../stores/gameStore';
  import { GAME_PHASES } from '../../game';
  import { createEventDispatcher } from 'svelte';
  import Icon from '../icons/Icon.svelte';
  
  const dispatch = createEventDispatcher();

  // Phase badge color mapping (daisyUI classes)
  const phaseColors = {
    [GAME_PHASES.SETUP]: 'badge-neutral',
    [GAME_PHASES.BIDDING]: 'badge-info',
    [GAME_PHASES.TRUMP_SELECTION]: 'badge-secondary',
    [GAME_PHASES.PLAYING]: 'badge-success',
    [GAME_PHASES.SCORING]: 'badge-warning',
    [GAME_PHASES.GAME_END]: 'badge-error',
    [GAME_PHASES.ONE_HAND_COMPLETE]: 'badge-success'
  };

  // Phase display names - shorter for mobile
  const phaseNames = {
    [GAME_PHASES.SETUP]: 'Setup',
    [GAME_PHASES.BIDDING]: 'Bidding',
    [GAME_PHASES.TRUMP_SELECTION]: 'Trump',
    [GAME_PHASES.PLAYING]: 'Playing',
    [GAME_PHASES.SCORING]: 'Scoring',
    [GAME_PHASES.GAME_END]: 'Game End',
    [GAME_PHASES.ONE_HAND_COMPLETE]: 'Complete'
  };

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
  let selectedPerspective = $state($currentPerspective);
  $effect(() => {
    selectedPerspective = $currentPerspective;
  });

  async function handlePerspectiveChange(event: Event) {
    const target = event.currentTarget as HTMLSelectElement;
    const sessionId = target.value;
    await game.setPerspective(sessionId);
  }
  
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
        <Icon name="verticalDots" size="md" />
      </button>
      {#if menuOpen}
        <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
        <ul 
          tabindex="0" 
          class="dropdown-content menu p-2 shadow-lg bg-base-100 rounded-box w-52 border border-base-300 z-50"
          onmouseleave={closeMenu}
        >
          <li class="px-2 py-1 text-xs uppercase opacity-60">View As</li>
          <li class="px-2 pb-2">
            <select
              class="select select-ghost select-xs w-full"
              bind:value={selectedPerspective}
              onchange={async (event) => {
                await handlePerspectiveChange(event);
                closeMenu();
              }}
            >
              {#each $availablePerspectives as option}
                <option value={option.id}>{option.label}</option>
              {/each}
            </select>
          </li>
          <li>
            <button
              class="new-game-btn flex items-center justify-between"
              onclick={() => {
                game.resetGame();
                closeMenu();
              }}
            >
              <span class="flex items-center gap-2">
                <Icon name="dice" size="sm" />
                New Game
              </span>
            </button>
          </li>
          <li>
            <button
              class="one-hand-btn flex items-center justify-between"
              onclick={() => {
                modes.oneHand.start();
                closeMenu();
              }}
            >
              <span class="flex items-center gap-2">
                <Icon name="handRaised" size="sm" />
                Play One Hand
              </span>
            </button>
          </li>
          <li>
            <button
              class="theme-colors-btn flex items-center justify-between"
              onclick={() => {
                dispatch('openThemeEditor');
                closeMenu();
              }}
            >
              <span class="flex items-center gap-2">
                <Icon name="paintBrush" size="sm" />
                Colors
              </span>
            </button>
          </li>
          <li class="text-base-content/50 text-xs mt-2">
            <div class="flex items-center justify-between px-2">
              <div class="flex items-center gap-2">
                <span>Turn:</span>
                <span class="font-bold">P{$viewProjection.currentPlayer}</span>
              </div>
              <button
                class="settings-btn btn btn-ghost btn-circle btn-xs"
                onclick={() => {
                  dispatch('openSettings');
                  closeMenu();
                }}
                aria-label="Settings"
              >
                <Icon name="cog" size="sm" />
              </button>
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
