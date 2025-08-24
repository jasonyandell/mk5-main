<script lang="ts">
  import { eventGame } from '../../stores/eventGame';
  
  interface Props {
    onclose: () => void;
  }

  let { onclose }: Props = $props();

  const events = eventGame.events;
  const gameState = eventGame.getGameState();

  function handleTimeTravel(index: number) {
    eventGame.timeTravel(index);
  }

  function handleClear() {
    eventGame.clear();
    onclose();
  }

  function handleEnableQuickplay() {
    eventGame.enableQuickplay('fast');
  }

  function handleDisableQuickplay() {
    eventGame.disableQuickplay();
  }
</script>

<div class="p-4 h-full overflow-y-auto">
  <div class="flex justify-between items-center mb-4">
    <h2 class="text-xl font-bold">Debug Panel</h2>
    <button class="btn btn-sm btn-circle" onclick={onclose}>
      <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  </div>

  <div class="space-y-4">
    <div class="card bg-base-200">
      <div class="card-body">
        <h3 class="card-title text-sm">Game Controls</h3>
        <div class="flex gap-2">
          <button class="btn btn-sm btn-primary" onclick={handleEnableQuickplay}>
            Enable Quickplay
          </button>
          <button class="btn btn-sm btn-secondary" onclick={handleDisableQuickplay}>
            Disable Quickplay
          </button>
          <button class="btn btn-sm btn-error" onclick={handleClear}>
            New Game
          </button>
        </div>
      </div>
    </div>

    <div class="card bg-base-200">
      <div class="card-body">
        <h3 class="card-title text-sm">Current State</h3>
        <pre class="text-xs overflow-x-auto">{JSON.stringify(gameState, null, 2)}</pre>
      </div>
    </div>

    <div class="card bg-base-200">
      <div class="card-body">
        <h3 class="card-title text-sm">Event History</h3>
        <div class="space-y-1 max-h-64 overflow-y-auto">
          {#each $events as event, index}
            <div class="flex items-center gap-2 text-xs">
              <button 
                class="btn btn-xs"
                onclick={() => handleTimeTravel(index)}
              >
                {index}
              </button>
              <span class="font-mono">{event.payload.type}</span>
            </div>
          {/each}
        </div>
      </div>
    </div>
  </div>
</div>