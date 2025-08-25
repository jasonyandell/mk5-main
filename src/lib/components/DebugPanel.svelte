<script lang="ts">
  import { gameState, actionHistory, gameActions, initialState } from '../../stores/gameStore';
  import { quickplayState, quickplayActions } from '../../stores/quickplayStore';
  import { compressGameState, compressActionId, encodeURLData } from '../../game/core/url-compression';
  import StateTreeView from './StateTreeView.svelte';

  interface Props {
    onclose: () => void;
  }

  let { onclose }: Props = $props();

  let activeTab = $state('state');
  let showDiff = $state(false);
  let showTreeView = $state(true);
  let showHistoricalTreeView = $state(true);
  let previousState: any = $state(null);
  let changedPaths = $state(new Set<string>());

  // JSON stringify with indentation
  function prettyJson(obj: any): string {
    return JSON.stringify(obj, null, 2);
  }

  // Copy to clipboard
  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
  }

  // Generate shareable URL
  function copyShareableUrl() {
    const url = window.location.href;
    copyToClipboard(url);
  }

  // Time travel to a specific action
  function timeTravel(index: number) {
    // Get actions up to the specified index
    const actionsToReplay = $actionHistory.slice(0, index + 1);
    
    // Use gameActions to properly load the state
    // This ensures controllers are notified and state is properly managed
    const historyState = {
      initialState: $initialState,
      actions: actionsToReplay.map(a => a.id),
      timestamp: Date.now()
    };
    
    // Use loadFromHistoryState which handles everything correctly:
    // - Deep clones to prevent mutations
    // - Updates all stores properly
    // - Notifies controllers for AI
    gameActions.loadFromHistoryState(historyState);
    
    // Update URL to reflect the new state
    const urlData = {
      v: 1 as const,
      s: compressGameState($initialState),
      a: actionsToReplay.map(a => ({ i: compressActionId(a.id) }))
    };
    
    const encoded = encodeURLData(urlData);
    const newURL = `${window.location.pathname}?d=${encoded}`;
    
    // Update browser history
    window.history.pushState(historyState, '', newURL);
  }

  // Find differences between two objects
  function findDifferences(obj1: any, obj2: any, path: string = 'root'): Set<string> {
    const differences = new Set<string>();
    
    if (obj1 === obj2) return differences;
    
    if (typeof obj1 !== typeof obj2) {
      differences.add(path);
      return differences;
    }
    
    if (typeof obj1 !== 'object' || obj1 === null || obj2 === null) {
      if (obj1 !== obj2) {
        differences.add(path);
      }
      return differences;
    }
    
    // For arrays
    if (Array.isArray(obj1) && Array.isArray(obj2)) {
      const maxLength = Math.max(obj1.length, obj2.length);
      for (let i = 0; i < maxLength; i++) {
        const childDiffs = findDifferences(obj1[i], obj2[i], `${path}[${i}]`);
        childDiffs.forEach(diff => differences.add(diff));
      }
      return differences;
    }
    
    // For objects
    const allKeys = new Set([...Object.keys(obj1), ...Object.keys(obj2)]);
    for (const key of allKeys) {
      const childDiffs = findDifferences(obj1[key], obj2[key], `${path}.${key}`);
      childDiffs.forEach(diff => differences.add(diff));
    }
    
    return differences;
  }

  // Track state changes
  $effect(() => {
    if (showDiff && previousState) {
      changedPaths = findDifferences(previousState, $gameState);
    } else {
      changedPaths = new Set();
    }
    previousState = JSON.parse(JSON.stringify($gameState));
  });
</script>

<!-- Modal backdrop -->
<div class="modal modal-open">
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="modal-box max-w-6xl h-[90vh] p-0 flex flex-col" onclick={(e) => e.stopPropagation()}>
    <!-- Header -->
    <div class="flex justify-between items-center p-4 border-b border-base-300">
      <h2 class="text-xl font-bold">Debug Panel</h2>
      <button 
        class="btn btn-sm btn-circle btn-ghost"
        onclick={onclose}
        aria-label="Close debug panel"
      >
        ✕
      </button>
    </div>

    <!-- Tabs -->
    <div class="tabs tabs-boxed bg-base-200 rounded-none">
      <button 
        class="tab {activeTab === 'state' ? 'tab-active' : ''}"
        onclick={() => activeTab = 'state'}
      >
        Game State
      </button>
      <button 
        class="tab {activeTab === 'history' ? 'tab-active' : ''}"
        onclick={() => activeTab = 'history'}
      >
        History
      </button>
      <button 
        class="tab {activeTab === 'quickplay' ? 'tab-active' : ''}"
        onclick={() => activeTab = 'quickplay'}
      >
        QuickPlay
      </button>
      <button 
        class="tab {activeTab === 'historical' ? 'tab-active' : ''}"
        onclick={() => activeTab = 'historical'}
      >
        Historical
      </button>
    </div>

    <!-- Content -->
    <div class="flex-1 overflow-hidden p-4">
      {#if activeTab === 'state'}
        <div class="flex flex-col h-full gap-4">
          <div class="flex flex-wrap gap-2">
            <label class="label cursor-pointer gap-2">
              <input 
                type="checkbox" 
                class="checkbox checkbox-sm"
                bind:checked={showTreeView}
              />
              <span class="label-text">Tree View</span>
            </label>
            <label class="label cursor-pointer gap-2">
              <input 
                type="checkbox" 
                class="checkbox checkbox-sm"
                bind:checked={showDiff}
              />
              <span class="label-text">Diff Mode</span>
            </label>
            <button 
              class="btn btn-sm btn-primary"
              onclick={() => copyToClipboard(prettyJson($gameState))}
            >
              Copy State
            </button>
            <button 
              class="btn btn-sm btn-secondary"
              onclick={copyShareableUrl}
            >
              Copy URL
            </button>
          </div>
          
          <div class="flex-1 overflow-auto bg-base-200 rounded-lg p-4">
            {#if showTreeView}
              <StateTreeView 
                data={$gameState} 
                searchQuery=""
                changedPaths={showDiff ? changedPaths : new Set()}
              />
            {:else}
              <pre class="text-xs">{prettyJson($gameState)}</pre>
            {/if}
          </div>
        </div>
      {/if}

      {#if activeTab === 'history'}
        <div class="flex flex-col h-full gap-4">
          <div class="flex justify-between items-center">
            <h3 class="text-lg font-semibold">Action History ({$actionHistory.length})</h3>
            <div class="flex gap-2">
              <button 
                class="btn btn-sm btn-warning"
                onclick={gameActions.undo}
                disabled={$actionHistory.length === 0}
              >
                Undo Last
              </button>
              <button 
                class="btn btn-sm btn-error"
                onclick={gameActions.resetGame}
              >
                Reset Game
              </button>
            </div>
          </div>
          
          <div class="flex-1 overflow-y-auto bg-base-200 rounded-lg p-4">
            {#if $actionHistory.length === 0}
              <div class="text-center py-8 text-base-content/60">
                No actions taken yet. Start playing to see history.
              </div>
            {:else}
              <div class="space-y-2">
                {#each [...$actionHistory].reverse() as action, reverseIndex}
                  {@const actualIndex = $actionHistory.length - 1 - reverseIndex}
                  <div class="flex items-center gap-2 p-2 bg-base-100 rounded-lg">
                    <span class="badge badge-neutral">#{actualIndex + 1}</span>
                    <span class="flex-1 text-sm">{action.label}</span>
                    <button 
                      class="btn btn-xs btn-info"
                      onclick={() => timeTravel(actualIndex)}
                      title="Time travel to this point"
                    >
                      ⏪
                    </button>
                  </div>
                {/each}
              </div>
            {/if}
          </div>
        </div>
      {/if}

      {#if activeTab === 'quickplay'}
        <div class="space-y-4">
          <div class="flex flex-wrap gap-4 items-center">
            <label class="label cursor-pointer gap-2">
              <input 
                type="checkbox" 
                class="toggle toggle-primary"
                bind:checked={$quickplayState.enabled}
                onchange={() => quickplayActions.toggle()}
              />
              <span class="label-text">QuickPlay Active</span>
            </label>
            
            <div class="form-control">
              <label class="label">
                <span class="label-text">Speed:</span>
              </label>
              <select 
                class="select select-bordered select-sm"
                bind:value={$quickplayState.speed}
                onchange={(e) => quickplayActions.setSpeed(e.currentTarget.value as any)}
              >
                <option value="instant">Instant</option>
                <option value="fast">Fast</option>
                <option value="normal">Normal</option>
                <option value="slow">Slow</option>
              </select>
            </div>

            <button 
              class="btn btn-sm btn-primary"
              onclick={quickplayActions.step}
              disabled={$quickplayState.enabled}
            >
              Step
            </button>

            <button 
              class="btn btn-sm btn-secondary"
              onclick={quickplayActions.playToEndOfHand}
            >
              End of Hand
            </button>

            <button 
              class="btn btn-sm btn-accent"
              onclick={quickplayActions.playToEndOfGame}
            >
              End of Game
            </button>
          </div>

          <div class="card bg-base-200">
            <div class="card-body">
              <h4 class="card-title text-base">Status</h4>
              <div class="grid grid-cols-2 gap-2 text-sm">
                <span class="font-semibold">Active:</span>
                <span>{$quickplayState.enabled}</span>
                <span class="font-semibold">Speed:</span>
                <span>{$quickplayState.speed}</span>
                <span class="font-semibold">Phase:</span>
                <span>{$gameState.phase}</span>
                <span class="font-semibold">Current Player:</span>
                <span>P{$gameState.currentPlayer}</span>
              </div>
            </div>
          </div>
          
          <!-- Theme Switcher -->
          <div class="card bg-base-200">
            <div class="card-body">
              <h4 class="card-title text-base">Theme</h4>
              <select 
                class="select select-bordered w-full"
                data-choose-theme
                value={typeof window !== 'undefined' && localStorage.getItem('theme') || 'light'}
                onchange={(e) => {
                  const theme = e.currentTarget.value;
                  document.documentElement.setAttribute('data-theme', theme);
                  localStorage.setItem('theme', theme);
                }}>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
                <option value="cupcake">Cupcake</option>
                <option value="bumblebee">Bumblebee</option>
                <option value="emerald">Emerald</option>
                <option value="corporate">Corporate</option>
                <option value="retro">Retro</option>
                <option value="cyberpunk">Cyberpunk</option>
                <option value="valentine">Valentine</option>
                <option value="garden">Garden</option>
                <option value="forest">Forest</option>
                <option value="lofi">Lo-Fi</option>
                <option value="pastel">Pastel</option>
                <option value="wireframe">Wireframe</option>
                <option value="luxury">Luxury</option>
                <option value="dracula">Dracula</option>
                <option value="autumn">Autumn</option>
                <option value="business">Business</option>
                <option value="coffee">Coffee</option>
                <option value="winter">Winter</option>
              </select>
            </div>
          </div>
        </div>
      {/if}

      {#if activeTab === 'historical'}
        <div class="flex flex-col h-full gap-4">
          <div class="flex items-center gap-4">
            <h3 class="text-lg font-semibold">Event Sourcing State</h3>
            <label class="label cursor-pointer gap-2">
              <input 
                type="checkbox" 
                class="checkbox checkbox-sm"
                bind:checked={showHistoricalTreeView}
              />
              <span class="label-text">Tree View</span>
            </label>
            <button 
              class="btn btn-sm btn-primary"
              onclick={() => {
                const historicalData = {
                  initialState: $initialState,
                  actions: $actionHistory
                };
                copyToClipboard(prettyJson(historicalData));
              }}
            >
              Copy JSON
            </button>
          </div>
          
          <div class="flex-1 overflow-auto">
            {#if showHistoricalTreeView}
              <div class="bg-base-200 rounded-lg p-4">
                <StateTreeView 
                  data={{
                    initialState: $initialState,
                    actions: $actionHistory
                  }}
                  searchQuery=""
                  changedPaths={new Set()}
                />
              </div>
            {:else}
              <div class="space-y-4">
                <div class="card bg-base-200">
                  <div class="card-body">
                    <h4 class="card-title text-base">Initial State</h4>
                    <pre class="text-xs overflow-auto max-h-64">{prettyJson($initialState)}</pre>
                  </div>
                </div>
                
                <div class="card bg-base-200">
                  <div class="card-body">
                    <h4 class="card-title text-base">Actions ({$actionHistory.length})</h4>
                    <pre class="text-xs overflow-auto max-h-64">{prettyJson($actionHistory)}</pre>
                  </div>
                </div>
              </div>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  </div>
  
  <!-- Modal backdrop click to close -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <form method="dialog" class="modal-backdrop" onclick={onclose}>
    <button type="button" aria-label="Close modal">close</button>
  </form>
</div>