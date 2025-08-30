<script lang="ts">
  import { gameState, actionHistory, gameActions, initialState } from '../../stores/gameStore';
  import { encodeGameUrl } from '../../game/core/url-compression';
  import StateTreeView from './StateTreeView.svelte';

  interface Props {
    onclose: () => void;
  }

  let { onclose }: Props = $props();

  let activeTab = $state('theme');
  let showDiff = $state(false);
  let showTreeView = $state(true);
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
    const newURL = window.location.pathname + encodeGameUrl(
      $initialState.shuffleSeed,
      actionsToReplay.map(a => a.id),
      $initialState.playerTypes
    );
    
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

<!-- Full screen mobile drawer -->
<div class="fixed inset-0 z-50 bg-base-100 flex flex-col" data-testid="settings-panel">
  <!-- Header with mobile-friendly close -->
  <div class="navbar bg-base-200 shadow-lg">
    <div class="flex-1">
      <h2 class="text-lg font-semibold">Settings</h2>
    </div>
    <button 
      class="btn btn-ghost btn-square"
      data-testid="settings-close-button"
      onclick={onclose}
      aria-label="Close settings panel"
    >
      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-6 h-6">
        <path stroke-linecap="round" stroke-linejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    </button>
  </div>

  <!-- Bottom tabs for mobile navigation -->
  <div class="flex-1 overflow-hidden flex flex-col">
    <!-- Content -->
    <div class="flex-1 overflow-auto p-4 pb-16">
      {#if activeTab === 'theme'}
        {@const themes = [
          { value: 'cupcake', emoji: '🧁', name: 'Cupcake' },
          { value: 'light', emoji: '☀️', name: 'Light' },
          { value: 'dark', emoji: '🌙', name: 'Dark' },
          { value: 'bumblebee', emoji: '🐝', name: 'Bumblebee' },
          { value: 'emerald', emoji: '💚', name: 'Emerald' },
          { value: 'corporate', emoji: '💼', name: 'Corporate' },
          { value: 'retro', emoji: '📻', name: 'Retro' },
          { value: 'cyberpunk', emoji: '🤖', name: 'Cyberpunk' },
          { value: 'valentine', emoji: '💝', name: 'Valentine' },
          { value: 'garden', emoji: '🌸', name: 'Garden' },
          { value: 'forest', emoji: '🌲', name: 'Forest' },
          { value: 'lofi', emoji: '🎵', name: 'Lo-Fi' },
          { value: 'pastel', emoji: '🎨', name: 'Pastel' },
          { value: 'wireframe', emoji: '📐', name: 'Wireframe' },
          { value: 'luxury', emoji: '💎', name: 'Luxury' },
          { value: 'dracula', emoji: '🧛', name: 'Dracula' },
          { value: 'autumn', emoji: '🍂', name: 'Autumn' },
          { value: 'business', emoji: '👔', name: 'Business' },
          { value: 'coffee', emoji: '☕', name: 'Coffee' },
          { value: 'winter', emoji: '❄️', name: 'Winter' }
        ]}
        
        {@const currentTheme = typeof document !== 'undefined' ? 
          document.documentElement.getAttribute('data-theme') || 'coffee' : 
          'coffee'}
        
        <div class="space-y-4">
          <div class="prose prose-sm">
            <h3>Choose Theme</h3>
          </div>
          
          <div class="grid grid-cols-2 gap-3">
            {#each themes as theme}
              <button
                class="relative overflow-hidden rounded-lg transition-transform active:scale-95 {currentTheme === theme.value ? 'ring-2 ring-primary ring-offset-2 ring-offset-base-100' : ''}"
                onclick={() => {
                  // Use gameActions to update theme (first-class state)
                  gameActions.updateTheme(theme.value, {});
                }}
              >
                <div data-theme={theme.value} class="bg-base-100 p-3">
                  <!-- Color preview dots -->
                  <div class="flex gap-1.5 mb-2">
                    <div class="bg-primary w-6 h-6 rounded-full"></div>
                    <div class="bg-secondary w-6 h-6 rounded-full"></div>
                    <div class="bg-accent w-6 h-6 rounded-full"></div>
                    <div class="bg-neutral w-6 h-6 rounded-full"></div>
                  </div>
                  
                  <!-- Theme name -->
                  <div class="flex items-center gap-1 text-sm font-medium text-base-content">
                    <span>{theme.emoji}</span>
                    <span>{theme.name}</span>
                  </div>
                  
                  <!-- Mini component preview -->
                  <div class="flex gap-1 mt-2">
                    <div class="badge badge-primary badge-xs">A</div>
                    <div class="btn btn-secondary btn-xs">B</div>
                    <div class="bg-base-200 rounded px-1 text-xs text-base-content">C</div>
                  </div>
                </div>
                
                <!-- Selected indicator -->
                {#if currentTheme === theme.value}
                  <div class="absolute top-1 right-1 bg-primary text-primary-content rounded-full p-0.5">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-3 w-3" viewBox="0 0 20 20" fill="currentColor">
                      <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                    </svg>
                  </div>
                {/if}
              </button>
            {/each}
          </div>
          
          <!-- Reset Game button -->
          <div class="divider">Game Controls</div>
          <button 
            class="btn btn-warning btn-block"
            onclick={() => {
              // Reset the game state (theme is now preserved automatically)
              gameActions.resetGame();
            }}
          >
            🎮 Reset Game
          </button>
          <p class="text-xs text-base-content/60 text-center mt-2">
            Starts a new game while keeping your theme settings
          </p>
        </div>
      {/if}

      {#if activeTab === 'state'}
        <div class="space-y-4">
          <!-- Toggle switches -->
          <div class="flex flex-col gap-3">
            <div class="form-control">
              <label class="label cursor-pointer">
                <span class="label-text">Tree View</span>
                <input 
                  type="checkbox" 
                  class="toggle toggle-primary"
                  bind:checked={showTreeView}
                />
              </label>
            </div>
            <div class="form-control">
              <label class="label cursor-pointer">
                <span class="label-text">Show Changes</span>
                <input 
                  type="checkbox" 
                  class="toggle toggle-secondary"
                  bind:checked={showDiff}
                />
              </label>
            </div>
          </div>

          <!-- Action buttons -->
          <div class="grid grid-cols-2 gap-2">
            <button 
              class="btn btn-primary"
              onclick={() => copyToClipboard(prettyJson($gameState))}
            >
              📋 Copy State
            </button>
            <button 
              class="btn btn-secondary"
              onclick={copyShareableUrl}
            >
              🔗 Copy URL
            </button>
          </div>
          
          <!-- State viewer -->
          <div class="bg-base-200 rounded-lg p-3 max-h-96 overflow-auto">
            {#if showTreeView}
              <StateTreeView 
                data={$gameState} 
                searchQuery=""
                changedPaths={showDiff ? changedPaths : new Set()}
              />
            {:else}
              <pre class="text-xs font-mono">{prettyJson($gameState)}</pre>
            {/if}
          </div>
        </div>
      {/if}

      {#if activeTab === 'history'}
        <div class="space-y-4">
          <!-- Header with count -->
          <div class="flex items-center justify-between">
            <h3 class="text-base font-semibold">
              History 
              <span class="badge badge-neutral ml-2">{$actionHistory.length}</span>
            </h3>
          </div>

          <!-- Action buttons -->
          <div class="grid grid-cols-2 gap-2">
            <button 
              class="btn btn-warning"
              onclick={gameActions.undo}
              disabled={$actionHistory.length === 0}
            >
              ↩️ Undo
            </button>
            <button 
              class="btn btn-error"
              onclick={gameActions.resetGame}
            >
              🔄 Reset
            </button>
          </div>
          
          <!-- History list -->
          {#if $actionHistory.length === 0}
            <div class="alert">
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="stroke-current shrink-0 w-6 h-6"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
              <span>No moves yet. Start playing!</span>
            </div>
          {:else}
            <div class="space-y-2">
              {#each [...$actionHistory].reverse() as action, reverseIndex}
                {@const actualIndex = $actionHistory.length - 1 - reverseIndex}
                <!-- Make entire card clickable on mobile -->
                <button 
                  class="card bg-base-200 w-full text-left active:scale-95 transition-transform"
                  data-testid="history-item"
                  onclick={() => timeTravel(actualIndex)}
                  onpointerdown={(e) => {
                    e.preventDefault();
                    timeTravel(actualIndex);
                  }}
                >
                  <div class="card-body p-3 flex-row items-center gap-3">
                    <div class="badge badge-lg badge-primary font-bold">
                      {actualIndex + 1}
                    </div>
                    <div class="flex-1 text-sm">
                      {action.label}
                    </div>
                    <div class="text-primary">
                      <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-5 h-5">
                        <path stroke-linecap="round" stroke-linejoin="round" d="M9 15L3 9m0 0l6-6M3 9h12a6 6 0 010 12h-3" />
                      </svg>
                    </div>
                  </div>
                </button>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    </div>
    
    <!-- Bottom navigation tabs -->
    <div class="btm-nav btm-nav-sm bg-base-200 border-t border-base-300">
      <button 
        class="{activeTab === 'theme' ? 'active text-primary' : ''}"
        onclick={() => activeTab = 'theme'}
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zm0 0h12a2 2 0 002-2v-4a2 2 0 00-2-2h-2.343M11 7.343l1.657-1.657a2 2 0 012.828 0l2.829 2.829a2 2 0 010 2.828l-8.486 8.485M7 17h.01" />
        </svg>
        <span class="btm-nav-label">Theme</span>
      </button>
      <button 
        class="{activeTab === 'state' ? 'active text-primary' : ''}"
        onclick={() => activeTab = 'state'}
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
        </svg>
        <span class="btm-nav-label">State</span>
      </button>
      <button 
        class="{activeTab === 'history' ? 'active text-primary' : ''}"
        data-testid="history-tab"
        onclick={() => activeTab = 'history'}
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        <span class="btm-nav-label">History</span>
      </button>
    </div>
  </div>
  
  <!-- Modal backdrop click to close -->
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
  <form method="dialog" class="modal-backdrop" onclick={onclose}>
    <button type="button" aria-label="Close modal">close</button>
  </form>
</div>