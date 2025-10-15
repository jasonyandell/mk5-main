<script lang="ts">
  import { gameState, gameActions } from '../../stores/gameStore';
  import StateTreeView from './StateTreeView.svelte';
  import Icon from '../icons/Icon.svelte';
  import { shareContent, canNativeShare } from '../utils/share';

  interface Props {
    onclose: () => void;
    initialTab?: 'state' | 'theme';
  }

  let { onclose, initialTab = 'state' }: Props = $props();

  let activeTab = $state(initialTab);
  let showTreeView = $state(true);

  // JSON stringify with indentation
  function prettyJson(obj: any): string {
    return JSON.stringify(obj, null, 2);
  }

  // Copy to clipboard (for non-shareable content like JSON)
  function copyToClipboard(text: string) {
    navigator.clipboard.writeText(text);
  }

  // Generate shareable URL (simplified for new architecture)
  async function shareUrl() {
    // In the new architecture, we don't have action history
    // Just share the current URL with game seed
    const url = window.location.href;

    await shareContent({
      title: 'Texas 42 Game',
      text: 'Check out this Texas 42 game!',
      url: url
    });
  }

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
      <Icon name="xMark" size="lg" />
    </button>
  </div>

  <!-- Bottom tabs for mobile navigation -->
  <div class="flex-1 overflow-hidden flex flex-col">
    <!-- Content -->
    <div class="flex-1 overflow-auto p-4 pb-16">
      {#if activeTab === 'theme'}
        {@const themes = [
          { value: 'cupcake', icon: 'cupcake' as const, name: 'Cupcake' },
          { value: 'light', icon: 'sun' as const, name: 'Light' },
          { value: 'dark', icon: 'moon' as const, name: 'Dark' },
          { value: 'bumblebee', icon: 'sparkles' as const, name: 'Bumblebee' },
          { value: 'emerald', icon: 'sparkles' as const, name: 'Emerald' },
          { value: 'corporate', icon: 'briefcase' as const, name: 'Corporate' },
          { value: 'retro', icon: 'radio' as const, name: 'Retro' },
          { value: 'cyberpunk', icon: 'cpuChip' as const, name: 'Cyberpunk' },
          { value: 'valentine', icon: 'heart' as const, name: 'Valentine' },
          { value: 'garden', icon: 'sparkles' as const, name: 'Garden' },
          { value: 'forest', icon: 'tree' as const, name: 'Forest' },
          { value: 'lofi', icon: 'musicalNote' as const, name: 'Lo-Fi' },
          { value: 'pastel', icon: 'paintBrush' as const, name: 'Pastel' },
          { value: 'wireframe', icon: 'ruler' as const, name: 'Wireframe' },
          { value: 'luxury', icon: 'sparkles' as const, name: 'Luxury' },
          { value: 'dracula', icon: 'vampire' as const, name: 'Dracula' },
          { value: 'autumn', icon: 'tree' as const, name: 'Autumn' },
          { value: 'business', icon: 'briefcase' as const, name: 'Business' },
          { value: 'coffee', icon: 'coffee' as const, name: 'Coffee' },
          { value: 'winter', icon: 'snowflake' as const, name: 'Winter' }
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
                  // In new architecture, themes are handled client-side only
                  document.documentElement.setAttribute('data-theme', theme.value);
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
                    <Icon name={theme.icon} size="sm" />
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
                    <Icon name="checkSolid" size="sm" className="w-3 h-3" />
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
              // Reset the game state
              gameActions.resetGame();
            }}
          >
            <span class="flex items-center justify-center gap-2">
              <Icon name="gameController" size="md" />
              Reset Game
            </span>
          </button>
          <p class="text-xs text-base-content/60 text-center mt-2">
            Starts a new game
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
          </div>

          <!-- Action buttons -->
          <div class="grid grid-cols-2 gap-2">
            <button
              class="btn btn-primary"
              onclick={() => copyToClipboard(prettyJson($gameState))}
            >
              <span class="flex items-center justify-center gap-1">
                <Icon name="clipboard" size="sm" />
                Copy State
              </span>
            </button>
            <button
              class="btn btn-secondary"
              onclick={shareUrl}
            >
              <span class="flex items-center justify-center gap-1">
                <Icon name={canNativeShare() ? 'share' : 'arrowPath'} size="sm" />
                {canNativeShare() ? 'Share' : 'Copy'} URL
              </span>
            </button>
          </div>

          <!-- State viewer -->
          <div class="bg-base-200 rounded-lg p-3 max-h-96 overflow-auto">
            {#if showTreeView}
              <StateTreeView
                data={$gameState}
                searchQuery=""
                changedPaths={new Set()}
              />
            {:else}
              <pre class="text-xs font-mono">{prettyJson($gameState)}</pre>
            {/if}
          </div>
        </div>
      {/if}
    </div>

    <!-- Bottom navigation tabs -->
    <div class="btm-nav btm-nav-sm bg-base-200 border-t border-base-300">
      <button
        class="{activeTab === 'state' ? 'active text-primary' : ''}"
        onclick={() => activeTab = 'state'}
      >
        <Icon name="code" size="md" />
        <span class="btm-nav-label">State</span>
      </button>
      <button
        class="{activeTab === 'theme' ? 'active text-primary' : ''}"
        onclick={() => activeTab = 'theme'}
      >
        <Icon name="paintBrush" size="md" />
        <span class="btm-nav-label">Theme</span>
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