<script lang="ts">
  import { onMount } from 'svelte';
  import Header from './lib/components/Header.svelte';
  import PlayingArea from './lib/components/PlayingArea.svelte';
  import ActionPanel from './lib/components/ActionPanel.svelte';
  import SettingsPanel from './lib/components/SettingsPanel.svelte';
  import ThemeColorEditor from './lib/components/ThemeColorEditor.svelte';
  import SeedFinderModal from './lib/components/SeedFinderModal.svelte';
  import OneHandCompleteModal from './lib/components/OneHandCompleteModal.svelte';
  import { gameState, viewProjection, gameVariants } from './stores/gameStore';
  import { fly, fade } from 'svelte/transition';

  let showSettingsPanel = $state(false);
  let showThemeEditor = $state(false);
  let activeView = $state<'game' | 'actions'>('game');
  let settingsInitialTab = $state<'state' | 'theme'>('state');

  // Handle keyboard shortcuts
  function handleKeydown(e: KeyboardEvent) {
    if (e.ctrlKey && e.shiftKey && e.key === 'D') {
      e.preventDefault();
      showSettingsPanel = !showSettingsPanel;
    } else if (e.key === 'Escape') {
      if (showThemeEditor) {
        showThemeEditor = false;
      } else if (showSettingsPanel) {
        showSettingsPanel = false;
      }
    }
  }

  onMount(() => {
    // Check for one-hand URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    const oneHandParam = urlParams.get('onehand');

    if (oneHandParam === 'auto') {
      // Start with competitive seed (server finds one)
      gameVariants.startOneHand();
    } else if (oneHandParam) {
      // Start with specific seed
      const seed = parseInt(oneHandParam, 10);
      if (!isNaN(seed)) {
        gameVariants.startOneHand(seed);
      }
    }
    // Note: loadFromURL is deprecated - server handles all state
  });
  
  // Smart panel switching based on game phase
  // This handles both URL loading and normal game flow
  $effect(() => {
    // Use the ViewProjection's computed activeView
    activeView = $viewProjection.ui.activeView;
  });
  
  // Theme is a first-class citizen - reactively apply to DOM
  $effect(() => {
    // Cleanup any legacy override element from older implementations
    document.getElementById('theme-color-overrides')?.remove();

    // Apply theme
    if ($gameState.theme) {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      if (currentTheme !== $gameState.theme) {
        // Theme is changing
        document.documentElement.setAttribute('data-theme', $gameState.theme);
        
        // CRITICAL: Force DaisyUI to recalculate all CSS variables
        // This is the most reliable way to ensure theme changes are applied immediately
        requestAnimationFrame(() => {
          // Step 1: Toggle a class to force style recalculation
          document.documentElement.classList.add('theme-changing');
          
          // Step 2: Force browser to recalculate all CSS variables by reading them
          const style = window.getComputedStyle(document.documentElement);
          // Read all DaisyUI CSS variables to force recalculation
          const cssVars = ['--p', '--s', '--a', '--n', '--b1', '--b2', '--b3', '--bc', '--pc', '--sc', '--ac', '--nc'];
          cssVars.forEach(varName => style.getPropertyValue(varName));
          
          // Step 3: Force a complete reflow by modifying then restoring critical styles
          const html = document.documentElement;
          const body = document.body;
          
          // Save original styles
          const originalHtmlDisplay = html.style.display;
          const originalBodyDisplay = body.style.display;
          
          // Force reflow with display changes
          html.style.display = 'none';
          html.offsetHeight; // Force reflow
          html.style.display = originalHtmlDisplay || '';
          
          // Extra aggressive for mobile/touch devices
          if ('ontouchstart' in window) {
            body.style.display = 'none';
            body.offsetHeight; // Force reflow
            body.style.display = originalBodyDisplay || '';
          }
          
          // Step 4: Remove the temporary class
          requestAnimationFrame(() => {
            document.documentElement.classList.remove('theme-changing');
          });
        });
      }
    }
    
    // Apply color overrides
    const overrides = $gameState.colorOverrides;
    if (overrides && Object.keys(overrides).length > 0) {
      let styleEl = document.getElementById('theme-overrides') as HTMLStyleElement;
      if (!styleEl) {
        styleEl = document.createElement('style');
        styleEl.id = 'theme-overrides';
        document.head.appendChild(styleEl);
      }
      
      let css = ':root {\n';
      Object.entries(overrides).forEach(([varName, value]) => {
        css += `  ${varName}: ${value} !important;\n`;
      });
      css += '}\n';
      styleEl.textContent = css;
    } else {
      // Remove overrides if none
      document.getElementById('theme-overrides')?.remove();
    }
  });
</script>

<svelte:window onkeydown={handleKeydown} />

<div 
  class="app-container flex flex-col h-screen bg-base-100 text-base-content font-sans overflow-hidden"
  style="height: 100dvh;"
  role="application" 
  data-phase={$gameState.phase}
>
  <Header 
    on:openSettings={() => {
      settingsInitialTab = 'state';
      showSettingsPanel = true;
    }} 
    on:openThemeEditor={() => showThemeEditor = true}
  />
  
  <main class="flex-1 overflow-y-auto overflow-x-hidden touch-pan-y pb-safe relative {activeView === 'actions' ? 'overflow-hidden' : ''}">
    {#if activeView === 'game'}
      <div transition:fade={{ duration: 200 }}>
        <PlayingArea />
      </div>
    {:else}
      <div transition:fade={{ duration: 200 }} class="h-full flex flex-col overflow-hidden">
        <ActionPanel onswitchToPlay={() => activeView = 'game'} />
      </div>
    {/if}
  </main>
</div>

{#if showSettingsPanel}
  <div transition:fly={{ y: 200, duration: 300 }}>
    <SettingsPanel 
      onclose={() => showSettingsPanel = false} 
      initialTab={settingsInitialTab}
    />
  </div>
{/if}

<ThemeColorEditor 
  bind:isOpen={showThemeEditor} 
  onClose={() => showThemeEditor = false}
  on:openSettings={() => {
    showThemeEditor = false;
    settingsInitialTab = 'theme';
    showSettingsPanel = true;
  }}
/>

<SeedFinderModal />
<OneHandCompleteModal />
