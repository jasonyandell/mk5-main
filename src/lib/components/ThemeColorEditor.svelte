<script lang="ts">
  import { onMount } from 'svelte';
  import { formatHex, parse, converter } from 'culori';
  import { gameState, gameActions } from '../../stores/gameStore';
  import ColorPicker from 'svelte-awesome-color-picker';
  
  interface Props {
    isOpen: boolean;
    onClose: () => void;
  }
  
  let { isOpen = $bindable(false), onClose }: Props = $props();
  
  // DaisyUI CSS variables we want to customize
  const daisyVariables = [
    { var: '--p', name: 'Primary', desc: 'Main brand color' },
    { var: '--pf', name: 'Primary Focus', desc: 'Primary hover state' },
    { var: '--pc', name: 'Primary Content', desc: 'Text on primary' },
    
    { var: '--s', name: 'Secondary', desc: 'Secondary brand color' },
    { var: '--sf', name: 'Secondary Focus', desc: 'Secondary hover state' },
    { var: '--sc', name: 'Secondary Content', desc: 'Text on secondary' },
    
    { var: '--a', name: 'Accent', desc: 'Accent color' },
    { var: '--af', name: 'Accent Focus', desc: 'Accent hover state' },
    { var: '--ac', name: 'Accent Content', desc: 'Text on accent' },
    
    { var: '--n', name: 'Neutral', desc: 'Neutral color' },
    { var: '--nf', name: 'Neutral Focus', desc: 'Neutral hover state' },
    { var: '--nc', name: 'Neutral Content', desc: 'Text on neutral' },
    
    { var: '--b1', name: 'Base 100', desc: 'Page background' },
    { var: '--b2', name: 'Base 200', desc: 'Card background' },
    { var: '--b3', name: 'Base 300', desc: 'Deep background' },
    { var: '--bc', name: 'Base Content', desc: 'Main text color' },
    
    { var: '--in', name: 'Info', desc: 'Info color' },
    { var: '--inc', name: 'Info Content', desc: 'Text on info' },
    
    { var: '--su', name: 'Success', desc: 'Success color' },
    { var: '--suc', name: 'Success Content', desc: 'Text on success' },
    
    { var: '--wa', name: 'Warning', desc: 'Warning color' },
    { var: '--wac', name: 'Warning Content', desc: 'Text on warning' },
    
    { var: '--er', name: 'Error', desc: 'Error color' },
    { var: '--erc', name: 'Error Content', desc: 'Text on error' },
  ];
  
  // Store current colors (HSL format)
  let currentColors = $state<Record<string, string>>({});
  // Custom colors now come from gameState
  let customColors = $derived($gameState.colorOverrides || {});
  let styleElement: HTMLStyleElement | null = null;
  let showShareDialog = $state(false);
  let shareURL = $state('');
  let copySuccess = $state<'link' | 'css' | null>(null);
  
  // Convert DaisyUI color string to hex for color picker
  // DaisyUI uses either HSL or OKLCH format
  function colorToHex(colorString: string): string {
    try {
      const parts = colorString.trim().split(/\s+/);
      
      // Check if it's OKLCH format (3 numbers with decimal points)
      if (parts.length === 3 && parts[0] && parts[0].includes('.')) {
        // OKLCH format: L C H
        const l = parseFloat(parts[0]) / 100; // L is 0-100 in DaisyUI
        const c = parseFloat(parts[1] || '0');
        const h = parseFloat(parts[2] || '0');
        
        const color = { mode: 'oklch', l, c, h };
        const hex = formatHex(color);
        return hex || '#808080';
      }
      
      // HSL format (legacy, some themes might still use it)
      if (parts.length === 3) {
        const h = parseFloat(parts[0] || '0');
        const s = parseFloat(parts[1] || '0') / 100;
        const l = parseFloat(parts[2] || '0') / 100;
        
        const color = { mode: 'hsl', h, s, l };
        const hex = formatHex(color);
        return hex || '#808080';
      }
      
      return '#808080';
    } catch (e) {
      console.warn('Failed to convert color to hex:', colorString, e);
      return '#808080';
    }
  }
  
  // Convert hex to DaisyUI color string (OKLCH format)
  function hexToColor(hex: string): string {
    try {
      // Parse hex and convert to OKLCH
      const parsed = parse(hex);
      if (!parsed) return '50% 0 0';
      
      // Convert to OKLCH color space using culori's converter
      const toOklch = converter('oklch');
      const oklchColor = toOklch(parsed);
      
      if (!oklchColor) return '50% 0 0';
      
      // Format as DaisyUI expects: L% C H
      const l = (oklchColor.l || 0.5) * 100;
      const c = oklchColor.c || 0;
      const h = oklchColor.h || 0;
      
      return `${l.toFixed(4)}% ${c.toFixed(6)} ${h.toFixed(6)}`;
    } catch (e) {
      console.warn('Failed to convert hex to OKLCH:', hex, e);
      return '50% 0 0';
    }
  }
  
  // Read current CSS variables from the theme
  function readCurrentColors() {
    const styles = getComputedStyle(document.documentElement);
    const colors: Record<string, string> = {};
    
    daisyVariables.forEach(({ var: varName }) => {
      const value = styles.getPropertyValue(varName).trim();
      if (value) {
        colors[varName] = value;
      }
    });
    
    return colors;
  }
  
  // Apply color overrides
  function applyColors() {
    if (!styleElement) {
      styleElement = document.createElement('style');
      styleElement.id = 'theme-color-overrides';
      document.head.appendChild(styleElement);
    }
    
    let css = ':root {\n';
    Object.entries(customColors).forEach(([varName, hslValue]) => {
      css += `  ${varName}: ${hslValue} !important;\n`;
    });
    css += '}\n';
    
    styleElement.textContent = css;
  }
  
  // Handle color change from picker
  function handleColorChange(varName: string, hexColor: string) {
    const colorValue = hexToColor(hexColor);
    
    // Update colors through gameActions
    const newColors = {
      ...$gameState.colorOverrides,
      [varName]: colorValue
    };
    gameActions.updateTheme($gameState.theme, newColors);
    applyColors();
  }
  
  // Reset all colors
  function resetColors() {
    // Clear colors through gameActions (keeps theme)
    gameActions.updateTheme($gameState.theme, {});
    
    // Remove style overrides
    if (styleElement) {
      styleElement.remove();
      styleElement = null;
    }
    
    // Force a small delay to ensure styles are cleared before re-reading
    setTimeout(() => {
      // Re-read current theme colors after reset
      currentColors = readCurrentColors();
    }, 10);
  }
  
  // Share colors via URL
  function shareColorsViaURL() {
    // The URL already has everything since theme is first-class state!
    shareURL = window.location.href;
    showShareDialog = true;
    
    navigator.clipboard.writeText(shareURL).catch(e => {
      console.error('Could not copy to clipboard:', e);
    });
  }
  
  // Export as CSS
  function exportCSS(): string {
    // Use theme from gameState (first-class citizen)
    const currentTheme = $gameState.theme || 'coffee';
    let css = `/* Custom theme colors for ${currentTheme} */\n`;
    css += `[data-theme="${currentTheme}"] {\n`;
    
    // Export color overrides from gameState
    Object.entries($gameState.colorOverrides || {}).forEach(([varName, colorValue]) => {
      css += `  ${varName}: ${colorValue};\n`;
    });
    
    css += '}\n';
    return css;
  }
  
  
  // Initialize colors when component opens
  $effect(() => {
    if (isOpen) {
      // Always re-read colors when opening the panel
      // This ensures we get the latest theme's colors
      currentColors = readCurrentColors();
    }
  });
  
  
  // Load from URL on mount
  onMount(() => {
    // Read current theme colors
    currentColors = readCurrentColors();
    
    // Color overrides are now applied reactively by App.svelte
  });
</script>

{#if isOpen}
  <!-- Backdrop - invisible, just for click detection -->
  <button
    class="fixed inset-0 z-40"
    onclick={() => onClose()}
    aria-label="Close color editor"
    type="button"
  ></button>
  
  <!-- Color Editor Panel -->
  <div class="theme-editor-panel fixed right-0 top-16 bottom-0 w-96 bg-base-100 shadow-2xl border-l border-base-300 overflow-y-auto transition-transform duration-300 ease-out z-50">
    <!-- Header -->
    <div class="sticky top-0 bg-base-100 border-b border-base-300 p-4 flex items-center justify-between backdrop-blur-sm bg-opacity-95">
      <h3 class="font-semibold text-lg">Theme Colors</h3>
      <button 
        class="btn btn-ghost btn-sm btn-circle"
        onclick={onClose}
        aria-label="Close"
        type="button"
      >
        ✕
      </button>
    </div>
    
    <!-- Current theme indicator -->
    <div class="px-4 pt-3 pb-1 text-sm text-base-content/60">
      Base theme: <strong>{document.documentElement.getAttribute('data-theme') || 'default'}</strong>
    </div>
    
    <!-- Color variables list -->
    <div class="p-4 space-y-3">
      {#each daisyVariables as variable}
        {@const currentValue = currentColors[variable.var] || '50% 0.02 0'}
        {@const customValue = customColors[variable.var]}
        {@const displayValue = customValue || currentValue}
        {@const hexValue = colorToHex(displayValue)}
        
        <div class="flex items-center gap-3 p-2 rounded-lg hover:bg-base-200/50 transition-colors">
          <!-- Variable info -->
          <div class="flex-1">
            <div class="text-sm font-medium text-base-content/90">{variable.name}</div>
            <div class="text-xs text-base-content/50">{variable.desc}</div>
            {#if customValue}
              <div class="text-xs text-primary mt-1">Modified</div>
            {/if}
          </div>
          
          <!-- Color picker -->
          <div class="color-picker-compact">
            <ColorPicker 
              hex={hexValue}
              label=""
              onInput={(e: any) => {
                // Get the new hex value from the event
                const newHex = e.hex || hexValue;
                handleColorChange(variable.var, newHex);
              }}
            />
          </div>
          
          <!-- Current value -->
          <div class="text-xs font-mono text-base-content/60 w-24">
            <div>{hexValue}</div>
            <div class="text-[10px] opacity-60">{displayValue}</div>
          </div>
        </div>
      {/each}
    </div>
    
    <!-- Action buttons -->
    <div class="sticky bottom-0 p-4 bg-base-100 border-t border-base-300 space-y-2 backdrop-blur-sm bg-opacity-95">
      <div class="flex gap-2">
        <button 
          class="btn btn-primary flex-1"
          onclick={shareColorsViaURL}
          type="button"
        >
          📤 Share Link
        </button>
        <button 
          class="btn btn-primary flex-1"
          onclick={() => {
            const css = exportCSS();
            navigator.clipboard.writeText(css);
          }}
          type="button"
        >
          📋 Copy CSS
        </button>
      </div>
      <button 
        class="btn btn-outline btn-sm w-full"
        onclick={resetColors}
        type="button"
      >
        Reset to Theme Defaults
      </button>
    </div>
  </div>
  
  <!-- Share Dialog -->
  {#if showShareDialog}
    <div class="share-dialog fixed inset-0 z-[60] flex items-center justify-center p-4">
      <div class="bg-base-100 rounded-xl shadow-2xl max-w-2xl w-full flex flex-col">
        <div class="p-4 border-b border-base-300 flex items-center justify-between">
          <h3 class="font-semibold">Share Your Theme</h3>
          <button 
            class="btn btn-ghost btn-sm btn-circle"
            onclick={() => showShareDialog = false}
            type="button"
          >
            ✕
          </button>
        </div>
        <div class="p-4">
          <p class="text-sm text-base-content/70 mb-3">
            This link includes your <strong>{document.documentElement.getAttribute('data-theme') || 'default'}</strong> theme and all color customizations.
          </p>
          <div class="bg-base-200 p-3 rounded-lg break-all relative">
            <code class="text-xs select-all" style="user-select: text;">{shareURL}</code>
          </div>
          <div class="mt-3 flex gap-2">
            <button
              class="btn btn-sm flex-1 {copySuccess === 'link' ? 'btn-success' : 'btn-primary'}"
              onclick={() => {
                navigator.clipboard.writeText(shareURL).then(() => {
                  copySuccess = 'link';
                  setTimeout(() => copySuccess = null, 2000);
                }).catch(e => {
                  console.error('Failed to copy:', e);
                });
              }}
              type="button"
            >
              {copySuccess === 'link' ? '✓ Copied!' : '📋 Copy Link'}
            </button>
            <button
              class="btn btn-sm flex-1 {copySuccess === 'css' ? 'btn-success' : 'btn-secondary'}"
              onclick={() => {
                const css = exportCSS();
                navigator.clipboard.writeText(css).then(() => {
                  copySuccess = 'css';
                  setTimeout(() => copySuccess = null, 2000);
                }).catch(e => {
                  console.error('Failed to copy CSS:', e);
                });
              }}
              type="button"
            >
              {copySuccess === 'css' ? '✓ Copied!' : '📄 Copy CSS'}
            </button>
          </div>
          <div class="mt-4 p-3 bg-base-300/50 rounded-lg">
            <p class="text-xs text-base-content/70">
              <strong>Tip:</strong> The link has been automatically copied to your clipboard. You can also select the text above to copy manually.
            </p>
          </div>
        </div>
        <div class="p-4 border-t border-base-300">
          <button 
            class="btn btn-primary w-full"
            onclick={() => showShareDialog = false}
            type="button"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  {/if}
{/if}

<style>
  .theme-editor-panel {
    max-width: 400px;
  }
  
  .share-dialog {
    background-color: rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(4px);
  }
  
  /* Compact color picker styling using theme variables */
  .color-picker-compact :global(.color-picker) {
    width: 3rem;
    height: 3rem;
    border-radius: 0.375rem;
    border: 1px solid oklch(var(--b3));
    cursor: pointer;
  }
  
  .color-picker-compact :global(.color-picker__input) {
    display: none;
  }
  
  .color-picker-compact :global(.color-picker__label) {
    display: none;
  }
  
  /* Fix mobile scrolling: ensure picker popup breaks out of scrolling context */
  .color-picker-compact :global(.picker) {
    z-index: 99999 !important;
    position: fixed !important;
  }
  
  /* Ensure the color picker wrapper properly positions the popup */
  .color-picker-compact :global(.wrapper) {
    position: fixed !important;
    z-index: 99999 !important;
  }
  
  /* Additional mobile-specific fixes */
  @media (max-width: 768px) {
    .theme-editor-panel {
      position: fixed;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
    }
    
    /* Ensure picker popup isn't constrained by panel */
    .color-picker-compact :global(.color-picker-wrapper) {
      position: static !important;
    }
  }
</style>