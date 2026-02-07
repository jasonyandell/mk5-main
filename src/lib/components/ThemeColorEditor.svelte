<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import { formatHex, parse, converter } from 'culori';
  // No need to re-encode full URL for sharing; preserve current URL and patch theme if needed
  import ColorPicker from 'svelte-awesome-color-picker';
  import Icon from '../icons/Icon.svelte';
  import { shareContent, canNativeShare } from '../utils/share';
  
  const dispatch = createEventDispatcher();
  
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
  
  // Initialize picker states for all variables  
  const initStates: Record<string, boolean> = {};
  daisyVariables.forEach(v => {
    initStates[v.var] = false;
  });
  let pickerStates = $state<Record<string, boolean>>(initStates);
  
  // Helper to get picker state with guaranteed boolean
  const getPickerState = (varName: string): boolean => {
    return pickerStates[varName] || false;
  };
  
  // Check if any picker is open
  let isAnyPickerOpen = $derived(Object.values(pickerStates).some(isOpen => isOpen));
  
  // Store current colors (HSL format)
  let currentColors = $state<Record<string, string>>({});
  // Custom colors are not persisted in the new architecture
  let customColors = $state<Record<string, string>>({});
  // No direct style injection here; App.svelte handles overrides centrally
  // Keep a reference only to clean up any legacy element from older builds
  let legacyStyleElement: HTMLStyleElement | null = null;
  let showShareDialog = $state(false);
  let shareURL = $state('');
  let copySuccess = $state<'link' | 'css' | null>(null);
  
  // Simple debounce timer for color updates
  let updateTimer: ReturnType<typeof setTimeout> | null = null;
  let pendingUpdates: Record<string, string> = {};
  
  // Convert DaisyUI color string to hex for color picker
  // DaisyUI uses OKLCH format: L% C H
  function colorToHex(colorString: string): string {
    try {
      const parts = colorString.trim().split(/\s+/);
      
      // OKLCH format: L% C H
      if (parts.length === 3) {
        // Remove % sign from L value if present
        const lStr = (parts[0] ?? '').replace('%', '');
        const l = parseFloat(lStr) / 100; // Convert percentage to 0-1 range
        const c = parseFloat(parts[1] ?? '0');
        const h = parseFloat(parts[2] ?? '0');
        
        const color = { mode: 'oklch', l, c, h };
        const hex = formatHex(color);
        return hex || '#808080';
      }
      
      // Fallback for unexpected formats
      console.warn('Unexpected color format:', colorString);
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

  // Approximate equality in OKLCH space to avoid tiny diffs from formatting
  function approxEqualOklch(a: string, b: string): boolean {
    try {
      const pa = a.trim().split(/\s+/);
      const pb = b.trim().split(/\s+/);
      if (pa.length !== 3 || pb.length !== 3) return false;
      const l1 = parseFloat((pa[0] ?? '').replace('%', ''));
      const c1 = parseFloat(pa[1] ?? '');
      const h1 = parseFloat(pa[2] ?? '');
      const l2 = parseFloat((pb[0] ?? '').replace('%', ''));
      const c2 = parseFloat(pb[1] ?? '');
      const h2 = parseFloat(pb[2] ?? '');
      if (![l1,c1,h1,l2,c2,h2].every(Number.isFinite)) return false;
      return Math.abs(l1 - l2) < 0.02 && Math.abs(c1 - c2) < 0.0005 && Math.abs(h1 - h2) < 0.1;
    } catch {
      return false;
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
  
  // Centralized overrides are applied in App.svelte; no local CSS injection
  
  // Handle color change from picker with debouncing
  function handleColorChange(varName: string, hexColor: string) {
    const colorValue = hexToColor(hexColor);
    // If the computed OKLCH equals the current theme value, ignore
    const styles = getComputedStyle(document.documentElement);
    const baseVal = styles.getPropertyValue(varName).trim();
    if (baseVal && approxEqualOklch(baseVal, colorValue)) return;
    // If an override exists and equals requested, ignore
    const existing = customColors[varName];
    if (existing && approxEqualOklch(existing, colorValue)) return;

    // Accumulate this change
    pendingUpdates[varName] = colorValue;
    
    // Clear existing timer if any
    if (updateTimer) {
      clearTimeout(updateTimer);
      updateTimer = null;
    }
    
    // Set new timer to update after delay
    updateTimer = setTimeout(() => {
      // Apply custom colors locally
      customColors = { ...customColors, ...pendingUpdates };

      // Apply to DOM
      const style = document.createElement('style');
      style.id = 'custom-colors';
      const existing = document.getElementById('custom-colors');
      if (existing) existing.remove();

      let css = ':root {\n';
      Object.entries(customColors).forEach(([varName, value]) => {
        css += `  ${varName}: ${value} !important;\n`;
      });
      css += '}\n';
      style.textContent = css;
      document.head.appendChild(style);

      pendingUpdates = {};
      updateTimer = null;
    }, 100);
  }
  
  // Reset all colors
  function resetColors() {
    // Clear any pending timer and updates
    if (updateTimer) {
      clearTimeout(updateTimer);
      updateTimer = null;
    }
    pendingUpdates = {};
    
    // Clear any custom color overrides
    customColors = {};
    const style = document.getElementById('custom-colors');
    if (style) style.remove();
    
    // Remove any legacy style overrides (from older approach)
    if (legacyStyleElement) {
      legacyStyleElement.remove();
      legacyStyleElement = null;
    }
    
    // Force a small delay to ensure styles are cleared before re-reading
    setTimeout(() => {
      // Re-read current theme colors after reset
      currentColors = readCurrentColors();
    }, 10);
  }
  
  // Share colors via URL (simplified for new architecture)
  function shareColorsViaURL() {
    // In new architecture, just share current URL with theme info
    shareURL = window.location.href;
    showShareDialog = true;
  }
  
  // Export as CSS
  function exportCSS(): string {
    // Get current theme from DOM
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'coffee';
    let css = `/* Custom theme colors for ${currentTheme} */\n`;
    css += `[data-theme="${currentTheme}"] {\n`;

    // Export custom colors
    Object.entries(customColors).forEach(([varName, colorValue]) => {
      css += `  ${varName}: ${colorValue};\n`;
    });

    css += '}\n';
    return css;
  }
  
  
  // Initialize colors when component opens
  $effect(() => {
    if (isOpen) {
      // Force complete style recalculation before reading colors
      // This ensures DaisyUI theme variables are fully computed
      requestAnimationFrame(() => {
        // Force browser to recalculate all CSS variables
        const style = window.getComputedStyle(document.documentElement);
        const cssVars = ['--p', '--s', '--a', '--n', '--b1', '--b2', '--b3', '--bc', '--pc', '--sc', '--ac', '--nc'];
        cssVars.forEach(varName => style.getPropertyValue(varName));
        
        // Now read the colors with fresh computed values
        currentColors = readCurrentColors();
        
        // If we still got invalid colors, try again after a small delay
        // (sometimes needed when transitioning from settings panel)
        if (Object.values(currentColors).some(color => color === '#808080')) {
          setTimeout(() => {
            currentColors = readCurrentColors();
          }, 50);
        }
      });
    }
  });
  
  // Re-read colors when theme changes (fixes mobile issue)
  // Use a more reliable approach with MutationObserver
  $effect(() => {
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
          // Theme changed, wait for App.svelte to complete its force reflow
          // then update our colors
          setTimeout(() => {
            requestAnimationFrame(() => {
              // Force style recalculation by reading all DaisyUI variables
              const style = window.getComputedStyle(document.documentElement);
              const cssVars = ['--p', '--s', '--a', '--n', '--b1', '--b2', '--b3', '--bc', '--pc', '--sc', '--ac', '--nc'];
              cssVars.forEach(varName => style.getPropertyValue(varName));
              
              // Then read all colors
              currentColors = readCurrentColors();
              // Force component update
              currentColors = { ...currentColors };
            });
          }, 150); // Delay to ensure theme is fully applied after App.svelte's force reflow
        }
      });
    });
    
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme']
    });
    
    return () => observer.disconnect();
  });
  
  
  // Load from URL on mount
  onMount(() => {
    // Cleanup: remove any legacy style element that might persist from older sessions
    const old = document.getElementById('theme-color-overrides') as HTMLStyleElement | null;
    if (old) {
      old.remove();
      legacyStyleElement = null;
    }
    // Read current theme colors
    currentColors = readCurrentColors();
    
    // Color overrides are now applied reactively by App.svelte
    
    // Cleanup function - clear timer on unmount
    return () => {
      if (updateTimer) {
        clearTimeout(updateTimer);
        updateTimer = null;
      }
    };
  });
</script>

{#if isOpen}
  <!-- Backdrop - only show when no picker is active -->
  {#if !isAnyPickerOpen}
    <button
      class="fixed inset-0 z-40"
      onclick={() => onClose()}
      aria-label="Close color editor"
      type="button"
    ></button>
  {/if}
  
  <!-- Color Editor Panel - Minimal vertical -->
  <div class="theme-editor-panel fixed right-0 top-16 bottom-0 w-16 bg-base-100 shadow-2xl border-l border-base-300 overflow-y-auto transition-all duration-300 ease-out z-50" class:picker-open={isAnyPickerOpen}>
    <!-- All content in single scrollable container -->
    <div class="p-2 space-y-2">
      {#if !isAnyPickerOpen}
        <!-- Top action buttons -->
        <button 
          class="btn btn-ghost btn-xs btn-block justify-center"
          onclick={() => dispatch('openSettings')}
          title="Change theme"
          type="button"
        >
          <Icon name="paintBrush" size="sm" />
        </button>
        <button 
          class="btn btn-ghost btn-xs btn-block justify-center"
          onclick={shareColorsViaURL}
          title="Share theme"
          type="button"
        >
          <Icon name="share" size="sm" />
        </button>
        <div class="divider my-2"></div>
      {/if}
      
      <!-- Color circles list -->
      {#each daisyVariables as variable (variable.var)}
        {@const currentValue = currentColors[variable.var] || '50% 0.02 0'}
        {@const customValue = customColors[variable.var]}
        {@const displayValue = customValue || currentValue}
        {@const hexValue = colorToHex(displayValue)}
        {@const isPickerOpen = getPickerState(variable.var)}
        
        {#if !isAnyPickerOpen || isPickerOpen}
          <div class="flex justify-center" title={variable.name}>
            <!-- Color picker circle only -->
            <div class="color-picker-compact">
              <ColorPicker 
                bind:isOpen={pickerStates[variable.var]!}
                hex={hexValue}
                label=""
                isTextInput={false}
                sliderDirection="vertical"
                onInput={(e: any) => {
                  // Only handle actual user changes; ignore initial emissions
                  const newHex = e?.hex;
                  if (!newHex || newHex.toLowerCase() === hexValue.toLowerCase()) return;
                  handleColorChange(variable.var, newHex);
                }}
              />
            </div>
          </div>
        {/if}
      {/each}
      
      <!-- Bottom action button -->
      {#if !isAnyPickerOpen}
        <div class="divider my-2"></div>
        <button 
          class="btn btn-ghost btn-xs btn-block justify-center"
          onclick={resetColors}
          title="Reset to theme defaults"
          type="button"
        >
          <Icon name="arrowPath" size="sm" />
        </button>
      {/if}
    </div>
  </div>
  
  <!-- Share Dialog - Compact Mobile Friendly -->
  {#if showShareDialog}
    <div class="share-dialog fixed inset-0 z-[60] flex items-center justify-center p-4">
      <div class="bg-base-100 rounded-lg shadow-2xl max-w-[12rem] w-full">
        <div class="p-3 border-b border-base-300 flex items-center justify-between">
          <h3 class="font-semibold text-sm">Share</h3>
          <button 
            class="btn btn-ghost btn-xs btn-circle"
            onclick={() => showShareDialog = false}
            type="button"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div class="p-3 space-y-2">
          <!-- Copy buttons -->
          <div class="flex flex-col gap-2">
            <button
              class="btn btn-sm flex-1 {copySuccess === 'link' ? 'btn-success' : 'btn-primary'}"
              onclick={async () => {
                const theme = document.documentElement.getAttribute('data-theme') || 'coffee';
                const success = await shareContent({
                  title: `Texas 42 - ${theme} Theme`,
                  text: `Check out my custom ${theme} theme!`,
                  url: shareURL
                });
                if (success) {
                  copySuccess = 'link';
                  setTimeout(() => copySuccess = null, 2000);
                }
              }}
              type="button"
            >
              {#if copySuccess === 'link'}
                <Icon name="check" size="sm" /> {canNativeShare() ? 'Shared' : 'Link'}
              {:else}
                <Icon name={canNativeShare() ? 'share' : 'clipboard'} size="sm" /> {canNativeShare() ? 'Share' : 'Link'}
              {/if}
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
              {#if copySuccess === 'css'}
                <Icon name="check" size="sm" /> CSS
              {:else}
                <Icon name="clipboard" size="sm" /> CSS
              {/if}
            </button>
          </div>
          
          <!-- Theme name -->
          <div class="text-xs text-center text-base-content/60">
            {document.documentElement.getAttribute('data-theme') || 'default'}
          </div>
          
          <!-- URL in scrollable box -->
          <div class="bg-base-200 p-2 rounded overflow-x-auto max-h-20">
            <code class="text-xs whitespace-nowrap select-all" style="user-select: text;">{shareURL}</code>
          </div>
        </div>
      </div>
    </div>
  {/if}
{/if}

<style>
  .share-dialog {
    background-color: hsl(var(--nc) / 0.5);
    backdrop-filter: blur(4px);
  }
  
  /* Compact color picker styling - circular */
  .color-picker-compact :global(.color-picker) {
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 9999px;
    cursor: pointer;
    transition: transform 0.2s;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin: 0 !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    outline: none !important;
    filter: none !important;
    backdrop-filter: none !important;
    line-height: 0 !important;
    vertical-align: middle;
  }

  /* (intentionally no pseudo on .color-picker; backdrop handled on label) */
  
  /* Neutralize any wrapper/label backgrounds that create a pill behind the swatch */
  .color-picker-compact :global(label),
  .color-picker-compact :global(.label),
  .color-picker-compact :global(.color-picker__label) {
    --input-size: 2.5rem !important; /* match our outer circle */
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
    height: var(--input-size) !important;
    border-radius: 9999px !important;
    gap: 0 !important;
    line-height: 0 !important;
  }

  /* Ensure swatch wrapper itself has no background/shadow */
  .color-picker-compact :global(.color-picker) {
    background: transparent !important;
    box-shadow: none !important;
  }

  /* Keep the color circle round and free of extra layers */
  .color-picker-compact :global(.result) {
    border-radius: 9999px !important;
    box-shadow: none !important;
    overflow: hidden !important; /* ensure inner layers don't bleed as ellipse */
  }
  .color-picker-compact :global(.container) {
    /* Ensure inner container matches our circle */
    width: var(--input-size, 2.5rem) !important;
    height: var(--input-size, 2.5rem) !important;
  }
  .color-picker-compact :global(.alpha),
  .color-picker-compact :global(.color) {
    width: var(--input-size, 2.5rem) !important;
    height: var(--input-size, 2.5rem) !important;
    border-radius: 9999px !important;
    background-clip: padding-box !important;
  }

  /* Hide alpha checkerboard layer behind the swatch to avoid faint ring/ellipse */
  .color-picker-compact :global(.alpha) {
    display: none !important;
    background: transparent !important;
  }

  /* Remove focus outlines that can appear as faint ovals around the swatch */
  .color-picker-compact :global(input:focus ~ .color),
  .color-picker-compact :global(input:focus-visible ~ .color) {
    outline: none !important;
  }

  /* Cover any faint background with a clean circular backdrop matching panel bg */
  .color-picker-compact :global(label) {
    position: relative;
    z-index: 1;
  }
  .color-picker-compact :global(label)::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 9999px;
    background: oklch(var(--b1));
    pointer-events: none;
    z-index: 0;
  }
  .color-picker-compact :global(.result::before),
  .color-picker-compact :global(.result::after) {
    content: none !important;
  }

  /* Remove WebKit tap highlight and focus rings that appear as an oval */
  .color-picker-compact :global(.color-picker),
  .color-picker-compact :global(label),
  .color-picker-compact :global(.color-picker__label),
  .color-picker-compact :global(.result) {
    -webkit-tap-highlight-color: transparent !important;
    outline: none !important;
  }
  .color-picker-compact :global(.color-picker:focus),
  .color-picker-compact :global(label:focus),
  .color-picker-compact :global(.color-picker__label:focus),
  .color-picker-compact :global(.result:focus),
  .color-picker-compact :global(.color-picker:focus-visible),
  .color-picker-compact :global(label:focus-visible),
  .color-picker-compact :global(.color-picker__label:focus-visible),
  .color-picker-compact :global(.result:focus-visible),
  .color-picker-compact :global(.color-picker:focus-within),
  .color-picker-compact :global(label:focus-within),
  .color-picker-compact :global(.color-picker__label:focus-within),
  .color-picker-compact :global(.result:focus-within) {
    outline: none !important;
    box-shadow: none !important;
  }
  
  /* Remove hover scaling to avoid layout/position thrash with popup */
  /* .color-picker-compact :global(.color-picker:hover) {
    transform: scale(1.1);
  } */
  
  .color-picker-compact :global(.color-picker__input) {
    display: none;
  }
  
  .color-picker-compact :global(.color-picker__label) {
    display: none;
  }
  
  /* Basic picker styling (keep defaults; avoid forced positioning to prevent flicker) */
  
  /* Remove wrapper padding and margins for compact layout */
  .color-picker-compact :global(.wrapper) {
    z-index: 99999 !important;
    padding: 4px !important;
    margin: 0 !important;
    border-radius: 8px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
  }
  
  /* When picker is open, ensure it's above everything */
  .color-picker-compact :global(.color-picker-wrapper) {
    z-index: 100000 !important;
  }

  /* Allow popup to overflow panel when any picker is open */
  .theme-editor-panel.picker-open {
    overflow: visible !important;
  }
  
  /* Remove margins from sliders to fit in window */
  .color-picker-compact :global(.h),
  .color-picker-compact :global(.a) {
    margin: 2px 0 !important;
    --track-width: 180px !important;
  }
  
  /* Optional: compact picker size — enable if needed
  .color-picker-compact :global(.picker) {
    width: 180px !important;
    height: 180px !important;
    --picker-width: 180px !important;
    --picker-height: 180px !important;
  }
  */
  
  /* Remove any layout overrides - let the picker use its default layout */

  /* Additional mobile-specific fixes */
  @media (max-width: 768px) {
    .theme-editor-panel {
      position: fixed;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
    }
  }
</style>
