/* global HTMLStyleElement, getComputedStyle */
import { converter } from 'culori';
import type { GameState, StateTransition } from '../../game/types';
import { encodeGameUrl } from '../../game/core/url-compression';

// Track current section/scenario for URL (e.g., 'one_hand')
let currentScenario: string | null = null;

export function setCurrentScenario(name: string | null): void {
  const toSnake = (s: string) => s
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/\s+/g, '_')
    .toLowerCase();
  currentScenario = name ? toSnake(name) : null;
}

// Centralized URL builder for outbound navigations and sharing
// Always routes through encodeGameUrl, then applies include/exclude flags and merges unknown params.
export function buildUrl(options: {
  initialState: GameState,
  actionIds?: string[],
  scenarioName?: string | null,
  includeSeed?: boolean,
  includeActions?: boolean,
  includeScenario?: boolean,
  includeTheme?: boolean,
  includeOverrides?: boolean,
  preserveUnknownParams?: boolean,
  absolute?: boolean,
  path?: string
}): string {
  const {
    initialState,
    actionIds,
    scenarioName,
    includeSeed = true,
    includeActions = true,
    includeScenario = true,
    includeTheme = true,
    includeOverrides = true,
    preserveUnknownParams = true,
    absolute = true,
    path
  } = options;

  // Compact action ids (filter out consensus) if provided
  const compactActions = (actionIds || []).filter(id => !id.startsWith('agree-'));
  const theme = includeTheme ? initialState.theme : undefined as unknown as string | undefined;
  const overrides = includeOverrides ? initialState.colorOverrides : undefined as unknown as Record<string,string> | undefined;
  const scenario = includeScenario ? (scenarioName ?? currentScenario ?? undefined) : undefined;

  // Encode state to query string (encodeGameUrl always includes s and a)
  const qs = encodeGameUrl(
    initialState.shuffleSeed,
    includeActions ? compactActions : [],
    initialState.playerTypes,
    initialState.dealer,
    initialState.tournamentMode,
    theme,
    overrides,
    scenario
  );

  const basePath = path || (typeof window !== 'undefined' ? window.location.pathname : '/');
  const origin = typeof window !== 'undefined' ? window.location.origin : '';
  const url = new URL(basePath + qs, absolute ? origin : 'http://local');

  // Optionally drop seed/actions from the final URL (for seedless scenario links or clean starts)
  if (!includeSeed) url.searchParams.delete('s');
  if (!includeActions) url.searchParams.delete('a');

  // Optionally merge unknown params from the current URL (e.g., testMode)
  if (preserveUnknownParams && typeof window !== 'undefined') {
    const current = new URL(window.location.href);
    current.searchParams.forEach((value, key) => {
      // Known keys encoded by encodeGameUrl
      const known = new Set(['s','a','p','d','tm','t','v','h']);
      if (!known.has(key) && !url.searchParams.has(key)) {
        url.searchParams.set(key, value);
      }
    });
  }

  return absolute ? url.toString() : (url.pathname + (url.search || ''));
}

// Helper function to update URL with initial state and actions
let urlBatchDepth = 0;
let pendingUrlUpdate: { initialState: GameState; actions: StateTransition[]; usePushState: boolean } | null = null;

export function beginUrlBatch(): void {
  urlBatchDepth++;
}

export function endUrlBatch(): void {
  if (urlBatchDepth > 0) urlBatchDepth--;
  if (urlBatchDepth === 0 && pendingUrlUpdate) {
    const { initialState, actions, usePushState } = pendingUrlUpdate;
    pendingUrlUpdate = null;
    updateURLWithState(initialState, actions, usePushState);
  }
}

export function updateURLWithState(initialState: GameState, actions: StateTransition[], usePushState = false) {
  // If batching, store the latest request and return
  if (urlBatchDepth > 0) {
    pendingUrlUpdate = { initialState, actions, usePushState };
    return;
  }
  if (typeof window !== 'undefined') {
    // Filter color overrides to only include values that differ from the base theme defaults
    let filteredOverrides = initialState.colorOverrides || {};
    try {
      const currentThemeAttr = document.documentElement.getAttribute('data-theme');
      if (currentThemeAttr === initialState.theme && filteredOverrides && Object.keys(filteredOverrides).length > 0) {
        const styleEl = document.getElementById('theme-overrides') as HTMLStyleElement | null;
        const prevDisabled = styleEl ? styleEl.disabled : undefined;
        if (styleEl) styleEl.disabled = true;
        const styles = getComputedStyle(document.documentElement);
        const minimal: Record<string, string> = {};
        // Compare defaults vs overrides in OKLCH space with tolerance to avoid formatting/precision issues
        const toOklch = converter('oklch');
        const approxEqual = (a: string, b: string): boolean => {
          const parse = (val: string): [number, number, number] | null => {
            const v = val.trim();
            if (!v) return null;
            const parts = v.split(/\s+/);
            if (parts.length !== 3) return null;
            // OKLCH if first part has '%'
            if (parts[0] && parts[0].includes('%')) {
              const l = parseFloat(parts[0].replace('%', ''));
              const c = parseFloat(parts[1] || '0');
              const h = parseFloat(parts[2] || '0');
              if (Number.isFinite(l) && Number.isFinite(c) && Number.isFinite(h)) return [l, c, h];
              return null;
            }
            // HSL (legacy) otherwise
            const h = parseFloat(parts[0] || '0');
            const s = parseFloat((parts[1] || '0').replace('%', '')) / 100;
            const l = parseFloat((parts[2] || '0').replace('%', '')) / 100;
            if (!Number.isFinite(h) || !Number.isFinite(s) || !Number.isFinite(l)) return null;
            const ok = toOklch({ mode: 'hsl', h, s, l });
            if (!ok) return null;
            return [ (ok.l || 0) * 100, ok.c || 0, ok.h || 0 ];
          };
          const A = parse(a);
          const B = parse(b);
          if (!A || !B) return false;
          const [l1, c1, h1] = A;
          const [l2, c2, h2] = B;
          const el = Math.abs(l1 - l2);
          const ec = Math.abs(c1 - c2);
          const eh = Math.abs(h1 - h2);
          return el < 0.02 && ec < 0.0005 && eh < 0.1;
        };
        for (const [varName, value] of Object.entries(filteredOverrides)) {
          const defaultVal = styles.getPropertyValue(varName).trim();
          const overrideVal = String(value).trim();
          if (!defaultVal || !approxEqual(defaultVal, overrideVal)) {
            minimal[varName] = value;
          }
        }
        filteredOverrides = minimal;
        if (styleEl && prevDisabled !== undefined) styleEl.disabled = prevDisabled;
      }
    } catch {
      // If anything goes wrong, fall back to using provided overrides
    }
    // Compact consensus: filter out agree-* actions from URL/history encoding
    const compactActionIds = actions.map(a => a.id).filter(id => !id.startsWith('agree-'));

    // Preserve testMode parameter if present
    const currentParams = new URLSearchParams(window.location.search);
    const testMode = currentParams.get('testMode');
    
    // Use v2 compressed format with theme as first-class citizen
    // Always encode URL properly, even with no actions (preserves theme)
    let newURL = window.location.pathname + encodeGameUrl(
      initialState.shuffleSeed,
      compactActionIds,
      initialState.playerTypes,
      initialState.dealer,
      initialState.tournamentMode,
      initialState.theme,
      filteredOverrides,
      currentScenario || undefined
    );
    
    // Append testMode if it was present
    if (testMode === 'true') {
      newURL = `${newURL}${newURL.includes('?') ? '&' : '?'}testMode=true`;
    }
    
    // Store state in history for easy access
    const historyState = { initialState, actions: compactActionIds, timestamp: Date.now() };
    
    if (usePushState) {
      window.history.pushState(historyState, '', newURL);
    } else {
      window.history.replaceState(historyState, '', newURL);
    }
  }
}

// Utility: reset URL to minimal (initial state only), discarding action list
export function setURLToMinimal(initial: GameState): void {
  if (typeof window === 'undefined') return;
  
  // Preserve testMode parameter if present
  const currentParams = new URLSearchParams(window.location.search);
  const testMode = currentParams.get('testMode');
  
  let newURL = window.location.pathname + encodeGameUrl(
    initial.shuffleSeed,
    [],
    initial.playerTypes,
    initial.dealer,
    initial.tournamentMode,
    initial.theme,
    initial.colorOverrides
  );
  
  // Append testMode if it was present
  if (testMode === 'true') {
    newURL = `${newURL}${newURL.includes('?') ? '&' : '?'}testMode=true`;
  }
  
  const historyState = { initialState: initial, actions: [], timestamp: Date.now() };
  window.history.replaceState(historyState, '', newURL);
}
