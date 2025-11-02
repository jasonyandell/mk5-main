# Theme Color System

## Overview
Theme and color customization is a **first-class citizen** in the game state. Theme configuration lives directly in `GameState.theme` and `GameState.colorOverrides`. A single reactive effect in App.svelte automatically syncs all theme changes to the DOM - no manual updates needed anywhere else.

## Architecture

### Reactive Theme System
```
GameState.theme / GameState.colorOverrides change
    ↓ (Svelte reactivity)
$effect in App.svelte triggers
    ↓ (automatic)
Updates DOM: data-theme attribute + style element
    ↓ (immediate)
All UI components update via CSS variables
```

### Color Flow
```
DaisyUI Theme CSS Variables (OKLCH format)
    ↓ (read on panel open)
Convert OKLCH → Hex for color picker display
    ↓ (user interaction)
HTML Color Picker (hex values)
    ↓ (on change)
Convert Hex → OKLCH 
    ↓ (via gameActions.updateTheme)
Updates GameState → Triggers reactive effect
```

### Storage Format
Modern DaisyUI themes use OKLCH color space (perceptually uniform):
- `--p: 71.9967% 0.123825 62.756393` (primary in OKLCH: Lightness% Chroma Hue)
- `--b2: 26.8053% 0.020556 277.508664` (base-200 color)
- etc.

Legacy themes may use HSL format:
- `--p: 259 94% 51%` (primary in HSL: Hue Saturation% Lightness%)

The system automatically detects the format (OKLCH has decimal points in first value).

## CSS Variables

### Primary Variables
- `--p`, `--pf`, `--pc` - Primary, focus, content
- `--s`, `--sf`, `--sc` - Secondary, focus, content  
- `--a`, `--af`, `--ac` - Accent, focus, content
- `--n`, `--nf`, `--nc` - Neutral, focus, content

### Base Colors
- `--b1` - Base 100 (main background)
- `--b2` - Base 200 (slightly darker)
- `--b3` - Base 300 (darkest base)
- `--bc` - Base content (text on base colors)

### Semantic Colors
- `--in`, `--inc` - Info and content
- `--su`, `--suc` - Success and content
- `--wa`, `--wac` - Warning and content
- `--er`, `--erc` - Error and content

## URL Format

### Structure (Theme First!)
```
?t=<theme>&v=<colors>&s=<seed>&a=<actions>...
```
Theme parameters come FIRST to survive URL truncation in messaging apps.

### Examples

#### OKLCH Format (modern themes)
```
?t=autumn&v=p:73.9228,0.131773,88.104434;s:34.465,0.029849,199.19444
```

This sets:
- Base theme: autumn
- Primary: OKLCH(73.92%, 0.132, 88.10°)
- Secondary: OKLCH(34.47%, 0.030, 199.19°)

#### HSL Format (legacy)
```
?t=retro&v=p:259,94,51;s:314,100,47
```

This sets:
- Base theme: retro
- Primary: HSL(259, 94%, 51%)
- Secondary: HSL(314, 100%, 47%)

## Export Format

The system exports clean CSS that can be added to your stylesheet:

### OKLCH Export (modern)
```css
/* Custom theme colors for autumn */
[data-theme="autumn"] {
  --p: 73.9228% 0.131773 88.104434;
  --s: 34.465% 0.029849 199.19444;
  --a: 42.6213% 0.074405 224.389184;
  /* ... other overrides ... */
}
```

### HSL Export (legacy)
```css
[data-theme="retro"] {
  --p: 259 94% 51%;
  --s: 314 100% 47%;
  /* ... other overrides ... */
}
```

## User Workflow

1. **Select base theme** - Choose from DaisyUI themes (coffee, dracula, etc.)
2. **Customize colors** - Use color pickers to adjust any theme color
3. **Preview live** - Changes apply immediately to the entire UI
4. **Share link** - Generate URL with theme + customizations
5. **Export CSS** - Get production-ready CSS variables

## Technical Details

### Color Space Support
- **OKLCH** (Oklab Lightness Chroma Hue): Modern perceptually uniform color space
  - Used by newer DaisyUI themes (dracula, autumn, coffee, etc.)
  - Format: `L% C H` where L=lightness(0-100%), C=chroma(0-0.4), H=hue(0-360)
  - Better color interpolation and contrast ratios
  
- **HSL** (Hue Saturation Lightness): Traditional color space
  - Used by some older themes
  - Format: `H S% L%` where H=hue(0-360), S=saturation(0-100%), L=lightness(0-100%)

### Conversion Functions
- **OKLCH to Hex**: For displaying in color picker
  - Parse OKLCH: "71.9967% 0.123825 62.756393"
  - Convert via culori: "#db924b"
  
- **Hex to OKLCH**: For storing as CSS variable
  - Parse hex: "#db924b"
  - Convert via culori: "71.9967% 0.123825 62.756393"

### Persistence
1. **URL parameters**: Complete theme + color configuration
   - `t`: Theme name (e.g., "autumn")
   - `v`: Color overrides in compact format
2. **CSS injection**: Runtime style element with `!important` overrides
3. **No localStorage**: Keeps configuration ephemeral and shareable

### Reset Behavior
- Clears all custom CSS variables
- Removes style element
- Reverts to base DaisyUI theme colors
- Removes `t` and `v` URL parameters
- Preserves game state parameters

## Benefits

1. **Native DaisyUI**: Works with all DaisyUI components automatically
2. **Complete control**: Every theme color is customizable
3. **Clean exports**: Production-ready CSS variables
4. **Shareable**: URL contains complete theme configuration
5. **Live preview**: See changes instantly across entire UI
6. **Theme-aware**: Builds on DaisyUI's theme system

## Implementation Notes

### State Management
- **First-class state**: Theme/colors in `GameState`, not separate URL params
- **Reactive application**: Single `$effect` in App.svelte handles all theme DOM updates
- **gameActions.updateTheme()**: Updates state, DOM updates automatically via reactivity
- **Reset Game**: Preserves theme while clearing game state
- **URL persistence**: Theme parameters always included, even with no actions
- **No localStorage**: Everything flows through game state

### UI Components
- **All use DaisyUI classes**: `bg-warning`, `text-error`, `btn-primary`, etc.
- **Exception - Domino**: Keeps white bg & black pips (physical game piece)
- **Winner pill**: Uses `bg-warning text-warning-content`
- **Error dialogs**: Use `bg-error/10 text-error`

### Key Files
- `src/App.svelte`: Contains the reactive `$effect` that syncs theme to DOM
- `src/game/types.ts`: GameState includes theme & colorOverrides
- `src/stores/gameStore.ts`: updateTheme() method updates state only
- `src/lib/components/ThemeColorEditor.svelte`: UI for customization
- `src/game/core/url-compression.ts`: Theme params first in URL