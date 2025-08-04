# Debug UI Specification v2

## Core Philosophy

The Debug UI for Texas 42 is a **game-first debugging interface** where the actual game state and playable actions are always visible and interactive on the main screen. The full debug panel (accessed via Ctrl+Shift+D) provides deep inspection capabilities, but the game must be fully playable without opening it.

## Key Principles

1. **Game Visibility First**: The current game state, player's hand, and available actions must be visible at all times on the main screen
2. **Direct Interaction**: Players can click dominoes in their hand to play them - no need to use action buttons for play moves
3. **Debug as Overlay**: The debug panel is an optional overlay for deep inspection, not required for gameplay
4. **No Scrolling**: All essential game information fits on one screen without scrolling
5. **Color-Coded Clarity**: Use color to indicate playable vs non-playable dominoes and game state

## Main Screen Layout (Always Visible)

### Layout Structure
```
┌──────────────────────────────────────────────────────────────────┐
│ Header: Title | Phase | AI: [▶️Play] [⏸️] [⏭️Step] | Scores | Turn │
├─────────────────┬───────────────────────┬───────────────────────┤
│                 │                       │               │
│  Trick History  │   Game Center Area    │   Actions     │
│  (Left Panel)   │   (Center Panel)      │  (Right Panel)│
│                 │                       │               │
│  - Shows all 7  │  ┌─────────────────┐ │  - Non-play   │
│    tricks with  │  │ TRUMP | LED SUIT│ │    actions    │
│    dominoes     │  └─────────────────┘ │  - Bidding    │
│                 │                       │  - Trump sel. │
│                 │   Current Trick:      │  - Other      │
│                 │   [P0: 5-3] [P1: 6-2]│               │
│                 │                       │               │
│                 │   Player's Hand:      │               │
│                 │   ┌─┐ ┌─┐ ┌─┐ ┌─┐   │               │
│                 │   │5│ │6│ │3│ │4│   │               │
│                 │   │-│ │-│ │-│ │-│   │               │
│                 │   │3│ │4│ │2│ │1│   │               │
│                 │   └─┘ └─┘ └─┘ └─┘   │               │
└─────────────────┴───────────────────────┴───────────────┘
│ Footer: Debug Mode • Press Ctrl+Shift+D for full Debug UI│
└─────────────────────────────────────────────────────────┘
```

### Component Details

#### Header Bar (Compact)
- **Texas 42** title with current phase badge (color-coded)
- **AI Controls** (centered): Play/Pause button and Step button
  - Play/Pause: Toggles continuous AI play
  - Step: Executes one AI action
  - Instant execution only - no delays
- Team scores and marks displayed
- Current player turn indicator
- Height: ~40px to maximize game area

#### Left Panel: Trick History (250px wide)
- Shows all completed tricks (up to 7)
- Each trick displays:
  - Trick number, winner, and points
  - All 4 dominoes played in 2x2 grid
  - Winner's domino highlighted in green
- Compact design to fit all 7 tricks without scrolling

#### Center Panel: Game State & Interactive Hand
1. **Trump & Suit Display** (Prominent, top center)
   - TRUMP in large red text (24px)
   - LED SUIT in large blue text when active
   - Clearly visible box with border

2. **Current Trick Display**
   - Shows dominoes played so far in current trick
   - Player labels above each domino
   - Centered layout with spacing

3. **Player's Hand** (Interactive dominoes)
   - Displayed as clickable domino tiles
   - Visual states:
     - **Playable**: Bright green background, white text
     - **Non-playable**: Gray background, 50% opacity
     - **Counting (5s/10s)**: Yellow border glow
   - Hover effects on playable dominoes
   - Click to play - no need for action buttons

#### Right Panel: Non-Play Actions (250px wide)
- Shows all available actions EXCEPT play actions
- Includes: bidding, pass, trump selection, complete trick, etc.
- Simple button list, categorized by action type
- No duplicates - each action appears once

## Technical Implementation Details

### Action ID Formats (Critical for Implementation)
```javascript
// Bidding actions
"bid-30"           // Points bid
"bid-1-marks"      // Marks bid
"pass"             // Pass bid

// Trump selection
"trump-blanks"     // Suit trump
"trump-ones"
"trump-doubles"    // Doubles as trump
"trump-no-trump"   // No trump

// Play actions (handled by domino clicks)
"play-5-3"         // Play domino 5-3
"play-6-6"         // Play double six

// Other actions
"complete-trick"
"score-hand"
"redeal"
```

### Domino Playability Detection
```javascript
// Extract playable dominoes from available actions
$: playableDominoes = (() => {
  const dominoes = new Set();
  $availableActions
    .filter(action => action.id.startsWith('play-'))
    .forEach(action => {
      const dominoId = action.id.replace('play-', '');
      dominoes.add(dominoId);
      // Add reversed version (5-3 and 3-5)
      const parts = dominoId.split('-');
      if (parts.length === 2) {
        dominoes.add(`${parts[1]}-${parts[0]}`);
      }
    });
  return dominoes;
})();
```

### Color Usage Guidelines
- **Game Phases**: Each phase should have a distinct color for immediate recognition
- **Domino States**:
  - Playable dominoes: Bright, saturated color (e.g., green) that stands out
  - Non-playable dominoes: Muted, desaturated color with reduced opacity
  - Counting dominoes (5s/10s): Special accent or glow to highlight their value
- **Game Information**:
  - Trump: Bold, attention-grabbing color (e.g., red)
  - Led suit: Different prominent color (e.g., blue)
  - Winning elements: Consistent success color throughout UI

## AI Player Functionality

### Core Requirements
- **Instant Mode Only**: AI executes actions immediately with no delays or animations
- **No Speed Settings**: Remove any speed/delay configuration - AI always runs at maximum speed
- **Simple Controls**: Just Play/Pause and Step buttons in the main header
- **Strategy Configuration**: Remains in debug panel only (not in header)

### Header AI Controls
- **Play Button (▶️)**: Starts continuous AI play for all players
- **Pause Button (⏸️)**: Stops AI execution (shows when playing)
- **Step Button (⏭️)**: Executes exactly one AI action
- Controls are centered in header for easy access
- No configuration in header - just start/stop/step

### AI Behavior
- Takes actions for any player whose turn it is
- Uses the first available action by default (configurable in debug panel)
- Continues until game ends or paused
- No delays between actions - instant execution
- Respects game rules and available actions only

### Debug Panel AI Configuration
The full QuickPlay panel in the debug overlay provides:
- Strategy selection (random, first, aggressive, conservative)
- Detailed status information
- Batch action execution
- Phase skipping functionality

Note: The header controls are just shortcuts - full configuration remains in the debug panel to keep the main UI clean.

## Debug Panel (Ctrl+Shift+D Overlay)

The full debug panel provides deep inspection without interfering with gameplay:

### Tabs
1. **State**: Complete GameState display with all fields
2. **Actions**: Categorized view of all available actions
3. **History**: Action timeline with time travel
4. **JSON**: Raw state with syntax highlighting
5. **Tools**: Bug report, test generation, auto-play

### Key Features
- **Time Travel**: Click any past action to restore game state
- **State Validation**: Automatic detection of invalid states
- **Test Generation**: Create Playwright tests from game sessions
- **Bug Reporting**: Capture complete state for issue reports
- **Auto-Play**: Simulate games with configurable strategies

## Critical Discoveries During Implementation

1. **Action Format**: Game engine uses simple `play-${dominoId}` format, not complex player-prefixed IDs
2. **Domino Storage**: Dominoes may be stored in either orientation (5-3 or 3-5), must check both
3. **No Duplicate Actions**: The game engine already prevents duplicates, don't filter by phase
4. **State is Reliable**: All game state is consistently available through stores - trust it
5. **Debug First**: This is a debug UI - show everything, hide nothing

## Error Handling & Edge Cases

### State Validation Errors
- Display validation errors prominently at the top of the debug panel
- Show full error details in a copy-friendly format
- Highlight invalid state fields in red
- Never hide or minimize error messages

### Empty States
- When no tricks have been played: Show "No tricks played yet"
- When no current trick: Show "Waiting for first play..."
- When no available actions: Show appropriate message based on phase
- Empty hands: Should never occur, but handle gracefully

### Game End States
- Clearly indicate when game has ended
- Show final scores and winner prominently
- Disable play controls but keep history/debug features active
- Allow starting new game or resetting

## Responsive Behavior

### Minimum Requirements
- Minimum viewport width: 1024px
- If viewport is smaller, panels should stack vertically
- Priority order for small screens: Game state > Actions > History
- Debug panel should be scrollable if content exceeds viewport

### Panel Resizing
- Fixed widths for side panels (250px each)
- Center panel expands to fill available space
- No user-resizable panels to keep layout stable
- Content within panels scrolls if needed

## State Persistence & URL Sharing

### URL State Encoding
- Game state can be encoded in URL for sharing
- Clicking "Copy State URL" creates shareable link
- Loading URL restores exact game state
- URL includes: current state, action history, and random seed

### Local Storage
- Remember debug panel open/closed state
- Persist AI strategy preference
- Store recently used game states
- Clear storage option in debug panel

## Integration Points

### Store Connections
Required Svelte stores:
- `gameState`: Current game state
- `availableActions`: Valid actions for current state
- `actionHistory`: List of all actions taken
- `stateValidationError`: Any validation errors
- `gameActions`: Object with executeAction, undo, reset methods

### Event System
- All state changes through gameActions.executeAction()
- No direct state manipulation in UI components
- Actions trigger immediate re-render
- Error boundaries around action execution

### Test Integration
- Every interactive element must have a data-testid
- Consistent naming: `play-${dominoId}`, `bid-${value}`, etc.
- Debug panel sections have section-level test IDs
- Actions must be accessible via both UI and test helpers

## Performance Requirements

- No scrolling on main game view
- Instant response to domino clicks
- Smooth hover animations (< 200ms transitions)
- Efficient re-rendering on state changes
- Support for 100+ actions in history

## Accessibility Notes

- All clickable elements have appropriate hover states
- Disabled dominoes show not-allowed cursor
- Color coding is supplemented with other indicators (opacity, borders)
- Test IDs on all interactive elements for E2E testing

## Future Enhancements

- Keyboard shortcuts for common actions
- State diff visualization  
- Network play debugging
- Performance profiling overlay
- Advanced AI strategies
- Multi-game batch testing