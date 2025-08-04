# Debug UI Specification v4

## Core Philosophy

The Debug UI for Texas 42 is a **development-friendly game interface** that balances debugging capabilities with an excellent playing experience. It provides full visibility into game mechanics while maintaining an intuitive, enjoyable interface that respects the traditions of Texas 42.

## Key Design Principles

1. **Game Visibility First**: Current game state, player's hand, and available actions are visible at all times
2. **Direct Interaction**: Click dominoes to play them - no action buttons needed for play moves
3. **Debug as Overlay**: Debug panel is optional overlay (Ctrl+Shift+D) for deep inspection
4. **Minimal Scrolling**: Essential game information fits on screen, with compact trick history
5. **Color-Coded Clarity**: Visual states for playable dominoes, trump, winners, and highlighting
6. **Intuitive Game Flow**: Natural Texas 42 patterns - bid, declare trump, play dominoes
7. **Visual Hierarchy**: Most important info (hand, current trick, trump) prominently displayed
8. **Contextual Actions**: Only show relevant actions for current game phase
9. **Progressive Disclosure**: Basic gameplay on main screen, advanced debugging via overlay
10. **Respectful Design**: Honor the game's Texas heritage with appropriate visual treatment

## Visual Design System

### Color Palette
- **Primary Colors**:
  - Texas Blue (#002868) - Headers, bid buttons, important text
  - White (#FFFFFF) - Background, contrast
  - Warm Cream (#F5F3F0) - Domino backgrounds
- **Game State Colors**:
  - Playable Green (#22C55E) - Legal moves with green background
  - Inactive Gray (#9CA3AF) - Unplayable dominoes grayed out
  - Trump Red (#DC2626) - Trump indicators and pass button
  - Count Gold (#F59E0B) - Counting domino badges
  - Winner Purple (#8B5CF6) - Winning domino borders in tricks
- **Highlighting Colors** (Bidding/Trump Selection):
  - Primary Yellow (#FEF3C7) - Main suit matches
  - Secondary Blue (#DBEAFE) - Secondary matches (non-doubles when hovering double)

### Typography
- Headers: Bold sans-serif (18-24px)
- Game text: Clear sans-serif (14-16px)
- Small text: Sans-serif (11-12px)
- Domino pips: Geometric shapes, not text

## Layout Structure

### Three-Panel Layout
```
┌────────────────────────────────────────────────────────────────────┐
│ Texas 42 | [PLAYING] | [▶️ Play All] [⏭️ Step] | US: 4 • THEM: 3 | P1│
├────────────────┬─────────────────────────┬────────────────────────┤
│ Game Progress  │    Playing Area         │   Actions              │
│   (300px)      │      (flex: 1)          │    (200px)             │
├────────────────┴─────────────────────────┴────────────────────────┤
│ Debug Mode • Ctrl+Shift+D for advanced debugging • v4.0            │
└────────────────────────────────────────────────────────────────────┘
```

## Component Specifications

### Header Bar
- **Title**: "Texas 42" (24px, bold, Texas Blue)
- **Phase Badge**: Color-coded current phase
  - Setup (Gray #6B7280)
  - Bidding (Blue #3B82F6)
  - Trump Selection (Purple #8B5CF6)
  - Playing (Green #22C55E)
  - Scoring (Gold #F59E0B)
  - Game End (Red #DC2626)
- **AI Controls**: Centered, minimal
  - Toggle button: ▶️ Play All / ⏸️ Pause
  - Step button: ⏭️ (disabled when AI active)
  - Status text: "AI Playing" when active
- **Score Display**: "US: X • THEM: Y" format
- **Turn Indicator**: "Turn: PX" with player badge

### Left Panel: Game Progress (300px wide)

#### Progress Summary
- **Points Display**: "Points: X/42" (no trick count)
- **Progress Bar**: Visual representation of points toward bid
- **Need Text**: "Need: X more" when bid active

#### Trick History (Compact Display)
- **Trick Cards**: Minimal height (padding: 6px)
- **Layout**: Horizontal flex row
  - 4 tiny dominoes (36x56px) in a row
  - Trick info on right (number, points)
- **Visual Indicators**:
  - Purple border on winning domino
  - Faded opacity (0.75) for completed tricks
  - 4px margin between tricks
- **No Scrollbar**: Fits 4-5 tricks comfortably

### Center Panel: Playing Area

#### Trump & Led Suit Display
- **Two boxes** side by side:
  - TRUMP: [Suit/Doubles/No-Trump] (red text)
  - Led: [Current Suit] (blue text, only during play)

#### Current Trick (2x2 Grid)
```
P0    P1
[🁣]  [???]

P3    P2
[???] [🁢]
```
- Shows played dominoes immediately
- Unplayed positions show "???"
- Player labels above each position

#### Player's Hand
- **Large dominoes** (70x110px) for easy clicking
- **Visual States**:
  - Playable: Green background, elevated, pointer cursor
  - Unplayable: Grayed out, disabled cursor
  - Counting: Gold badge with point value
- **Suit Highlighting** (Bidding/Trump Selection only):
  - Hover over domino half highlights all matching suits
  - Primary (yellow): Direct suit matches
  - Secondary (blue): When hovering double, shows non-doubles with that suit
  - Indicator text: "Highlighting: [Suit]" or "Highlighting: Doubles & [Suit]"

### Right Panel: Actions (200px)

#### Bidding Phase
- **Pass button** at top (red background)
- **Separator line**
- **All bid buttons** below (blue background)
  - Equal treatment for all bids
  - Shows actual bid text (e.g., "Bid 31", "1 Mark")

#### Trump Selection
- **Suit buttons** with clear labels
- Includes Doubles and No-Trump options

#### Play Phase
- Only shows non-play actions
- Complete Trick, Score Hand, etc.

#### Team Status (Always Visible)
- US: X marks (Y pts)
- THEM: X marks (Y pts)
- Bid: X (PY)
- Need: X more

## Enhanced UX Features

### Contextual Tooltips
- Hover over domino: Show why it's playable/unplayable
- Hover over bid: Show what points are needed
- Hover over trick: Show detailed point breakdown
- Hover over action: Show what it will do

### Animation & Feedback
- **Domino Play**: Smooth slide from hand to trick (200ms)
- **Trick Completion**: Brief fade animation
- **Invalid Action**: Gentle shake animation
- **Score Updates**: Number roll animation
- **Phase Changes**: Color transition in badge

## Enhanced Features

### Domino Hover Highlighting (Bidding/Trump Only)
1. **Non-double hover**: Highlights matching suit dominoes
   - Hover top half → highlight high pip value
   - Hover bottom half → highlight low pip value
2. **Double hover**: Highlights ALL doubles + matching suit
   - Doubles get primary highlight (yellow)
   - Non-doubles with suit get secondary (blue)
3. **Implementation**: Uses `{#key hoveredSuit}` for reactivity

### AI Integration

#### Core Requirements
- **Instant Mode Only**: AI executes actions immediately with no delays or animations
- **No Speed Settings**: No speed/delay configuration - AI always runs at maximum speed
- **Simple Controls**: Just Play/Pause and Step buttons in the main header
- **Strategy Configuration**: Remains in debug panel only (not in header)

#### AI Behavior
- Takes actions for any player whose turn it is
- Uses the first available action by default (configurable in debug panel)
- Continues until game ends or paused
- No delays between actions - instant execution
- Respects game rules and available actions only

#### Debug Panel AI Configuration
The full QuickPlay panel in the debug overlay provides:
- Strategy selection (random, first, aggressive, conservative)
- Detailed status information
- Batch action execution
- Phase skipping functionality

Note: The header controls are just shortcuts - full configuration remains in the debug panel to keep the main UI clean.

### Debug Panel (Ctrl+Shift+D)

#### Panel Design
- **Semi-transparent backdrop** (rgba(0,0,0,0.5))
- **Centered modal** (80% viewport coverage)
- **Tabbed interface** with clear navigation
- **Close button** and ESC key support

#### Tab Structure (4 Tabs)

##### 1. Game State Tab
- **Tree View**: Expandable/collapsible state sections
- **Diff Mode**: Highlight recent changes in yellow
- **Copy Button**: For each section (JSON format)
- **Raw JSON View**: 
  - Full GameState with syntax highlighting
  - Copy-to-clipboard functionality
  - Monospace font for readability
- **No Search**: Direct display of current game state

##### 2. History Tab
- **Timeline View**: Complete action history with sequential numbering
- **Time Travel**: Click any action to restore that state
- **Undo Button**: Revert last action
- **Reset Game**: Clear history and start fresh
- **Empty State**: Helpful message when no actions taken

##### 3. QuickPlay Tab
- **AI Strategy**: Selection per player (random, first, aggressive, conservative)
- **Batch Actions**: Execute multiple actions at once
- **Phase Skip**: Jump to specific game phases
- **Auto-Play Controls**: Start/stop continuous play
- **Status Display**: Current AI state and decisions

##### 4. Historical State Tab
- **Event Sourcing View**: Shows initial state and all actions for replay
- **Tree View Toggle**: Expandable/collapsible JSON tree view
- **Initial State**: Complete starting game state with all player hands
- **Actions List**: All executed actions with their IDs and labels
  - Shows commands like: `bid-30`, `pass`, `trump-sixes`, `play-5-3`
  - Each action includes id, label, and any parameters
- **Copy Historical JSON**: One-click export of complete event sourcing data
- **Critical for Debugging**: 
  - Complete state history for exact game replay
  - Includes all randomness (initial shuffle)
  - Perfect for bug replication

#### Quick Access Toolbar
When debug panel is closed, show floating toolbar:
- Bug Report button
- Game state indicator (active/paused)
- Quick undo button
- Toggle auto-play
- Open debug panel button

#### Keyboard Shortcuts
- `Ctrl+Shift+D`: Toggle debug panel
- `←/→`: Navigate history (when panel open)
- `Escape`: Close debug panel
- `Ctrl+Z`: Undo last action

## Technical Implementation

### Action ID Formats (shown in History)
```javascript
// Bidding
"bid-30" to "bid-41"    // Point bids
"bid-1-marks"           // Mark bids
"pass"                  // Pass

// Trump
"trump-blanks" through "trump-sixes"
"trump-doubles"
"trump-no-trump"

// Play (automatic from domino clicks)
"play-5-3"              // Domino play actions

// Other
"complete-trick"
"score-hand"
```

### Store Connections & Game Integration

#### Core Stores (from `src/stores/gameStore.ts`)
- `gameState`: Complete GameState object containing:
  - `phase`: Current game phase
  - `currentPlayer`: Active player ID
  - `players`: Array of Player objects with hands
  - `tricks`: Completed tricks array
  - `currentTrick`: In-progress trick
  - `teamScores`: Current hand scores
  - `teamMarks`: Game scores
  - `trump`: Trump selection
  - `currentBid`: Winning bid
- `actionHistory`: Complete history for time travel
- `stateValidationError`: Current validation errors
- `gameActions`: Methods object with:
  - `executeAction(action)`: Execute any game action
  - `undo()`: Revert last action
  - `reset()`: Start new game
  - `loadState(state)`: Load specific state

#### Derived Stores
- `currentPlayer`: Current player with full hand
- `gamePhase`: Just the phase string
- `biddingInfo`: Bidding-specific state
- `teamInfo`: Team scores and status
- `quickplayState`: AI control state
- `quickplayActions`: AI control methods

#### Game Engine Integration
- **Rules Engine** (`src/rules/`): All game logic
  - `getAvailableActions()`: Legal moves
  - `executeAction()`: State transitions
  - `validateState()`: Rule checking
- **Types** (`src/game/types.ts`): All TypeScript interfaces
  - `GameState`, `Player`, `Domino`, `Action`, `Trick`, etc.
- **AI Engine** (`src/ai/`): Decision making
  - `makeAIDecision()`: Choose best action
  - Strategy implementations

### Domino Sizes
- **Regular**: 70x110px (player hand)
- **Small**: 50x80px (current trick)
- **Tiny**: 36x56px (trick history)

### Performance Optimizations
- Reactive statements for computed values
- `{#key}` blocks for forced reactivity
- No unnecessary re-renders
- Compact trick display prevents scrolling

## Critical Implementation Details

1. **Direct domino clicking**: No play buttons needed
2. **Highlight reactivity**: Requires `{#key hoveredSuit}` wrapper
3. **Compact tricks**: Horizontal layout with tiny dominoes
4. **Purple winner borders**: Visual trick winner indication
5. **Suit detection**: Based on mouse position within domino
6. **No trick count**: Removed "Trick X of 7" to save space

## Error Handling & User Experience

### State Validation Errors
- Display validation errors prominently in debug panel Validation tab
- Show full error details in copy-friendly monospace format
- Highlight invalid state fields with red border
- Never hide or minimize error messages
- Include stack trace for debugging

### Empty States
- When no tricks played: "No tricks played yet" message
- When no current trick: Show empty 2x2 grid with "???" placeholders
- When no available actions: "Waiting for other players..." or phase-specific message
- Empty hands: Should never occur, show error state if detected

### Game End States
- Phase badge changes to red "GAME END"
- Show final scores prominently in header
- Disable play controls but keep debug features active
- Show "New Game" button in actions panel

### User-Friendly Error Messages
Instead of technical errors, show:
- "That domino can't be played right now" (with reason on hover)
- "Bid must be higher than X" 
- "Please complete the current trick first"
- "It's not your turn" with current player highlight

### Recovery Options
- Undo button always visible in debug toolbar
- Reset game option in debug panel
- Report Bug captures full context
- Suggested actions based on error type

## Testing Requirements

### Test ID Conventions
All interactive elements must have `data-testid` attributes:
- **Actions**: `action-{actionId}` (e.g., `action-bid-30`, `action-pass`)
- **Dominoes**: `domino-{player}-{high}-{low}` (e.g., `domino-0-5-3`)
- **Panels**: `panel-{name}` (e.g., `panel-game-progress`, `panel-actions`)
- **Debug Elements**: `debug-{element}` (e.g., `debug-toggle`, `debug-state-tab`)
- **Game Elements**: `game-{element}` (e.g., `game-phase`, `game-trump`)

### E2E Helper Integration
- All actions accessible via playwrightHelper.clickAction()
- State verification through playwrightHelper.verifyGameState()
- Consistent selectors for automated testing
- 5-second timeout for all E2E tests per project rules
- Tests must hit Debug UI via unified helper in `src/tests/e2e/helpers/playwrightHelper.ts`


## Performance Considerations

### Rendering Optimization
- Use Svelte's reactive statements (`$:`) for computed values
- Implement `{#key}` blocks only where needed for forced reactivity
- Minimize DOM updates with proper component boundaries
- Debounce rapid state changes (100ms)

### State Management
- Keep derived values in reactive statements, not stores
- Use store subscriptions efficiently
- Batch related updates together
- Avoid unnecessary store creations

### Memory Management
- Limit action history to 500 items
- Virtualize long lists in debug panel
- Clean up event listeners properly
- Clear timeouts on component destroy

## Implementation Status

### Completed Features
- ✅ Three-panel responsive layout
- ✅ Compact horizontal trick display
- ✅ Direct domino clicking for play
- ✅ Suit highlighting on hover
- ✅ AI instant execution controls
- ✅ Basic debug panel overlay
- ✅ All core game state display
- ✅ Color-coded visual states

### Pending Features
- ⏳ Enhanced debug panel tabs
- ⏳ Time travel UI
- ⏳ Raw JSON viewer
- ⏳ Contextual tooltips
- ⏳ Animation polish
- ⏳ Error recovery UI
- ⏳ Quick access toolbar

This specification represents the complete design for the Texas 42 Debug UI, incorporating all essential features from previous versions while documenting both implemented and planned functionality.