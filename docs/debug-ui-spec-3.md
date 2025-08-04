# Debug UI Specification v3

## Core Philosophy

The Debug UI for Texas 42 is a **development-friendly game interface** that balances debugging capabilities with an excellent playing experience. It provides full visibility into game mechanics while maintaining an intuitive, enjoyable interface that respects the traditions of Texas 42.

## Key Design Principles

1. **Game Visibility First**: The current game state, player's hand, and available actions must be visible at all times on the main screen
2. **Direct Interaction**: Players can click dominoes in their hand to play them - no need to use action buttons for play moves
3. **Debug as Overlay**: The debug panel is an optional overlay for deep inspection, not required for gameplay
4. **No Scrolling**: All essential game information fits on one screen without scrolling
5. **Color-Coded Clarity**: Use color to indicate playable vs non-playable dominoes and game state
6. **Intuitive Game Flow**: The interface follows natural Texas 42 gameplay patterns - bid, declare trump, play dominoes
7. **Visual Hierarchy**: Most important information (hand, current trick, trump) is prominently displayed
8. **Contextual Actions**: Only show relevant actions for the current game phase
9. **Progressive Disclosure**: Basic gameplay on main screen, advanced debugging via overlay
10. **Respectful Design**: Honor the game's Texas heritage with appropriate visual treatment

## Main Screen Layout

### Visual Design System

#### Color Palette
- **Primary Colors**:
  - Texas Blue (#002868) - Headers, important text
  - White (#FFFFFF) - Background, contrast
  - Warm Wood (#8B4513) - Domino backgrounds
- **Game State Colors**:
  - Playable Green (#22C55E) - Legal moves
  - Inactive Gray (#9CA3AF) - Unplayable dominoes
  - Trump Red (#DC2626) - Trump indicators
  - Count Gold (#F59E0B) - Counting dominoes
  - Winner Purple (#8B5CF6) - Winning plays

#### Typography
- Headers: Bold sans-serif (20-24px)
- Game text: Clear sans-serif (16-18px)
- Domino pips: Bold monospace (20px)

### Layout Structure
```
┌────────────────────────────────────────────────────────────────────┐
│ Texas 42 | [Phase Badge] | [AI Controls] | Team Scores | Turn: P0  │
├──────────────────┬────────────────────────┬────────────────────────┤
│                  │                        │                        │
│  Game Progress   │    Playing Area        │   Action Panel         │
│                  │                        │                        │
│  Trick 1 ✓ (7)   │  ┌──────────────────┐ │  ┌──────────────────┐ │
│  P2 won          │  │ TRUMP: FIVES     │ │  │ Available Actions │ │
│  [5-5][6-4]      │  │ Led: SIXES       │ │  │                  │ │
│  [5-3][5-2]      │  └──────────────────┘ │  │ [Bid 30 Points]  │ │
│                  │                        │  │ [Bid 1 Mark]     │ │
│  Trick 2 ✓ (5)   │   Current Trick:       │  │ [Pass]           │ │
│  P0 won          │   ┌─────┬─────┐       │  │                  │ │
│  [3-2][4-1]      │   │ P0  │ P1  │       │  │ ─────────────    │ │
│  [6-1][3-0]      │   │[5-3]│[6-2]│       │  │                  │ │
│                  │   └─────┴─────┘       │  │ Quick Actions:   │ │
│                  │   ┌─────┬─────┐       │  │ [Complete Trick] │ │
│  Current: 3/7    │   │ P3  │ P2  │       │  │ [Score Hand]     │ │
│  Points: 12/42   │   │ ??? │ ??? │       │  │                  │ │
│                  │   └─────┴─────┘       │  └──────────────────┘ │
│                  │                        │                        │
│                  │   Your Hand:           │  Team Status:          │
│                  │   ┌───┐ ┌───┐ ┌───┐   │  US: 2 marks (12 pts) │
│                  │   │ 5 │ │ 6 │ │ 3 │   │  THEM: 3 marks        │
│                  │   │ ─ │ │ ─ │ │ ─ │   │                        │
│                  │   │ 3 │ │ 4 │ │ 2 │   │  Bid: 35 (P0)         │
│                  │   └───┘ └───┘ └───┘   │  Need: 23 more        │
└──────────────────┴────────────────────────┴────────────────────────┘
│ Debug Mode • Ctrl+Shift+D for advanced debugging • v3.0            │
└────────────────────────────────────────────────────────────────────┘
```

### Component Specifications

#### Header Bar (Clean & Functional)
- **Title**: "Texas 42" with version badge
- **Phase Indicator**: Color-coded badge showing current phase
  - Setup (Gray) → Bidding (Blue) → Trump Selection (Purple) → Playing (Green) → Scoring (Gold)
- **AI Controls** (Centered, minimal):
  - Single button that toggles between ▶️ Play All and ⏸️ Pause
  - Step button (⏭️) for single AI moves
  - No speed controls - AI always plays instantly
- **Score Display**: "US: 2 • THEM: 3" format with marks
- **Turn Indicator**: Highlights current player with arrow

#### Left Panel: Game Progress (200px)
**Purpose**: Show game progression and trick history

**Trick Display**:
- Completed tricks shown as cards with:
  - Header: "Trick N ✓" with points won
  - Winner indication (colored background)
  - 2x2 grid of dominoes played
  - Subtle animation when trick completes

**Progress Indicators**:
- Current trick number (e.g., "Trick 3 of 7")
- Running point total with progress bar
- Visual indication of bid target

**Design Details**:
- Completed tricks slightly faded
- Current trick highlighted
- Smooth scroll if > 4 tricks

#### Center Panel: Playing Area
**Trump & Led Suit Display**:
- Prominent box at top with clear labels
- TRUMP in red with suit icon/text
- Led suit in blue (only during play)
- Visual pip representations when possible

**Current Trick Area**:
- 2x2 grid showing player positions
- Player labels (P0, P1, P2, P3) with position indicators
- Played dominoes show immediately
- Unplayed positions show "???" placeholder
- Winning domino gets subtle glow effect

**Player's Hand**:
- Large, clickable domino tiles
- Visual states:
  - **Playable**: Bright background, slight elevation, cursor pointer
  - **Counting**: Gold corner badge with point value
  - **Unplayable**: Grayed out, no hover effect
  - **Hover**: Slight scale-up animation
- Smart arrangement (counts on ends when possible)

#### Right Panel: Contextual Actions (200px)
**Action Organization**:
- Group by action type with subtle headers
- Only show phase-appropriate actions
- Visual hierarchy (primary vs secondary actions)

**Bidding Phase**:
- Common bids as quick buttons (30, 35, 40, 1 mark)
- "Other bid..." for custom amounts
- Pass button prominently displayed

**Trump Selection**:
- Visual suit buttons with pip icons
- Special options clearly marked (Doubles, No-Trump)

**Play Phase**:
- Only non-play actions shown (play via domino clicks)
- Quick actions: Complete Trick, Score Hand

**Team Status Widget**:
- Always visible in right panel
- Shows current scores, bid, and target
- Updates in real-time

### Enhanced UX Features

#### Smart Domino Arrangement
```javascript
// Arrange hand for optimal visibility
function arrangeHand(dominoes) {
  // 1. Group doubles together
  // 2. Place counting dominoes (5-5, 6-4, etc.) on edges
  // 3. Sort by suit connections for easier play
  return optimizedArrangement;
}
```

#### Contextual Tooltips
- Hover over domino: Show why it's playable/unplayable
- Hover over bid: Show what you need to make it
- Hover over trick: Show detailed point breakdown

#### Animation & Feedback
- **Domino Play**: Smooth slide from hand to trick
- **Trick Completion**: Brief celebration for winner
- **Invalid Action**: Gentle shake animation
- **Score Updates**: Number roll animation

### AI Integration

#### Simplified Controls
Header shows only essential controls:
- **Play/Pause Toggle**: Single button that changes based on state
- **Step Button**: Execute one AI action
- **Status Indicator**: Small text showing "AI Playing" or "Paused"

#### AI Behavior
- Takes actions for any player whose turn it is
- Uses the first available action by default (configurable in debug panel)
- Continues until game ends or paused
- No delays between actions - instant execution
- Respects game rules and available actions only
- Shows thinking indicator during complex decisions

#### Smart Defaults
- AI automatically takes over for non-human players
- Pauses after each hand for review
- Can be set to pause at phase transitions

#### Debug Panel AI Configuration
The full QuickPlay panel in the debug overlay provides:
- Strategy selection (random, first, aggressive, conservative)
- Detailed status information
- Batch action execution
- Phase skipping functionality

Note: The header controls are just shortcuts - full configuration remains in the debug panel to keep the main UI clean.

## Debug Panel Overlay (Ctrl+Shift+D)

### Panel Design
- Semi-transparent backdrop
- Centered modal (80% viewport)
- Tabbed interface with clear navigation
- Close button and ESC key support

### Enhanced Tabs

#### 1. Game State
- **Tree View**: Expandable/collapsible state sections
- **Search**: Filter state properties
- **Diff Mode**: Highlight recent changes
- **Copy Button**: For each section

#### 2. Actions & History
- **Timeline View**: Visual representation of game flow
- **Action Groups**: Collapsed by phase
- **Time Travel**: Click to restore any state
- **Action Preview**: Hover to see state changes

#### 3. Analysis
- **Statistics**: Win probabilities, optimal plays
- **Hand Strength**: Visual analysis of current hand
- **Mistake Detection**: Highlight suboptimal plays
- **Pattern Recognition**: Common play sequences

#### 4. Testing
- **Scenario Builder**: Create specific game states
- **Test Recorder**: Generate Playwright tests
- **Replay Mode**: Load and replay saved games
- **Batch Testing**: Run multiple game variations

#### 5. Settings
- **AI Configuration**: Strategy selection per player
- **Visual Preferences**: Color themes, animations
- **Debug Options**: Validation, logging levels
- **Export/Import**: Save/load preferences

### Debug Panel UX Improvements

#### Quick Access Toolbar
Floating toolbar when debug panel is closed:
- Bug Report button
- State validation indicator
- Quick time travel (←→ arrows)
- Toggle auto-play

#### Keyboard Shortcuts
- `Ctrl+Shift+D`: Toggle debug panel
- `←/→`: Navigate history
- `Space`: Play/pause AI
- `S`: Step AI
- `R`: Reset game
- `1-7`: Quick bid shortcuts during bidding

## Mobile & Responsive Considerations

### Breakpoints
- **Desktop** (>1200px): Full three-panel layout
- **Tablet** (768-1200px): Stacked layout, collapsible panels
- **Mobile** (<768px): Single column, swipe between panels

### Touch Optimizations
- Larger tap targets (min 44px)
- Swipe gestures for panel navigation
- Long-press for tooltips
- Pinch to zoom on game area

## Accessibility Enhancements

### Screen Reader Support
- Semantic HTML structure
- ARIA labels for all controls
- Announce game state changes
- Keyboard navigation for all features

### Visual Accessibility
- High contrast mode option
- Colorblind-friendly palette
- Adjustable text size
- Pattern/texture options for color coding

## Performance Optimizations

### Rendering
- Virtual scrolling for long histories
- Memoized components
- Debounced state updates
- Progressive rendering for debug data

### State Management
- Efficient diff algorithms
- Compressed history storage
- Lazy loading for debug features
- Background validation

## Performance Requirements

- No scrolling on main game view
- Instant response to domino clicks
- Smooth hover animations (< 200ms transitions)
- Efficient re-rendering on state changes
- Support for 100+ actions in history

## Polish & Delight

### Subtle Animations
- Smooth transitions between phases
- Gentle hover effects
- Satisfying click feedback
- Victory celebrations

### Sound Design (Optional)
- Subtle click sounds
- Phase transition chimes
- Victory/defeat themes
- Mutable by default

### Easter Eggs
- Special animations for rare hands
- Texas-themed victory messages
- Achievement notifications
- Historical facts during loading

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

## Critical Discoveries During Implementation

1. **Action Format**: Game engine uses simple `play-${dominoId}` format, not complex player-prefixed IDs
2. **Domino Storage**: Dominoes may be stored in either orientation (5-3 or 3-5), must check both
3. **No Duplicate Actions**: The game engine already prevents duplicates, don't filter by phase
4. **State is Reliable**: All game state is consistently available through stores - trust it
5. **Debug First**: This is a debug UI - show everything, hide nothing

## Error Handling

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

### User-Friendly Messages
Instead of technical errors, show:
- "Oops! That domino can't be played right now"
- "This bid needs to be higher than 30"
- "Let's finish the current trick first"

### Recovery Options
- Undo button for recent actions
- Reset hand option
- Report bug with context
- Suggested fixes

## Summary of Key Improvements

1. **Better Visual Hierarchy**: Clear separation of game area, history, and actions
2. **Contextual UI**: Show only relevant actions and information
3. **Enhanced Feedback**: Animations and tooltips guide players
4. **Smarter Defaults**: AI integration that doesn't interfere with play
5. **Progressive Disclosure**: Advanced features hidden until needed
6. **Texas Character**: Respectful design honoring the game's heritage
7. **Accessibility**: Full keyboard and screen reader support
8. **Mobile Ready**: Responsive design for all devices
9. **Delightful Details**: Polish that makes debugging enjoyable
10. **Error Prevention**: Guide users away from invalid actions

This specification creates a Debug UI that serves both as an excellent development tool and an enjoyable way to play Texas 42, making testing and debugging a pleasure rather than a chore.