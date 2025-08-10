# Mobile UI Specification

## Core Philosophy

The Mobile UI for Texas 42 is a **mobile-first game interface** that provides an excellent playing experience on mobile web browsers. The interface prioritizes touch-friendly interactions and maintains full visibility into game mechanics while respecting mobile screen constraints.

## Key Design Principles

1. **Pure UI Implementation**: UI exclusively displays state from src/game and src/stores - NO game logic in UI components
2. **Mobile-First Design**: Optimized specifically for mobile web browsers with touch interactions
3. **Game Flow in Play Area**: Actions are invoked through direct interaction with game elements
4. **Hybrid Input Support**: Primary touch interactions with optional keyboard shortcuts for power users
5. **Smart Navigation**: Automatic panel switching based on game phase reduces cognitive load
6. **Touch-Optimized**: All interactive elements meet mobile touch target requirements
7. **Correct by Construction**: UI correctness through pure functions and error detection
8. **Visual Consistency**: Well-defined styles enable easy designer customization
9. **Flawless Mobile Compatibility**: Works seamlessly across standard mobile browsers

## Technical Architecture Philosophy

**CRITICAL**: The UI is a pure presentation layer that implements what's already in src/game. The only logic in UI components should be about displaying state and creating a pleasurable, modern gaming experience.

- **Game Logic**: EXCLUSIVELY in src/game (pure functional game engine)
- **State Management**: EXCLUSIVELY in src/stores (reactive state management)
- **UI Responsibility**: Display state, handle user interactions, provide visual feedback
- **Forbidden in UI**: Game rules, scoring calculations, state transitions, business logic

This separation ensures the UI remains maintainable, testable, and focused on user experience while the game engine remains pure and reliable.

## Visual Design System

### Color Semantic Roles
- **Playable State**: Dominoes that can be legally played
- **Inactive State**: Unplayable dominoes (grayed out)
- **Trump Indicators**: Trump suit and trump-related elements
- **Counting Dominoes**: Point-bearing dominoes with badges
- **Winner Highlighting**: Winning dominoes in completed tricks
- **Phase Indicators**: Color-coded game phase badges
- **Team Identification**: Visual distinction between US/THEM teams

*Note: Specific color values are at implementer's discretion. Styles must be consistent and well-defined for easy designer modification.*

### Typography Requirements
- Headers: Bold, readable at mobile sizes
- Game text: Clear, legible for gameplay
- Small text: Readable for secondary information
- Domino pips: Geometric shapes, not text characters

### Touch Target Requirements
- Minimum touch target size for reliable mobile interaction
- Adequate spacing between interactive elements
- Visual feedback on touch interactions
- No reliance on hover states

## Mobile Layout Structure

### Two-Panel Architecture with Bottom Navigation
```
┌─────────────────────────────────────────────────────────┐
│ Header: Phase Badge | AI Controls | Scores | Turn       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                Main Content Area                        │
│            (Game View OR Actions View)                  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│ Bottom Nav: [🎯 Play] [🎲 Actions] [🔧 Debug]          │
└─────────────────────────────────────────────────────────┘
```

### Smart Panel Switching Logic
- **Bidding/Trump Selection**: Auto-switch to Actions panel
- **Playing/Setup**: Auto-switch to Game panel  
- **Scoring**: Stay on Game panel for score display
- **Game End**: Stay on Game panel for final state
- **Manual Override**: Users can manually switch panels anytime

## Component Specifications

### Header Bar
- **Phase Badge**: Visually distinct current game phase with smooth transitions
  - Setup, Bidding, Trump Selection, Playing, Scoring, Game End
  - Animated phase changes with slide-in effects when phase updates
- **AI Controls**: Floating action button for play/pause
  - Play icon when AI stopped, pause icon when AI active
  - Pulsing animation when AI is actively playing
  - Single-tap toggle for instant AI execution
- **Score Display**: Team scores with animated progress indicators
  - Animated score updates with scaling effects when scores change
  - Progress bars showing marks toward game victory (7 marks)
  - Winner highlighting when one team is ahead
- **Turn Indicator**: Current player with clear identification
- **Advanced Animations**: Reactive animations triggered by state changes
  - Phase transitions animate only when phase actually changes
  - Score animations trigger only when scores update
  - Smooth progress bar width transitions

### Game Panel (PlayingArea)

#### Trump & Led Suit Display
- **Trump Badge**: Shows current trump selection with visual emphasis
- **Led Suit Badge**: Displays led suit during trick play when applicable
- **Visual States**: Pulsing animation during trump selection phase
- **Conditional Display**: Trump badge only shows when selected, led suit only during active tricks
- **Responsive Styling**: Mobile-optimized sizing with proper touch targets

#### Current Trick Display (NSEW Layout)
```
      [P1]
       N

[P0] W   E [P2]
       S
      [P3]
```
- **Primary Visual Focus**: Central game board that resembles traditional Texas 42 table layout
- **NSEW Player Positions**: Fixed compass-style layout with clear player identification
- **Clockwise Play Order**: Dominoes played and displayed in clockwise sequence (P0→P1→P2→P3)
- **Played Dominoes**: Immediate display when played with appropriately sized domino components
- **Waiting Positions**: Visual placeholders for unplayed positions in current trick
- **Fresh Play Animation**: Brief animation for newly played dominoes
- **Winner Highlighting**: Visual emphasis on winning domino in completed tricks
- **Game State Indicators**:
  - Bid status display (BID MADE/BID SET) with trick point totals during scoring
  - Trick completion state with visual highlighting
  - Score hand action indicator when hand scoring is available
- **Trick Counter**: Displays current trick progress (e.g., "Trick 3/7")
  - Positioned on same line as Trump and Led Suit badges
  - Clickable when completed tricks exist to expand/collapse history
  - Shows full domino layouts efficiently in compact format
  - Expandable history panel with slide transition showing all completed tricks
- **Interactive Elements**:
  - Click completed trick table to collect trick
  - "Complete Trick" button/action for smooth trick progression
  - "Score Hand" action with clear visual indication



#### Player Hand
- **Large Dominoes**: Touch-optimized size for easy interaction
- **Visual States**:
  - Playable: Enhanced visual state with pulsing animation and elevation
  - Unplayable: Reduced opacity, disabled interaction, grayscale filter
  - Counting: Point value badges on scoring dominoes
  - Winner: Special highlighting for winning dominoes in completed tricks
- **Direct Interaction**: Click domino to play with immediate visual feedback
- **Advanced Visual Effects**:
  - Hover elevation with subtle rotation for enhanced depth perception
  - Gradient backgrounds indicating different states
  - Realistic shadows and inset lighting effects
  - Smooth transitions between all visual states
- **Responsive Layout**: Horizontal scrolling hand with proper spacing
- **Animation**: Staggered entrance animations for hand reveal
- **Tooltips**: Contextual information about playability and suit following

#### Game Flow Actions
- **Trick Collection**: Click completed trick table to collect
- **Hand Scoring**: Click scoring area to score hand
- **Phase Progression**: Automatic progression through game state management
- **Score Hand Button**: Available during scoring phase for manual scoring
- **Redeal Functionality**: Automatic redeal when all players pass
  - Game logic handled by purely functional game engine (src/game)
  - UI integrates with engine but does not implement shuffle or redeal logic
  - Return to bidding phase with game state (from src/game), which will indicate bidding phase

### Actions Panel (ActionPanel)

#### Hand Display (Bidding/Trump Phases)
- **Compact Hand View**: Smaller dominoes for reference during decision-making
- **Responsive Grid**: Proper spacing and mobile-optimized layout
- **Staggered Animation**: Entrance animations with custom delay properties

#### Bidding Interface
- **Pass Button**: Prominent placement with distinct styling
- **Bid Buttons**: Grid layout with all available bids (30, 32, 34, 36, 38, 40, 42)
- **Visual Hierarchy**: Pass button separated and styled differently from bid options
- **Tooltips**: Contextual information on bid requirements
- **Touch Feedback**: Scale animations and proper touch targets
- **Error Handling**: Visual feedback for invalid actions
  - Shake animation for failed action attempts
  - Graceful error recovery without UI disruption
  - Clear visual indication when actions cannot be performed

#### Trump Selection
- **Suit Buttons**: Clear labels for all trump options (Blanks through Sixes, Doubles, No-Trump)
- **Responsive Layout**: Flexible column layout adapting to available actions
- **Visual Feedback**: Hover states and touch feedback with transitions
- **Action Validation**: Only valid trump selections available based on game state

#### Team Status (Always Visible)
- **Score Tracking**: Team scores displayed in ActionPanel with current marks
- **Bid Information**: Active bid and bidding player information
- **Progress Indicators**: Visual progress bars showing points needed to make bid
- **Responsive Layout**: Adapts to mobile screen constraints

### Bottom Navigation
- **Three Tabs**: Play (🎯), Actions (🎲), Debug (🔧) with emoji icons
- **Visual Indicator**: Animated sliding indicator with smooth transitions
- **Touch Feedback**: Scale animation on button press with proper timing
- **Smart Panel Auto-Switching**: Automatic panel switching based on game phase
  - Bidding/Trump Selection phases → Actions panel for decision-making
  - Playing/Setup/Scoring phases → Game panel for board interaction
  - Game End phase → Game panel for final state display
  - Works seamlessly with URL loading and natural game progression
- **Post-Action Transitions**: Automatic panel switching after key actions
  - After trump selection → Switch to Game panel with brief delay
  - After hand scoring → Switch to Actions panel for next bidding round
- **Manual Control**: Users can override auto-switching anytime via navigation buttons
- **Accessibility**: Proper touch targets and test identifiers

## Enhanced Mobile Features

### Touch Interactions
- **Swipe Gestures**: Swipe up from bottom to open debug panel
- **Direct Manipulation**: Touch dominoes to play, touch navigation elements
- **Visual Feedback**: Scale animations on button press, immediate response to touch
- **No Hover Dependencies**: All functionality accessible through touch
- **Keyboard Shortcuts**: Optional power-user shortcuts (Ctrl+Shift+D, Ctrl+Z, Escape)
- **Back Button**: Browser back button should undo last action

### Smart UX Behaviors
- **Contextual Interface**: UI adapts to current game phase automatically
- **Progressive Disclosure**: Advanced features hidden in debug panel
- **Minimal Cognitive Load**: Automatic navigation reduces decision fatigue
- **Error Prevention**: UI prevents invalid actions through state management

### Animation & Feedback
- **Panel Transitions**: Smooth fade transitions between Game and Actions panels
- **Touch Response**: Scale animations on button press with proper timing
- **State Changes**: Reactive animations for phase transitions and score updates
- **Fresh Content**: Staggered entrance animations for dominoes and new game elements
- **Navigation**: Smooth sliding indicator for bottom navigation
- **Debug Panel**: Fly-in transition for debug panel appearance
- **Trick History**: Slide transitions for expandable history panel
- **Advanced Effects**:
  - Pulsing animations for playable dominoes and active AI
  - Elevation and rotation effects on domino hover
  - Progress bar width animations for score tracking
  - Winner highlighting with scaling effects
- **Deterministic Transitions**: All transitions must be deterministic and state-driven
- **No setTimeout for Transitions**: setTimeout NEVER used for UI transitions - use Svelte transitions and reactive statements only

*Animation system uses reactive triggers to ensure animations occur only on actual state changes.*

## Debug Panel Overlay

### Panel Design
- **Modal Overlay**: Semi-transparent backdrop with centered panel
- **Touch-Optimized**: Responsive sizing with mobile breakpoints
- **Gesture Control**: Swipe up to open, tap backdrop to close
- **Keyboard Shortcuts**: Ctrl+Shift+D to toggle, Escape to close (hybrid approach)

### Four-Tab Structure

#### 1. Game State Tab
- **Tree View Toggle**: Expandable/collapsible state sections with mobile-optimized touch targets
  - Collapsible sections: Players, Tricks, Scores, Game Settings
  - Touch-friendly expand/collapse icons with adequate spacing
  - Nested indentation with clear visual hierarchy
  - Smooth expand/collapse animations for mobile experience
- **Diff Mode**: Highlight recent state changes with color coding
  - Yellow highlighting for recently changed values
  - Timestamp indicators for when changes occurred
  - Clear visual distinction between old and new values
- **Raw JSON View**: Complete GameState display with syntax highlighting
  - Monospace font optimized for mobile readability
  - Collapsible JSON tree structure with touch-friendly controls
  - Copy-to-clipboard functionality for debugging
- **Copy Functions**: Export current game state to clipboard
  - Individual section copying (players, tricks, scores)
  - Complete state export in JSON format
  - Mobile-optimized copy confirmation feedback
- **URL Sharing**: Generate shareable game state URLs
  - One-tap URL generation with automatic clipboard copy
  - Compressed URL format for efficient sharing

#### 2. History Tab
- **Action Timeline**: Complete history in reverse chronological order (newest first)
- **Sequential Numbering**: Actions numbered with proper indexing despite reverse order
- **Time Travel**: Click any action to restore that game state
  - Immediate state restoration without confirmation dialogs
  - Visual feedback showing selected historical point
  - Maintains action history integrity (no lost actions)
  - Works seamlessly with URL state management for shareable historical states
- **Action Display Format**: Shows action IDs and labels for clarity
  - Bidding: `bid-30`, `bid-1-marks`, `pass`
  - Trump: `trump-blanks` through `trump-sixes`, `trump-doubles`, `trump-no-trump`
  - Play: `play-5-3` (domino play actions)
  - Other: `complete-trick`, `score-hand`
- **Control Buttons**: Undo Last and Reset Game buttons in header
- **Navigation Shortcuts**: Touch-friendly previous/next action navigation
- **Empty State**: Clear message when no actions have been taken

#### 3. QuickPlay Tab
- **AI Controls**: Toggle continuous AI play with configurable players
- **Player Configuration**: Individual AI toggle for each player (P0-P3)
- **Speed Controls**: Instant execution for immediate AI decisions
- **Step Function**: Manual step-through for debugging AI decisions
- **Status Display**: Real-time AI state and activity indicators
- **Error Handling**: Automatic AI disable on decision errors
- **Integration**: Connected to quickplayStore for state management

#### 4. Historical State Tab
- **Event Sourcing**: Complete initial state and action history display
  - Shows complete game replay data including initial shuffle
  - Perfect for bug replication and game analysis
  - Includes all randomness and deterministic state progression
- **Tree View Toggle**: Toggle between tree view and raw JSON display
  - Mobile-optimized collapsible tree structure
  - Touch-friendly expand/collapse controls with adequate spacing
  - Clear visual hierarchy with proper indentation
  - Smooth animations for tree expansion/collapse
- **Initial State Display**: Complete starting game state with all player hands
  - Expandable sections for each player's initial hand
  - Game settings and configuration display
  - Shuffle seed and randomization information
- **Actions List**: All executed actions with their IDs and labels
  - Chronological action sequence with clear formatting
  - Action parameters and results clearly displayed
  - Touch-friendly action selection for detailed inspection
- **Copy Historical JSON**: Button to copy complete replay data to clipboard
  - One-tap export of complete event sourcing data
  - Mobile-optimized copy confirmation with visual feedback
  - Formatted JSON for easy debugging and sharing
- **Structured Display**: Initial state and actions clearly separated
  - Clear visual separation between initial state and action history
  - Collapsible sections for organized information display
  - Mobile-friendly layout with proper spacing and typography
- **Debug Support**: Full context for bug reproduction with detailed state information
  - Complete context for exact game replay
  - All necessary data for bug replication
  - Integration with URL state management for shareable debug sessions

## Technical Requirements

### Mobile Browser Compatibility
- **Standard Mobile Browsers**: Flawless operation across iOS Safari, Chrome Mobile, etc.
- **Responsive Design**: Adapts to various mobile screen sizes
- **Touch Optimization**: Proper touch event handling
- **Performance**: Smooth operation on mobile hardware
- **Safe Areas**: iPhone notch support

### UI Correctness by Construction
- **Pure Presentation**: UI state derived from game state through pure display functions only
- **No Game Logic**: UI components forbidden from implementing game rules or calculations
- **Error Detection**: Built-in validation prevents invalid states (validation logic in src/game)
- **State Management**: Centralized state with predictable updates (managed by src/stores)
- **Type Safety**: Strong typing prevents runtime errors
- **Separation of Concerns**: Clear boundary between presentation (UI) and logic (src/game + src/stores)

### Accessibility & UX
- **Touch Targets**: Meet mobile accessibility guidelines
- **Visual Hierarchy**: Clear information architecture
- **Consistent Styling**: Systematic approach enabling easy design changes
- **Error Prevention**: UI design prevents user errors
- **Screen Reader Support**: Semantic HTML and proper ARIA roles

## Implementation Notes

### State Management Integration
- **Pure UI Pattern**: UI components consume state from stores, never modify game state directly
- **Reactive Updates**: UI automatically reflects game state changes from src/stores
- **Action Dispatch**: UI dispatches user actions to stores, which handle all game logic via src/game
- **No UI State Logic**: UI components contain no game rules, calculations, or state transitions
- **Action Validation**: Only valid actions available at any time (computed by src/game)
- **History Tracking**: Complete action history for debugging and replay (managed by src/stores)
- **URL State Management**: **CRITICAL FEATURE** - Complete game state sharing and restoration
  - Compressed URL format for efficient sharing
  - Event sourcing with initial state + action replay
  - Automatic URL updates on every user action
  - URL loading on application mount for seamless game restoration
  - Implementation provided by src/game/core/url-compression.ts and src/stores/gameStore.ts
  - UI integrates with existing URL functionality but does not implement compression logic
  - Essential for game sharing, debugging, and session continuity
- **AI System**: `quickplayStore` manages AI player configuration and decision-making
  - Configurable AI players with individual toggle capability
  - Instant speed execution for immediate AI decisions
  - Error handling with automatic disable on decision failures
  - Integration with main game state for seamless AI/human player mixing

### Testing Requirements
- **Touch Interaction Testing**: Verify all touch-based interactions
- **Mobile Browser Testing**: Cross-browser compatibility validation
- **State Validation**: Ensure UI correctness across all game states
- **Performance Testing**: Smooth operation on mobile devices

This specification documents the complete mobile-first Texas 42 interface, prioritizing touch interactions and mobile browser compatibility while maintaining full game functionality through direct manipulation and smart interface behaviors.
