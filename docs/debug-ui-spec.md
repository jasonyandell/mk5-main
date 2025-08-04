# Debug UI Specification

## Overview

The Debug UI is a comprehensive development and debugging interface for the Texas 42 game that provides complete visibility into game state and transitions. It maintains a 1:1 relationship with the game state, displaying all state variables and values using the game's ubiquitous language, with additional UI-only features for debugging and development.

## Core Principles

1. **Complete State Visibility**: Display ALL game state without hiding any information
2. **1:1 State Mapping**: Every game state field must be visible and labeled exactly as it appears in the code
3. **Ubiquitous Language**: Use exact terminology from the game types and state (e.g., "phase", "currentPlayer", "teamMarks")
4. **Display-Only Logic**: The Debug UI contains only display and basic interaction logic - no game logic
5. **Real-time Updates**: Automatically reflect all state changes immediately

## State Display Requirements

### Core Game State (Always Visible)
The Debug UI must display the following state fields exactly as they exist in `GameState`:

- **phase**: Current game phase (setup, bidding, trump_selection, playing, scoring, game_end)
- **currentPlayer**: Player ID whose turn it is (0-3)
- **dealer**: Current dealer player ID
- **winningBidder**: Player who won the bid (-1 if none)
- **trump**: Current trump selection (type and suit if applicable)
- **currentSuit**: Led suit for current trick (-1 if no trick in progress)
- **currentBid**: The winning bid details
- **teamScores**: Points scored by each team in current hand [team0, team1]
- **teamMarks**: Total marks accumulated by each team [team0, team1]

### Bidding State
- **bids**: Complete history of all bids made
- Display format: Player ID + bid type + value (e.g., "P0: 31pts", "P1: Pass", "P2: 1m")
- Highlight current winning bid

### Trick State
- **tricks**: All completed tricks with:
  - Play sequence (player + domino)
  - Winner
  - Points value
  - Led suit
- **currentTrick**: In-progress trick plays
- Visual indicators for:
  - Winning play
  - Counting dominoes
  - Trump plays

### Player State
- **players**: For each player:
  - ID and name
  - Team assignment
  - Current hand (all dominoes visible)
  - Suit analysis when available
- No hidden information - all hands visible

### Scores and Progress
- Team scores for current hand
- Team marks (game score)
- Progress indicators (e.g., "Trick 3/7", "Actions: 15")

## UI-Only Features

### 1. Action History & Undo
- **Purpose**: Track all state transitions for debugging and replay
- **Display**:
  - Chronological list of all actions taken
  - Action type, player, and description
  - Event numbers for time travel
- **Interaction**:
  - Click any event to restore game to that state
  - Undo functionality to revert last action
  - Clear visual indication of current position

### 2. State Validation & Error Display
- **Purpose**: Identify invalid states or rule violations
- **Display**:
  - Validation error messages with full details
  - Highlighted invalid state fields
  - Copy-friendly error text for bug reports

### 3. Available Actions Panel
- **Purpose**: Show all legal actions for current state
- **Display**:
  - Categorized by type (bid, trump, play, etc.)
  - Color-coded by action category
  - Test IDs for E2E testing
- **Interaction**:
  - Click to execute action
  - Hover for detailed description

### 4. Bug Reporting
- **Purpose**: Capture complete state for bug reports
- **Features**:
  - Snapshot current state
  - Include action history
  - Generate shareable bug report
  - Copy state to clipboard

### 5. Test Generation
- **Purpose**: Generate test cases from current state
- **Features**:
  - Export state as test fixture
  - Generate test scenario code
  - Include assertions for current state

### 6. Quick Play / Simulation
- **Purpose**: Rapidly advance game state for testing
- **Features**:
  - Auto-play with configurable AI
  - Skip to specific game phases
  - Batch action execution

## Component Structure

### Main Components

1. **DebugGameState**: Core state display panel
   - Phase, players, trump, scores
   - Bidding history
   - Current game status

2. **DebugPreviousTricks**: Trick history visualization
   - Compact trick display
   - Winner highlighting
   - Point tracking

3. **DebugPlayerHands**: All player hands display
   - Domino organization
   - Suit analysis
   - Playable domino highlighting

4. **DebugActions**: Available actions panel
   - Action buttons with test IDs
   - Action categorization
   - Click to execute

5. **DebugReplay**: History and time travel
   - Action log
   - State restoration
   - Undo functionality

6. **DebugJsonView**: Raw state viewer
   - Full JSON state
   - Syntax highlighting
   - Copy functionality

## Test Integration

### Test IDs
All interactive elements must have appropriate `data-testid` attributes:
- Action buttons: `bid-P0-30`, `set-trump-5s`, `play-domino-P2-{id}`
- State displays: `phase`, `current-player`, `team-0-score`
- Panels: `debug-panel`, `action-panel`, `history-panel`

### E2E Helper Integration
- All actions must be accessible via the playwrightHelper
- Consistent selectors for state verification
- Reliable element targeting for automated tests

## Visual Design

### Layout
- Overlay panel activated by debug button
- Organized sections for different state aspects
- Responsive grid layout
- Scrollable sub-panels for long content

### Color Coding
- Phase colors: bidding (blue), playing (green), scoring (purple)
- Action categories: bids (blue), trump (orange), plays (green)
- State indicators: current (yellow), winner (green), error (red)

### Typography
- Monospace for IDs and technical values
- Clear hierarchy with consistent sizing
- High contrast for readability

## Performance Considerations

- Efficient re-rendering on state changes
- Virtualized lists for long histories
- Debounced updates for rapid state changes
- Minimal computational overhead

## Future Enhancements

- State diff visualization
- Performance profiling integration
- Multiplayer session debugging
- State persistence/loading
- Advanced filtering and search
- Custom state manipulation tools