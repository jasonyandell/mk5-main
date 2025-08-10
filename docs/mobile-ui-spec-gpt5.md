## Mobile UI Spec — Essentials

### Principles & Scope
- Mobile-only; no desktop layouts
- UI renders state from src/game and src/stores; no rules/scoring in UI
- Direct manipulation: play happens on the board (tap table to collect trick)
- Deterministic, state-driven transitions; never use setTimeout

### Integration (src/game · src/stores)
- Valid transitions via src/stores/availableActions (wraps src/game/core/gameEngine.getNextStates)
- Read state from src/stores/gameStore: gameState, initialState, actionHistory; derived: currentPlayer, gamePhase, biddingInfo, teamInfo, trickInfo
- URL share/restore: src/game/core/url-compression.ts (compress/expand/encode/decode) wired in gameStore.loadFromURL/updateURLWithState
- Display helpers (src/game): rules.getCurrentSuit for led suit labels; dominoes.getDominoPoints for counting badges

### Layout & Navigation
- Header, Main, Bottom tabs: Play 🎯, Actions 🎲, Debug 🔧
- Auto-switch (user can override):
  - Setup/Playing/Scoring/Game End → Play panel
  - Bidding/Trump → Actions panel
  - After Trump chosen → Play; after Hand scored → Actions

### Header
- Phase badge (Setup, Bidding, Trump, Playing, Scoring, Game End)
- AI quickplay toggle (play/pause)
- Team scores with marks to 7; leading team highlighted
- Turn indicator

### Play Panel (PlayingArea)
- Badges: Trump (GameState.trump once chosen), Led suit (GameState.currentSuit; label via src/game/core/rules.getCurrentSuit(state))
- Board: NSEW layout; clockwise P0→P1→P2→P3; show plays immediately; placeholder for waiting seats
- Trick counter ("Trick X/7"); X = tricks.length + 1; 7 = GAME_CONSTANTS.TRICKS_PER_HAND; tap to expand/collapse "Previous Tricks"
- Previous Tricks (Expandable)
  - Collapsed by default on mobile; toggled by tapping the Trick counter
  - Source: GameState.tricks (completed tricks only); display newest first
  - Each trick renders ALL four dominoes from Trick.plays using the same seat mapping as the main board (NSEW mini-board)
    - Maintain clockwise play order; annotate with seat/initials if needed for clarity
    - Highlight the winning seat from Trick.winner
    - Show a small points badge from Trick.points
    - Show a led-suit badge from Trick.ledSuit (map 0–6 to Blanks..Sixes; 7 = Doubles)
- Interactions
  - Collect trick when availableActions contains id "complete-trick"
  - Show "Score Hand" when availableActions contains id "score-hand"
- Player hand: large touch dominoes, horizontal scroll
  - States
    - playable if availableActions contains id `play-${domino.id}`, disabled otherwise
    - counting badge if src/game/core/dominoes.getDominoPoints(domino) > 0
    - winner shown in history via Trick.winner
  - Tap to play with immediate feedback
- Mobile-only: no hover-only affordances

### Actions Panel
- Compact hand preview
- Bidding: render from availableActions (pass and bid-*); Points 30–41, Marks 1–4 per rules; only valid enabled; invalid attempts get clear feedback
- Trump: render from availableActions (trump-* actions: Blanks–Sixes, Doubles, No‑Trump)
- Team status visible: scores, active bid, bidding player, progress toward making bid; optionally show "hand decided" via src/game/core/handOutcome.checkHandOutcome(state)

### Debug Overlay
- Modal overlay; swipe up to open; tap backdrop to close
- Tabs
  1) Game State
     - Toggle tree view (Players, Tricks, Scores, Settings)
     - Raw JSON; copy section or full state; shareable URL
  2) History
     - Reverse-chronological action list; stable indexing
     - Time travel by tapping an action
     - Header: Undo Last, Reset Game
  3) QuickPlay
     - Toggle continuous AI; per-player AI (P0–P3)
     - Instant execution; Step
     - Auto-disable on error; visible status
  4) Historical State
     - Initial state + full action list
     - Optional tree view or raw JSON
     - Copy Historical JSON button

### Visual System & Animation
- Roles/colors: playable, inactive, trump, counting badges, winner, phase, team identity
- Mobile-legible typography; domino pips are graphics
- Touch targets meet mobile guidelines; clear pressed feedback
- Centralize styles for easy theme changes
- Animation: minimal and state-driven (phase changes, score updates, tab selection, new plays)

### Technical Notes
- URL state: compressed, auto-updated on every action; load on mount
- AI via quickplayStore; UI only configures/reflects
- Type-safe updates via stores
- Mobile browser compatibility: iOS Safari, Chrome; responsive and safe-area aware

### Accessibility & UX
- Adequate touch sizes/spacing; clear hierarchy
- Error prevention: only valid actions enabled; invalid attempts get non-disruptive feedback

### Developer Testing
- Touch interactions: play domino, collect trick, bid, choose trump, navigation
- Cross-browser mobile checks
- URL share/restore
- History time travel and Debug overlay
- Performance on typical mobile devices

### Out of Scope
- Desktop layouts
- Keyboard shortcuts
- Any game logic in UI