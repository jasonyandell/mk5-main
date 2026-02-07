# Texas 42 Playing Area UI Inventory

Comprehensive inventory of all visual information displayed in the playing area.

## Header (Top Bar)

| Element | Description |
|---------|-------------|
| **Phase Badge** | Color-coded indicator (Setup, Bidding, Trump, Playing, Scoring, Game End, Complete) |
| **Team Scores** | "US [score]" (primary) • "THEM [score]" (secondary) |
| **Menu Button** | Three vertical dots, opens dropdown |

### Menu Items

- View As dropdown (select perspective: P0-P3)
- New Game (dice icon)
- Play One Hand (hand icon)
- Colors (paintbrush icon)
- Current Player indicator ("Turn: P[X]")
- Settings (cog icon)

---

## Game Info Bar (Status Line)

**During Playing Phase:**
- Current player: "P[X] turn"
- Trump suit: "[suit] trump"
- Bid info: "P[X] bid [value]"
- Led suit: "[suit] led"

**Other Phases:** Phase-specific messages (current bid, trump selection status, etc.)

---

## Central Table Area

### During Playing

- Four player positions (top/bottom/left/right)
- Played dominoes with pip display
- Player labels: "P[0-3]"
- Winner banner: "Winner!" with sparkle icon (animated)
- Waiting indicator: Dashed spinner with "P[X]"
- AI thinking badge: "P[X] is thinking..." with CPU icon

### During Scoring

- Bid winner: "P[X] ([team]) bid [amount]"
- Result badge: ✓ SUCCESS or ✗ FAILURE
- Team points comparison: US vs THEM

### Proceed Indicator

"Click to proceed" with hand icon (when available)

---

## Trick History Panel (Expandable Drawer)

- Trick counter: "Tricks ([X]/7)"
- Current hand points: "US: [X] | THEM: [Y]"
- Completed tricks table (1-7) with winner and points
- Current trick in progress (if applicable)
- Vertical tab: "Tricks [X]/7"

---

## Player Hand (Bottom Section)

- Label: "Your Hand" or "Selected Hand"
- Domino grid with:
  - **Playable dominoes**: Green border, lifted, scaled 105%
  - **Points badges**: 5 or 10 (warning color, top-right)
  - **Tooltips**: Domino value on hover
- Empty state: "No dominoes" message

---

## Action Panel (Bidding/Trump Selection)

### During Bidding

- Compact bid status (P0-P3 with bids/passes)
- High bid indicator
- Dealer badge
- Your hand preview
- **Action buttons:**
  - Pass (error color)
  - Redeal (warning color, if applicable)
  - Bid amounts (30, 31, 32... up to marks)

### During Trump Selection

- Winning bid card: "P[X] - [value]"
- Trump suit buttons (Spades, Hearts, Diamonds, Clubs, No Trump, etc.)

---

## Settings Panel (Modal)

### Tabs

- State tab (code icon)
- Theme tab (paintbrush icon)

### State Tab

- Tree View toggle
- Copy State button
- Copy/Share URL button
- State viewer (JSON)

### Theme Tab

- 20 theme options in grid (color previews + names)
- Reset Game button

---

## One Hand Complete Modal

- Title: "Victory!" or "Defeat"
- Score display: "US [X] • THEM [Y]"
- Attempt counter (if retried)
- Retry Same Hand button
- New Hand button
- Share Challenge button
- Exit One Hand Mode button

---

## Domino Visual States

| State | Appearance |
|-------|------------|
| Playable | Green 4px border, lifted, scaled, ring effect |
| Winner | Primary 4px border |
| Default | Base-300 2px border |
| Points domino | Badge showing 5 or 10 |

---

## Animations

| Animation | Usage |
|-----------|-------|
| `animate-phase-in` | Phase badge transitions |
| `animate-hand-slide` | Hand entrance (staggered) |
| `animate-drop-in` | Domino played to table |
| `animate-winner-glow` | Winner domino pulse |
| `animate-tap-bounce` | Click to proceed indicator |
| `animate-pulse` | AI thinking messages |

---

## Data Test IDs

Key testing markers:
- `playing-area` - Main play area
- `trick-area` - Trick table section
- `app-header` - Header
- `action-panel` - Action panel
- `settings-panel` - Settings modal
- `domino-[high]-[low]` - Individual dominoes
- `trump-display` - Trump display
