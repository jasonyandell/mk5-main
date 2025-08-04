# Hand Outcome Detection Implementation Plan

## Goal
Detect when a hand's outcome is mathematically certain and end it early, improving game flow.

## Core Tasks

### 1. Detection Logic (`src/game/core/handOutcome.ts`)

- [ ] Detect when bidding team has secured their bid
- [ ] Detect when bidding team cannot possibly make their bid
- [ ] Calculate remaining points in unplayed dominoes
- [ ] Handle all bid types: points (30-41), marks, nello

### 2. Engine Integration

- [ ] Check for determined outcome after each trick completion
- [ ] Auto-transition to scoring phase when outcome is certain
- [ ] No state changes needed - just phase transition

### 3. Debug UI

- [ ] Show when hand was decided (trick #)
- [ ] Display reason hand ended early

### 4. Testing

- [ ] Unit tests for detection logic with various game states
- [ ] E2E tests for early hand termination scenarios

## Detection Rules

**Points Bids (30-41)**
- End when: bidding team score ≥ bid value
- End when: opponent score > (42 - bid value)

**Marks Bids (1+ marks)**
- End when: opponent team scores any points

**Nello**
- End when: bidding team wins any trick

## Success Criteria

1. Hands end as soon as outcome is certain
2. No false positives
3. Clear indication in Debug UI when/why hand ended
4. All tests pass