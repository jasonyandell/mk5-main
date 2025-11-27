---
name: texas-42
description: Texas 42 dominoes game development - activates for architecture questions, layer system implementation, testing patterns, URL debugging workflow, game rule clarifications, and codebase navigation. Use when working on game logic, special contracts (nello/splash/plunge/sevens), or debugging game states.
---

# Texas 42 Development Guide

## North Star

This project is a "crystal palace in the sky" - elegance, simplicity, and correctness above all. Every line of code is a liability. Prefer spending extra time to make things perfect over expedient shortcuts. We're on the 8th major overhaul; 100 more would just mean we had fun.

## Architecture Quick Reference

### Core Pattern
```
STATE → ACTION → NEW STATE
```

**Event Sourcing**: `state = replayActions(config, history)` - state is derived, actions are truth.

### The Stack
```
UI (Svelte 5)
    ↓
Svelte Stores (gameStore)
    ↓
GameClient (fire-and-forget)
    ↓ Socket interface
Room (★ COMPOSITION POINT ★)
    ↓
Pure Helpers (kernel.ts)
    ↓
Unified Layer System
    ↓
Core Engine (pure utilities)
```

### Key Abstractions

| Abstraction | Purpose |
|-------------|---------|
| **GameRules** | 14 execution methods (WHO/WHEN/HOW/VALIDATION/SCORING/LIFECYCLE) |
| **Layer** | Unified execution rules + action generation |
| **Capability** | Permission token (`act-as-player`, `observe-hands`) |
| **ExecutionContext** | Bundles layers + rules + getValidActions |

### Composition Pattern

**Single Composition Point**: Only `Room` and `HeadlessRoom` compose ExecutionContext.

```typescript
const layers = [baseLayer, ...getEnabledLayers(config)];
const rules = composeRules(layers);
const getValidActionsComposed = composeLayerActions(layers);
this.ctx = Object.freeze({ layers, rules, getValidActions: getValidActionsComposed });
```

**Parametric Polymorphism**: Executors call `rules.method()`, never `if (mode === 'nello')`.

## Development Commands

```bash
npm run test:all      # REQUIRED before closing beads issues
npm run typecheck     # Run often
npm test              # Unit tests (Vitest)
npm run test:e2e      # Playwright E2E
npm run dev           # Dev server
```

## Critical: URL Debugging Workflow

When user provides a localhost URL with bug report:

```bash
# 1. Generate test automatically
node scripts/replay-from-url.js "<url>" --generate-test

# 2. Debug with focused options
--action-range 87 92   # Show only actions 87-92
--hand 4               # Focus on just hand 4
--show-tricks          # Display trick winners and points
--compact              # One line per action with score changes
--stop-at N            # Stop replay at action N
```

## Task Tracking (Beads)

- **Check ready issues before starting**: Use beads MCP tools
- **File issues for discovered bugs** as you go
- **Run `npm run test:all`** before closing any issue

## Key Anti-Patterns

| DON'T | DO |
|-------|-----|
| `if (mode === 'nello')` in executors | Add rule method, delegate to `rules.method()` |
| Mutate state directly | Create new objects via spread operator |
| Check player identity for permissions | Use Capability tokens |
| `setTimeout()` in Playwright tests | Use proper waits |
| Skip failing tests | Fix them - greenfield project |
| Add game logic to Room | Put in pure helpers or Layers |
| Client-side validation | Trust server's validActions completely |

## File Map

**Core Engine** (pure utilities):
- `src/game/core/actions.ts` - Action executors (thread rules through)
- `src/game/core/state.ts` - State creation and transitions
- `src/game/types.ts` - Core types

**Layer System**:
- `src/game/layers/types.ts` - GameRules interface, Layer type
- `src/game/layers/compose.ts` - composeRules() reduce pattern
- `src/game/layers/base.ts` - Standard Texas 42
- `src/game/layers/{nello,splash,plunge,sevens,tournament,oneHand,hints,speed}.ts` - Special contracts

**Multiplayer**:
- `src/multiplayer/authorization.ts` - authorizeAndExecute()
- `src/multiplayer/capabilities.ts` - Capability builders

**Server**:
- `src/server/Room.ts` - Production orchestrator (composition point)
- `src/server/HeadlessRoom.ts` - Tools/scripts API (composition point)
- `src/kernel/kernel.ts` - Pure helpers

**Client**:
- `src/multiplayer/GameClient.ts` - ~43 lines, fire-and-forget
- `src/stores/gameStore.ts` - Svelte facade

## Testing Patterns

| Test Type | Tool | When |
|-----------|------|------|
| **Unit** | `createTestContext()` | Layer composition, pure functions |
| **Integration** | `HeadlessRoom` | Full game flows |
| **E2E** | Playwright + `PlaywrightGameHelper` | UI interactions |

**Key Rule**: Tests use same composition paths as production (HeadlessRoom → Room → ExecutionContext).

**Note**: Vitest uses `environment: 'node'` (not jsdom) - unit tests are pure logic with no DOM dependencies.

## Supporting Files

For deep dives, see companion files in this skill directory:
- `architecture.md` - Layer system, GameRules, how to extend
- `workflows.md` - URL replay, testing, debugging tips
- `game-rules.md` - Texas 42 rules for feature implementation
