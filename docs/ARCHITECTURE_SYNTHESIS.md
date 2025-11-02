# Texas 42 Architecture: Core Components Synthesis

**Purpose**: Distilled overview of the essential architectural components and their relationships.

---

## Fundamental Pattern

```
STATE → ACTION → NEW STATE
```

The architecture centers on immutable state transformation. The system:
1. **Generates** valid actions from current state
2. **Executes** actions deterministically
3. **Filters** results based on observer permissions

---

## Two-Level Composition System

The architecture addresses varying game rules through orthogonal composition:

```
LAYERS (execution rules) × VARIANTS (action transformation) = Game Configuration
```

- **Layers**: Override how the game executes (13 GameRules methods)
- **Variants**: Transform what actions are available (filter/annotate/script/replace)
- Compose independently without modifying core engine

---

## Three-Tier Authority Structure

```
KERNEL → SERVER → TRANSPORT
(logic)  (orchestration)  (routing)
```

Each tier has distinct responsibilities:
- **GameKernel**: Game logic, state storage, authorization
- **GameServer**: Lifecycle management, AI coordination, subscriptions
- **Transport**: Message routing (in-process, worker, edge deployment)

---

## Client-Server Architecture

```
SERVER (authoritative) ← PROTOCOL → CLIENT (delegating)
```

- Server validates all game logic
- Clients trust server's validActions list
- AI clients use identical protocol to human clients

---

## Core Abstractions

### Essential Types
1. **GameState**: Immutable representation of game position
2. **GameAction**: Atomic unit of state change
3. **PlayerSession**: Links identity to seat with capabilities
4. **ExecutionContext**: Bundles composed rules, layers, and variants
5. **ViewProjection**: Transforms state for UI consumption

### Capability Model
Permissions granted through tokens rather than identity checks:
- `act-as-player`: Execute actions for a seat
- `observe-hand`: View specific player hands
- `see-hints`: Access AI recommendations

---

## Parametric Polymorphism

Executors delegate behavior to injected rules rather than inspecting state:

```typescript
// Instead of:
if (state.trump === 'nello') { /* special logic */ }

// The system uses:
rules.isTrickComplete(state)
```

This pattern enables new game modes without modifying executor code.

---

## Event Sourcing

State derives from action history:

```
state = replayActions(config, history)
```

- Actions serve as source of truth
- State is computed, not stored
- Enables replay, debugging, and game sharing

---

## AI Integration

AI players operate as standard clients:
- Receive state updates through protocol
- Analyze positions using pure functions
- Submit actions via standard messages
- No privileged state access

---

## Functional Programming Principles

State transformations are:
- **Deterministic**: Identical inputs produce identical outputs
- **Pure**: No side effects or mutations
- **Composable**: Functions combine predictably

---

## Architectural Invariants

Critical rules that maintain system integrity:

1. **Pure state storage**: Filter on-demand, not at rest
2. **Server authority**: Clients accept server decisions
3. **Capability-based access**: Use tokens, not identity checks
4. **Single composition point**: GameKernel constructor only
5. **Zero coupling**: Core engine unaware of multiplayer/transport
6. **Parametric execution**: Rules injected, not inspected
7. **Event sourcing**: Actions determine state
8. **Clean separation**: Each component has one responsibility

---

## System Mental Model

The architecture can be understood as:
- **State machine**: Games progress through defined positions
- **Composition system**: Behaviors combine like stackable filters
- **Pure transformation**: State changes through function application
- **Trust hierarchy**: Server validates, clients execute

---

## Architectural Benefits

### Correctness
- Type system prevents invalid states
- Pure functions ensure predictability
- Immutability eliminates mutation bugs

### Extensibility
- New features via composition, not modification
- Layers and variants combine orthogonally
- Core engine remains unchanged

### Testability
- Pure functions simplify unit testing
- Deterministic replay enables regression testing
- Event sourcing provides complete audit trail

### Deployment Flexibility
- GameKernel runs in browser, worker, or edge
- Transport abstraction enables various deployments
- Clean separation allows independent scaling

---

## Key Design Decisions

### Composition Over Configuration
Behavior emerges from function composition rather than configuration flags or conditional logic.

### Immutability by Default
State objects are never modified. New states are created through transformation.

### Explicit Dependencies
All dependencies passed as parameters. No hidden state or global variables.

### Single Source of Truth
Each piece of information has one authoritative location. Everything else derives from it.

---

## Implementation Overview

The system achieves complexity through composition of simple components:
- No inheritance hierarchies
- No monolithic classes
- No implicit behavior
- Clear boundaries between concerns

Each component focuses on a single responsibility, with behavior emerging from their composition rather than from complex individual components.

---

**Document Version**: 1.0
**Last Updated**: October 2025