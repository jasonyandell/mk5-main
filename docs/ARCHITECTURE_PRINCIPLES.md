# Texas 42 Architecture: Principles & Philosophy

**Purpose**: High-level architectural principles, design philosophy, and mental models for the Texas 42 codebase. Focuses on WHY we designed the system this way and WHAT patterns we use.

**Audience**: Architects, senior developers, product managers, and anyone seeking to understand the design philosophy.

**Related Documentation**:
- **Vision**: [VISION.md](VISION.md) - Strategic direction and north star outcomes
- **Orientation**: [ORIENTATION.md](ORIENTATION.md) - Developer onboarding and navigation
- **Reference**: [CONCEPTS.md](CONCEPTS.md) - Complete implementation reference

---

## Table of Contents

1. [Fundamental Pattern](#fundamental-pattern)
2. [Core Architecture Principles](#core-architecture-principles)
3. [Two-Level Composition System](#two-level-composition-system)
4. [Three-Tier Authority Structure](#three-tier-authority-structure)
5. [Mental Models](#mental-models)
6. [Design Philosophy](#design-philosophy)
7. [Architectural Invariants](#architectural-invariants)
8. [Common Patterns](#common-patterns)
9. [Concept Relationships](#concept-relationships)
10. [Architectural Benefits](#architectural-benefits)
11. [Key Design Decisions](#key-design-decisions)
12. [Glossary of Terms](#glossary-of-terms)

---

## Fundamental Pattern

```
STATE → ACTION → NEW STATE
```

The architecture centers on immutable state transformation. The system:
1. **Generates** valid actions from current state
2. **Executes** actions deterministically
3. **Filters** results based on observer permissions

Everything in the system exists to support this fundamental transformation.

---

## Core Architecture Principles

### Event Sourcing
The fundamental architectural pattern where game state is derived from a sequence of actions:

```
state = replayActions(initialConfig, actionHistory)
```

State is never mutated directly; instead, actions are applied to produce new states. This enables:
- Perfect replay from any point in history
- Time-travel debugging
- Game sharing via compressed URLs
- Complete audit trail
- Deterministic testing

**Key Insight**: Actions are the source of truth, state is computed.

---

### Pure Functions & Immutability
All game logic is implemented as pure functions with no side effects. Given the same inputs, functions always produce the same outputs. State objects are never modified; new states are created through transformation.

**Benefits**:
- Predictability: Same inputs always produce same outputs
- Testability: Functions can be tested in isolation
- Composability: Pure functions combine reliably
- Debugging: No hidden state changes to track
- Parallelization: Safe to execute concurrently

**Application**: Executors, state transformations, filtering, composition, utilities.

---

### Two-Level Composition System
A unique architectural innovation solving the challenge of special contracts that need to modify both execution rules AND available actions.

```
RULESETS (execution rules) × ACTION_TRANSFORMERS (action transformation) = Game Configuration
```

#### Level 1: RuleSets → Execution Rules
**Purpose**: Define HOW the game executes (who acts, when tricks complete, how winners determined)

**Mechanism**: Override specific methods in the 13-method `GameRules` interface
- WHO: getTrumpSelector, getFirstLeader, getNextPlayer
- WHEN: isTrickComplete, checkHandOutcome
- HOW: getLedSuit, calculateTrickWinner
- VALIDATION: isValidPlay, getValidPlays, isValidBid
- SCORING: getBidComparisonValue, isValidTrump, calculateScore

**Composition**: RuleSets override only what differs from base, compose via reduce pattern

**Example**: Nello partner sits out → override `isTrickComplete` to return true at 3 plays instead of 4.

#### Level 2: ActionTransformers → Action Transformation
**Purpose**: Transform WHAT actions are possible (filter, annotate, script, replace)

**Operations**:
- **Filter**: Remove actions (tournament removes special bids)
- **Annotate**: Add metadata (hints, autoExecute flags)
- **Script**: Inject actions (oneHand scripts bidding)
- **Replace**: Swap action types (oneHand replaces score-hand with end-game)

**Composition**: ActionTransformers wrap the state machine via function composition

**Key Insight**: RuleSets and ActionTransformers compose independently without modifying core engine. Executors have ZERO conditional logic.

---

### Parametric Polymorphism
Executors never inspect state to determine behavior. Instead, they accept rule functions as parameters and delegate all decisions to them.

**Pattern**:
```typescript
// Instead of:
if (state.trump === 'nello') { /* special logic */ }

// The system uses:
rules.isTrickComplete(state)
```

This allows the same executor code to work for all game variants without conditional logic, eliminating complexity and enabling unlimited extensibility.

**Key Benefit**: Add new game modes without modifying executors.

---

### Separation of Concerns
Each architectural layer has a single, well-defined responsibility:

```
KERNEL → SERVER → TRANSPORT
(logic)  (orchestration)  (routing)
```

**Boundaries**:
- **Engine**: Pure game logic, zero coupling
- **Kernel**: Game logic + multiplayer (authorization, filtering)
- **Server**: Orchestration + lifecycle management
- **Transport**: Message routing (in-process, worker, edge)
- **UI**: Rendering + interaction

**Benefit**: Changes in one layer don't cascade to others. Each component focuses on one thing.

---

## Three-Tier Authority Structure

```
KERNEL (authority) → SERVER (orchestrator) → TRANSPORT (routing)
```

Each tier has distinct responsibilities:

### GameKernel - Pure Game Authority
- Stores unfiltered state
- Composes layers/variants (single composition point)
- Authorizes actions via capability-based filtering
- Filters views per-request based on observer permissions
- Zero knowledge of transport, networking, or AI spawning

### GameServer - Orchestrator
- Creates and owns GameKernel
- Creates and owns AIManager
- Routes protocol messages to handlers
- Manages client subscriptions
- Broadcasts state updates via Transport

### Transport - Message Router
- Abstracts message delivery mechanism
- Enables multiple implementations (in-process, Worker, edge)
- GameServer is transport-agnostic

**Key Insight**: Clean separation enables deployment flexibility—same kernel runs in browser, Worker, or edge.

---

## Mental Models

Understanding the architecture requires thinking about it in multiple ways:

### The Game as a State Machine
Every game position has defined transitions to next positions. Actions are edges, states are nodes. AI explores this graph to make decisions. Games flow deterministically through well-defined positions.

### RuleSets as Lenses
Each ruleset provides a different lens through which to view game rules. Stack lenses to create new game modes. Nello adds a "3-player trick" lens, Plunge adds a "partner leads" lens, Sevens adds a "distance-from-7" lens.

### ActionTransformers as Decorators
ActionTransformers wrap and transform the base game, adding features without modifying core logic. Tournament filters, Speed annotates with auto-execute, OneHand scripts bidding, Hints adds recommendations—each wraps the state machine in a decorator pattern.

### Capabilities as Keys
Each capability unlocks specific functionality. Collect keys to gain more power:
- `act-as-player` → Execute actions for a seat
- `observe-all-hands` → Full visibility
- `see-hints` → AI recommendations
- `see-ai-intent` → Strategy metadata

### The Kernel as a Pure Function
Given state and action, always produces same new state. No hidden state or side effects:
```
newState = kernel(oldState, action)
```

### Trust Hierarchy
Server validates, clients trust completely. Clear security boundary: server is authoritative for all game logic, clients only handle UI and send action requests.

---

## Design Philosophy

These principles guide all architectural decisions:

### Simplicity Through Composition
Complex behavior emerges from composing simple, pure functions. No monolithic classes or deep inheritance hierarchies. Each component is understandable in isolation.

**Application**: RuleSet composition, ActionTransformer wrapping, rule delegation.

### Correct by Construction
Use the type system to prevent errors at compile time. Make illegal states unrepresentable.

**Examples**:
- GameAction union type (can't create invalid actions)
- GamePhase enum (can't have invalid phase)
- Capability types (type-safe permissions)

**Benefit**: Compile errors catch bugs before runtime.

### Explicit Over Implicit
All behavior is explicitly defined through composition. No hidden magic or implicit conventions. What you see is what happens.

**Application**: No global state, no hidden dependencies, all parameters passed explicitly.

### Immutability as Default
State is never mutated, only transformed. Enables reasoning, debugging, and time-travel. New states are created through transformation via spread operators.

**Benefit**: Can't have "action at a distance" bugs where distant code mutates shared state.

### Single Source of Truth
Each piece of information has one authoritative location. Everything else derives from it.

**Examples**:
- GameKernel stores unfiltered state (clients get filtered copies)
- Action history is source of truth (state is computed)
- ExecutionContext created once (used everywhere)

### Trust Through Verification
Server validates everything, clients trust completely. Clear security boundary enables simple client code and guaranteed consistency.

**Pattern**: Client sends action → server validates → server executes → server broadcasts → client displays. No client-side validation.

---

## Architectural Invariants

**CRITICAL**: These principles must never be violated. Violations constitute architectural regressions.

### 1. Pure State Storage
GameKernel stores unfiltered GameState. Filtering happens per-request, never at rest. Single source of truth principle.

**Why**: Enables per-observer filtering, ensures consistency, prevents state divergence.

### 2. Server Authority
Clients trust server completely for game logic. Clients never revalidate, refilter, or recompute server decisions.

**Why**: Ensures consistency, simplifies client code, prevents cheating.

### 3. Capability-Based Access
All permissions via capability tokens. Never use identity checks (`playerId === X`) or role comparisons.

**Why**: Transparent, composable security model. Capabilities are data, not magic.

### 4. Single Composition Point
RuleSets and ActionTransformers compose only in GameKernel constructor. ExecutionContext created once, used everywhere.

**Why**: Ensures consistent behavior, prevents composition drift, enables parametric polymorphism.

### 5. Zero Coupling
Core engine has no awareness of multiplayer, networking, or transport. RuleSets and ActionTransformers don't reference multiplayer concepts.

**Why**: Clean architectural boundaries, enables reuse, simplifies testing.

### 6. Parametric Execution
Executors delegate to injected rules, never inspect state for behavior. No conditional logic based on game mode checks.

**Why**: Enables unlimited extensibility, eliminates conditional complexity.

### 7. Event Sourcing Foundation
State must be derivable from action replay: `state = replayActions(config, history)`. Actions are immutable and append-only.

**Why**: Perfect reproducibility, enables sharing, debugging, and testing.

### 8. Clean Separation
Each component has single responsibility. GameServer orchestrates, GameKernel executes, Transport routes. No responsibility bleeding.

**Why**: Changes don't cascade, enables independent evolution, clear ownership.

---

## Common Patterns

### Filter-Map-Reduce
Used throughout for data transformation:
1. **Filter**: Select valid items from collection
2. **Map**: Transform items to new form
3. **Reduce**: Combine into single result

**Application**: RuleSet composition, action filtering, score calculation.

### Delegation Over Inspection
Functions accept behavior as parameters rather than inspecting data to determine behavior.

**Example**: Executors call `rules.method()` instead of checking `state.type`.

### Composition Over Configuration
Build behavior by composing functions, not by setting flags or options.

**Example**: RuleSets override methods (not `enableNello: true` flag with conditionals).

### Fail-Fast Validation
Validate early and explicitly. Make errors impossible through types where feasible.

**Application**: Type guards, discriminated unions, compile-time checks.

### Single Source of Truth
One authoritative location for each piece of state. Everything else derives from it.

**Application**: GameKernel state, action history, ExecutionContext.

---

## Concept Relationships

### Data Flow
```
User input → Client → Transport → Server → GameKernel →
Authorization → Execution → State Update → Filtering →
Transport → Client → UI Update
```

### Composition Flow
```
RuleSets define rules → Rules compose via reduce →
Base state machine uses rules → ActionTransformers wrap state machine →
ExecutionContext bundles all → Executors use context
```

### Authority Hierarchy
```
GameKernel (source of truth) →
GameServer (orchestration) →
Transport (routing) →
Clients (presentation)
```

### State Transformation
```
GameState (full) → Authorization → Execution → New GameState →
Filtering → FilteredGameState → ViewProjection → UI State
```

---

## Architectural Benefits

### Correctness
- **Type system**: Prevents invalid states at compile time
- **Pure functions**: Ensure predictability and testability
- **Immutability**: Eliminates mutation bugs
- **Event sourcing**: Enables perfect replay verification

### Extensibility
- **Composition**: New features via composition, not modification
- **Orthogonality**: RuleSets and ActionTransformers combine independently
- **Zero coupling**: Core engine unchanged as system grows
- **Parametric polymorphism**: Same code works for all rulesets

### Testability
- **Pure functions**: Simple unit testing
- **Deterministic replay**: Regression testing from URLs
- **Event sourcing**: Complete audit trail
- **No side effects**: Isolated component testing

### Deployment Flexibility
- **Transport abstraction**: Browser, Worker, or edge
- **Clean separation**: Independent scaling
- **Location-agnostic**: Kernel runs anywhere
- **Progressive enhancement**: Start simple, scale up

### Maintainability
- **Single responsibility**: Each component focused
- **Explicit dependencies**: No hidden coupling
- **Clear boundaries**: Changes don't cascade
- **Type-driven**: Compiler catches breaking changes

---

## Key Design Decisions

### Composition Over Configuration
**Decision**: Achieve variant behavior through function composition rather than configuration flags and conditional logic.

**Rationale**: Eliminates complexity, enables unlimited combinations, prevents flag explosion.

**Trade-off**: Requires understanding functional composition patterns.

### Immutability by Default
**Decision**: State objects are never modified. New states created through transformation.

**Rationale**: Prevents "action at a distance," enables time-travel debugging, simplifies reasoning.

**Trade-off**: More object allocation (mitigated by modern JS engines).

### Explicit Dependencies
**Decision**: All dependencies passed as parameters. No hidden state or global variables.

**Rationale**: Makes data flow explicit, enables testing, prevents hidden coupling.

**Trade-off**: More parameter passing (mitigated by ExecutionContext bundling).

### Single Composition Point
**Decision**: RuleSets and ActionTransformers compose only in GameKernel constructor.

**Rationale**: Ensures consistency, prevents drift, enables parametric polymorphism.

**Trade-off**: Can't dynamically change composition mid-game (not a requirement).

### Client Trust Model
**Decision**: Clients trust server's validActions list completely, never revalidate.

**Rationale**: Simplifies client code, ensures consistency, prevents cheating.

**Trade-off**: Requires robust server validation (which we need anyway).

### Capability-Based Security
**Decision**: Use capability tokens instead of identity checks or roles.

**Rationale**: More flexible, composable, transparent. Capabilities are data.

**Trade-off**: Requires understanding capability pattern (well-documented).

---

## Glossary of Terms

**Action**: Atomic unit of game progression (bid, play, trump selection)
**ActionTransformer**: Action set transformer (tournament, oneHand, hints)
**Capability**: Permission token for access control (observe-hand, act-as-player)
**Composition**: Building complex behavior from simple parts (rulesets, action-transformers)
**Executor**: Pure function that applies action to state
**Filtering**: Hiding information based on permissions (hand visibility)
**Kernel**: Pure game logic authority (composition, authorization, filtering)
**Parametric**: Behavior determined by injected parameters, not inspection
**Projection**: Transformation from state to UI-ready data
**RuleSet**: Rule modifier for execution semantics (nello, plunge, splash)
**Session**: Player identity and permissions (playerId + capabilities)
**Transport**: Message routing abstraction (in-process, Worker, edge)

---

## Key Takeaways

1. **Event sourcing** enables perfect replay and debugging
2. **Pure functions** ensure predictability and testability
3. **Two-level composition** provides unlimited extensibility without core changes
4. **Capability-based security** offers flexible, transparent permissions
5. **Clean separation** keeps changes isolated to single components
6. **Server authority** ensures consistency and prevents cheating
7. **AI as clients** maintains fairness through protocol equality
8. **Immutability** simplifies reasoning and enables time-travel debugging
9. **Parametric polymorphism** eliminates conditional complexity
10. **Single composition point** ensures consistent behavior

---

## Implementation Overview

The system achieves complexity through composition of simple components:
- No inheritance hierarchies
- No monolithic classes
- No implicit behavior
- Clear boundaries between concerns

Each component focuses on a single responsibility, with behavior emerging from their composition rather than from complex individual components.

**Result**: A system that is correct, extensible, testable, and maintainable.

---

**Document Version**: 1.0
**Architecture Version**: mk5-tailwind
**Last Updated**: January 2025
