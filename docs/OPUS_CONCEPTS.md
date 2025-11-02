# Texas 42 Architecture: Conceptual Overview

**Purpose**: High-level architectural concepts reference for the Texas 42 codebase, focusing on mental models and design patterns without implementation details.

**Audience**: Architects, developers, product managers, and anyone seeking to understand the system's design philosophy.

---

## Core Architecture Principles

### Event Sourcing
The fundamental architectural pattern where game state is derived from a sequence of actions. State is never mutated directly; instead, actions are applied to produce new states. This enables perfect replay, debugging, and game sharing via compressed URLs.

### Pure Functions & Immutability
All game logic is implemented as pure functions with no side effects. Given the same inputs, functions always produce the same outputs. State objects are never modified; new states are created. This ensures predictability, testability, and enables time-travel debugging.

### Two-Level Composition System
A unique architectural innovation solving the challenge of special contracts that need to modify both execution rules AND available actions. Level 1 (Layers) modifies HOW the game executes through rule overrides. Level 2 (Variants) transforms WHAT actions are possible through filtering, annotation, and scripting.

### Parametric Polymorphism
Executors never inspect state to determine behavior. Instead, they accept rule functions as parameters and delegate all decisions to them. This allows the same executor code to work for all game variants without conditional logic, eliminating complexity.

### Separation of Concerns
Each architectural layer has a single, well-defined responsibility. The engine handles pure game logic, the kernel adds multiplayer concerns, the server orchestrates, transport routes messages, and UI renders. Changes in one layer don't cascade to others.

---

## Game Logic Concepts

### GameState
The complete representation of a game at any point in time. Contains players, hands, tricks, scores, current phase, and all information needed to continue play. GameState is immutable - actions create new states rather than modifying existing ones.

### GameAction
A union type representing all possible player actions: bidding, trump selection, playing dominoes, and consensus actions. Actions are the atomic units of game progression and form the event log that enables replay.

### Executor Pattern
Pure functions that apply GameActions to GameState, producing new GameState. Each action type has a corresponding executor. Executors are parameterized by rules, never inspecting state directly to determine behavior.

### State Transitions
Representations of possible future game states. Each transition includes the action to execute, the resulting state, and a human-readable label. Used by AI for decision-making and UI for showing available moves.

### Deterministic Replay
The ability to perfectly recreate any game state by replaying its action history from the initial configuration. Same seed + same actions always produces identical game state, enabling debugging and game sharing.

### Action History
The immutable sequence of all GameActions executed in a game. Serves as the authoritative record, audit trail, and enables the event sourcing pattern. History is append-only, never modified.

---

## Composition Systems

### Layers (Execution Semantics)
Composable units that modify HOW the game executes by overriding specific rule methods. Nello makes tricks 3-player instead of 4-player. Plunge changes who selects trump. Layers compose via reduce pattern, with later layers overriding earlier ones.

### Variants (Action Transformation)
Functions that transform WHAT actions are available by wrapping the state machine. Tournament variant filters out special contracts. Speed variant adds auto-execute flags. OneHand variant scripts entire bidding sequences.

### GameRules Interface
The contract of 13 pure methods that completely define game execution behavior. Methods determine WHO acts (player selection), WHEN things happen (completion checks), HOW mechanics work (trick winning), and WHAT is valid (legal moves).

### StateMachine Pattern
A function that takes GameState and returns available GameActions. The base state machine generates actions according to layers. Variants wrap this state machine, transforming its output through filtering, annotation, or replacement.

### ExecutionContext
An immutable container bundling layers, composed rules, and the variant-wrapped state machine. Created once in GameKernel constructor, then passed throughout the system. Ensures consistent composition and enables parametric polymorphism.

---

## Multiplayer & Authorization

### PlayerSession
Separates player identity from game seat. A session links a player ID to a seat index, control type (human/AI), and capabilities. This enables players to reconnect, switch seats, and have different permissions independent of their seat.

### Capability-Based Security
Permissions are granted through composable capability tokens rather than identity checks. Capabilities determine what actions a player can execute and what state they can observe. More flexible and transparent than role-based systems.

### Authorization Flow
When a player attempts an action, the system finds their session, gets all valid actions, filters by their capabilities, then checks if the requested action is allowed. Only authorized actions execute, ensuring game integrity.

### Visibility Filtering
State and actions are filtered based on observer capabilities before transmission. Players only see hands they're authorized to see. The kernel stores unfiltered state and filters on-demand per request, ensuring single source of truth.

### Consensus Mechanisms
Actions requiring agreement from multiple players before proceeding. Complete-trick and score-hand actions use set-based tracking to record which players have agreed. When all required players agree, the action executes.

---

## Server Architecture

### GameKernel
The pure game authority that owns state, composes layers/variants, authorizes actions, and filters views. Has zero knowledge of transport, networking, or AI. Stores unfiltered state and applies capability-based filtering on each request.

### GameServer
The orchestrator that creates and owns the GameKernel and AIManager. Routes protocol messages to appropriate handlers, manages client subscriptions, and broadcasts state updates. Bridges between transport layer and game logic.

### Transport Abstraction
Interface enabling different message routing implementations without changing game logic. InProcessTransport for single-player browser games, WorkerTransport for web workers, CloudflareTransport for edge computing. GameServer is transport-agnostic.

### Message Protocol
Standardized client-server communication using typed messages. Clients send CREATE_GAME, EXECUTE_ACTION, SUBSCRIBE. Server responds with STATE_UPDATE, ACTION_CONFIRMED, ERROR. Protocol is transport-independent.

### Client-Server Trust Model
Clients fully trust the server's validActions list and never re-validate or re-filter. Server is authoritative for all game logic. Clients only handle UI and send action requests. This simplifies client code and ensures consistency.

---

## AI System

### Protocol-Speaking AI Clients
AI players are independent actors that communicate via the same protocol as human players. They receive state updates, analyze positions, and send action messages. No special privileges or direct state access, ensuring fairness.

### AI Strategy Pattern
Pure functions that select actions based on game state and available transitions. Strategies are composable and parameterized by difficulty. Beginner uses simple heuristics, intermediate adds hand analysis, expert employs game tree search.

### Hand Strength Analysis
Sophisticated evaluation system considering multiple factors: control (unbeatable dominoes), trump quality, count safety (protecting high-value dominoes), defensive strength, and synergy bonuses. Produces numerical scores guiding bidding decisions.

### Lexicographic Evaluation
Advanced hand comparison technique that ranks dominoes by how many others can beat them, then compares hands like comparing words alphabetically. More nuanced than simple point counting, capturing positional strength.

### Difficulty Levels
Three tiers affecting both decision quality and thinking time. Beginner (800-2000ms) makes random moves except consensus. Intermediate (500-1500ms) uses hand strength analysis. Expert (200-800ms) employs full strategic evaluation.

### Consensus Priority
AI immediately executes consensus actions (complete-trick, score-hand) without delay, ensuring smooth game flow. These actions take precedence over strategic thinking since they're procedural rather than strategic.

---

## Client & UI Architecture

### NetworkGameClient
Client-side game interface that caches filtered state from the server and trusts the server's validActions list. Sends action requests and receives state updates. Never performs game logic, only UI state management.

### ViewProjection Pipeline
Pure transformation from GameState to UI-ready data structures. Groups actions by type, calculates display properties, adds tooltips and hints. Same state always produces same projection, enabling predictable reactive updates.

### Reactive State Management
Svelte stores provide reactive bindings between game state and UI components. When state updates arrive, stores update, ViewProjection recomputes, and components automatically re-render. No manual DOM manipulation needed.

### Filtered vs Unfiltered State
Server maintains unfiltered GameState with all information. Clients receive FilteredGameState with only authorized information visible. Hand visibility, bid information, and action availability filtered by capabilities.

---

## Special Features

### Special Contracts
Game modes that dramatically alter rules and objectives. Nello: bidder's partner sits out, bidder must lose all tricks. Plunge: partner selects trump, bidder must win all tricks. Splash: like plunge but requires 3+ doubles. Sevens: closest to 42 points wins.

### URL Compression & Sharing
Complete games encoded in compact URLs using custom compression. Actions map to single characters for common events, two for rare ones. Enables sharing games via link, perfect replay from URL, and regression testing.

### Auto-Execute Actions
Actions marked with autoExecute flag that play automatically without user interaction. Used for forced plays (only one legal move), procedural actions (deal, shuffle), and speed variant (auto-play when one option).

### Tournament Mode
Variant that disables special contracts, enforcing standard Texas 42 rules for competitive play. Achieved through action filtering rather than conditional logic, demonstrating the power of the composition system.

### Hand Scenarios
Predefined interesting game situations for testing and training. Encoded as action sequences that can be replayed to reach specific positions. Used for debugging, AI training, and player education.

---

## Architectural Invariants

These principles must never be violated:

### Pure State Storage
GameKernel stores unfiltered state. Filtering happens per-request, never at rest. Single source of truth principle.

### Server Authority
Clients trust server completely for game logic. No client-side validation or filtering. Ensures consistency and simplifies client code.

### Capability-Based Access
All permissions via capability tokens. Never use identity checks or role comparisons. Transparent, composable security model.

### Single Composition Point
Layers and variants compose only in GameKernel constructor. ExecutionContext created once, used everywhere. Ensures consistent behavior.

### Zero Coupling
Core engine has no awareness of multiplayer, networking, or transport. Layers and variants don't reference multiplayer concepts. Clean architectural boundaries.

### Parametric Execution
Executors delegate to injected rules, never inspect state for behavior. No conditional logic based on game mode. Enables unlimited extensibility.

### Event Sourcing Foundation
State must be derivable from action replay. Actions are immutable and append-only. Perfect reproducibility guaranteed.

### Clean Separation
Each component has single responsibility. GameServer orchestrates, GameKernel executes, Transport routes. No responsibility bleeding.

---

## Concept Relationships

### Data Flow
User input → Client → Transport → Server → GameKernel → Authorization → Execution → State Update → Filtering → Transport → Client → UI Update

### Composition Flow
Layers define rules → Rules compose via reduce → Base state machine uses rules → Variants wrap state machine → ExecutionContext bundles all → Executors use context

### Authority Hierarchy
GameKernel (source of truth) → GameServer (orchestration) → Transport (routing) → Clients (presentation)

### State Transformation
GameState (full) → Authorization → Execution → New GameState → Filtering → FilteredGameState → ViewProjection → UI State

---

## Design Philosophy

### Simplicity Through Composition
Complex behavior emerges from composing simple, pure functions. No monolithic classes or deep inheritance hierarchies.

### Correct by Construction
Use type system to prevent errors at compile time. Make illegal states unrepresentable.

### Explicit Over Implicit
All behavior explicitly defined through composition. No hidden magic or implicit conventions.

### Immutability as Default
State never mutated, only transformed. Enables reasoning, debugging, and time-travel.

### Trust Through Verification
Server validates everything, clients trust completely. Clear security boundary.

---

## Mental Models

### The Game as a State Machine
Every game position has defined transitions to next positions. Actions are edges, states are nodes. AI explores this graph to make decisions.

### Layers as Lenses
Each layer provides a different lens through which to view game rules. Stack lenses to create new game modes.

### Variants as Decorators
Variants wrap and transform the base game, adding features without modifying core logic.

### Capabilities as Keys
Each capability unlocks specific functionality. Collect keys to gain more power.

### The Kernel as a Pure Function
Given state and action, always produces same new state. No hidden state or side effects.

---

## Common Patterns

### Filter-Map-Reduce
Used throughout for data transformation. Filter valid items, map to new form, reduce to result.

### Delegation Over Inspection
Functions accept behavior as parameters rather than inspecting data to determine behavior.

### Composition Over Configuration
Build behavior by composing functions, not by setting flags or options.

### Fail-Fast Validation
Validate early and explicitly. Make errors impossible through types where feasible.

### Single Source of Truth
One authoritative location for each piece of state. Everything else derives from it.

---

## Future Architecture

### Planned Enhancements
- Worker-based transport for true parallelism
- Cloudflare Durable Objects for edge deployment
- WebRTC transport for peer-to-peer play
- Machine learning AI strategies
- Spectator mode with commentary

### Architectural Flexibility
The clean separation and composition patterns enable these enhancements without restructuring. New transports plug in, new variants compose, new AI strategies swap in.

---

## Glossary of Terms

**Action**: Atomic unit of game progression
**Capability**: Permission token for access control
**Composition**: Building complex behavior from simple parts
**Executor**: Function that applies action to state
**Filtering**: Hiding information based on permissions
**Kernel**: Pure game logic authority
**Layer**: Rule modifier for execution semantics
**Parametric**: Behavior determined by parameters
**Projection**: Transformation from state to UI
**Session**: Player identity and permissions
**Transport**: Message routing abstraction
**Variant**: Action set transformer

---

## Key Takeaways

1. **Event sourcing** enables perfect replay and debugging
2. **Pure functions** ensure predictability and testability
3. **Two-level composition** provides unlimited extensibility
4. **Capability-based security** offers flexible permissions
5. **Clean separation** keeps changes isolated
6. **Server authority** ensures consistency
7. **AI as clients** maintains fairness
8. **Immutability** simplifies reasoning

---

**Document Version**: 1.0
**Architecture Version**: mk5-tailwind
**Last Updated**: October 2025