# ADR-20250111: System Authority for Scripted Actions

**Status**: Implemented
**Date**: 2025-01-11
**Deciders**: Architecture Review

## Context

The Texas 42 architecture uses a capability-based authorization system where every action must be authorized by a player session's capabilities. This works well for user-initiated actions (player clicks button → action executes), but creates conceptual problems for scripted actions.

### The Problem

**Scripted actions** (like one-hand transformer setup) are deterministic game scripts, not player requests:

```typescript
// one-hand transformer generates:
{ type: 'pass', player: 1, autoExecute: true }
{ type: 'pass', player: 2, autoExecute: true }
{ type: 'pass', player: 3, autoExecute: true }
{ type: 'bid', player: 0, bid: 'points', value: 30, autoExecute: true }
```

**Current problem**: The system forces these through player session authorization:
1. `processAutoExecuteActions()` finds the auto-execute action
2. `resolveSessionForAction()` searches for a session with `act-as-player` capability
3. `authorizeAndExecute()` validates the session can execute the action

**This is wrong because**:
- The script doesn't care about session capabilities
- `player: 1` means "affects player 1's position", NOT "requires player 1's session to authorize"
- Capability checks for scripts are meaningless (same seed = same actions, regardless of sessions)

### Why This Matters

**1. Conceptual Confusion**
Developers must think "which session should authorize this script action?" when the real answer is "none - it's a script"

**2. Artificial Complexity**
`resolveSessionForAction()` has complex fallback logic to find "some session that can execute this" when the script just wants to execute

**3. Original Intent Violated**
The one-hand transformer is a **system-level game setup script**, not a player action. Forcing it through player authorization violates its purpose.

**4. Capability Tightening Blocked**
Attempts to strengthen capability enforcement (original issue) fail because they assume every action needs a capable session - but scripts don't.

## Decision

Introduce **two authority models** for action execution:

### 1. Player Authority (Default)
Action must be authorized by a player session's capabilities
- User-initiated actions (clicks, button presses)
- Requires session to have `act-as-player` capability
- Standard authorization flow

### 2. System Authority (New)
Action executes as part of deterministic game script, bypassing capability checks
- Scripted setup actions (one-hand transformer)
- Auto-executed game flow (forced moves in speed mode)
- Still validated for structural correctness (must be valid in current state)
- Set via `action.meta.authority = 'system'`

### Implementation

**Add authority field to ActionMeta**:
```typescript
interface ActionMeta {
  authority?: 'player' | 'system';
  autoExecute?: boolean;
  // ... other fields
}
```

**Update authorization logic** (authorization.ts):
```typescript
function authorizeAndExecute(mpState, request, ctx) {
  const meta = normalizeMeta(request.action.meta);

  // System authority: bypass capability checks
  if (meta.authority === 'system') {
    // Still validate structural correctness
    const validActions = ctx.getValidActions(mpState.coreState);
    if (!validActions.some(a => actionsMatch(a, request.action))) {
      return err('Action not valid in current state');
    }
    return ok(executeAction(mpState.coreState, request.action, ctx.rules));
  }

  // Player authority: standard capability checks (existing logic)
  // ...
}
```

**Update transformers** to mark scripted actions:
```typescript
// one-hand transformer
{
  type: 'pass',
  player: 1,
  autoExecute: true,
  meta: { authority: 'system', scriptId: 'one-hand-setup' }
}
```

## Consequences

### Positive

**1. Conceptual Clarity**
Two distinct execution paths with clear semantics:
- Player actions → require authorization
- System scripts → deterministic execution

**2. Simpler Auto-Execute Logic**
No more complex session resolution for scripts. System actions just execute.

**3. Enables Capability Tightening**
Can now strengthen player authorization without affecting scripts.

**4. Matches Reality**
System authority accurately models what one-hand transformer actually is: a deterministic game setup script.

**5. Maintains Determinism**
System actions still validated structurally. Same seed + config = same execution.

### Negative

**1. Two Code Paths**
Authorization now has two branches (system vs player). Increases complexity slightly.

**Mitigation**: Clear documentation, explicit checks, good tests.

**2. New Concept to Learn**
Contributors must understand when to use system authority.

**Mitigation**: Document in ORIENTATION.md, provide examples in transformers.

**3. Potential for Misuse**
Developer could mark player actions as system authority to bypass checks.

**Mitigation**: Code review. Clear guidance: "Use system authority only for deterministic scripts and forced game flow"

## Alternative Considered: Tighten Capability Enforcement

**Approach**: Force every action through strict capability checks, annotate all scripted actions with explicit `requiredCapabilities`

**Why rejected**:
- Fundamentally wrong model: scripts don't need "capabilities", they need to execute
- Added complexity without conceptual clarity
- Would require every transformer to maintain capability annotations
- Treats symptoms (complex resolution logic) rather than cause (wrong abstraction)

## Capability System Simplification (Same ADR)

As part of this change, we simplified the capability system itself.

### Removed Capabilities

The following capability types were removed as they were not used in production:
- `observe-own-hand` → merged into `observe-hands`
- `observe-hand` → merged into `observe-hands`
- `observe-all-hands` → merged into `observe-hands`
- `observe-full-state` → merged into `observe-hands`
- `see-hints` (future feature - not yet implemented)
- `see-ai-intent` (future feature - not yet implemented)
- `replace-ai` (future feature - not yet implemented)
- `configure-action-transformer` (future feature - not yet implemented)
- `undo-actions` (future feature - not yet implemented)

### New Simplified Capability Model

Only 2 capability types remain:

```typescript
type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-hands'; playerIndices: number[] | 'all' };
```

**Standard capability sets**:
- **Player**: `[{ type: 'act-as-player', playerIndex: N }, { type: 'observe-hands', playerIndices: [N] }]`
- **AI**: Same as player
- **Spectator**: `[{ type: 'observe-hands', playerIndices: 'all' }]`

### Rationale for Simplification

**1. Over-Specification**
10 capability types defined, but only 2 were actually enforced in production code.

**2. Future Features**
Most capabilities existed for features not yet implemented (hints, tutorials, undo). Can be re-added when those features are built.

**3. Maintenance Burden**
589 lines of capability code (types, builders, filtering) when current needs could be met with ~240 lines.

**4. Redundant Types**
Multiple observe capabilities (`observe-own-hand`, `observe-hand`, `observe-all-hands`, `observe-full-state`) served similar purposes and could be unified.

### Impact

**Code Reduction**: ~315 lines removed
- Simplified filtering logic
- Removed metadata pruning for future capabilities
- Unified observation capabilities into single type

**Spectator Support Preserved**:
- `observe-hands: 'all'` provides same functionality as deleted `observe-all-hands` and `observe-full-state`
- No loss of functionality for planned features

**Future Flexibility Maintained**:
- Can re-add capability types when features are implemented
- Architecture supports extension (capability-based model preserved)
- Clear documentation of removed capabilities for future reference

## Files Modified

### Core Implementation
- `src/game/multiplayer/capabilityUtils.ts` - Added authority field, simplified filtering
- `src/game/multiplayer/authorization.ts` - System authority bypass
- `src/game/action-transformers/oneHand.ts` - Added system authority
- `src/game/action-transformers/speed.ts` - Added system authority

### Type Simplification
- `src/game/multiplayer/types.ts` - Reduced to 2 capability types
- `src/game/multiplayer/capabilities.ts` - Simplified builders

### Tests
- `src/tests/unit/authorization.test.ts` - System authority tests
- `src/tests/integration/gamehost-autoexec.test.ts` - Verify system authority
- `src/tests/unit/capability-utils.test.ts` - Updated for new capability types
- All test fixtures updated to use new capability model

### Documentation
- `docs/ORIENTATION.md` - Document system authority and simplified capabilities
- `docs/adrs/ADR-20250111-system-authority.md` - This document

## Status

**Implemented**: All changes complete, tests passing

**Test Results**:
- 991 tests total (989 passing, 1 skipped, 1 pre-existing failure)
- All capability and authorization tests passing
- Zero TypeScript compilation errors

## Future Work

**When implementing hints/tutorials/undo**:
- Re-add capability types as needed (`see-hints`, `undo-actions`, etc.)
- Update capability builders (coachCapabilities, tutorialCapabilities)
- Re-implement metadata filtering in capabilityUtils.ts
- Keep system authority distinction (hints are player features, not scripts)

## References

- VISION.md - "Capability-based access" principle
- ORIENTATION.md - Capability and authorization documentation
- Original issue: "Tighten Capability Enforcement For Scripted/Auto Actions"
