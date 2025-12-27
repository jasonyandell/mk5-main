# GPU Solver: Complete Regret Tables

**Goal:** Generate exhaustive training data capturing optimal play at every reachable state.

**Output:** For each (seed, state, move): the value of making that move.

**Constraint:** Fully GPU. No CPU enumeration. CPU only for loop control and I/O.

---

## Overview

For each seed (deal + declaration), we solve the complete game tree:

```
1. GPU BFS enumeration    → all reachable states (~3M)
2. GPU child index build  → torch.searchsorted
3. GPU backward induction → minimax values
4. Write parquet          → one file per seed
```

All state manipulation is **packed int64 + bitwise ops**. No Python objects.

---

## State Representation (47 bits in int64)

```
Bits 0-27: Remaining hands (4 × 7-bit masks)
  [0:7]   remaining[0] - Player 0's local indices still in hand
  [7:14]  remaining[1] - Player 1's
  [14:21] remaining[2] - Player 2's
  [21:28] remaining[3] - Player 3's

Bits 28-35: Game progress
  [28:34] score       - Team 0's points (0-42, 6 bits)
  [34:36] leader      - Current trick leader (0-3, 2 bits)

Bits 36-47: Current trick
  [36:38] trick_len   - Plays so far (0-3, 2 bits)
  [38:41] play0       - Leader's local index (0-6, or 7 if N/A)
  [41:44] play1       - Second player's local index
  [44:47] play2       - Third player's local index
```

**Total: 47 bits.** Fits int64 with 17 bits to spare.

### Why Local Indices?

Each player holds 7 dominoes. We use indices 0-6 within each hand, not global domino IDs 0-27.

- Smaller state space (7-bit masks vs 28-bit)
- Context tables map local → global when needed
- Same state packing works for any deal

---

## Context Tables (Per Seed, ~20KB)

```python
@dataclass
class SeedContext:
    # Local → global domino mapping
    L: torch.Tensor                  # (4, 7) int8

    # Follow rules: which local indices can follow a led suit?
    # LOCAL_FOLLOW[leader][lead_local_idx][follower_offset] → 7-bit mask
    LOCAL_FOLLOW: torch.Tensor       # (4, 7, 4) → flattened to (112,)

    # Trick outcomes (precomputed for all possible 4-domino tricks)
    # TRICK_WINNER[leader][p0][p1][p2][p3] → winner offset (0-3)
    TRICK_WINNER: torch.Tensor       # (4, 7, 7, 7, 7) → flattened to (9604,)

    # TRICK_POINTS[leader][p0][p1][p2][p3] → points (1-11)
    TRICK_POINTS: torch.Tensor       # (4, 7, 7, 7, 7) → flattened to (9604,)
```

These tables are built once per seed from the deal and declaration.

---

## Core Functions

### pack_state / unpack

```python
def pack_state(remaining: torch.Tensor, score, leader, trick_len, p0, p1, p2) -> torch.Tensor:
    """
    remaining: (N, 4) tensor of 7-bit masks
    others: (N,) tensors
    returns: (N,) int64 packed states
    """
    return (
        (remaining[:, 0].to(torch.int64)) |
        (remaining[:, 1].to(torch.int64) << 7) |
        (remaining[:, 2].to(torch.int64) << 14) |
        (remaining[:, 3].to(torch.int64) << 21) |
        (score.to(torch.int64) << 28) |
        (leader.to(torch.int64) << 34) |
        (trick_len.to(torch.int64) << 36) |
        (p0.to(torch.int64) << 38) |
        (p1.to(torch.int64) << 41) |
        (p2.to(torch.int64) << 44)
    )

def unpack_remaining(states: torch.Tensor) -> torch.Tensor:
    """states: (N,) int64 → remaining: (N, 4) int64"""
    r0 = (states >> 0) & 0x7F
    r1 = (states >> 7) & 0x7F
    r2 = (states >> 14) & 0x7F
    r3 = (states >> 21) & 0x7F
    return torch.stack([r0, r1, r2, r3], dim=1)
```

### compute_level / compute_team / compute_terminal_value

```python
# Popcount lookup table (128 entries for 7-bit values)
POPCOUNT = torch.tensor([bin(i).count('1') for i in range(128)], device='cuda')

def compute_level(states: torch.Tensor) -> torch.Tensor:
    """Level = total dominoes remaining across all hands."""
    r0 = (states >> 0) & 0x7F
    r1 = (states >> 7) & 0x7F
    r2 = (states >> 14) & 0x7F
    r3 = (states >> 21) & 0x7F
    return POPCOUNT[r0] + POPCOUNT[r1] + POPCOUNT[r2] + POPCOUNT[r3]

def compute_team(states: torch.Tensor) -> torch.Tensor:
    """Is current player on team 0?"""
    leader = (states >> 34) & 0x3
    trick_len = (states >> 36) & 0x3
    player = (leader + trick_len) % 4
    return (player % 2 == 0)

def compute_terminal_value(states: torch.Tensor) -> torch.Tensor:
    """Terminal value = 2 * score - 42 (team 0's advantage)."""
    score = (states >> 28) & 0x3F
    return (2 * score - 42).to(torch.int8)
```

---

## expand_gpu: The Core Operation

This is the heart of the solver. Given N packed states, produce (N, 7) children.

```python
def expand_gpu(states: torch.Tensor, ctx: SeedContext) -> torch.Tensor:
    """
    states: (N,) int64 packed states
    returns: (N, 7) int64 children, -1 for illegal moves
    """
    N = len(states)
    device = states.device

    # === UNPACK ===
    r0 = (states >> 0) & 0x7F
    r1 = (states >> 7) & 0x7F
    r2 = (states >> 14) & 0x7F
    r3 = (states >> 21) & 0x7F
    score = (states >> 28) & 0x3F
    leader = (states >> 34) & 0x3
    trick_len = (states >> 36) & 0x3
    p0 = (states >> 38) & 0x7
    p1 = (states >> 41) & 0x7
    p2 = (states >> 44) & 0x7

    remaining = torch.stack([r0, r1, r2, r3], dim=1)  # (N, 4)

    # === WHOSE TURN? ===
    player = (leader + trick_len) % 4  # (N,)

    # === PLAYER'S HAND ===
    hand = remaining.gather(1, player.unsqueeze(1)).squeeze(1)  # (N,)

    # === LEGAL MOVES ===
    leading = (trick_len == 0)

    # Follow mask lookup: LOCAL_FOLLOW[leader * 28 + p0 * 4 + trick_len]
    safe_p0 = p0.clamp(0, 6)
    follow_idx = leader * 28 + safe_p0 * 4 + trick_len
    follow_mask = ctx.LOCAL_FOLLOW[follow_idx]

    can_follow = hand & follow_mask
    must_slough = (can_follow == 0) & ~leading
    legal = torch.where(leading | must_slough, hand, can_follow)

    # === COMPUTE CHILDREN ===
    children = torch.full((N, 7), -1, dtype=torch.int64, device=device)

    for move in range(7):
        move_bit = 1 << move
        is_legal = (legal & move_bit) != 0

        if not is_legal.any():
            continue

        # Remove domino from player's hand
        new_hand = hand ^ move_bit
        new_remaining = remaining.scatter(1, player.unsqueeze(1), new_hand.unsqueeze(1))

        # Does this complete a trick?
        completes = (trick_len == 3)

        # --- BRANCH A: Mid-trick ---
        new_trick_len = trick_len + 1
        new_p0 = torch.where(trick_len == 0, move, p0)
        new_p1 = torch.where(trick_len == 1, move, p1)
        new_p2 = torch.where(trick_len == 2, move, p2)

        child_mid = pack_state(new_remaining, score, leader,
                               new_trick_len, new_p0, new_p1, new_p2)

        # --- BRANCH B: Trick completes ---
        trick_idx = leader * 2401 + p0 * 343 + p1 * 49 + p2 * 7 + move
        winner_offset = ctx.TRICK_WINNER[trick_idx]
        points = ctx.TRICK_POINTS[trick_idx]

        winner = (leader + winner_offset) % 4
        team0_wins = (winner % 2 == 0)
        new_score = score + torch.where(team0_wins, points, torch.zeros_like(points))

        child_done = pack_state(new_remaining, new_score, winner,
                                torch.zeros_like(trick_len),
                                torch.full_like(p0, 7),
                                torch.full_like(p1, 7),
                                torch.full_like(p2, 7))

        # Select branch
        child = torch.where(completes, child_done, child_mid)
        children[:, move] = torch.where(is_legal, child, -1)

    return children
```

### Key Design Points

| Aspect | Approach |
|--------|----------|
| **Indexing player's hand** | `remaining.gather(1, player.unsqueeze(1))` |
| **Table lookups** | Flatten to 1D, compute index arithmetically |
| **Branching** | Compute both branches, select with `torch.where` |
| **Move loop** | Python for-loop (7 iterations, not over states) |
| **Invalid indices** | Use `.clamp()` before lookup, result is masked anyway |

---

## The Three Phases

### Phase 1: GPU BFS Enumeration

```python
def enumerate_gpu(ctx: SeedContext) -> torch.Tensor:
    """Returns sorted tensor of all reachable states."""

    initial = ctx.pack_initial_state()
    frontier = torch.tensor([initial], dtype=torch.int64, device='cuda')
    all_levels = [frontier]

    for level in range(28, 0, -1):
        children = expand_gpu(frontier, ctx)
        children = children[children >= 0]      # Remove illegal
        children = torch.unique(children)       # GPU dedup
        all_levels.append(children)
        frontier = children

    return torch.sort(torch.cat(all_levels)).values
```

### Phase 2: Build Child Index

```python
def build_child_index(all_states: torch.Tensor, ctx: SeedContext) -> torch.Tensor:
    """Build (N, 7) index mapping each move to child's position in all_states."""

    all_children = expand_gpu(all_states, ctx)  # (N, 7)

    # GPU binary search
    child_idx = torch.searchsorted(all_states, all_children.clamp(min=0))
    child_idx = torch.where(all_children >= 0, child_idx, -1)

    return child_idx
```

### Phase 3: Backward Induction

```python
def solve_gpu(all_states: torch.Tensor, child_idx: torch.Tensor, ctx: SeedContext):
    """Compute minimax values for all states."""

    N = len(all_states)
    V = torch.zeros(N, dtype=torch.int8, device='cuda')
    move_values = torch.full((N, 7), -128, dtype=torch.int8, device='cuda')

    level_of = compute_level(all_states)
    is_team0 = compute_team(all_states)

    # Terminal states (level 0)
    terminal = (level_of == 0)
    V[terminal] = compute_terminal_value(all_states[terminal])

    # Backward pass
    for level in range(1, 29):
        mask = (level_of == level)
        if not mask.any():
            continue

        idx = mask.nonzero(as_tuple=True)[0]
        cidx = child_idx[idx].clamp(min=0)  # (K, 7)
        legal = (child_idx[idx] >= 0)       # (K, 7)

        cv = V[cidx]
        move_values[idx] = torch.where(legal, cv, -128)

        # Max for team0, min for team1
        cv_max = torch.where(legal, cv, -128)
        cv_min = torch.where(legal, cv, 127)
        V[idx] = torch.where(is_team0[idx], cv_max.max(dim=1).values, cv_min.min(dim=1).values)

    return V, move_values
```

---

## Robustness: Crash Recovery

**Problem:** Long runs that crash with no output teach us nothing.

**Solution:** One file per seed, verbose logging, resume capability.

### Per-Seed Output

```python
def solve_and_save(seed: int, decl_id: int, output_dir: Path) -> bool:
    """Solve one seed, write immediately. Returns True if newly solved."""

    output_path = output_dir / f"seed_{seed:08d}_decl_{decl_id}.parquet"

    if output_path.exists():
        return False  # Already done

    timer = SeedTimer(seed, decl_id)

    ctx = build_context(seed, decl_id)
    timer.phase("setup")

    all_states = enumerate_gpu(ctx)
    timer.phase("enumerate", f"states={len(all_states):,}")

    child_idx = build_child_index(all_states, ctx)
    timer.phase("child_index")

    V, move_values = solve_gpu(all_states, child_idx, ctx)
    root_value = int(V[0])
    timer.phase("solve", f"root={root_value:+d}")

    write_parquet(output_path, seed, decl_id, all_states, V, move_values)
    timer.done(root_value)

    return True
```

### Example Output

```
10:15:32 | seed=0 decl=0 | setup | 0.01s
10:15:33 | seed=0 decl=0 | enumerate | 0.82s | states=2,847,291
10:15:33 | seed=0 decl=0 | child_index | 0.34s
10:15:34 | seed=0 decl=0 | solve | 0.91s | root=+14
10:15:34 | seed=0 decl=0 | DONE | 2.12s
```

If crash at seed=47, restart sees:
```
10:20:01 | Skipping seed=0..46 (already exist)
10:20:01 | seed=47 decl=0 | setup | 0.01s
```

---

## File Structure

```
scripts/solver/
├── __init__.py
├── tables.py       # Global tables (computed once at import)
├── context.py      # SeedContext: build L, LOCAL_FOLLOW, TRICK_WINNER, TRICK_POINTS
├── state.py        # pack_state, unpack, compute_level, compute_team
├── expand.py       # expand_gpu
├── solve.py        # enumerate_gpu, build_child_index, solve_gpu
├── output.py       # write_parquet
└── main.py         # CLI entry point
```

---

## Memory Budget (RTX 3050, 4GB VRAM)

```
all_states:  3M × 8 bytes (int64)     = 0.024 GB
children:    3M × 7 × 8 bytes         = 0.168 GB
child_idx:   3M × 7 × 4 bytes (int32) = 0.084 GB
V:           3M × 1 byte              = 0.003 GB
move_values: 3M × 7 × 1 byte          = 0.021 GB
context:     ~20 KB                   = 0.00002 GB
───────────────────────────────────────────────
Total per seed:                         ~0.3 GB  ✓
```

Plenty of headroom on 4GB GPU.

---

## Summary

```
For each seed (deal + declaration):
  1. enumerate_gpu    → ~3M states via BFS + torch.unique
  2. build_child_index → torch.searchsorted for (N, 7) lookup
  3. solve_gpu        → backward induction over 29 levels
  4. write_parquet    → immediate, one file per seed

~2-3 seconds per seed on RTX 3050
~1M seeds in ~1 weekend
```

### What We Do

- int64 state packing (47 bits)
- Pure tensor ops (no Python loops over states)
- Flattened table lookups
- `torch.where` for branchless logic
- One file per seed (crash-safe)

### What We Avoid

- CPU enumeration
- Python objects for states
- Batched output (lose batch on crash)
- Silent progress
