"""GPU-native MCTS forest for batched tree search.

Manages N parallel MCTS trees on GPU. All operations are fully vectorized
with no Python loops in the hot path.

Key design decisions:
- Pre-allocated node pools to avoid dynamic allocation
- Flattened state storage for efficient GPU access
- Virtual loss during selection for exploration diversity
- Single-child expansion: each expand_gpu call creates ONE child per tree,
  matching Python MCTS behavior (100 sims = ~100 nodes, not ~700)
- Selection only descends through fully-expanded nodes (all legal children exist)
- Uniform prior (1/n_legal_actions) - no policy network guidance yet

Usage with oracle:

```python
from forge.zeb.gpu_mcts import (
    create_forest, select_leaves_gpu, expand_gpu, backprop_gpu,
    get_root_policy_gpu, get_leaf_states, get_terminal_values,
    prepare_oracle_inputs,
)
from forge.zeb.gpu_game_state import deal_random_gpu

# Create forest with initial game states
deals = deal_random_gpu(n_trees=16, device='cuda', decl_ids=0)
forest = create_forest(n_trees=16, max_nodes=1024, initial_states=deals, device='cuda')

for _ in range(n_simulations):
    # Select leaves for all trees in parallel
    leaf_indices, paths = select_leaves_gpu(forest)

    # Get states for oracle evaluation
    leaf_states = get_leaf_states(forest, leaf_indices)

    # Prepare inputs for oracle.batch_evaluate_gpu()
    oracle_inputs = prepare_oracle_inputs(forest, leaf_states)

    # Evaluate with oracle
    values = oracle.batch_evaluate_gpu(**oracle_inputs)

    # Handle terminal states
    terminal_values, is_terminal = get_terminal_values(forest, leaf_indices, leaf_states)
    values = torch.where(is_terminal, terminal_values, values)

    # Expand and backpropagate
    expand_gpu(forest, leaf_indices)
    backprop_gpu(forest, leaf_indices, values, paths)

# Get final policies
policies = get_root_policy_gpu(forest)  # (N, 7) slot probabilities
```
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from forge.zeb.cuda_only import require_cuda, require_cuda_tensor
from forge.zeb.gpu_game_state import (
    GPUGameState,
    current_player_gpu,
    deal_random_gpu,
    get_domino_tables,
    is_terminal_gpu,
    led_suit_for_lead_domino_gpu,
    legal_actions_gpu,
    trick_rank_gpu,
)
from forge.zeb.gpu_preprocess import compute_legal_mask_gpu, compute_remaining_bitmask_gpu

if TYPE_CHECKING:
    pass


@dataclass
class GPUMCTSForest:
    """N parallel MCTS trees on GPU.

    Each tree has max M nodes. Uses pre-allocated tensors to avoid
    dynamic allocation during search.

    Memory layout:
    - Node tensors have shape (N, M) for N trees with M nodes each
    - States are flattened to (N*M, ...) and indexed as tree*max_nodes + node_idx
    """
    n_trees: int
    max_nodes: int
    device: torch.device

    # Node statistics (N, M)
    visit_counts: Tensor   # int32 - visits to each node
    value_sums: Tensor     # float32 - sum of backpropagated values

    # Tree structure (N, M)
    parents: Tensor        # int64 - parent node index (-1 for root)
    depths: Tensor         # int32 - node depth (root=0)

    # Children pointers (N, M, 7) - one child per slot
    children: Tensor       # int64 - child node index (-1 if unexpanded)
    child_actions: Tensor  # int32 - slot index that led to this child

    # Cached node properties (N, M) - avoid recomputing in select_leaves_gpu
    n_legal: Tensor        # int64 - number of legal actions per node
    is_terminal: Tensor    # bool - whether node is terminal

    # Cached player-to-act at each node (N, M) int32 in [0..3].
    # Equivalent to current_player_gpu(state_at_node), but avoids gathering full state.
    to_play: Tensor

    # Policy priors (N, M, 7) - from neural network policy head
    # priors[tree, node, slot] = prior probability for taking action at slot
    # Used in UCB calculation: Q + c_puct * prior * sqrt(parent) / (1 + visits)
    priors: Tensor         # float32 - policy priors per node

    # Node states - flattened (N * M, ...)
    # Indexed as: states[tree_idx * max_nodes + node_idx]
    states: GPUGameState

    # Allocation tracking
    n_nodes: Tensor        # (N,) int64 - number of allocated nodes per tree

    # Cached indices (avoid per-step torch.arange allocations)
    tree_idx: Tensor       # (N,) int64 - [0..N-1]

    # Configuration
    c_puct: float = 1.414

    # Original hands for oracle queries (N, 4, 7)
    original_hands: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return self.n_trees


def create_forest(
    n_trees: int,
    max_nodes: int,
    initial_states: GPUGameState,
    device: torch.device,
    c_puct: float = 1.414,
    original_hands: Tensor | None = None,
) -> GPUMCTSForest:
    """Create a forest with root nodes initialized to initial_states.

    Args:
        n_trees: Number of parallel trees (must match initial_states.batch_size)
        max_nodes: Maximum nodes per tree
        initial_states: GPUGameState with N games for root nodes
        device: Target device
        c_puct: Exploration constant for UCB
        original_hands: Optional (N, 4, 7) tensor of original deal hands.
                        If None, uses initial_states.hands (only valid at game start).
                        Pass this when creating forest mid-game for oracle queries.

    Returns:
        GPUMCTSForest ready for search
    """
    device = require_cuda(device, where="create_forest")
    require_cuda_tensor(initial_states.hands, where="create_forest", name="initial_states")
    if original_hands is not None:
        require_cuda_tensor(original_hands, where="create_forest", name="original_hands")

    assert initial_states.batch_size == n_trees, (
        f"initial_states batch size {initial_states.batch_size} != n_trees {n_trees}"
    )

    # Allocate node statistics
    visit_counts = torch.zeros(n_trees, max_nodes, dtype=torch.int32, device=device)
    value_sums = torch.zeros(n_trees, max_nodes, dtype=torch.float32, device=device)

    # Tree structure
    parents = torch.full((n_trees, max_nodes), -1, dtype=torch.int64, device=device)
    depths = torch.zeros(n_trees, max_nodes, dtype=torch.int32, device=device)

    # Children pointers - 7 slots per node
    children = torch.full((n_trees, max_nodes, 7), -1, dtype=torch.int64, device=device)
    child_actions = torch.full((n_trees, max_nodes, 7), -1, dtype=torch.int32, device=device)

    # Cached node properties - avoid recomputing in select_leaves_gpu
    n_legal = torch.zeros(n_trees, max_nodes, dtype=torch.int64, device=device)
    is_terminal_tensor = torch.zeros(n_trees, max_nodes, dtype=torch.bool, device=device)
    to_play = torch.zeros(n_trees, max_nodes, dtype=torch.int32, device=device)

    # Policy priors - initialized to uniform, updated by neural network
    # priors[tree, node, slot] = prior probability for action at slot
    priors = torch.zeros(n_trees, max_nodes, 7, dtype=torch.float32, device=device)

    # Initialize root node properties
    root_terminal = is_terminal_gpu(initial_states)  # (N,)
    root_legal_mask = legal_actions_gpu(initial_states)  # (N, 7)
    root_n_legal = root_legal_mask.sum(dim=1)  # (N,) int64
    root_to_play = current_player_gpu(initial_states)  # (N,)

    n_legal[:, 0] = root_n_legal
    is_terminal_tensor[:, 0] = root_terminal
    to_play[:, 0] = root_to_play

    # Initialize root priors to uniform over legal actions
    # Will be overwritten by neural network if set_node_priors is called
    root_priors = root_legal_mask.float()
    root_priors = root_priors / root_priors.sum(dim=1, keepdim=True).clamp(min=1e-8)
    priors[:, 0, :] = root_priors

    # Pre-allocate state storage for all N*M potential nodes
    # Initialize with copies of initial states (only root nodes are valid initially)
    total_nodes = n_trees * max_nodes

    # Expand initial_states to fill the entire node pool
    # Each tree's initial state goes into index [tree_idx * max_nodes + 0]
    # We'll use torch.repeat_interleave to tile the states
    states = _allocate_state_pool(initial_states, max_nodes, device)

    # Track allocation: each tree starts with 1 node (the root)
    n_nodes = torch.ones(n_trees, dtype=torch.int64, device=device)

    tree_idx = torch.arange(n_trees, device=device, dtype=torch.int64)

    # Store original hands for oracle queries
    # Use provided original_hands if available, otherwise clone from initial_states
    if original_hands is not None:
        stored_original_hands = original_hands.clone()
    else:
        stored_original_hands = initial_states.hands.clone()

    return GPUMCTSForest(
        n_trees=n_trees,
        max_nodes=max_nodes,
        device=device,
        visit_counts=visit_counts,
        value_sums=value_sums,
        parents=parents,
        depths=depths,
        children=children,
        child_actions=child_actions,
        n_legal=n_legal,
        is_terminal=is_terminal_tensor,
        to_play=to_play,
        priors=priors,
        states=states,
        n_nodes=n_nodes,
        tree_idx=tree_idx,
        c_puct=c_puct,
        original_hands=stored_original_hands,
    )


def _allocate_state_pool(
    initial_states: GPUGameState,
    max_nodes: int,
    device: torch.device,
) -> GPUGameState:
    """Allocate state pool with initial states at root positions.

    Creates a flattened state array of size N*M where:
    - states[tree_idx * max_nodes + 0] = initial_states[tree_idx]
    - Other positions are initialized but not yet valid
    """
    n_trees = initial_states.batch_size
    total_nodes = n_trees * max_nodes

    # Allocate tensors for all nodes
    # Initialize with zeros/defaults - only root positions will be valid initially
    hands = torch.zeros(total_nodes, 4, 7, dtype=torch.int32, device=device)
    played_mask = torch.zeros(total_nodes, 28, dtype=torch.bool, device=device)
    play_history = torch.zeros(total_nodes, 28, 3, dtype=torch.int32, device=device)
    n_plays = torch.zeros(total_nodes, dtype=torch.int32, device=device)
    current_trick = torch.zeros(total_nodes, 4, 2, dtype=torch.int32, device=device)
    trick_len = torch.zeros(total_nodes, dtype=torch.int32, device=device)
    leader = torch.zeros(total_nodes, dtype=torch.int32, device=device)
    decl_id = torch.zeros(total_nodes, dtype=torch.int32, device=device)
    scores = torch.zeros(total_nodes, 2, dtype=torch.int32, device=device)

    # Copy initial states to root positions (index 0 in each tree's node pool)
    root_indices = torch.arange(n_trees, device=device) * max_nodes  # [0, M, 2M, ...]

    hands[root_indices] = initial_states.hands
    played_mask[root_indices] = initial_states.played_mask
    play_history[root_indices] = initial_states.play_history
    n_plays[root_indices] = initial_states.n_plays
    current_trick[root_indices] = initial_states.current_trick
    trick_len[root_indices] = initial_states.trick_len
    leader[root_indices] = initial_states.leader
    decl_id[root_indices] = initial_states.decl_id
    scores[root_indices] = initial_states.scores

    return GPUGameState(
        hands=hands,
        played_mask=played_mask,
        play_history=play_history,
        n_plays=n_plays,
        current_trick=current_trick,
        trick_len=trick_len,
        leader=leader,
        decl_id=decl_id,
        scores=scores,
    )


def _get_node_states(
    forest: GPUMCTSForest,
    node_indices: Tensor,
    tree_indices: Tensor | None = None,
) -> GPUGameState:
    """Extract states for specific nodes.

    Args:
        forest: The MCTS forest
        node_indices: (K,) int32 - node index within each tree
        tree_indices: (K,) int64 - which tree each node belongs to.
                      If None, assumes K=N and uses arange(N).

    Returns:
        GPUGameState with K states
    """
    device = forest.device

    # Compute flat indices: tree_idx * max_nodes + node_idx
    if tree_indices is None:
        tree_indices = forest.tree_idx
    flat_indices = tree_indices * forest.max_nodes + node_indices

    return GPUGameState(
        hands=forest.states.hands[flat_indices],
        played_mask=forest.states.played_mask[flat_indices],
        play_history=forest.states.play_history[flat_indices],
        n_plays=forest.states.n_plays[flat_indices],
        current_trick=forest.states.current_trick[flat_indices],
        trick_len=forest.states.trick_len[flat_indices],
        leader=forest.states.leader[flat_indices],
        decl_id=forest.states.decl_id[flat_indices],
        scores=forest.states.scores[flat_indices],
    )


def _set_node_states(
    forest: GPUMCTSForest,
    node_indices: Tensor,
    states: GPUGameState,
) -> None:
    """Set states for specific nodes (in-place).

    Args:
        forest: The MCTS forest
        node_indices: (N,) int32 - node index within each tree
        states: GPUGameState with N states to write
    """
    n = forest.n_trees
    device = forest.device

    # Compute flat indices
    tree_indices = forest.tree_idx
    flat_indices = tree_indices * forest.max_nodes + node_indices

    forest.states.hands[flat_indices] = states.hands
    forest.states.played_mask[flat_indices] = states.played_mask
    forest.states.play_history[flat_indices] = states.play_history
    forest.states.n_plays[flat_indices] = states.n_plays
    forest.states.current_trick[flat_indices] = states.current_trick
    forest.states.trick_len[flat_indices] = states.trick_len
    forest.states.leader[flat_indices] = states.leader
    forest.states.decl_id[flat_indices] = states.decl_id
    forest.states.scores[flat_indices] = states.scores


def _compute_ucb_scores(
    visit_counts: Tensor,      # (N, 7) child visit counts (int32)
    value_sums: Tensor,        # (N, 7) child value sums (float32)
    parent_visits: Tensor,     # (N,) parent visit counts (int32)
    priors: Tensor,            # (N, 7) policy priors from neural network
    c_puct: float,
) -> Tensor:
    """Compute UCB scores for child selection.

    UCB = Q + c_puct * prior * sqrt(parent_visits) / (1 + visits)

    where prior comes from neural network policy head (not uniform).

    Args:
        visit_counts: (N, 7) visit counts for each child slot
        value_sums: (N, 7) value sums for each child slot
        parent_visits: (N,) visit counts of parent nodes
        priors: (N, 7) policy priors from neural network
        c_puct: Exploration constant

    Returns:
        UCB scores (N, 7)
    """
    # Q-value: value_sum / visit_count (0 if unvisited)
    q_value = torch.where(
        visit_counts > 0,
        value_sums / visit_counts.float(),
        torch.zeros_like(value_sums),
    )

    # Expand parent_visits to (N, 7)
    parent_visits = parent_visits.unsqueeze(-1)  # (N, 1)

    # Exploration term: c_puct * prior * sqrt(parent_visits) / (1 + visits)
    exploration = (
        c_puct * priors * torch.sqrt(parent_visits.float()) /
        (1.0 + visit_counts.float())
    )

    # UCB score
    ucb = q_value + exploration

    # Unvisited nodes get boosted by their prior (not infinite, so prior matters)
    # This is key: high-prior unvisited actions get explored first
    ucb = torch.where(
        visit_counts == 0,
        priors * 1000.0,  # Large but finite, so prior ordering matters
        ucb
    )

    return ucb


def select_leaves_gpu(
    forest: GPUMCTSForest,
    max_depth: int = 28,
) -> tuple[Tensor, Tensor]:
    """Select leaves for all N trees in parallel using UCB.

    Traverses from root to leaf for each tree, applying virtual loss
    (incrementing visit counts) along the way to encourage diversity.

    A node is only descended if it's "fully expanded" (all legal actions have
    children), matching Python MCTS behavior. This ensures the tree grows wide
    before deep when using single-child expansion.

    Args:
        forest: The MCTS forest
        max_depth: Maximum depth to traverse (game has max 28 moves)

    Returns:
        leaf_indices: (N,) int64 - node index of selected leaf per tree
        paths: (N, max_depth) int64 - node indices along path, -1 padded
    """
    n = forest.n_trees
    device = forest.device

    # Start at root (node 0) for all trees
    current_nodes = torch.zeros(n, dtype=torch.int64, device=device)

    # Track paths
    paths = torch.full((n, max_depth), -1, dtype=torch.int64, device=device)
    path_lengths = torch.zeros(n, dtype=torch.int64, device=device)

    # Track which trees are still traversing
    active = torch.ones(n, dtype=torch.bool, device=device)

    batch_idx = forest.tree_idx

    for depth in range(max_depth):
        # CUDA-only: do not add CPU-only early-exit fallbacks here.

        # Record current node in path
        paths[batch_idx, path_lengths] = torch.where(
            active, current_nodes, paths[batch_idx, path_lengths]
        )
        path_lengths = torch.where(active, path_lengths + 1, path_lengths)

        # Apply virtual loss to current nodes
        forest.visit_counts[batch_idx, current_nodes] += active.to(forest.visit_counts.dtype)

        # Get children of current nodes: (N, 7)
        current_children = forest.children[batch_idx, current_nodes]  # (N, 7)

        # Check if node is fully expanded using cached n_legal and is_terminal
        # (avoids expensive _get_node_states, is_terminal_gpu, legal_actions_gpu calls)
        terminal = forest.is_terminal[batch_idx, current_nodes]  # (N,)
        node_n_legal = forest.n_legal[batch_idx, current_nodes]  # (N,)
        n_children = (current_children >= 0).sum(dim=1)  # (N,) int64

        # Node is fully expanded if it has all legal children (or is terminal)
        fully_expanded = (n_children >= node_n_legal) | terminal  # (N,)

        # Only descend through fully expanded nodes
        has_children = (current_children >= 0).any(dim=1)  # (N,)
        should_descend = active & fully_expanded & has_children

        # For nodes we'll descend, select best child using UCB
        # Create mask for valid children
        valid_children = current_children >= 0  # (N, 7)

        # Get child visit counts and value sums
        # Safe indexing: clamp negative indices to 0 for gather, then mask results
        safe_child_indices = current_children.clamp(min=0)  # (N, 7) int64

        child_visits = forest.visit_counts.gather(
            1, safe_child_indices
        )  # (N, 7)
        child_values = forest.value_sums.gather(
            1, safe_child_indices
        )  # (N, 7)

        # Get parent (current node) visit counts for UCB
        parent_visits = forest.visit_counts[batch_idx, current_nodes]  # (N,) float32

        # Get policy priors for current nodes (from neural network)
        node_priors = forest.priors[batch_idx, current_nodes]  # (N, 7)

        # Compute UCB scores for all children using policy priors
        ucb_scores = _compute_ucb_scores(
            child_visits, child_values, parent_visits, node_priors, forest.c_puct
        )  # (N, 7)

        # Mask invalid children with -inf
        ucb_scores = ucb_scores.masked_fill(~valid_children, float("-inf"))

        # Select best child
        best_child_slot = ucb_scores.argmax(dim=1)  # (N,)
        best_child_node = current_children[batch_idx, best_child_slot]  # (N,)

        # Update current nodes (only for trees that should descend)
        current_nodes = torch.where(should_descend, best_child_node, current_nodes)

        # Trees that can't descend (not fully expanded or no children) become inactive
        active = should_descend

    return current_nodes, paths


def expand_gpu(
    forest: GPUMCTSForest,
    leaf_indices: Tensor,
    leaf_states: GPUGameState | None = None,
) -> Tensor:
    """Expand leaves by creating ONE child node per tree.

    Matches Python MCTS _expand() behavior: finds the first unexplored legal
    action and creates a single child. This is more efficient than expanding
    all children at once (100 sims = ~100 nodes instead of ~700).

    Args:
        forest: The MCTS forest
        leaf_indices: (N,) int64 - node indices of leaves to expand

    Returns:
        n_expanded: (N,) int32 - 1 if child created, 0 if terminal/fully-expanded
    """
    n = forest.n_trees
    device = forest.device
    batch_idx = forest.tree_idx

    # Get states for the leaves (used for legal mask + transition)
    if leaf_states is None:
        leaf_states = _get_node_states(forest, leaf_indices)

    # Use leaf state for terminal/to-play to avoid relying on cache correctness
    # (tests and some callers may mutate state tensors directly).
    terminal = leaf_states.n_plays == 28  # (N,)
    players_i32 = current_player_gpu(leaf_states)  # (N,) int32

    # Get legal actions for non-terminal leaves
    legal_mask = legal_actions_gpu(leaf_states)  # (N, 7) bool

    # Don't expand terminal nodes
    legal_mask = legal_mask & ~terminal.unsqueeze(1)

    # Check which slots already have children (already expanded)
    existing_children = forest.children[batch_idx, leaf_indices, :]  # (N, 7)
    has_child = existing_children >= 0  # (N, 7) - True if slot already has a child

    # Find unexplored legal actions: legal AND not yet expanded
    unexplored_legal = legal_mask & ~has_child  # (N, 7)

    # Find the FIRST unexplored legal slot for each tree
    # Use argmax on int tensor - returns index of first True (1)
    # But argmax returns 0 if all are False, so we need to check if any exist
    has_unexplored = unexplored_legal.any(dim=1)  # (N,) - True if tree has unexplored action

    # Also check room constraints
    room_available = forest.max_nodes - forest.n_nodes  # (N,)
    can_expand = has_unexplored & (room_available > 0)  # (N,)

    # Find the first unexplored slot index for each tree
    # argmax returns index of first True when converted to int
    first_slot = unexplored_legal.int().argmax(dim=1)  # (N,) int64

    # Allocate new node indices (fixed-shape, no compaction).
    # new_node_idx is the allocation point BEFORE incrementing n_nodes.
    new_node_idx = forest.n_nodes.clone()  # (N,) int64
    safe_new_node_idx = new_node_idx.clamp(min=0, max=forest.max_nodes - 1)  # (N,) int64

    # Further guard: a "legal but empty" slot would yield domino_id == -1.
    players_i64 = players_i32.to(torch.int64)
    slots_i64 = first_slot.to(torch.int64)
    domino_ids = leaf_states.hands[batch_idx, players_i64, slots_i64]  # (N,) int32
    can_expand = can_expand & (domino_ids >= 0)

    # Advance allocation pointer for expanding trees only (after all guards).
    forest.n_nodes.add_(can_expand.to(dtype=forest.n_nodes.dtype))

    # Update child link in parent (children + child_actions) without compaction.
    # Use fixed-length index vectors and no-op writeback for non-expanding rows.
    old_child_ptr = forest.children[batch_idx, leaf_indices, first_slot]  # (N,) int64
    new_child_ptr = torch.where(can_expand, new_node_idx, old_child_ptr)
    forest.children[batch_idx, leaf_indices, first_slot] = new_child_ptr

    old_child_action = forest.child_actions[batch_idx, leaf_indices, first_slot]  # (N,) int32
    new_child_action = torch.where(
        can_expand,
        first_slot.to(torch.int32),
        old_child_action,
    )
    forest.child_actions[batch_idx, leaf_indices, first_slot] = new_child_action

    # --- Child transition (masked, fixed-shape) ---
    # Reuse leaf_states tensors as a scratch child state by applying masked updates in-place.
    # This avoids apply_action_gpu() cloning and avoids any dynamic-shape indexing.
    safe_domino_i64 = domino_ids.to(torch.int64).clamp(min=0, max=27)  # (N,)
    safe_play_idx = leaf_states.n_plays.to(torch.int64).clamp(min=0, max=27)  # (N,)
    safe_trick_pos = leaf_states.trick_len.to(torch.int64).clamp(min=0, max=3)  # (N,)

    # Remove domino from hand (set to -1)
    old_hand_val = leaf_states.hands[batch_idx, players_i64, slots_i64]
    leaf_states.hands[batch_idx, players_i64, slots_i64] = torch.where(
        can_expand,
        torch.full_like(old_hand_val, -1),
        old_hand_val,
    )

    # Mark domino as played
    old_played = leaf_states.played_mask[batch_idx, safe_domino_i64]
    leaf_states.played_mask[batch_idx, safe_domino_i64] = torch.where(
        can_expand,
        torch.ones_like(old_played, dtype=torch.bool),
        old_played,
    )

    # Determine lead domino for this play
    is_leading = leaf_states.trick_len == 0
    lead_domino = torch.where(is_leading, domino_ids, leaf_states.current_trick[:, 0, 1])

    # Add to play history (masked, safe index)
    old_ph0 = leaf_states.play_history[batch_idx, safe_play_idx, 0]
    old_ph1 = leaf_states.play_history[batch_idx, safe_play_idx, 1]
    old_ph2 = leaf_states.play_history[batch_idx, safe_play_idx, 2]
    leaf_states.play_history[batch_idx, safe_play_idx, 0] = torch.where(can_expand, players_i32, old_ph0)
    leaf_states.play_history[batch_idx, safe_play_idx, 1] = torch.where(can_expand, domino_ids, old_ph1)
    leaf_states.play_history[batch_idx, safe_play_idx, 2] = torch.where(can_expand, lead_domino, old_ph2)
    leaf_states.n_plays = leaf_states.n_plays + can_expand.to(dtype=leaf_states.n_plays.dtype)

    # Add to current trick (masked, safe index)
    old_ct0 = leaf_states.current_trick[batch_idx, safe_trick_pos, 0]
    old_ct1 = leaf_states.current_trick[batch_idx, safe_trick_pos, 1]
    leaf_states.current_trick[batch_idx, safe_trick_pos, 0] = torch.where(can_expand, players_i32, old_ct0)
    leaf_states.current_trick[batch_idx, safe_trick_pos, 1] = torch.where(can_expand, domino_ids, old_ct1)
    leaf_states.trick_len = leaf_states.trick_len + can_expand.to(dtype=leaf_states.trick_len.dtype)

    # Resolve completed tricks (masked; avoid Python control-flow on tensors).
    trick_complete = leaf_states.trick_len == 4
    resolve_mask = trick_complete & can_expand
    tables = get_domino_tables(device)

    trick_dominoes = leaf_states.current_trick[:, :, 1]  # (N, 4)
    lead_domino_complete = leaf_states.current_trick[:, 0, 1]  # (N,)
    led_suit = led_suit_for_lead_domino_gpu(lead_domino_complete, leaf_states.decl_id, tables)  # (N,)

    led_suit_exp = led_suit.unsqueeze(1).expand(-1, 4)
    decl_exp = leaf_states.decl_id.unsqueeze(1).expand(-1, 4)

    ranks = trick_rank_gpu(trick_dominoes, led_suit_exp, decl_exp, tables)  # (N, 4)
    winner_offset = ranks.argmax(dim=1).to(torch.int32)  # (N,)
    winner_player = ((leaf_states.leader + winner_offset) % 4).to(torch.int32)  # (N,)

    # Compute trick points: 1 base + count points
    safe_ids = trick_dominoes.clamp(min=0)
    points = tables.points[safe_ids].sum(dim=1) + 1  # (N,) int32

    winner_team = winner_player % 2
    team0_points = torch.where(winner_team == 0, points, torch.zeros_like(points))
    team1_points = torch.where(winner_team == 1, points, torch.zeros_like(points))

    leaf_states.leader = torch.where(resolve_mask, winner_player, leaf_states.leader)
    leaf_states.trick_len = torch.where(
        resolve_mask,
        torch.zeros_like(leaf_states.trick_len),
        leaf_states.trick_len,
    )
    leaf_states.scores[:, 0] = leaf_states.scores[:, 0] + torch.where(
        resolve_mask, team0_points, torch.zeros_like(team0_points)
    )
    leaf_states.scores[:, 1] = leaf_states.scores[:, 1] + torch.where(
        resolve_mask, team1_points, torch.zeros_like(team1_points)
    )

    # Clear trick for resolved rows (masked_fill avoids allocating a full zeros_like tensor).
    leaf_states.current_trick.masked_fill_(resolve_mask.view(-1, 1, 1), 0)

    # --- Commit child states to forest pool (masked no-op writeback) ---
    flat_child = batch_idx * forest.max_nodes + safe_new_node_idx  # (N,) int64

    def _write_pool(field: Tensor, values: Tensor) -> None:
        old = field[flat_child]
        if values.ndim > 1:
            mask = can_expand.view(-1, *([1] * (values.ndim - 1)))
        else:
            mask = can_expand
        write_values = torch.where(mask, values, old)
        field.index_copy_(0, flat_child, write_values)

    _write_pool(forest.states.hands, leaf_states.hands)
    _write_pool(forest.states.played_mask, leaf_states.played_mask)
    _write_pool(forest.states.play_history, leaf_states.play_history)
    _write_pool(forest.states.n_plays, leaf_states.n_plays)
    _write_pool(forest.states.current_trick, leaf_states.current_trick)
    _write_pool(forest.states.trick_len, leaf_states.trick_len)
    _write_pool(forest.states.leader, leaf_states.leader)
    _write_pool(forest.states.decl_id, leaf_states.decl_id)
    _write_pool(forest.states.scores, leaf_states.scores)

    # --- Update cached node properties for new children (masked) ---
    child_terminal = leaf_states.n_plays == 28  # (N,)
    child_legal_mask = legal_actions_gpu(leaf_states)  # (N, 7)
    child_n_legal = child_legal_mask.sum(dim=1)  # (N,) int64
    child_to_play = ((leaf_states.leader + leaf_states.trick_len) % 4).to(torch.int32)  # (N,)

    # Uniform priors over legal actions
    child_priors = child_legal_mask.float()
    child_priors = child_priors / child_priors.sum(dim=1, keepdim=True).clamp(min=1e-8)

    flat_nodes = batch_idx * forest.max_nodes + safe_new_node_idx  # (N,)

    def _write_node_flat(flat_field: Tensor, values_1d: Tensor) -> None:
        old = flat_field[flat_nodes]
        write_values = torch.where(can_expand, values_1d, old)
        flat_field.index_copy_(0, flat_nodes, write_values)

    _write_node_flat(forest.parents.view(-1), leaf_indices)
    _write_node_flat(forest.is_terminal.view(-1), child_terminal)
    _write_node_flat(forest.n_legal.view(-1), child_n_legal)
    _write_node_flat(forest.to_play.view(-1), child_to_play)

    # depths is int32; update with masked writeback
    parent_depths = forest.depths[batch_idx, leaf_indices]  # (N,) int32
    new_depths = parent_depths + 1
    depths_flat = forest.depths.view(-1)
    old_depth = depths_flat[flat_nodes]
    depths_flat.index_copy_(0, flat_nodes, torch.where(can_expand, new_depths, old_depth))

    # Priors: (N*M, 7) update at flat_nodes
    priors_flat = forest.priors.view(-1, 7)
    old_priors = priors_flat[flat_nodes]
    priors_flat.index_copy_(
        0,
        flat_nodes,
        torch.where(can_expand.view(-1, 1), child_priors, old_priors),
    )

    return can_expand.to(torch.int32)


def reset_forest_inplace(
    forest: GPUMCTSForest,
    initial_states: GPUGameState,
    original_hands: Tensor | None = None,
) -> None:
    """Reset an existing forest in-place for a new root position (CUDA-only).

    This enables reusing preallocated tensors (and CUDA graphs) across moves.
    It does NOT allocate new tensors for any forest fields used in hot paths.
    """
    require_cuda_tensor(initial_states.hands, where="reset_forest_inplace(initial_states.hands)")
    if initial_states.batch_size != forest.n_trees:
        raise ValueError(
            f"initial_states.batch_size ({initial_states.batch_size}) must match forest.n_trees ({forest.n_trees})"
        )
    states_device = initial_states.device
    if states_device.type != forest.device.type:
        raise ValueError(
            f"initial_states.device.type ({states_device.type}) must match forest.device.type ({forest.device.type})"
        )
    if forest.device.index is not None and states_device.index != forest.device.index:
        raise ValueError(
            f"initial_states.device ({states_device}) must match forest.device ({forest.device})"
        )

    # Reset allocation counters.
    forest.n_nodes.fill_(1)

    # Clear tree stats/structure (fixed-shape).
    forest.visit_counts.zero_()
    forest.value_sums.zero_()
    forest.parents.fill_(-1)
    forest.depths.zero_()
    forest.children.fill_(-1)
    forest.child_actions.fill_(-1)

    # Reset state pool root nodes.
    root_flat = forest.tree_idx * forest.max_nodes
    forest.states.hands.index_copy_(0, root_flat, initial_states.hands)
    forest.states.played_mask.index_copy_(0, root_flat, initial_states.played_mask)
    forest.states.play_history.index_copy_(0, root_flat, initial_states.play_history)
    forest.states.n_plays.index_copy_(0, root_flat, initial_states.n_plays)
    forest.states.current_trick.index_copy_(0, root_flat, initial_states.current_trick)
    forest.states.trick_len.index_copy_(0, root_flat, initial_states.trick_len)
    forest.states.leader.index_copy_(0, root_flat, initial_states.leader)
    forest.states.decl_id.index_copy_(0, root_flat, initial_states.decl_id)
    forest.states.scores.index_copy_(0, root_flat, initial_states.scores)

    # Recompute cached node properties for roots only.
    root_terminal = is_terminal_gpu(initial_states)
    root_legal = legal_actions_gpu(initial_states)
    root_n_legal = root_legal.sum(dim=1)
    root_to_play = current_player_gpu(initial_states)

    forest.is_terminal[:, 0] = root_terminal
    forest.n_legal[:, 0] = root_n_legal
    forest.to_play[:, 0] = root_to_play

    # Reset priors to 0, then set uniform legal priors at root.
    forest.priors.zero_()
    root_priors = root_legal.float()
    root_priors = root_priors / root_priors.sum(dim=1, keepdim=True).clamp(min=1e-8)
    forest.priors[:, 0, :] = root_priors

    # Update original_hands if provided.
    if original_hands is not None:
        require_cuda_tensor(original_hands, where="reset_forest_inplace(original_hands)")
        if original_hands.shape != (forest.n_trees, 4, 7):
            raise ValueError(f"original_hands must have shape {(forest.n_trees, 4, 7)}, got {tuple(original_hands.shape)}")
        if forest.original_hands is None:
            forest.original_hands = original_hands.clone()
        else:
            forest.original_hands.copy_(original_hands)


class MCTSCUDAGraphRunner:
    """CUDA Graph runner for GPU-only MCTS.

    Captures the small-kernel-heavy parts of MCTS as two CUDA graphs:
    - Graph A: select -> gather leaf states -> terminal evaluation
    - Graph B: expand -> backprop (consumes oracle values buffer)

    Oracle/model evaluation stays outside the graphs so callers can use any
    runtime (PyTorch, TensorRT, etc.) without requiring graph-capture support.
    """

    def __init__(self, forest: GPUMCTSForest, max_depth: int = 28, *, pool=None):
        self.forest = forest
        self.max_depth = max_depth
        require_cuda(forest.device, where="MCTSCUDAGraphRunner.__init__")

        self._pool = pool if pool is not None else torch.cuda.graphs.graph_pool_handle()
        self._graph_select = torch.cuda.CUDAGraph()
        self._graph_update = torch.cuda.CUDAGraph()
        self._captured = False
        self._oracle_supports_indices: bool | None = None

        self.oracle_values = torch.zeros(
            forest.n_trees, dtype=torch.float32, device=forest.device
        )

        # Populated during capture.
        self.leaf_indices: Tensor | None = None
        self.paths: Tensor | None = None
        self.leaf_states: GPUGameState | None = None
        self.terminal_values: Tensor | None = None
        self.is_terminal: Tensor | None = None

    @staticmethod
    def _supports_state_leaf_tree_signature(fn: callable) -> bool:
        """Best-effort check whether fn can accept (states, leaf_indices, tree_indices).

        Falls back to False for callables without inspectable signatures.
        """
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return False

        params = list(sig.parameters.values())
        if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params):
            return True

        # Count positional params (positional-only or positional-or-keyword).
        positional = [
            p for p in params
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        return len(positional) >= 3

    def capture(self) -> None:
        if self._captured:
            return

        with torch.inference_mode():
            # Warm up kernels to reduce capture-time variability.
            torch.cuda.synchronize()
            for _ in range(2):
                leaf_indices, paths = select_leaves_gpu(self.forest, max_depth=self.max_depth)
                leaf_states = get_leaf_states(self.forest, leaf_indices)
                terminal_values, is_terminal = get_terminal_values(
                    self.forest, leaf_indices, leaf_states
                )
                values = torch.where(is_terminal, terminal_values, self.oracle_values)
                expand_gpu(self.forest, leaf_indices, leaf_states=leaf_states)
                backprop_gpu(self.forest, leaf_indices, values, paths)

            torch.cuda.synchronize()

            # Capture Graph A: select + gather + terminal values.
            with torch.cuda.graph(self._graph_select, pool=self._pool):
                leaf_indices, paths = select_leaves_gpu(self.forest, max_depth=self.max_depth)
                leaf_states = get_leaf_states(self.forest, leaf_indices)
                terminal_values, is_terminal = get_terminal_values(
                    self.forest, leaf_indices, leaf_states
                )

        # Stash outputs to be read after replay (these tensors are reused).
        self.leaf_indices = leaf_indices
        self.paths = paths
        self.leaf_states = leaf_states
        self.terminal_values = terminal_values
        self.is_terminal = is_terminal

        with torch.inference_mode():
            # Capture Graph B: expand + backprop using external oracle values buffer.
            with torch.cuda.graph(self._graph_update, pool=self._pool):
                values = torch.where(self.is_terminal, self.terminal_values, self.oracle_values)
                expand_gpu(self.forest, self.leaf_indices, leaf_states=self.leaf_states)
                backprop_gpu(self.forest, self.leaf_indices, values, self.paths)

        self._captured = True

    def step(self, oracle_fn: callable) -> None:
        if not self._captured:
            self.capture()

        with torch.inference_mode():
            self._graph_select.replay()

            # Skip oracle/model eval for terminal leaves (outside graphs).
            # This can materially reduce oracle work near the end of games.
            nonterm_idx = (~self.is_terminal).nonzero(as_tuple=True)[0]
            self.oracle_values.zero_()

            if nonterm_idx.numel() > 0:
                nonterm_states = GPUGameState(
                    hands=self.leaf_states.hands.index_select(0, nonterm_idx),
                    played_mask=self.leaf_states.played_mask.index_select(0, nonterm_idx),
                    play_history=self.leaf_states.play_history.index_select(0, nonterm_idx),
                    n_plays=self.leaf_states.n_plays.index_select(0, nonterm_idx),
                    current_trick=self.leaf_states.current_trick.index_select(0, nonterm_idx),
                    trick_len=self.leaf_states.trick_len.index_select(0, nonterm_idx),
                    leader=self.leaf_states.leader.index_select(0, nonterm_idx),
                    decl_id=self.leaf_states.decl_id.index_select(0, nonterm_idx),
                    scores=self.leaf_states.scores.index_select(0, nonterm_idx),
                )
                nonterm_leaf_indices = self.leaf_indices.index_select(0, nonterm_idx)

                if self._oracle_supports_indices is None:
                    self._oracle_supports_indices = self._supports_state_leaf_tree_signature(oracle_fn)

                if self._oracle_supports_indices:
                    oracle_out = oracle_fn(nonterm_states, nonterm_leaf_indices, nonterm_idx)
                else:
                    oracle_out = oracle_fn(nonterm_states)

                if oracle_out.shape != (nonterm_states.batch_size,):
                    raise ValueError(
                        f"oracle_fn must return shape {(nonterm_states.batch_size,)}, got {tuple(oracle_out.shape)}"
                    )
                if oracle_out.device != self.oracle_values.device:
                    raise ValueError(
                        f"oracle_fn must return on device {self.oracle_values.device}, got {oracle_out.device}"
                    )
                if oracle_out.dtype != self.oracle_values.dtype:
                    oracle_out = oracle_out.to(dtype=self.oracle_values.dtype)

                self.oracle_values.index_copy_(0, nonterm_idx, oracle_out)

            self._graph_update.replay()


class MCTSFullStepCUDAGraphRunner:
    """Single CUDA-graph runner for one full MCTS simulation step (incl oracle).

    This is Phase 4: capture select + oracle eval + expand + backprop in one replay.
    Uses fixed batch size N (n_trees) and evaluates oracle for all leaves, then
    overrides terminal leaves with terminal values.
    """

    def __init__(self, forest: GPUMCTSForest, oracle_value_fn, *, max_depth: int = 28, pool=None):
        self.forest = forest
        self.oracle_value_fn = oracle_value_fn
        self.max_depth = max_depth
        require_cuda(forest.device, where="MCTSFullStepCUDAGraphRunner.__init__")

        stage1 = getattr(oracle_value_fn, "oracle", None)
        if stage1 is None or not hasattr(stage1, "gpu_tokenizer"):
            raise ValueError("oracle_value_fn must wrap a Stage1Oracle with gpu_tokenizer for Phase 4 capture")
        self.stage1_oracle = stage1

        self._pool = pool if pool is not None else torch.cuda.graphs.graph_pool_handle()
        self._graph = torch.cuda.CUDAGraph()
        self._captured = False

        # Cached constants/buffers.
        self._position_idx = torch.arange(3, device=forest.device, dtype=torch.int64).unsqueeze(0)  # (1, 3)

        # Outputs (populated on capture; reused on replay).
        self.leaf_indices: Tensor | None = None
        self.paths: Tensor | None = None
        self.leaf_states: GPUGameState | None = None
        self.terminal_values: Tensor | None = None
        self.is_terminal: Tensor | None = None
        self.values: Tensor | None = None

    @property
    def captured(self) -> bool:
        return self._captured

    def capture(self) -> None:
        if self._captured:
            return

        # Ensure tokenizer buffers are sized before capture (avoid realloc during capture).
        n = self.forest.n_trees
        dummy_worlds = self.forest.original_hands
        dummy_decl = torch.zeros(n, dtype=torch.int32, device=self.forest.device)
        dummy_players = torch.zeros(n, dtype=torch.int32, device=self.forest.device)
        dummy_leaders = torch.zeros(n, dtype=torch.int32, device=self.forest.device)
        dummy_remaining = torch.zeros(n, 4, dtype=torch.int64, device=self.forest.device)
        dummy_tp = torch.full((n, 3), -1, dtype=torch.int32, device=self.forest.device)
        dummy_td = torch.full((n, 3), -1, dtype=torch.int32, device=self.forest.device)
        _ = self.stage1_oracle.gpu_tokenizer.tokenize(
            worlds=dummy_worlds,
            decl_ids=dummy_decl,
            actors=dummy_players,
            leaders=dummy_leaders,
            remaining=dummy_remaining,
            trick_players=dummy_tp,
            trick_dominoes=dummy_td,
        )

        # Warm up one step to stabilize capture.
        with torch.inference_mode():
            leaf_indices, paths = select_leaves_gpu(self.forest, max_depth=self.max_depth)
            leaf_states = get_leaf_states(self.forest, leaf_indices)
            terminal_values, is_terminal = get_terminal_values(self.forest, leaf_indices, leaf_states)
            _ = terminal_values + is_terminal.float()
            # Oracle eval warmup
            actors = current_player_gpu(leaf_states)
            tp = leaf_states.current_trick[:, :3, 0]
            td = leaf_states.current_trick[:, :3, 1]
            empty = self._position_idx >= leaf_states.trick_len.unsqueeze(1)
            tp = torch.where(empty, torch.full_like(tp, -1), tp)
            td = torch.where(empty, torch.full_like(td, -1), td)
            remaining = compute_remaining_bitmask_gpu(self.forest.original_hands, leaf_states.hands)
            tokens, masks = self.stage1_oracle.gpu_tokenizer.tokenize(
                worlds=self.forest.original_hands,
                decl_ids=leaf_states.decl_id,
                actors=actors,
                leaders=leaf_states.leader,
                remaining=remaining,
                trick_players=tp,
                trick_dominoes=td,
            )
            q_values, _ = self.stage1_oracle.model(tokens, masks, actors.long())
            legal_mask = compute_legal_mask_gpu(
                self.forest.original_hands,
                leaf_states.hands,
                actors,
                batch_idx=self.forest.tree_idx,
            )
            best_q = q_values.masked_fill(~legal_mask, float("-inf")).max(dim=1).values
            root_players = self.forest.to_play[:, 0]
            best_q = torch.where((root_players % 2) == 1, -best_q, best_q)
            oracle_values = best_q / 42.0
            values = torch.where(is_terminal, terminal_values, oracle_values)
            expand_gpu(self.forest, leaf_indices, leaf_states=leaf_states)
            backprop_gpu(self.forest, leaf_indices, values, paths)

        torch.cuda.synchronize()

        with torch.inference_mode():
            with torch.cuda.graph(self._graph, pool=self._pool):
                leaf_indices, paths = select_leaves_gpu(self.forest, max_depth=self.max_depth)
                leaf_states = get_leaf_states(self.forest, leaf_indices)
                terminal_values, is_terminal = get_terminal_values(self.forest, leaf_indices, leaf_states)

                actors = current_player_gpu(leaf_states)

                tp = leaf_states.current_trick[:, :3, 0]
                td = leaf_states.current_trick[:, :3, 1]
                empty = self._position_idx >= leaf_states.trick_len.unsqueeze(1)
                tp = torch.where(empty, torch.full_like(tp, -1), tp)
                td = torch.where(empty, torch.full_like(td, -1), td)

                remaining = compute_remaining_bitmask_gpu(self.forest.original_hands, leaf_states.hands)
                tokens, masks = self.stage1_oracle.gpu_tokenizer.tokenize(
                    worlds=self.forest.original_hands,
                    decl_ids=leaf_states.decl_id,
                    actors=actors,
                    leaders=leaf_states.leader,
                    remaining=remaining,
                    trick_players=tp,
                    trick_dominoes=td,
                )

                q_values, _ = self.stage1_oracle.model(tokens, masks, actors.long())
                legal_mask = compute_legal_mask_gpu(
                    self.forest.original_hands,
                    leaf_states.hands,
                    actors,
                    batch_idx=self.forest.tree_idx,
                )
                best_q = q_values.masked_fill(~legal_mask, float("-inf")).max(dim=1).values
                root_players = self.forest.to_play[:, 0]
                best_q = torch.where((root_players % 2) == 1, -best_q, best_q)
                oracle_values = best_q / 42.0

                values = torch.where(is_terminal, terminal_values, oracle_values)

                expand_gpu(self.forest, leaf_indices, leaf_states=leaf_states)
                backprop_gpu(self.forest, leaf_indices, values, paths)

        self.leaf_indices = leaf_indices
        self.paths = paths
        self.leaf_states = leaf_states
        self.terminal_values = terminal_values
        self.is_terminal = is_terminal
        self.values = values
        self._captured = True

    def step(self) -> None:
        if not self._captured:
            self.capture()
        with torch.inference_mode():
            self._graph.replay()


class MCTSSelfPlayFullStepCUDAGraphRunner:
    """Single CUDA-graph runner for one full MCTS simulation step in self-play mode.

    Captures: select + tokenize + ZebModel forward (policy+value) + set priors +
    expand + backprop. Uses fixed batch size N (n_trees) and evaluates the model
    for all leaves, overriding terminal leaves with terminal values.
    """

    def __init__(self, forest: GPUMCTSForest, model, tokenizer, *, max_depth: int = 28, pool=None):
        self.forest = forest
        self.model = model
        self.tokenizer = tokenizer
        self.max_depth = max_depth
        require_cuda(forest.device, where="MCTSSelfPlayFullStepCUDAGraphRunner.__init__")

        if not hasattr(tokenizer, "tokenize_batch_into"):
            raise ValueError("tokenizer must provide tokenize_batch_into(...) for self-play full-step capture")

        self._pool = pool if pool is not None else torch.cuda.graphs.graph_pool_handle()
        self._graph = torch.cuda.CUDAGraph()
        self._captured = False

        n = forest.n_trees
        max_tokens = int(getattr(tokenizer, "MAX_TOKENS", 36))
        n_features = int(getattr(tokenizer, "N_FEATURES", 8))

        # Preallocated tokenizer outputs (reused on replay).
        self._tokens_i32 = torch.empty((n, max_tokens, n_features), dtype=torch.int32, device=forest.device)
        self._masks = torch.empty((n, max_tokens), dtype=torch.bool, device=forest.device)
        self._hand_masks = torch.empty((n, 7), dtype=torch.bool, device=forest.device)
        self._hand_indices = (
            torch.arange(1, 8, dtype=torch.int64, device=forest.device).unsqueeze(0).expand(n, -1).contiguous()
        )

        # Populated during capture.
        self.leaf_indices: Tensor | None = None
        self.paths: Tensor | None = None
        self.leaf_states: GPUGameState | None = None
        self.terminal_values: Tensor | None = None
        self.is_terminal: Tensor | None = None
        self.values: Tensor | None = None

    @property
    def captured(self) -> bool:
        return self._captured

    def capture(self) -> None:
        if self._captured:
            return

        self.model.eval()

        # Warm up one step to stabilize capture (and ensure tokenizer outputs are resident).
        with torch.inference_mode():
            leaf_indices, paths = select_leaves_gpu(self.forest, max_depth=self.max_depth)
            leaf_states = get_leaf_states(self.forest, leaf_indices)
            terminal_values, is_terminal = get_terminal_values(self.forest, leaf_indices, leaf_states)
            actors = current_player_gpu(leaf_states)
            _ = self.tokenizer.tokenize_batch_into(
                leaf_states,
                self.forest.original_hands,
                actors,
                out_tokens=self._tokens_i32,
                out_masks=self._masks,
                out_hand_masks=self._hand_masks,
                batch_idx=self.forest.tree_idx,
            )
            policy_logits, values = self.model(self._tokens_i32.long(), self._masks, self._hand_indices, self._hand_masks)
            _ = policy_logits + values.unsqueeze(1)
            values = torch.where(is_terminal, terminal_values, values)
            expand_gpu(self.forest, leaf_indices, leaf_states=leaf_states)
            backprop_gpu(self.forest, leaf_indices, values, paths)

        torch.cuda.synchronize()

        with torch.inference_mode():
            with torch.cuda.graph(self._graph, pool=self._pool):
                leaf_indices, paths = select_leaves_gpu(self.forest, max_depth=self.max_depth)
                leaf_states = get_leaf_states(self.forest, leaf_indices)
                terminal_values, is_terminal = get_terminal_values(self.forest, leaf_indices, leaf_states)

                actors = current_player_gpu(leaf_states)
                _ = self.tokenizer.tokenize_batch_into(
                    leaf_states,
                    self.forest.original_hands,
                    actors,
                    out_tokens=self._tokens_i32,
                    out_masks=self._masks,
                    out_hand_masks=self._hand_masks,
                    batch_idx=self.forest.tree_idx,
                )

                policy_logits, values = self.model(
                    self._tokens_i32.long(),
                    self._masks,
                    self._hand_indices,
                    self._hand_masks,
                )

                # Policy -> priors for this node.
                policy_logits = policy_logits.masked_fill(~self._hand_masks, float("-inf"))
                all_masked = self._hand_masks.sum(dim=-1) == 0
                safe_logits = torch.where(all_masked.unsqueeze(1), torch.zeros_like(policy_logits), policy_logits)
                policy_probs = torch.softmax(safe_logits, dim=-1)
                set_node_priors(self.forest, leaf_indices, policy_probs)

                # Values are from current player's perspective; convert to root.
                root_players = self.forest.to_play[:, 0]
                same_team = (actors % 2) == (root_players % 2)
                values = torch.where(same_team, values, -values)
                values = torch.where(is_terminal, terminal_values, values)

                expand_gpu(self.forest, leaf_indices, leaf_states=leaf_states)
                backprop_gpu(self.forest, leaf_indices, values, paths)

        self.leaf_indices = leaf_indices
        self.paths = paths
        self.leaf_states = leaf_states
        self.terminal_values = terminal_values
        self.is_terminal = is_terminal
        self.values = values
        self._captured = True

    def step(self) -> None:
        if not self._captured:
            self.capture()
        with torch.inference_mode():
            self._graph.replay()


def backprop_gpu(
    forest: GPUMCTSForest,
    leaf_indices: Tensor,
    values: Tensor,
    paths: Tensor,
) -> None:
    """Backpropagate values along paths.

    Note: visit_counts were already incremented during selection (virtual loss).
    This function only updates value_sums.

    Args:
        forest: The MCTS forest
        leaf_indices: (N,) int64 - leaf node indices (for perspective tracking)
        values: (N,) float32 - leaf evaluations from root player's perspective
        paths: (N, max_depth) int64 - node indices along path, -1 padded
    """
    n = forest.n_trees
    device = forest.device
    max_depth = paths.shape[1]
    batch_idx = forest.tree_idx

    # Player to act at root (cached)
    root_players = forest.to_play[:, 0]  # (N,)

    # Backpropagate along each path
    for depth in range(max_depth):
        node_indices = paths[:, depth]  # (N,)

        # Skip invalid path entries
        valid = node_indices >= 0  # (N,)
        safe_node_indices = node_indices.clamp(min=0)
        # CUDA-only: do not add CPU-only early-exit fallbacks here.

        # Get the parent state to determine whose perspective this value is from
        # For root nodes (parent == -1), use root player's perspective
        parents = forest.parents[batch_idx, safe_node_indices]  # (N,) int64
        is_root = parents < 0

        # Get parent player-to-act for perspective calculation (cached)
        parent_node_indices = torch.where(is_root, torch.zeros_like(parents), parents).clamp(min=0)
        parent_players = forest.to_play[batch_idx, parent_node_indices]  # (N,)

        # Use root player for root nodes
        parent_players = torch.where(is_root, root_players, parent_players)

        # Determine if same team as root player
        same_team = (parent_players % 2) == (root_players % 2)  # (N,)

        # Flip value for opponent's perspective
        node_values = torch.where(same_team, values, -values)

        # Update value_sums (masked; safe for invalid path entries)
        forest.value_sums[batch_idx, safe_node_indices] += node_values * valid


def get_root_policy_gpu(forest: GPUMCTSForest) -> Tensor:
    """Get visit distribution at root for all trees.

    Returns:
        policy: (N, 7) float32 - normalized visit probabilities
    """
    n = forest.n_trees
    device = forest.device
    batch_idx = forest.tree_idx

    # Get root children (node 0 for all trees)
    root_children = forest.children[:, 0, :]  # (N, 7)

    # Get visit counts for children
    valid_children = root_children >= 0  # (N, 7)
    safe_indices = root_children.clamp(min=0)  # (N, 7) int64

    child_visits = forest.visit_counts.gather(1, safe_indices)  # (N, 7) int32

    # Mask invalid children
    child_visits = torch.where(
        valid_children,
        child_visits,
        torch.zeros_like(child_visits)
    )

    child_visits = child_visits.float()

    # Normalize to probabilities
    total_visits = child_visits.sum(dim=1, keepdim=True)  # (N, 1)

    # Handle edge case where total is 0 (shouldn't happen after search)
    policy = torch.where(
        total_visits > 0,
        child_visits / total_visits,
        torch.zeros_like(child_visits)
    )

    return policy


def get_leaf_states(forest: GPUMCTSForest, leaf_indices: Tensor) -> GPUGameState:
    """Extract game states for leaf evaluation by oracle.

    Args:
        forest: The MCTS forest
        leaf_indices: (N,) int64 - leaf node indices

    Returns:
        GPUGameState with N states ready for oracle.batch_evaluate_gpu()
    """
    return _get_node_states(forest, leaf_indices)


def get_terminal_values(
    forest: GPUMCTSForest,
    leaf_indices: Tensor,
    leaf_states: GPUGameState,
) -> tuple[Tensor, Tensor]:
    """Compute values for terminal leaves.

    Terminal value is based on point difference:
    - Team with more points wins (+1)
    - Team with fewer points loses (-1)
    - Tie is 0

    Args:
        forest: The MCTS forest
        leaf_indices: (N,) int32 - leaf node indices
        leaf_states: GPUGameState for the leaves

    Returns:
        values: (N,) float32 - terminal values from root player's perspective
        is_terminal: (N,) bool - mask of which leaves are terminal
    """
    n = forest.n_trees
    device = forest.device

    # Check which leaves are terminal
    is_terminal = is_terminal_gpu(leaf_states)  # (N,)

    # Get root player for perspective (cached)
    root_players = forest.to_play[:, 0]  # (N,)
    root_team = root_players % 2  # (N,) - 0 or 1

    # Get scores
    team0_score = leaf_states.scores[:, 0]  # (N,)
    team1_score = leaf_states.scores[:, 1]  # (N,)

    # Compute root team's perspective
    my_score = torch.where(root_team == 0, team0_score, team1_score)
    opp_score = torch.where(root_team == 0, team1_score, team0_score)

    # Terminal value: +1 win, -1 loss, 0 tie
    values = torch.zeros(n, dtype=torch.float32, device=device)
    values = torch.where(my_score > opp_score, torch.ones_like(values), values)
    values = torch.where(my_score < opp_score, -torch.ones_like(values), values)

    return values, is_terminal


def prepare_oracle_inputs(
    forest: GPUMCTSForest,
    leaf_states: GPUGameState,
) -> dict:
    """Prepare inputs for oracle.batch_evaluate_gpu().

    Extracts all tensors needed for oracle evaluation from the forest
    and leaf states.

    Args:
        forest: The MCTS forest (provides original_hands)
        leaf_states: GPUGameState for leaves to evaluate

    Returns:
        Dict with keys matching oracle.batch_evaluate_gpu() signature:
        - original_hands: (N, 4, 7) int32
        - current_hands: (N, 4, 7) int32
        - decl_ids: (N,) int32
        - actors: (N,) int32
        - leaders: (N,) int32
        - trick_players: (N, 3) int32
        - trick_dominoes: (N, 3) int32
        - players: (N,) int32 (perspective player = root player)
    """
    n = forest.n_trees
    device = forest.device

    # Get root player for perspective (cached)
    root_players = forest.to_play[:, 0]  # (N,)

    # Current player at leaf
    actors = current_player_gpu(leaf_states)

    # Extract trick plays from current_trick tensor
    # current_trick is (N, 4, 2) with (player, domino) pairs
    # Oracle expects trick_players and trick_dominoes as (N, 3) tensors
    trick_players = leaf_states.current_trick[:, :3, 0]  # (N, 3)
    trick_dominoes = leaf_states.current_trick[:, :3, 1]  # (N, 3)

    # Mark empty trick positions as -1
    # A position is empty if trick_len is less than that position
    trick_len = leaf_states.trick_len.unsqueeze(1)  # (N, 1)
    position_idx = torch.arange(3, device=device).unsqueeze(0)  # (1, 3)
    empty_mask = position_idx >= trick_len  # (N, 3)

    trick_players = torch.where(empty_mask, torch.full_like(trick_players, -1), trick_players)
    trick_dominoes = torch.where(empty_mask, torch.full_like(trick_dominoes, -1), trick_dominoes)

    return {
        'original_hands': forest.original_hands,
        'current_hands': leaf_states.hands,
        'decl_ids': leaf_states.decl_id,
        'actors': actors,
        'leaders': leaf_states.leader,
        'trick_players': trick_players,
        'trick_dominoes': trick_dominoes,
        'players': root_players,  # Value from root player's perspective
    }


def run_mcts_search(
    forest: GPUMCTSForest,
    oracle_fn: callable,
    n_simulations: int,
    *,
    use_cudagraph: bool = False,
    max_depth: int = 28,
) -> Tensor:
    """Run complete MCTS search.

    This is the main entry point for MCTS search.

    Args:
        forest: The MCTS forest (will be modified in-place)
        oracle_fn: Function to evaluate non-terminal leaves.
                   Should accept GPUGameState and return (N,) float32 values.
        n_simulations: Number of simulations to run

    Returns:
        policy: (N, 7) float32 - normalized visit probabilities at root
    """
    if use_cudagraph:
        runner: MCTSCUDAGraphRunner | None = getattr(forest, "_mcts_cudagraph_runner", None)
        if runner is None or runner.max_depth != max_depth:
            runner = MCTSCUDAGraphRunner(forest, max_depth=max_depth)
            setattr(forest, "_mcts_cudagraph_runner", runner)

        if not runner._captured:
            # Capturing mutates the forest; restore to the current root position afterward.
            roots = torch.zeros(forest.n_trees, dtype=torch.int64, device=forest.device)
            root_states = _get_node_states(forest, roots)
            runner.capture()
            reset_forest_inplace(forest, root_states, original_hands=forest.original_hands)

        for _ in range(n_simulations):
            runner.step(oracle_fn)

        return get_root_policy_gpu(forest)

    for _ in range(n_simulations):
        # Select leaves for all trees
        leaf_indices, paths = select_leaves_gpu(forest, max_depth=max_depth)

        # Get leaf states for evaluation
        leaf_states = get_leaf_states(forest, leaf_indices)

        # Compute terminal values
        terminal_values, is_terminal = get_terminal_values(
            forest, leaf_indices, leaf_states
        )

        # Evaluate non-terminal leaves with oracle (avoid GPU->CPU sync from tensor.any()).
        oracle_values = torch.zeros(forest.n_trees, dtype=torch.float32, device=forest.device)
        nonterm_idx = (~is_terminal).nonzero(as_tuple=True)[0]
        if nonterm_idx.numel() > 0:
            nonterm_states = _get_node_states(forest, leaf_indices[nonterm_idx], nonterm_idx)
            oracle_values_nonterm = oracle_fn(nonterm_states)
            oracle_values[nonterm_idx] = oracle_values_nonterm

        # Use terminal values where applicable
        values = torch.where(is_terminal, terminal_values, oracle_values)

        # Expand leaves (only non-terminal)
        expand_gpu(forest, leaf_indices, leaf_states=leaf_states)

        # Backpropagate values
        backprop_gpu(forest, leaf_indices, values, paths)

    return get_root_policy_gpu(forest)


def set_node_priors(
    forest: GPUMCTSForest,
    node_indices: Tensor,
    priors: Tensor,
    tree_indices: Tensor | None = None,
) -> None:
    """Set policy priors for specific nodes.

    Called after evaluating nodes with neural network to store the policy
    output as priors for MCTS selection.

    Args:
        forest: The MCTS forest
        node_indices: (K,) int32 - node index within each selected tree
        priors: (K, 7) float32 - policy probabilities from neural network
        tree_indices: Optional (K,) int64 of which tree each node belongs to.
                      If None, assumes K == N and uses arange(N).
    """
    device = forest.device
    if tree_indices is None:
        batch_idx = torch.arange(forest.n_trees, device=device, dtype=torch.int64)
    else:
        batch_idx = tree_indices.to(device=device, dtype=torch.int64)

    # Store priors for these nodes
    forest.priors[batch_idx, node_indices] = priors
