"""GPU-native MCTS forest for batched tree search.

Manages N parallel MCTS trees on GPU. All operations are vectorized
with no Python loops in the hot path (except for the 7-slot expansion loop).

Key design decisions:
- Pre-allocated node pools to avoid dynamic allocation
- Flattened state storage for efficient GPU access
- Virtual loss during selection for exploration diversity
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
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from forge.zeb.gpu_game_state import (
    GPUGameState,
    apply_action_gpu,
    current_player_gpu,
    deal_random_gpu,
    is_terminal_gpu,
    legal_actions_gpu,
)

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
    parents: Tensor        # int32 - parent node index (-1 for root)
    depths: Tensor         # int32 - node depth (root=0)

    # Children pointers (N, M, 7) - one child per slot
    children: Tensor       # int32 - child node index (-1 if unexpanded)
    child_actions: Tensor  # int32 - slot index that led to this child

    # Node states - flattened (N * M, ...)
    # Indexed as: states[tree_idx * max_nodes + node_idx]
    states: GPUGameState

    # Allocation tracking
    n_nodes: Tensor        # (N,) int32 - number of allocated nodes per tree

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
    assert initial_states.batch_size == n_trees, (
        f"initial_states batch size {initial_states.batch_size} != n_trees {n_trees}"
    )

    # Allocate node statistics
    visit_counts = torch.zeros(n_trees, max_nodes, dtype=torch.int32, device=device)
    value_sums = torch.zeros(n_trees, max_nodes, dtype=torch.float32, device=device)

    # Tree structure
    parents = torch.full((n_trees, max_nodes), -1, dtype=torch.int32, device=device)
    depths = torch.zeros(n_trees, max_nodes, dtype=torch.int32, device=device)

    # Children pointers - 7 slots per node
    children = torch.full((n_trees, max_nodes, 7), -1, dtype=torch.int32, device=device)
    child_actions = torch.full((n_trees, max_nodes, 7), -1, dtype=torch.int32, device=device)

    # Pre-allocate state storage for all N*M potential nodes
    # Initialize with copies of initial states (only root nodes are valid initially)
    total_nodes = n_trees * max_nodes

    # Expand initial_states to fill the entire node pool
    # Each tree's initial state goes into index [tree_idx * max_nodes + 0]
    # We'll use torch.repeat_interleave to tile the states
    states = _allocate_state_pool(initial_states, max_nodes, device)

    # Track allocation: each tree starts with 1 node (the root)
    n_nodes = torch.ones(n_trees, dtype=torch.int32, device=device)

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
        states=states,
        n_nodes=n_nodes,
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


def _get_node_states(forest: GPUMCTSForest, node_indices: Tensor) -> GPUGameState:
    """Extract states for specific nodes.

    Args:
        forest: The MCTS forest
        node_indices: (N,) int32 - node index within each tree

    Returns:
        GPUGameState with N states (one per tree)
    """
    n = forest.n_trees
    device = forest.device

    # Compute flat indices: tree_idx * max_nodes + node_idx
    tree_indices = torch.arange(n, device=device, dtype=torch.int64)
    flat_indices = tree_indices * forest.max_nodes + node_indices.long()

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
    tree_indices = torch.arange(n, device=device, dtype=torch.int64)
    flat_indices = tree_indices * forest.max_nodes + node_indices.long()

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
    visit_counts: Tensor,      # (N, M) or (N, M, 7)
    value_sums: Tensor,        # (N, M) or (N, M, 7)
    parent_visits: Tensor,     # (N,) or (N, M)
    n_legal: Tensor,           # (N,) or (N, M) - number of legal actions
    c_puct: float,
) -> Tensor:
    """Compute UCB scores for nodes.

    UCB = Q + c_puct * prior * sqrt(parent_visits) / (1 + visits)

    where prior = 1 / n_legal (uniform prior)

    Returns:
        UCB scores with same shape as visit_counts
    """
    # Handle the case where visit_counts could be 0
    # Q-value: value_sum / visit_count (0 if unvisited)
    q_value = torch.where(
        visit_counts > 0,
        value_sums / visit_counts.float(),
        torch.zeros_like(value_sums, dtype=torch.float32),
    )

    # Prior: uniform = 1 / n_legal
    # Ensure n_legal is at least 1 to avoid division by zero
    prior = 1.0 / n_legal.float().clamp(min=1)

    # Expand parent_visits if needed
    if parent_visits.dim() < visit_counts.dim():
        for _ in range(visit_counts.dim() - parent_visits.dim()):
            parent_visits = parent_visits.unsqueeze(-1)

    # Expand prior if needed
    if prior.dim() < visit_counts.dim():
        for _ in range(visit_counts.dim() - prior.dim()):
            prior = prior.unsqueeze(-1)

    # Exploration term: c_puct * prior * sqrt(parent_visits) / (1 + visits)
    exploration = (
        c_puct * prior * torch.sqrt(parent_visits.float()) /
        (1.0 + visit_counts.float())
    )

    # UCB score
    ucb = q_value + exploration

    # Unvisited nodes get infinite UCB (exploration priority)
    ucb = torch.where(visit_counts == 0, torch.full_like(ucb, float('inf')), ucb)

    return ucb


def select_leaves_gpu(
    forest: GPUMCTSForest,
    max_depth: int = 28,
) -> tuple[Tensor, Tensor]:
    """Select leaves for all N trees in parallel using UCB.

    Traverses from root to leaf for each tree, applying virtual loss
    (incrementing visit counts) along the way to encourage diversity.

    Args:
        forest: The MCTS forest
        max_depth: Maximum depth to traverse (game has max 28 moves)

    Returns:
        leaf_indices: (N,) int32 - node index of selected leaf per tree
        paths: (N, max_depth) int32 - node indices along path, -1 padded
    """
    n = forest.n_trees
    device = forest.device

    # Start at root (node 0) for all trees
    current_nodes = torch.zeros(n, dtype=torch.int32, device=device)

    # Track paths
    paths = torch.full((n, max_depth), -1, dtype=torch.int32, device=device)
    path_lengths = torch.zeros(n, dtype=torch.int32, device=device)

    # Track which trees are still traversing
    active = torch.ones(n, dtype=torch.bool, device=device)

    batch_idx = torch.arange(n, device=device, dtype=torch.int64)

    for depth in range(max_depth):
        if not active.any():
            break

        # Record current node in path
        paths[batch_idx, path_lengths.long()] = torch.where(
            active, current_nodes, paths[batch_idx, path_lengths.long()]
        )
        path_lengths = torch.where(active, path_lengths + 1, path_lengths)

        # Apply virtual loss to current nodes
        forest.visit_counts[batch_idx, current_nodes.long()] += active.int()

        # Get children of current nodes: (N, 7)
        current_children = forest.children[batch_idx, current_nodes.long()]  # (N, 7)

        # Check if any children exist (not all -1)
        has_children = (current_children >= 0).any(dim=1)  # (N,)

        # For nodes with children, select best child using UCB
        # Get visit counts and value sums for children
        # Need to gather child statistics using the child indices

        # Create mask for valid children
        valid_children = current_children >= 0  # (N, 7)

        # Get child visit counts and value sums
        # Safe indexing: clamp negative indices to 0 for gather, then mask results
        safe_child_indices = current_children.clamp(min=0).long()  # (N, 7)

        child_visits = forest.visit_counts.gather(
            1, safe_child_indices
        )  # (N, 7)
        child_values = forest.value_sums.gather(
            1, safe_child_indices
        )  # (N, 7)

        # Get parent (current node) visit counts for UCB
        parent_visits = forest.visit_counts[batch_idx, current_nodes.long()]  # (N,)

        # Count legal actions for prior computation
        n_legal = valid_children.sum(dim=1)  # (N,)

        # Compute UCB scores for all children
        ucb_scores = _compute_ucb_scores(
            child_visits, child_values, parent_visits, n_legal, forest.c_puct
        )  # (N, 7)

        # Mask invalid children with -inf
        ucb_scores = torch.where(
            valid_children,
            ucb_scores,
            torch.full_like(ucb_scores, float('-inf'))
        )

        # Select best child
        best_child_slot = ucb_scores.argmax(dim=1)  # (N,)
        best_child_node = current_children[batch_idx, best_child_slot.long()]  # (N,)

        # Update current nodes (only for active trees with children)
        should_descend = active & has_children
        current_nodes = torch.where(
            should_descend,
            best_child_node.to(torch.int32),
            current_nodes
        )

        # Trees without children (leaves) become inactive
        active = active & has_children

    return current_nodes, paths


def expand_gpu(
    forest: GPUMCTSForest,
    leaf_indices: Tensor,
) -> Tensor:
    """Expand leaves by creating child nodes.

    For each leaf, creates child nodes for all legal actions.
    Uses vectorized operations where possible.

    Args:
        forest: The MCTS forest
        leaf_indices: (N,) int32 - node indices of leaves to expand

    Returns:
        n_expanded: (N,) int32 - number of children created per tree
    """
    n = forest.n_trees
    device = forest.device
    batch_idx = torch.arange(n, device=device, dtype=torch.int64)

    # Get states for the leaves
    leaf_states = _get_node_states(forest, leaf_indices)

    # Check which leaves are terminal (no expansion needed)
    terminal = is_terminal_gpu(leaf_states)  # (N,)

    # Get legal actions for non-terminal leaves
    legal_mask = legal_actions_gpu(leaf_states)  # (N, 7) bool

    # Don't expand terminal nodes
    legal_mask = legal_mask & ~terminal.unsqueeze(1)

    # Count legal actions per tree
    n_legal = legal_mask.sum(dim=1)  # (N,)

    # For each legal action, allocate a new node
    n_new_nodes = n_legal.sum().item()

    if n_new_nodes == 0:
        return torch.zeros(n, dtype=torch.int32, device=device)

    n_expanded = torch.zeros(n, dtype=torch.int32, device=device)

    for slot in range(7):
        # Check which trees have this slot as legal
        slot_legal = legal_mask[:, slot]  # (N,)

        if not slot_legal.any():
            continue

        # Check which trees have room for more nodes
        has_room = forest.n_nodes < forest.max_nodes
        can_expand = slot_legal & has_room

        if not can_expand.any():
            continue

        # Allocate new node indices
        new_node_indices = forest.n_nodes.clone()  # (N,)

        # Only increment for trees that can expand
        forest.n_nodes = torch.where(
            can_expand,
            forest.n_nodes + 1,
            forest.n_nodes
        )

        # Compute which trees to update (as boolean mask)
        expand_mask = can_expand  # (N,)
        expand_idx = batch_idx[expand_mask]  # (K,) where K = number of trees expanding

        # Set parent pointers
        forest.parents[expand_idx, new_node_indices[expand_mask].long()] = \
            leaf_indices[expand_mask]

        # Set depth
        parent_depths = forest.depths[expand_idx, leaf_indices[expand_mask].long()]
        forest.depths[expand_idx, new_node_indices[expand_mask].long()] = \
            parent_depths + 1

        # Set child pointer in parent
        forest.children[expand_idx, leaf_indices[expand_mask].long(), slot] = \
            new_node_indices[expand_mask]
        forest.child_actions[expand_idx, leaf_indices[expand_mask].long(), slot] = slot

        # Apply action to get child state - only for games that can expand
        # Create subset of leaf_states for games that can expand this slot
        n_can_expand = can_expand.sum().item()
        if n_can_expand == 0:
            continue

        # Extract states for games that can expand
        subset_states = GPUGameState(
            hands=leaf_states.hands[expand_mask],
            played_mask=leaf_states.played_mask[expand_mask],
            play_history=leaf_states.play_history[expand_mask],
            n_plays=leaf_states.n_plays[expand_mask],
            current_trick=leaf_states.current_trick[expand_mask],
            trick_len=leaf_states.trick_len[expand_mask],
            leader=leaf_states.leader[expand_mask],
            decl_id=leaf_states.decl_id[expand_mask],
            scores=leaf_states.scores[expand_mask],
        )

        actions = torch.full((n_can_expand,), slot, dtype=torch.int32, device=device)
        child_states = apply_action_gpu(subset_states, actions)

        # Compute flat indices for writing child states
        flat_indices = expand_idx * forest.max_nodes + new_node_indices[expand_mask].long()

        # Write child states
        forest.states.hands[flat_indices] = child_states.hands
        forest.states.played_mask[flat_indices] = child_states.played_mask
        forest.states.play_history[flat_indices] = child_states.play_history
        forest.states.n_plays[flat_indices] = child_states.n_plays
        forest.states.current_trick[flat_indices] = child_states.current_trick
        forest.states.trick_len[flat_indices] = child_states.trick_len
        forest.states.leader[flat_indices] = child_states.leader
        forest.states.decl_id[flat_indices] = child_states.decl_id
        forest.states.scores[flat_indices] = child_states.scores

        # Track expansion count
        n_expanded = torch.where(can_expand, n_expanded + 1, n_expanded)

    return n_expanded


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
        leaf_indices: (N,) int32 - leaf node indices (for perspective tracking)
        values: (N,) float32 - leaf evaluations from root player's perspective
        paths: (N, max_depth) int32 - node indices along path, -1 padded
    """
    n = forest.n_trees
    device = forest.device
    max_depth = paths.shape[1]
    batch_idx = torch.arange(n, device=device, dtype=torch.int64)

    # Get root player for perspective (player who acts at root)
    root_states = _get_node_states(forest, torch.zeros(n, dtype=torch.int32, device=device))
    root_players = current_player_gpu(root_states)  # (N,)

    # Backpropagate along each path
    for depth in range(max_depth):
        node_indices = paths[:, depth]  # (N,)

        # Skip invalid path entries
        valid = node_indices >= 0  # (N,)

        if not valid.any():
            break

        # Get the parent state to determine whose perspective this value is from
        # For root nodes (parent == -1), use root player's perspective
        parents = forest.parents[batch_idx, node_indices.long()]  # (N,)
        is_root = parents < 0

        # Get parent states for perspective calculation
        parent_node_indices = torch.where(is_root, torch.zeros_like(parents), parents)
        parent_states = _get_node_states(forest, parent_node_indices)
        parent_players = current_player_gpu(parent_states)  # (N,)

        # Use root player for root nodes
        parent_players = torch.where(is_root, root_players, parent_players)

        # Determine if same team as root player
        same_team = (parent_players % 2) == (root_players % 2)  # (N,)

        # Flip value for opponent's perspective
        node_values = torch.where(same_team, values, -values)

        # Update value_sums (only for valid path entries)
        forest.value_sums[batch_idx[valid], node_indices[valid].long()] += node_values[valid]


def get_root_policy_gpu(forest: GPUMCTSForest) -> Tensor:
    """Get visit distribution at root for all trees.

    Returns:
        policy: (N, 7) float32 - normalized visit probabilities
    """
    n = forest.n_trees
    device = forest.device
    batch_idx = torch.arange(n, device=device, dtype=torch.int64)

    # Get root children (node 0 for all trees)
    root_children = forest.children[:, 0, :]  # (N, 7)

    # Get visit counts for children
    valid_children = root_children >= 0  # (N, 7)
    safe_indices = root_children.clamp(min=0).long()  # (N, 7)

    child_visits = forest.visit_counts.gather(1, safe_indices)  # (N, 7)

    # Mask invalid children
    child_visits = torch.where(
        valid_children,
        child_visits.float(),
        torch.zeros_like(child_visits, dtype=torch.float32)
    )

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
        leaf_indices: (N,) int32 - leaf node indices

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

    # Get root player for perspective
    root_states = _get_node_states(forest, torch.zeros(n, dtype=torch.int32, device=device))
    root_players = current_player_gpu(root_states)  # (N,)
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

    # Get root player for perspective
    root_states = _get_node_states(forest, torch.zeros(n, dtype=torch.int32, device=device))
    root_players = current_player_gpu(root_states)  # (N,)

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
    for _ in range(n_simulations):
        # Select leaves for all trees
        leaf_indices, paths = select_leaves_gpu(forest)

        # Get leaf states for evaluation
        leaf_states = get_leaf_states(forest, leaf_indices)

        # Compute terminal values
        terminal_values, is_terminal = get_terminal_values(
            forest, leaf_indices, leaf_states
        )

        # Evaluate non-terminal leaves with oracle
        if (~is_terminal).any():
            oracle_values = oracle_fn(leaf_states)
        else:
            oracle_values = torch.zeros(forest.n_trees, dtype=torch.float32, device=forest.device)

        # Use terminal values where applicable
        values = torch.where(is_terminal, terminal_values, oracle_values)

        # Expand leaves (only non-terminal)
        expand_gpu(forest, leaf_indices)

        # Backpropagate values
        backprop_gpu(forest, leaf_indices, values, paths)

    return get_root_policy_gpu(forest)
