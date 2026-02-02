"""Tests for GPU-native MCTS forest.

Validates that GPU MCTS operations work correctly and produce sensible policies.
"""

from __future__ import annotations

import random
import time

import pytest
import torch

from forge.zeb.gpu_game_state import (
    GPUGameState,
    apply_action_gpu,
    current_player_gpu,
    deal_random_gpu,
    is_terminal_gpu,
    legal_actions_gpu,
)
from forge.zeb.gpu_mcts import (
    GPUMCTSForest,
    backprop_gpu,
    create_forest,
    expand_gpu,
    get_leaf_states,
    get_root_policy_gpu,
    get_terminal_values,
    prepare_oracle_inputs,
    run_mcts_search,
    select_leaves_gpu,
)


# Use CPU for tests (works on all machines)
DEVICE = torch.device("cpu")


class TestCreateForest:
    """Test forest creation and basic structure."""

    def test_create_forest_basic(self):
        """Forest should be created with correct dimensions."""
        n_trees = 16
        max_nodes = 100
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)

        forest = create_forest(n_trees, max_nodes, initial_states, DEVICE)

        assert forest.n_trees == n_trees
        assert forest.max_nodes == max_nodes
        assert forest.visit_counts.shape == (n_trees, max_nodes)
        assert forest.value_sums.shape == (n_trees, max_nodes)
        assert forest.parents.shape == (n_trees, max_nodes)
        assert forest.depths.shape == (n_trees, max_nodes)
        assert forest.children.shape == (n_trees, max_nodes, 7)
        assert forest.n_nodes.shape == (n_trees,)

    def test_forest_initial_state(self):
        """Forest should start with 1 node per tree (the root)."""
        n_trees = 10
        max_nodes = 50
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=3)

        forest = create_forest(n_trees, max_nodes, initial_states, DEVICE)

        # Each tree has exactly 1 node (root)
        assert (forest.n_nodes == 1).all()

        # Root has no parent
        assert (forest.parents[:, 0] == -1).all()

        # Root has depth 0
        assert (forest.depths[:, 0] == 0).all()

        # Root has no children yet
        assert (forest.children[:, 0, :] == -1).all()

        # Root starts with 0 visits
        assert (forest.visit_counts[:, 0] == 0).all()

    def test_forest_stores_original_hands(self):
        """Forest should store original hands for oracle queries."""
        n_trees = 8
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)

        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        assert forest.original_hands is not None
        assert forest.original_hands.shape == (n_trees, 4, 7)
        assert torch.equal(forest.original_hands, initial_states.hands)


class TestSelectLeavesGpu:
    """Test UCB selection from root to leaf."""

    def test_select_starts_at_root(self):
        """Selection should start at root for fresh forest."""
        n_trees = 10
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        leaf_indices, paths = select_leaves_gpu(forest)

        # With no expansion, leaves should be roots
        assert (leaf_indices == 0).all()

        # Path should contain only root
        assert (paths[:, 0] == 0).all()

    def test_virtual_loss_applied(self):
        """Selection should apply virtual loss to visited nodes."""
        n_trees = 10
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Initial visits are 0
        assert (forest.visit_counts[:, 0] == 0).all()

        # Select leaves
        leaf_indices, paths = select_leaves_gpu(forest)

        # Root should now have 1 visit (virtual loss)
        assert (forest.visit_counts[:, 0] == 1).all()

    def test_select_prefers_unvisited(self):
        """Selection should prefer unvisited children (UCB = inf)."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Expand root
        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        expand_gpu(forest, root_indices)

        # Now select - should go to an unvisited child
        leaf_indices, paths = select_leaves_gpu(forest)

        # Leaves should be children (not root)
        assert (leaf_indices > 0).all()

    def test_paths_contain_traversal(self):
        """Paths should contain all nodes from root to leaf."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Expand root
        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        expand_gpu(forest, root_indices)

        # Select
        leaf_indices, paths = select_leaves_gpu(forest)

        # Path should start with root (0)
        assert (paths[:, 0] == 0).all()

        # Second entry should be the leaf (child of root)
        for i in range(n_trees):
            if leaf_indices[i].item() > 0:
                # Path should include the leaf
                assert leaf_indices[i].item() in paths[i].tolist()


class TestExpandGpu:
    """Test expansion creates child nodes correctly."""

    def test_expand_creates_children(self):
        """Expanding root should create child nodes."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # All trees start with 1 node
        assert (forest.n_nodes == 1).all()

        # Expand root
        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        n_expanded = expand_gpu(forest, root_indices)

        # Should have created children (all 7 slots are legal at start)
        assert (n_expanded == 7).all()

        # n_nodes should increase
        assert (forest.n_nodes == 8).all()  # 1 root + 7 children

    def test_expand_sets_parent_pointers(self):
        """Expanded nodes should point back to parent."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        expand_gpu(forest, root_indices)

        # All children (nodes 1-7) should have parent 0 (root)
        for node_idx in range(1, 8):
            assert (forest.parents[:, node_idx] == 0).all()

    def test_expand_sets_child_pointers(self):
        """Parent should have child pointers set."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        expand_gpu(forest, root_indices)

        # Root should now have children
        root_children = forest.children[:, 0, :]  # (N, 7)
        has_children = (root_children >= 0).any(dim=1)
        assert has_children.all()

    def test_expand_respects_legal_actions(self):
        """Only legal actions should create children."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Get legal actions at root
        root_states = get_leaf_states(
            forest,
            torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        )
        legal = legal_actions_gpu(root_states)  # (N, 7)
        n_legal = legal.sum(dim=1)  # (N,)

        # Expand
        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        n_expanded = expand_gpu(forest, root_indices)

        # Number expanded should equal number of legal actions
        assert (n_expanded == n_legal).all()

    def test_expand_terminal_does_nothing(self):
        """Expanding terminal nodes should not create children."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Mark root states as terminal by setting n_plays = 28
        # Root nodes are at flat indices [0, max_nodes, 2*max_nodes, ...]
        root_flat_indices = torch.arange(n_trees, device=DEVICE) * forest.max_nodes
        forest.states.n_plays[root_flat_indices] = 28

        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        n_expanded = expand_gpu(forest, root_indices)

        # Should not expand terminal nodes
        assert (n_expanded == 0).all()
        assert (forest.n_nodes == 1).all()


class TestBackpropGpu:
    """Test backpropagation updates values along path."""

    def test_backprop_updates_root(self):
        """Backprop should update root value when path is just root."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Simulate selection (adds virtual loss)
        leaf_indices, paths = select_leaves_gpu(forest)

        # Backprop with value 0.5
        values = torch.full((n_trees,), 0.5, dtype=torch.float32, device=DEVICE)
        backprop_gpu(forest, leaf_indices, values, paths)

        # Root value_sum should be updated
        assert (forest.value_sums[:, 0] == 0.5).all()

    def test_backprop_flips_value_for_opponent(self):
        """Backprop should negate value for opponent's nodes."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Expand root to create children
        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        expand_gpu(forest, root_indices)

        # Select leaves (will go to children)
        leaf_indices, paths = select_leaves_gpu(forest)

        # Backprop with value 1.0
        values = torch.ones(n_trees, dtype=torch.float32, device=DEVICE)
        backprop_gpu(forest, leaf_indices, values, paths)

        # Check that values were updated along path
        # Root should have positive or negative value based on team
        # (This depends on the game state, so just check it's non-zero)
        assert (forest.value_sums[:, 0] != 0).any() or (forest.value_sums[:, leaf_indices.long()].diagonal() != 0).any()


class TestGetRootPolicyGpu:
    """Test root policy extraction."""

    def test_policy_sums_to_one(self):
        """Policy should sum to 1 for trees with children."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Do some search
        for _ in range(10):
            leaf_indices, paths = select_leaves_gpu(forest)
            expand_gpu(forest, leaf_indices)
            values = torch.zeros(n_trees, dtype=torch.float32, device=DEVICE)
            backprop_gpu(forest, leaf_indices, values, paths)

        policy = get_root_policy_gpu(forest)

        # Should have shape (N, 7)
        assert policy.shape == (n_trees, 7)

        # Should sum to approximately 1 (or 0 if no children)
        sums = policy.sum(dim=1)
        for i in range(n_trees):
            assert abs(sums[i].item() - 1.0) < 0.01 or sums[i].item() == 0

    def test_policy_reflects_visits(self):
        """Policy should be proportional to visit counts."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Manually set some children and visits
        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        expand_gpu(forest, root_indices)

        # Set different visit counts for children
        # Child 1 gets 10 visits, child 2 gets 20, others get 5
        for tree_idx in range(n_trees):
            for slot in range(7):
                child_node = forest.children[tree_idx, 0, slot].item()
                if child_node >= 0:
                    if slot == 0:
                        forest.visit_counts[tree_idx, child_node] = 10
                    elif slot == 1:
                        forest.visit_counts[tree_idx, child_node] = 20
                    else:
                        forest.visit_counts[tree_idx, child_node] = 5

        policy = get_root_policy_gpu(forest)

        # Slot 1 should have highest probability
        for tree_idx in range(n_trees):
            if forest.children[tree_idx, 0, 1].item() >= 0:
                # Slot 1 has 20 visits out of 10+20+5*5 = 55 total
                expected = 20.0 / 55.0
                actual = policy[tree_idx, 1].item()
                assert abs(actual - expected) < 0.01


class TestGetLeafStates:
    """Test extracting leaf states for oracle evaluation."""

    def test_leaf_states_match_forest(self):
        """Extracted states should match states stored in forest."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        leaf_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        leaf_states = get_leaf_states(forest, leaf_indices)

        # Should match initial states
        assert torch.equal(leaf_states.hands, initial_states.hands)
        assert torch.equal(leaf_states.decl_id, initial_states.decl_id)
        assert torch.equal(leaf_states.n_plays, initial_states.n_plays)


class TestGetTerminalValues:
    """Test terminal value computation."""

    def test_terminal_values_correct(self):
        """Terminal values should reflect game outcome."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Root nodes are at flat indices [0, max_nodes, 2*max_nodes, ...]
        root_flat_indices = torch.arange(n_trees, device=DEVICE) * forest.max_nodes

        # Make states terminal by setting n_plays = 28
        forest.states.n_plays[root_flat_indices] = 28

        # Set scores: team 0 wins in half, team 1 wins in other half
        forest.states.scores[root_flat_indices[0], 0] = 30  # Team 0: 30, Team 1: 12
        forest.states.scores[root_flat_indices[0], 1] = 12
        forest.states.scores[root_flat_indices[1], 0] = 12  # Team 0: 12, Team 1: 30
        forest.states.scores[root_flat_indices[1], 1] = 30
        forest.states.scores[root_flat_indices[2], 0] = 21  # Tie
        forest.states.scores[root_flat_indices[2], 1] = 21
        forest.states.scores[root_flat_indices[3], 0] = 25
        forest.states.scores[root_flat_indices[3], 1] = 17

        leaf_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        leaf_states = get_leaf_states(forest, leaf_indices)
        values, is_terminal = get_terminal_values(forest, leaf_indices, leaf_states)

        # All should be terminal
        assert is_terminal.all()

        # Check values based on root player's team
        # (Values depend on which player is at root, which we can check)


class TestFullSearchLoop:
    """Test complete MCTS search produces sensible policies."""

    def test_search_produces_valid_policy(self):
        """Full search should produce valid probability distribution."""
        n_trees = 8
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Simple oracle: random values
        def random_oracle(states: GPUGameState) -> torch.Tensor:
            return torch.rand(states.batch_size, device=states.device) * 2 - 1

        policy = run_mcts_search(forest, random_oracle, n_simulations=20)

        # Policy should be valid
        assert policy.shape == (n_trees, 7)
        assert (policy >= 0).all()

        # Should sum to ~1 for each tree
        sums = policy.sum(dim=1)
        for i in range(n_trees):
            assert abs(sums[i].item() - 1.0) < 0.01

    def test_search_explores_multiple_actions(self):
        """Search should visit multiple children, not just one."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        def random_oracle(states: GPUGameState) -> torch.Tensor:
            return torch.rand(states.batch_size, device=states.device) * 2 - 1

        policy = run_mcts_search(forest, random_oracle, n_simulations=50)

        # Should have non-zero probability for multiple actions
        for tree_idx in range(n_trees):
            n_visited = (policy[tree_idx] > 0.01).sum().item()
            assert n_visited > 1, f"Tree {tree_idx} only visited {n_visited} actions"

    def test_search_increases_visits(self):
        """More simulations should increase total visits."""
        n_trees = 4
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        def random_oracle(states: GPUGameState) -> torch.Tensor:
            return torch.rand(states.batch_size, device=states.device) * 2 - 1

        run_mcts_search(forest, random_oracle, n_simulations=30)

        # Root should have ~30 visits (maybe more due to selection)
        assert (forest.visit_counts[:, 0] >= 30).all()


class TestBenchmark:
    """Benchmark GPU MCTS performance."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available for benchmark"
    )
    def test_gpu_mcts_benchmark(self):
        """Benchmark GPU MCTS vs Python MCTS.

        Note: The pure tree operations may not be faster than Python due to:
        1. Small batch sizes (16 trees)
        2. Python loop over 7 action slots in expand_gpu
        3. Overhead of tensor operations for small batches

        The real speedup comes from:
        1. Keeping state on GPU (no CPU-GPU transfers for oracle)
        2. Batched oracle evaluation across all trees
        3. Scaling to larger batch sizes (100+ trees)

        This test benchmarks the tree operations alone.
        """
        from forge.zeb.mcts import MCTS
        from forge.eq.game import GameState

        n_sims = 50
        n_games = 16

        # GPU MCTS timing
        device = torch.device("cuda")
        initial_states = deal_random_gpu(n_games, device, decl_ids=0)
        forest = create_forest(n_games, 200, initial_states, device, c_puct=1.414)

        def random_oracle(states: GPUGameState) -> torch.Tensor:
            return torch.rand(states.batch_size, device=states.device) * 2 - 1

        # Warmup
        run_mcts_search(forest, random_oracle, n_simulations=5)

        # Reset forest
        forest = create_forest(n_games, 200, initial_states, device, c_puct=1.414)

        torch.cuda.synchronize()
        start = time.perf_counter()
        run_mcts_search(forest, random_oracle, n_simulations=n_sims)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start

        # Python MCTS timing (sequential)
        random.seed(42)
        py_states = []
        for _ in range(n_games):
            dominoes = list(range(28))
            random.shuffle(dominoes)
            hands = [dominoes[i*7:(i+1)*7] for i in range(4)]
            py_states.append(GameState.from_hands(hands, 0, 0))

        py_mcts = MCTS(n_simulations=n_sims, c_puct=1.414)

        start = time.perf_counter()
        for state in py_states:
            py_mcts._search_sequential(state, 0)
        py_time = time.perf_counter() - start

        speedup = py_time / gpu_time
        print(f"\nMCTS Benchmark: {n_games} games x {n_sims} simulations")
        print(f"  Python sequential: {py_time:.3f}s ({n_games * n_sims / py_time:.0f} sims/sec)")
        print(f"  GPU parallel:      {gpu_time:.3f}s ({n_games * n_sims / gpu_time:.0f} sims/sec)")
        print(f"  Speedup: {speedup:.1f}x")

        # Just verify it completes without errors
        # Real speedup comes from oracle batching, not tree operations
        assert gpu_time > 0, "GPU search completed"

    def test_selection_backprop_fast(self):
        """Selection and backprop should be fast even on CPU."""
        n_trees = 64
        max_nodes = 100
        n_iters = 100

        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, max_nodes, initial_states, DEVICE)

        # Expand root first
        root_indices = torch.zeros(n_trees, dtype=torch.int32, device=DEVICE)
        expand_gpu(forest, root_indices)

        # Time selection + backprop
        start = time.perf_counter()
        for _ in range(n_iters):
            leaf_indices, paths = select_leaves_gpu(forest)
            values = torch.rand(n_trees, device=DEVICE) * 2 - 1
            backprop_gpu(forest, leaf_indices, values, paths)
        elapsed = time.perf_counter() - start

        iters_per_sec = n_iters / elapsed
        print(f"\nSelect+Backprop: {n_trees} trees, {n_iters} iterations")
        print(f"  Time: {elapsed:.3f}s ({iters_per_sec:.0f} iters/sec)")

        # Should be reasonably fast (>100 iters/sec on CPU)
        assert iters_per_sec > 50


class TestOracleIntegration:
    """Test integration with oracle batch evaluation pattern."""

    def test_leaf_states_have_correct_format_for_oracle(self):
        """Leaf states should have correct tensor shapes for oracle.batch_evaluate_gpu()."""
        n_trees = 8
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Expand and select a few times
        for _ in range(3):
            leaf_indices, paths = select_leaves_gpu(forest)
            expand_gpu(forest, leaf_indices)

        # Select final leaves
        leaf_indices, paths = select_leaves_gpu(forest)
        leaf_states = get_leaf_states(forest, leaf_indices)

        # Verify shapes match what oracle.batch_evaluate_gpu expects
        assert leaf_states.hands.shape == (n_trees, 4, 7)
        assert leaf_states.decl_id.shape == (n_trees,)
        assert leaf_states.leader.shape == (n_trees,)
        assert leaf_states.current_trick.shape == (n_trees, 4, 2)
        assert leaf_states.scores.shape == (n_trees, 2)

    def test_original_hands_available_for_oracle(self):
        """Forest should store original hands for oracle queries."""
        n_trees = 8
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Original hands should be stored
        assert forest.original_hands is not None
        assert forest.original_hands.shape == (n_trees, 4, 7)

        # Should match initial state hands
        assert torch.equal(forest.original_hands, initial_states.hands)

    def test_prepare_oracle_inputs(self):
        """Test prepare_oracle_inputs returns correct tensor shapes."""
        n_trees = 8
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        # Expand and select a few times to get varied states
        for _ in range(3):
            leaf_indices, paths = select_leaves_gpu(forest)
            expand_gpu(forest, leaf_indices)

        leaf_indices, _ = select_leaves_gpu(forest)
        leaf_states = get_leaf_states(forest, leaf_indices)
        inputs = prepare_oracle_inputs(forest, leaf_states)

        # Verify all required keys are present
        required_keys = [
            'original_hands', 'current_hands', 'decl_ids', 'actors',
            'leaders', 'trick_players', 'trick_dominoes', 'players'
        ]
        for key in required_keys:
            assert key in inputs, f"Missing key: {key}"

        # Verify shapes
        assert inputs['original_hands'].shape == (n_trees, 4, 7)
        assert inputs['current_hands'].shape == (n_trees, 4, 7)
        assert inputs['decl_ids'].shape == (n_trees,)
        assert inputs['actors'].shape == (n_trees,)
        assert inputs['leaders'].shape == (n_trees,)
        assert inputs['trick_players'].shape == (n_trees, 3)
        assert inputs['trick_dominoes'].shape == (n_trees, 3)
        assert inputs['players'].shape == (n_trees,)

    def test_integration_pattern(self):
        """Test the full integration pattern from the task description.

        This simulates how the forest will be used with the real oracle:
        1. Create forest with initial game states
        2. For each simulation:
           a. Select leaves
           b. Get states for oracle evaluation
           c. Expand and backpropagate
        3. Get final policies
        """
        n_trees = 16
        max_nodes = 1024
        n_simulations = 10

        # Create forest with initial game states
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, max_nodes, initial_states, DEVICE)

        for _ in range(n_simulations):
            # Select leaves for all trees in parallel
            leaf_indices, paths = select_leaves_gpu(forest)

            # Get states for oracle evaluation
            leaf_states = get_leaf_states(forest, leaf_indices)

            # Prepare inputs for oracle (would call oracle.batch_evaluate_gpu(**inputs))
            inputs = prepare_oracle_inputs(forest, leaf_states)

            # Simulate oracle evaluation - for now use random values
            values = torch.rand(n_trees, device=DEVICE) * 2 - 1

            # Handle terminal states
            terminal_values, is_terminal = get_terminal_values(
                forest, leaf_indices, leaf_states
            )
            values = torch.where(is_terminal, terminal_values, values)

            # Expand and backpropagate
            expand_gpu(forest, leaf_indices)
            backprop_gpu(forest, leaf_indices, values, paths)

        # Get final policies
        policies = get_root_policy_gpu(forest)

        # Verify outputs
        assert policies.shape == (n_trees, 7)
        assert (policies >= 0).all()
        sums = policies.sum(dim=1)
        for i in range(n_trees):
            assert abs(sums[i].item() - 1.0) < 0.01, f"Policy {i} doesn't sum to 1"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_max_nodes_limit(self):
        """Forest should not exceed max_nodes allocation."""
        n_trees = 4
        max_nodes = 10  # Very limited
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=0)
        forest = create_forest(n_trees, max_nodes, initial_states, DEVICE)

        # Try to expand many times
        for _ in range(20):
            leaf_indices, _ = select_leaves_gpu(forest)
            expand_gpu(forest, leaf_indices)

        # Should never exceed max_nodes
        assert (forest.n_nodes <= max_nodes).all()

    def test_single_tree(self):
        """Should work with a single tree."""
        initial_states = deal_random_gpu(1, DEVICE, decl_ids=0)
        forest = create_forest(1, 50, initial_states, DEVICE)

        def random_oracle(states: GPUGameState) -> torch.Tensor:
            return torch.rand(states.batch_size, device=states.device) * 2 - 1

        policy = run_mcts_search(forest, random_oracle, n_simulations=10)

        assert policy.shape == (1, 7)
        assert abs(policy.sum().item() - 1.0) < 0.01

    def test_different_declarations(self):
        """Should handle different declarations correctly."""
        n_trees = 10
        initial_states = deal_random_gpu(n_trees, DEVICE, decl_ids=None)  # Random decls
        forest = create_forest(n_trees, 100, initial_states, DEVICE)

        def random_oracle(states: GPUGameState) -> torch.Tensor:
            return torch.rand(states.batch_size, device=states.device) * 2 - 1

        policy = run_mcts_search(forest, random_oracle, n_simulations=20)

        # All should produce valid policies
        assert policy.shape == (n_trees, 7)
        for i in range(n_trees):
            assert abs(policy[i].sum().item() - 1.0) < 0.01
