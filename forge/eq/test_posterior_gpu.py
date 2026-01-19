"""Tests for GPU-native historical state reconstruction."""
import pytest
import torch
from forge.eq.posterior_gpu import reconstruct_past_states_gpu, PastStatesGPU


def test_reconstruct_basic():
    """Test basic state reconstruction with a single trick."""
    # Create simple history: 4 plays, 1 trick
    # Play 0: player 0 plays domino 5, leads with 5
    # Play 1: player 1 plays domino 10, follows 5
    # Play 2: player 2 plays domino 15, follows 5
    # Play 3: player 3 plays domino 20, follows 5
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    history[0, 0] = torch.tensor([0, 5, 5])   # lead
    history[0, 1] = torch.tensor([1, 10, 5])  # follow
    history[0, 2] = torch.tensor([2, 15, 5])  # follow
    history[0, 3] = torch.tensor([3, 20, 5])  # follow
    history_len = torch.tensor([4])

    past = reconstruct_past_states_gpu(history, history_len, window_k=4)

    assert past.actors.shape == (1, 4)
    assert past.actors[0].tolist() == [0, 1, 2, 3]
    assert past.observed_actions[0].tolist() == [5, 10, 15, 20]
    assert past.leaders[0].tolist() == [0, 0, 0, 0]  # All same trick, player 0 led
    assert past.valid_mask[0].tolist() == [True, True, True, True]

    # Check trick_lens: step 0 has 0 plays before, step 1 has 1, etc.
    assert past.trick_lens[0].tolist() == [0, 1, 2, 3]

    # Check played_before
    # Step 0: nothing played before
    assert past.played_before[0, 0].sum().item() == 0

    # Step 1: domino 5 played before
    assert past.played_before[0, 1, 5].item() == True
    assert past.played_before[0, 1].sum().item() == 1

    # Step 2: dominoes 5, 10 played before
    assert past.played_before[0, 2, 5].item() == True
    assert past.played_before[0, 2, 10].item() == True
    assert past.played_before[0, 2].sum().item() == 2

    # Step 3: dominoes 5, 10, 15 played before
    assert past.played_before[0, 3].sum().item() == 3


def test_reconstruct_multiple_tricks():
    """Test with multiple tricks."""
    # Trick 1: plays 0-3 with lead_domino=5
    # Trick 2: plays 4-5 with lead_domino=8
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    history[0, 0] = torch.tensor([0, 5, 5])
    history[0, 1] = torch.tensor([1, 10, 5])
    history[0, 2] = torch.tensor([2, 15, 5])
    history[0, 3] = torch.tensor([3, 20, 5])
    history[0, 4] = torch.tensor([2, 8, 8])   # New trick, player 2 leads
    history[0, 5] = torch.tensor([3, 12, 8])  # Follow
    history_len = torch.tensor([6])

    past = reconstruct_past_states_gpu(history, history_len, window_k=4)

    # Window covers plays 2,3,4,5
    assert past.actors[0].tolist() == [2, 3, 2, 3]
    assert past.observed_actions[0].tolist() == [15, 20, 8, 12]

    # Leaders: plays 2,3 have leader 0 (trick 1), plays 4,5 have leader 2 (trick 2)
    assert past.leaders[0].tolist() == [0, 0, 2, 2]

    # Trick lens:
    # Step 2 (idx 2): 2 plays in trick so far (plays 0, 1)
    # Step 3 (idx 3): 3 plays in trick so far (plays 0, 1, 2)
    # Step 4 (idx 4): 0 plays in trick (new trick, player is leading)
    # Step 5 (idx 5): 1 play in trick so far (play 4)
    assert past.trick_lens[0].tolist() == [2, 3, 0, 1]


def test_reconstruct_short_history():
    """Test with history shorter than window_k."""
    # Only 2 plays, but window_k=4
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    history[0, 0] = torch.tensor([0, 5, 5])
    history[0, 1] = torch.tensor([1, 10, 5])
    history_len = torch.tensor([2])

    past = reconstruct_past_states_gpu(history, history_len, window_k=4)

    # Only last 2 steps should be valid
    assert past.valid_mask[0].tolist() == [False, False, True, True]

    # Valid steps should have correct data
    assert past.actors[0, 2:].tolist() == [0, 1]
    assert past.observed_actions[0, 2:].tolist() == [5, 10]
    assert past.leaders[0, 2:].tolist() == [0, 0]


def test_reconstruct_batched():
    """Test with multiple games in parallel."""
    # Game 0: 3 plays in one trick
    # Game 1: 2 complete tricks (4 plays + 2 plays)
    history = torch.full((2, 28, 3), -1, dtype=torch.int32)

    # Game 0
    history[0, 0] = torch.tensor([0, 5, 5])
    history[0, 1] = torch.tensor([1, 10, 5])
    history[0, 2] = torch.tensor([2, 15, 5])

    # Game 1
    history[1, 0] = torch.tensor([0, 1, 1])
    history[1, 1] = torch.tensor([1, 2, 1])
    history[1, 2] = torch.tensor([2, 3, 1])
    history[1, 3] = torch.tensor([3, 4, 1])
    history[1, 4] = torch.tensor([3, 6, 6])  # New trick
    history[1, 5] = torch.tensor([0, 7, 6])

    history_len = torch.tensor([3, 6])

    past = reconstruct_past_states_gpu(history, history_len, window_k=3)

    # Game 0: all 3 steps valid
    assert past.valid_mask[0].tolist() == [True, True, True]
    assert past.actors[0].tolist() == [0, 1, 2]

    # Game 1: covers last 3 steps (3, 4, 5)
    assert past.valid_mask[1].tolist() == [True, True, True]
    assert past.actors[1].tolist() == [3, 3, 0]
    assert past.observed_actions[1].tolist() == [4, 6, 7]

    # Game 1 leaders: step 3 has leader 0, steps 4-5 have leader 3
    assert past.leaders[1].tolist() == [0, 3, 3]


def test_reconstruct_trick_plays():
    """Test that trick_plays correctly captures plays before current step."""
    # Create a complete trick: 4 plays
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    history[0, 0] = torch.tensor([0, 5, 5])
    history[0, 1] = torch.tensor([1, 10, 5])
    history[0, 2] = torch.tensor([2, 15, 5])
    history[0, 3] = torch.tensor([3, 20, 5])
    history_len = torch.tensor([4])

    past = reconstruct_past_states_gpu(history, history_len, window_k=4)

    # Step 0: leading, no plays before
    assert past.trick_lens[0, 0].item() == 0
    assert (past.trick_plays[0, 0] == -1).all().item()

    # Step 1: 1 play before (0)
    assert past.trick_lens[0, 1].item() == 1
    assert past.trick_plays[0, 1, 0].tolist() == [0, 5]
    assert (past.trick_plays[0, 1, 1:] == -1).all().item()

    # Step 2: 2 plays before (0, 1)
    assert past.trick_lens[0, 2].item() == 2
    assert past.trick_plays[0, 2, 0].tolist() == [0, 5]
    assert past.trick_plays[0, 2, 1].tolist() == [1, 10]
    assert (past.trick_plays[0, 2, 2:] == -1).all().item()

    # Step 3: 3 plays before (0, 1, 2)
    assert past.trick_lens[0, 3].item() == 3
    assert past.trick_plays[0, 3, 0].tolist() == [0, 5]
    assert past.trick_plays[0, 3, 1].tolist() == [1, 10]
    assert past.trick_plays[0, 3, 2].tolist() == [2, 15]


def test_reconstruct_played_before_cumulative():
    """Test that played_before correctly accumulates across steps."""
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    history[0, 0] = torch.tensor([0, 1, 1])
    history[0, 1] = torch.tensor([1, 5, 1])
    history[0, 2] = torch.tensor([2, 10, 1])
    history[0, 3] = torch.tensor([3, 15, 1])
    history_len = torch.tensor([4])

    past = reconstruct_past_states_gpu(history, history_len, window_k=4)

    # Check each step's played_before
    expected_played = [
        [],              # Step 0: nothing before
        [1],             # Step 1: domino 1 played
        [1, 5],          # Step 2: dominoes 1, 5 played
        [1, 5, 10],      # Step 3: dominoes 1, 5, 10 played
    ]

    for k, expected in enumerate(expected_played):
        played = past.played_before[0, k].nonzero(as_tuple=True)[0].tolist()
        assert played == expected, f"Step {k}: expected {expected}, got {played}"


def test_reconstruct_empty_history():
    """Test with empty history (no plays yet)."""
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    history_len = torch.tensor([0])

    past = reconstruct_past_states_gpu(history, history_len, window_k=2)

    # All steps should be invalid
    assert past.valid_mask[0].tolist() == [False, False]


def test_reconstruct_device_placement():
    """Test that output tensors are on correct device."""
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    history[0, 0] = torch.tensor([0, 5, 5])
    history_len = torch.tensor([1])

    # Test CPU
    past_cpu = reconstruct_past_states_gpu(history, history_len, window_k=2, device='cpu')
    assert past_cpu.actors.device.type == 'cpu'
    assert past_cpu.played_before.device.type == 'cpu'

    # Test CUDA if available
    if torch.cuda.is_available():
        history_cuda = history.cuda()
        history_len_cuda = history_len.cuda()
        past_cuda = reconstruct_past_states_gpu(
            history_cuda, history_len_cuda, window_k=2, device='cuda'
        )
        assert past_cuda.actors.device.type == 'cuda'
        assert past_cuda.played_before.device.type == 'cuda'


def test_reconstruct_step_indices():
    """Test that step_indices are correctly computed."""
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    for i in range(6):
        history[0, i] = torch.tensor([i % 4, i, i])
    history_len = torch.tensor([6])

    past = reconstruct_past_states_gpu(history, history_len, window_k=3)

    # Should reconstruct steps 3, 4, 5
    assert past.step_indices[0].tolist() == [3, 4, 5]
    assert past.valid_mask[0].tolist() == [True, True, True]


def test_reconstruct_window_larger_than_history():
    """Test window_k much larger than actual history."""
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    history[0, 0] = torch.tensor([0, 5, 5])
    history_len = torch.tensor([1])

    past = reconstruct_past_states_gpu(history, history_len, window_k=10)

    # Only last step should be valid
    assert past.valid_mask[0, :9].tolist() == [False] * 9
    assert past.valid_mask[0, 9].item() == True
    assert past.actors[0, 9].item() == 0
    assert past.observed_actions[0, 9].item() == 5


def test_reconstruct_vectorized_vs_original():
    """Verify vectorized implementation matches original loop-based implementation.

    This is the key correctness test for the vectorization optimization.
    Tests multiple scenarios to ensure bit-for-bit compatibility.
    """
    import torch
    from forge.eq.posterior_gpu import reconstruct_past_states_gpu

    # Test scenarios with varying complexity
    test_cases = [
        # (name, history_len, window_k)
        ("single_trick", 4, 4),
        ("two_tricks", 8, 6),
        ("partial_history", 3, 8),
        ("large_window", 20, 12),
    ]

    for name, hist_len, window_k in test_cases:
        # Generate random but valid history
        N = 10  # Test with batch of 10 games
        max_len = 28  # Maximum history length

        history = torch.full((N, max_len, 3), -1, dtype=torch.int32)
        history_lens = torch.full((N,), hist_len, dtype=torch.int32)

        # Create realistic history with trick structure
        for n in range(N):
            trick_leader = 0
            lead_domino = n % 28  # Different lead for each game
            for step in range(hist_len):
                player = (trick_leader + step) % 4
                domino = (n * 7 + step) % 28  # Ensure unique dominoes per game
                # Start new trick every 4 plays
                if step % 4 == 0:
                    trick_leader = player
                    lead_domino = domino
                history[n, step] = torch.tensor([player, domino, lead_domino])

        # Run both implementations
        past_vectorized = reconstruct_past_states_gpu(
            history, history_lens, window_k, vectorized=True
        )
        past_original = reconstruct_past_states_gpu(
            history, history_lens, window_k, vectorized=False
        )

        # Compare all outputs
        assert torch.equal(past_vectorized.played_before, past_original.played_before), \
            f"{name}: played_before mismatch"
        assert torch.equal(past_vectorized.trick_plays, past_original.trick_plays), \
            f"{name}: trick_plays mismatch"
        assert torch.equal(past_vectorized.trick_lens, past_original.trick_lens), \
            f"{name}: trick_lens mismatch"
        assert torch.equal(past_vectorized.leaders, past_original.leaders), \
            f"{name}: leaders mismatch"
        assert torch.equal(past_vectorized.actors, past_original.actors), \
            f"{name}: actors mismatch"
        assert torch.equal(past_vectorized.observed_actions, past_original.observed_actions), \
            f"{name}: observed_actions mismatch"
        assert torch.equal(past_vectorized.step_indices, past_original.step_indices), \
            f"{name}: step_indices mismatch"
        assert torch.equal(past_vectorized.valid_mask, past_original.valid_mask), \
            f"{name}: valid_mask mismatch"


def test_reconstruct_vectorized_random_stress():
    """Stress test vectorized implementation with random data."""
    import torch
    from forge.eq.posterior_gpu import reconstruct_past_states_gpu

    # Run multiple random tests
    for seed in range(5):
        torch.manual_seed(seed)

        N = 50
        max_len = 28
        K = 8

        # Random history lengths
        history_lens = torch.randint(1, max_len, (N,), dtype=torch.int32)

        # Random history (realistic structure)
        history = torch.full((N, max_len, 3), -1, dtype=torch.int32)
        for n in range(N):
            current_lead = torch.randint(0, 28, (1,)).item()
            for step in range(history_lens[n].item()):
                player = step % 4
                domino = torch.randint(0, 28, (1,)).item()
                # New trick every 4 plays
                if step % 4 == 0:
                    current_lead = domino
                history[n, step] = torch.tensor([player, domino, current_lead])

        # Compare implementations
        past_vec = reconstruct_past_states_gpu(history, history_lens, K, vectorized=True)
        past_orig = reconstruct_past_states_gpu(history, history_lens, K, vectorized=False)

        assert torch.equal(past_vec.trick_plays, past_orig.trick_plays), \
            f"Seed {seed}: trick_plays mismatch"
        assert torch.equal(past_vec.trick_lens, past_orig.trick_lens), \
            f"Seed {seed}: trick_lens mismatch"
        assert torch.equal(past_vec.leaders, past_orig.leaders), \
            f"Seed {seed}: leaders mismatch"


def test_reconstruct_vs_cpu_reference():
    """Test that GPU implementation matches CPU reference algorithm.

    This test replicates the logic from forge/eq/posterior.py:_score_all_steps_batched
    lines 725-761 to verify our GPU implementation produces identical results.
    """
    # Create a complex history with multiple tricks
    # Trick 1: plays 0-3 (lead=2)
    # Trick 2: plays 4-7 (lead=7)
    # Trick 3: plays 8-9 (lead=1, incomplete)
    history_cpu = [
        (0, 2, 2),   # 0: player 0 leads with domino 2
        (1, 5, 2),   # 1: player 1 follows
        (2, 10, 2),  # 2: player 2 follows
        (3, 15, 2),  # 3: player 3 follows
        (3, 7, 7),   # 4: player 3 leads new trick with domino 7
        (0, 11, 7),  # 5: player 0 follows
        (1, 14, 7),  # 6: player 1 follows
        (2, 18, 7),  # 7: player 2 follows
        (2, 1, 1),   # 8: player 2 leads new trick with domino 1
        (3, 4, 1),   # 9: player 3 follows
    ]

    # Convert to GPU format
    history = torch.full((1, 28, 3), -1, dtype=torch.int32)
    for i, (player, domino, lead) in enumerate(history_cpu):
        history[0, i] = torch.tensor([player, domino, lead])
    history_len = torch.tensor([10])

    # Test with window_k=6 (covers plays 4-9)
    window_k = 6
    window_start = 10 - window_k  # = 4

    past = reconstruct_past_states_gpu(history, history_len, window_k=window_k)

    # Compute expected results using CPU algorithm
    for k in range(window_k):
        step_idx = window_start + k
        actor, observed_domino, lead_domino = history_cpu[step_idx]

        # Expected played_before
        expected_played = {history_cpu[i][1] for i in range(step_idx)}
        actual_played = set(past.played_before[0, k].nonzero(as_tuple=True)[0].tolist())
        assert actual_played == expected_played, (
            f"Step {k} (idx {step_idx}): played_before mismatch. "
            f"Expected {expected_played}, got {actual_played}"
        )

        # Expected trick_start
        trick_start = step_idx
        while trick_start > 0:
            prev_idx = trick_start - 1
            _, _, prev_lead = history_cpu[prev_idx]
            if prev_lead == lead_domino:
                trick_start = prev_idx
            else:
                break

        # Expected current_trick
        expected_trick = [(p, d) for p, d, _ in history_cpu[trick_start:step_idx]]
        expected_leader = expected_trick[0][0] if expected_trick else actor

        # Check leader
        actual_leader = past.leaders[0, k].item()
        assert actual_leader == expected_leader, (
            f"Step {k} (idx {step_idx}): leader mismatch. "
            f"Expected {expected_leader}, got {actual_leader}"
        )

        # Check trick_lens
        expected_trick_len = len(expected_trick)
        actual_trick_len = past.trick_lens[0, k].item()
        assert actual_trick_len == expected_trick_len, (
            f"Step {k} (idx {step_idx}): trick_len mismatch. "
            f"Expected {expected_trick_len}, got {actual_trick_len}"
        )

        # Check trick_plays
        for i, (p, d) in enumerate(expected_trick):
            assert past.trick_plays[0, k, i, 0].item() == p, (
                f"Step {k}, trick play {i}: player mismatch"
            )
            assert past.trick_plays[0, k, i, 1].item() == d, (
                f"Step {k}, trick play {i}: domino mismatch"
            )

        # Check remaining slots are -1
        for i in range(expected_trick_len, 3):
            assert past.trick_plays[0, k, i, 0].item() == -1
            assert past.trick_plays[0, k, i, 1].item() == -1

        # Check actor and observed_action
        assert past.actors[0, k].item() == actor
        assert past.observed_actions[0, k].item() == observed_domino


# =============================================================================
# Tests for compute_posterior_weights_gpu
# =============================================================================

def test_posterior_weights_basic():
    """Test basic posterior weight computation with simple Q-values."""
    from forge.eq.posterior_gpu import compute_posterior_weights_gpu

    # Setup: 1 game, 3 worlds, 2 steps
    N, M, K = 1, 3, 2
    device = 'cpu'

    # Create simple worlds: each world has different hands
    # World 0: player 0 has [0, 1, 2, 3, 4, 5, 6]
    # World 1: player 0 has [7, 8, 9, 10, 11, 12, 13]
    # World 2: player 0 has [14, 15, 16, 17, 18, 19, 20]
    worlds = torch.zeros(N, M, 4, 7, dtype=torch.int32, device=device)
    worlds[0, 0, 0] = torch.arange(0, 7)
    worlds[0, 1, 0] = torch.arange(7, 14)
    worlds[0, 2, 0] = torch.arange(14, 21)

    # Observed actions: player 0 played domino 1, then domino 2
    observed_actions = torch.tensor([[1, 2]], dtype=torch.int32, device=device)  # [N, K]
    actors = torch.tensor([[0, 0]], dtype=torch.int32, device=device)  # [N, K]

    # Q-values: [N, M, K, 7]
    # For simplicity, make Q-values favor observed actions in world 0
    q_past = torch.zeros(N, M, K, 7, dtype=torch.float32, device=device)

    # Step 0 (obs=1): world 0 slot 1 has high Q
    q_past[0, 0, 0, 1] = 10.0  # Observed action in world 0
    q_past[0, 1, 0, :] = 1.0   # World 1 has uniform low Q
    q_past[0, 2, 0, :] = 1.0   # World 2 has uniform low Q

    # Step 1 (obs=2): world 0 slot 2 has high Q
    q_past[0, 0, 1, 2] = 10.0  # Observed action in world 0
    q_past[0, 1, 1, :] = 1.0   # World 1 has uniform low Q
    q_past[0, 2, 1, :] = 1.0   # World 2 has uniform low Q

    # Legal masks: all actions legal for simplicity
    legal_masks = torch.ones(N, M, K, 7, dtype=torch.bool, device=device)

    # Compute weights
    weights, diag = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=1.0,
        uniform_mix=0.0,  # No uniform mix for this test
    )

    # World 0 should have highest weight (observed actions have high Q)
    assert weights.shape == (N, M)
    assert weights[0, 0] > weights[0, 1]
    assert weights[0, 0] > weights[0, 2]

    # Weights should sum to 1
    assert torch.allclose(weights.sum(dim=1), torch.ones(N))

    # Check diagnostics
    assert diag['ess'].shape == (N,)
    assert diag['max_w'].shape == (N,)
    assert diag['entropy'].shape == (N,)
    assert diag['k_eff'].shape == (N,)


def test_posterior_weights_invalid_observation():
    """Test that invalid observations (domino not in hand) get zero weight."""
    from forge.eq.posterior_gpu import compute_posterior_weights_gpu

    N, M, K = 1, 2, 1
    device = 'cpu'

    # World 0: player 0 has [0, 1, 2, 3, 4, 5, 6]
    # World 1: player 0 has [7, 8, 9, 10, 11, 12, 13]
    worlds = torch.zeros(N, M, 4, 7, dtype=torch.int32, device=device)
    worlds[0, 0, 0] = torch.arange(0, 7)
    worlds[0, 1, 0] = torch.arange(7, 14)

    # Observed action: domino 1 (only in world 0, not in world 1)
    observed_actions = torch.tensor([[1]], dtype=torch.int32, device=device)
    actors = torch.tensor([[0]], dtype=torch.int32, device=device)

    # Q-values: uniform
    q_past = torch.ones(N, M, K, 7, dtype=torch.float32, device=device)
    legal_masks = torch.ones(N, M, K, 7, dtype=torch.bool, device=device)

    weights, diag = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=1.0,
        uniform_mix=0.0,
    )

    # World 1 should have near-zero weight (invalid observation)
    assert weights[0, 0] > 0.99  # World 0 gets almost all weight
    assert weights[0, 1] < 0.01  # World 1 gets almost no weight


def test_posterior_weights_uniform_mix():
    """Test that uniform_mix parameter works correctly."""
    from forge.eq.posterior_gpu import compute_posterior_weights_gpu

    N, M, K = 1, 3, 1
    device = 'cpu'

    # All worlds have same basic hands, but we'll vary Q-values
    worlds = torch.zeros(N, M, 4, 7, dtype=torch.int32, device=device)
    for m in range(M):
        worlds[0, m, 0] = torch.arange(0, 7)  # All have dominoes 0-6

    observed_actions = torch.tensor([[1]], dtype=torch.int32, device=device)
    actors = torch.tensor([[0]], dtype=torch.int32, device=device)

    # Make world 0 strongly favored with very high Q for observed action
    q_past = torch.ones(N, M, K, 7, dtype=torch.float32, device=device)
    q_past[0, 0, 0, 1] = 100.0  # World 0: high Q for obs action
    q_past[0, 1, 0, 1] = 1.0    # World 1: low Q for obs action
    q_past[0, 2, 0, 1] = 1.0    # World 2: low Q for obs action
    legal_masks = torch.ones(N, M, K, 7, dtype=torch.bool, device=device)

    # Without uniform mix
    weights_no_mix, _ = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=1.0,
        uniform_mix=0.0,
    )

    # With uniform mix
    weights_with_mix, _ = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=1.0,
        uniform_mix=0.3,  # 30% uniform
    )

    # uniform_mix should make weights more balanced
    max_w_no_mix = weights_no_mix.max().item()
    max_w_with_mix = weights_with_mix.max().item()
    assert max_w_with_mix < max_w_no_mix


def test_posterior_weights_ess_computation():
    """Test ESS (Effective Sample Size) computation."""
    from forge.eq.posterior_gpu import compute_posterior_weights_gpu

    N, M, K = 2, 10, 1
    device = 'cpu'

    # Create dummy data
    worlds = torch.zeros(N, M, 4, 7, dtype=torch.int32, device=device)
    for m in range(M):
        worlds[:, m, 0] = torch.arange(m * 7, (m + 1) * 7) % 28

    observed_actions = torch.zeros(N, K, dtype=torch.int32, device=device)
    actors = torch.zeros(N, K, dtype=torch.int32, device=device)
    q_past = torch.randn(N, M, K, 7, dtype=torch.float32, device=device)
    legal_masks = torch.ones(N, M, K, 7, dtype=torch.bool, device=device)

    weights, diag = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=1.0,
        uniform_mix=0.1,
    )

    # Check ESS formula: ESS = 1 / sum(w^2)
    for n in range(N):
        expected_ess = 1.0 / (weights[n] ** 2).sum().item()
        actual_ess = diag['ess'][n].item()
        assert abs(expected_ess - actual_ess) < 1e-5

    # Check that ESS <= M (always true)
    assert (diag['ess'] <= M).all()

    # Check entropy and k_eff relationship
    for n in range(N):
        assert abs(diag['k_eff'][n].item() - torch.exp(diag['entropy'][n]).item()) < 1e-5


def test_posterior_weights_multi_game():
    """Test batched computation across multiple games."""
    from forge.eq.posterior_gpu import compute_posterior_weights_gpu

    N, M, K = 3, 5, 2
    device = 'cpu'

    # Create different worlds for each game
    worlds = torch.randint(0, 28, (N, M, 4, 7), dtype=torch.int32, device=device)

    # Make sure observed actions are in at least one world per game
    observed_actions = torch.randint(0, 7, (N, K), dtype=torch.int32, device=device)
    # Place observed actions in world 0 for each game
    actors = torch.zeros(N, K, dtype=torch.int32, device=device)
    for n in range(N):
        for k in range(K):
            worlds[n, 0, 0, observed_actions[n, k]] = observed_actions[n, k]

    q_past = torch.randn(N, M, K, 7, dtype=torch.float32, device=device)
    legal_masks = torch.ones(N, M, K, 7, dtype=torch.bool, device=device)

    weights, diag = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=1.0,
        uniform_mix=0.1,
    )

    # Check shapes
    assert weights.shape == (N, M)
    assert diag['ess'].shape == (N,)
    assert diag['max_w'].shape == (N,)

    # Each game should have normalized weights
    for n in range(N):
        assert torch.allclose(weights[n].sum(), torch.tensor(1.0), atol=1e-5)


def test_posterior_weights_temperature():
    """Test that tau (temperature) affects weight concentration."""
    from forge.eq.posterior_gpu import compute_posterior_weights_gpu

    N, M, K = 1, 5, 1
    device = 'cpu'

    worlds = torch.zeros(N, M, 4, 7, dtype=torch.int32, device=device)
    for m in range(M):
        worlds[0, m, 0] = torch.arange(m * 7, (m + 1) * 7) % 28

    observed_actions = torch.tensor([[0]], dtype=torch.int32, device=device)
    actors = torch.zeros(N, K, dtype=torch.int32, device=device)

    # World 0 has much higher Q for observed action
    q_past = torch.ones(N, M, K, 7, dtype=torch.float32, device=device)
    q_past[0, 0, 0, 0] = 10.0

    legal_masks = torch.ones(N, M, K, 7, dtype=torch.bool, device=device)

    # Low temperature (tau=0.1) should concentrate weights more
    weights_low_tau, diag_low = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=0.1,
        uniform_mix=0.0,
    )

    # High temperature (tau=10.0) should spread weights more
    weights_high_tau, diag_high = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=10.0,
        uniform_mix=0.0,
    )

    # Low tau should have higher max weight
    assert diag_low['max_w'][0] > diag_high['max_w'][0]

    # Low tau should have lower ESS (more concentrated)
    assert diag_low['ess'][0] < diag_high['ess'][0]


def test_posterior_weights_log_stability():
    """Test numerical stability with extreme Q-values."""
    from forge.eq.posterior_gpu import compute_posterior_weights_gpu

    N, M, K = 1, 3, 1
    device = 'cpu'

    worlds = torch.zeros(N, M, 4, 7, dtype=torch.int32, device=device)
    worlds[0, 0, 0] = torch.arange(0, 7)
    worlds[0, 1, 0] = torch.arange(7, 14)
    worlds[0, 2, 0] = torch.arange(14, 21)

    observed_actions = torch.tensor([[1]], dtype=torch.int32, device=device)
    actors = torch.zeros(N, K, dtype=torch.int32, device=device)

    # Extreme Q-values
    q_past = torch.zeros(N, M, K, 7, dtype=torch.float32, device=device)
    q_past[0, 0, 0, 1] = 1000.0  # Very high
    q_past[0, 1, 0, :] = -1000.0  # Very low
    q_past[0, 2, 0, :] = 0.0

    legal_masks = torch.ones(N, M, K, 7, dtype=torch.bool, device=device)

    # Should not produce NaN or Inf
    weights, diag = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=1.0,
        uniform_mix=0.0,
    )

    assert not torch.isnan(weights).any()
    assert not torch.isinf(weights).any()
    assert torch.allclose(weights.sum(dim=1), torch.ones(N))


def test_posterior_weights_legal_masking():
    """Test that illegal actions don't affect weight computation."""
    from forge.eq.posterior_gpu import compute_posterior_weights_gpu

    N, M, K = 1, 2, 1
    device = 'cpu'

    worlds = torch.zeros(N, M, 4, 7, dtype=torch.int32, device=device)
    worlds[0, 0, 0] = torch.arange(0, 7)
    worlds[0, 1, 0] = torch.arange(7, 14)

    observed_actions = torch.tensor([[1]], dtype=torch.int32, device=device)
    actors = torch.zeros(N, K, dtype=torch.int32, device=device)

    q_past = torch.ones(N, M, K, 7, dtype=torch.float32, device=device)

    # Make slot 1 have high Q in both worlds
    q_past[0, 0, 0, 1] = 10.0
    q_past[0, 1, 0, 1] = 10.0

    # But make slot 1 illegal in world 1
    legal_masks = torch.ones(N, M, K, 7, dtype=torch.bool, device=device)
    legal_masks[0, 1, 0, 1] = False  # Slot 1 illegal in world 1

    weights, _ = compute_posterior_weights_gpu(
        q_past=q_past,
        legal_masks=legal_masks,
        observed_actions=observed_actions,
        worlds=worlds,
        actors=actors,
        tau=1.0,
        uniform_mix=0.0,
    )

    # World 0 should still get weight even though world 1's action is illegal
    # (World 1 should compute advantage over legal actions only)
    assert weights[0, 0] > 0
    assert weights[0, 1] > 0
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
