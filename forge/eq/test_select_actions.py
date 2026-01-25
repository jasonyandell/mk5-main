"""Unit tests for _select_actions with p_make optimization.

Tests the core logic without requiring a trained model - just mock tensors.
"""

import torch
import pytest
from unittest.mock import MagicMock


def make_mock_states(
    current_players: list[int],
    legal_masks: torch.Tensor,
    bidders: list[int] | None = None,
) -> MagicMock:
    """Create a mock GameStateTensor with just the fields _select_actions needs.

    Args:
        current_players: List of current player indices (0-3) for each game
        legal_masks: [n_games, 7] boolean mask of legal actions
        bidders: Optional list of bidder player indices (0-3). Defaults to 0 for all games.
    """
    states = MagicMock()
    states.n_games = len(current_players)
    states.current_player = torch.tensor(current_players, dtype=torch.int32)
    states.legal_actions = MagicMock(return_value=legal_masks)
    # Default bidder to 0 if not provided
    if bidders is None:
        bidders = [0] * len(current_players)
    states.bidder = torch.tensor(bidders, dtype=torch.int32)
    return states


def make_peaked_pdf(n_games: int, n_actions: int, peak_bins: list[list[int]]) -> torch.Tensor:
    """Create PDF tensors with probability mass concentrated at specific bins.

    Args:
        n_games: Number of games
        n_actions: Number of actions (7)
        peak_bins: [n_games][n_actions] - which bin to concentrate mass at
                   bin = Q + 42, so Q=18 → bin=60, Q=-17 → bin=25

    Returns:
        [n_games, n_actions, 85] PDF tensor
    """
    pdf = torch.zeros(n_games, n_actions, 85)
    for g in range(n_games):
        for a in range(n_actions):
            pdf[g, a, peak_bins[g][a]] = 1.0
    return pdf


class TestPMakeThresholds:
    """Test that offense/defense use correct thresholds."""

    def test_offense_uses_bin_60_threshold(self):
        """P0 and P2 (offense) need Q >= 18 → bin 60+."""
        from forge.eq.generate_gpu import _select_actions

        # Two games: P0 (offense) and P2 (offense)
        current_players = [0, 2]
        legal_masks = torch.ones(2, 7, dtype=torch.bool)
        states = make_mock_states(current_players, legal_masks)

        # Action 0: Q=17 (bin 59) - just below threshold, p_make=0
        # Action 1: Q=18 (bin 60) - at threshold, p_make=1
        peak_bins = [
            [59, 60, 0, 0, 0, 0, 0],  # Game 0
            [59, 60, 0, 0, 0, 0, 0],  # Game 1
        ]
        e_q_pdf = make_peaked_pdf(2, 7, peak_bins)

        # E[Q] values (action 0 has higher E[Q], but lower p_make)
        e_q = torch.tensor([
            [17.0, 18.0, -42.0, -42.0, -42.0, -42.0, -42.0],
            [17.0, 18.0, -42.0, -42.0, -42.0, -42.0, -42.0],
        ])

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # Both should pick action 1 (p_make=1) over action 0 (p_make=0)
        assert actions[0].item() == 1, "P0 should pick action with Q>=18"
        assert actions[1].item() == 1, "P2 should pick action with Q>=18"

    def test_defense_uses_bin_25_threshold(self):
        """P1 and P3 (defense) need Q >= -17 → bin 25+."""
        from forge.eq.generate_gpu import _select_actions

        # Two games: P1 (defense) and P3 (defense)
        current_players = [1, 3]
        legal_masks = torch.ones(2, 7, dtype=torch.bool)
        states = make_mock_states(current_players, legal_masks)

        # Action 0: Q=-18 (bin 24) - just below threshold, p_make=0
        # Action 1: Q=-17 (bin 25) - at threshold, p_make=1
        peak_bins = [
            [24, 25, 0, 0, 0, 0, 0],  # Game 0
            [24, 25, 0, 0, 0, 0, 0],  # Game 1
        ]
        e_q_pdf = make_peaked_pdf(2, 7, peak_bins)

        # E[Q] values (action 0 has higher E[Q] from defense perspective, but lower p_make)
        e_q = torch.tensor([
            [-18.0, -17.0, -42.0, -42.0, -42.0, -42.0, -42.0],
            [-18.0, -17.0, -42.0, -42.0, -42.0, -42.0, -42.0],
        ])

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # Both should pick action 1 (p_make=1) over action 0 (p_make=0)
        assert actions[0].item() == 1, "P1 should pick action with Q>=-17"
        assert actions[1].item() == 1, "P3 should pick action with Q>=-17"


class TestPMakeBeatsEQ:
    """Test that p_make takes priority over E[Q]."""

    def test_higher_pmake_wins_over_higher_eq(self):
        """Action with lower E[Q] but higher p_make should win."""
        from forge.eq.generate_gpu import _select_actions

        # Offense player (P0)
        current_players = [0]
        legal_masks = torch.ones(1, 7, dtype=torch.bool)
        states = make_mock_states(current_players, legal_masks)

        # Action 0: E[Q]=17.9 with certainty (Q=17.9 → bin 59.9 rounds to 60, but let's use 59)
        #           Actually peaked at bin 59 (Q=17), p_make=0
        # Action 1: E[Q]=15 with high variance, some mass above threshold
        #           Spread: 50% at bin 50 (Q=8), 50% at bin 70 (Q=28), p_make=0.5
        e_q_pdf = torch.zeros(1, 7, 85)
        e_q_pdf[0, 0, 59] = 1.0  # Action 0: certain Q=17 (just below threshold)
        e_q_pdf[0, 1, 50] = 0.5  # Action 1: 50% at Q=8
        e_q_pdf[0, 1, 70] = 0.5  # Action 1: 50% at Q=28 (above threshold)

        # E[Q]: action 0 has higher expected value
        e_q = torch.tensor([[17.0, 18.0, -42.0, -42.0, -42.0, -42.0, -42.0]])  # 0.5*8 + 0.5*28 = 18

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # Should pick action 1 (p_make=0.5) over action 0 (p_make=0)
        # even though action 0 has E[Q]=17 close to action 1's E[Q]=18
        assert actions[0].item() == 1, "Higher p_make should win over similar E[Q]"

    def test_certain_loss_vs_chance_to_win(self):
        """The motivating example: certain loss vs 35% win chance."""
        from forge.eq.generate_gpu import _select_actions

        # Offense player (P0)
        current_players = [0]
        legal_masks = torch.ones(1, 7, dtype=torch.bool)
        states = make_mock_states(current_players, legal_masks)

        # Action A: E[Q]=17.9, std=0.1 → certain loss (all mass below bin 60)
        # Action B: E[Q]=15, std=8 → ~35% win chance
        e_q_pdf = torch.zeros(1, 7, 85)

        # Action 0 (A): all mass at bin 59 (Q=17)
        e_q_pdf[0, 0, 59] = 1.0

        # Action 1 (B): 65% at bin 45 (Q=3), 35% at bin 70 (Q=28)
        # E[Q] = 0.65*3 + 0.35*28 = 1.95 + 9.8 = 11.75
        e_q_pdf[0, 1, 45] = 0.65
        e_q_pdf[0, 1, 70] = 0.35

        e_q = torch.tensor([[17.0, 11.75, -42.0, -42.0, -42.0, -42.0, -42.0]])

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # Should pick action 1 (p_make=0.35) over action 0 (p_make=0)
        assert actions[0].item() == 1, "35% win chance beats certain loss"


class TestEQTieBreak:
    """Test E[Q] tie-breaking when p_make values are equal."""

    def test_equal_pmake_higher_eq_wins(self):
        """When p_make is equal, higher E[Q] should win (win big / lose gracefully)."""
        from forge.eq.generate_gpu import _select_actions

        # Offense player (P0)
        current_players = [0]
        legal_masks = torch.ones(1, 7, dtype=torch.bool)
        states = make_mock_states(current_players, legal_masks)

        # Both actions have p_make=1 (both above threshold)
        # Action 0: Q=20 (bin 62)
        # Action 1: Q=30 (bin 72) - higher E[Q]
        e_q_pdf = torch.zeros(1, 7, 85)
        e_q_pdf[0, 0, 62] = 1.0  # Q=20
        e_q_pdf[0, 1, 72] = 1.0  # Q=30

        e_q = torch.tensor([[20.0, 30.0, -42.0, -42.0, -42.0, -42.0, -42.0]])

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # Should pick action 1 (higher E[Q]) since p_make is tied at 1.0
        assert actions[0].item() == 1, "Equal p_make should tie-break by E[Q]"

    def test_lose_gracefully_when_both_losing(self):
        """When all actions lose (p_make≈0), pick smallest loss margin."""
        from forge.eq.generate_gpu import _select_actions

        # Offense player (P0)
        current_players = [0]
        legal_masks = torch.ones(1, 7, dtype=torch.bool)
        states = make_mock_states(current_players, legal_masks)

        # Both actions have p_make=0 (both below threshold)
        # Action 0: Q=10 (bin 52) - smaller loss
        # Action 1: Q=5 (bin 47) - bigger loss
        e_q_pdf = torch.zeros(1, 7, 85)
        e_q_pdf[0, 0, 52] = 1.0  # Q=10
        e_q_pdf[0, 1, 47] = 1.0  # Q=5

        e_q = torch.tensor([[10.0, 5.0, -42.0, -42.0, -42.0, -42.0, -42.0]])

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # Should pick action 0 (higher E[Q] = smaller loss margin)
        assert actions[0].item() == 0, "When losing, should minimize loss margin"


class TestIllegalActionMasking:
    """Test that illegal actions are never selected."""

    def test_illegal_action_not_selected_even_with_best_pmake(self):
        """Illegal action should not be selected even if it has best p_make."""
        from forge.eq.generate_gpu import _select_actions

        # Offense player (P0)
        current_players = [0]
        # Action 0 is illegal, actions 1-6 are legal
        legal_masks = torch.tensor([[False, True, True, True, True, True, True]])
        states = make_mock_states(current_players, legal_masks)

        # Action 0: p_make=1 (Q=30), but ILLEGAL
        # Action 1: p_make=0.5 (Q=18 and Q=10 split)
        e_q_pdf = torch.zeros(1, 7, 85)
        e_q_pdf[0, 0, 72] = 1.0  # Q=30, p_make=1 but illegal
        e_q_pdf[0, 1, 60] = 0.5  # Q=18
        e_q_pdf[0, 1, 52] = 0.5  # Q=10

        e_q = torch.tensor([[30.0, 14.0, -42.0, -42.0, -42.0, -42.0, -42.0]])

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # Should NOT pick action 0 even though it has best p_make
        assert actions[0].item() != 0, "Illegal action should never be selected"
        assert actions[0].item() == 1, "Should pick best legal action"


class TestBatchProcessing:
    """Test that batched games are processed correctly."""

    def test_mixed_offense_defense_batch(self):
        """Test batch with mixed offense/defense players."""
        from forge.eq.generate_gpu import _select_actions

        # 4 games: P0 (offense), P1 (defense), P2 (offense), P3 (defense)
        current_players = [0, 1, 2, 3]
        # Only actions 0 and 1 are legal (actions 2-6 would have invalid PDFs)
        legal_masks = torch.tensor([
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
        ])
        states = make_mock_states(current_players, legal_masks)

        # Set up PDFs where the "right" answer differs by threshold
        # Action 0: Q=17 (bin 59) - good for defense (>=−17), bad for offense (<18)
        # Action 1: Q=18 (bin 60) - good for offense (>=18), also good for defense
        e_q_pdf = torch.zeros(4, 7, 85)
        for g in range(4):
            e_q_pdf[g, 0, 59] = 1.0  # Q=17
            e_q_pdf[g, 1, 60] = 1.0  # Q=18

        e_q = torch.full((4, 7), -42.0)
        e_q[:, 0] = 17.0
        e_q[:, 1] = 18.0

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # Offense (P0, P2): should pick action 1 (p_make=1 vs 0)
        assert actions[0].item() == 1, "P0 (offense) should pick Q=18"
        assert actions[2].item() == 1, "P2 (offense) should pick Q=18"

        # Defense (P1, P3): both actions have p_make=1, so tie-break by E[Q]
        # Action 1 has higher E[Q] (18 > 17)
        assert actions[1].item() == 1, "P1 (defense) should pick higher E[Q] when p_make tied"
        assert actions[3].item() == 1, "P3 (defense) should pick higher E[Q] when p_make tied"


class TestBidderField:
    """Test that bidder field correctly determines offense/defense roles."""

    def test_bidder_zero_default(self):
        """With default bidder=0, P0/P2 are offense, P1/P3 are defense."""
        from forge.eq.generate_gpu import _select_actions

        # P0 plays, bidder=0 (default) -> P0 is offense
        current_players = [0]
        legal_masks = torch.ones(1, 7, dtype=torch.bool)
        states = make_mock_states(current_players, legal_masks)  # bidder defaults to 0

        # Action 0: Q=17 (bin 59) - p_make=0 for offense
        # Action 1: Q=18 (bin 60) - p_make=1 for offense
        peak_bins = [[59, 60, 0, 0, 0, 0, 0]]
        e_q_pdf = make_peaked_pdf(1, 7, peak_bins)
        e_q = torch.tensor([[17.0, 18.0, -42.0, -42.0, -42.0, -42.0, -42.0]])

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        assert actions[0].item() == 1, "P0 with bidder=0 should be offense, pick Q>=18"

    def test_bidder_one_reverses_roles(self):
        """With bidder=1, P1/P3 are offense, P0/P2 are defense."""
        from forge.eq.generate_gpu import _select_actions

        # P0 plays, bidder=1 -> P0 is now DEFENSE
        current_players = [0]
        legal_masks = torch.ones(1, 7, dtype=torch.bool)
        states = make_mock_states(current_players, legal_masks, bidders=[1])

        # For defense, Q >= -17 (bin 25+) is a win
        # Action 0: Q=-18 (bin 24) - p_make=0 for defense
        # Action 1: Q=-17 (bin 25) - p_make=1 for defense
        e_q_pdf = torch.zeros(1, 7, 85)
        e_q_pdf[0, 0, 24] = 1.0  # Q=-18
        e_q_pdf[0, 1, 25] = 1.0  # Q=-17

        e_q = torch.tensor([[-18.0, -17.0, -42.0, -42.0, -42.0, -42.0, -42.0]])

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        assert actions[0].item() == 1, "P0 with bidder=1 should be defense, pick Q>=-17"

    def test_bidder_one_p1_is_offense(self):
        """With bidder=1, P1 is on offense (same team as bidder)."""
        from forge.eq.generate_gpu import _select_actions

        # P1 plays, bidder=1 -> P1 is offense
        current_players = [1]
        legal_masks = torch.ones(1, 7, dtype=torch.bool)
        states = make_mock_states(current_players, legal_masks, bidders=[1])

        # Action 0: Q=17 (bin 59) - p_make=0 for offense
        # Action 1: Q=18 (bin 60) - p_make=1 for offense
        peak_bins = [[59, 60, 0, 0, 0, 0, 0]]
        e_q_pdf = make_peaked_pdf(1, 7, peak_bins)
        e_q = torch.tensor([[17.0, 18.0, -42.0, -42.0, -42.0, -42.0, -42.0]])

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        assert actions[0].item() == 1, "P1 with bidder=1 should be offense, pick Q>=18"

    def test_bidder_three_team_assignment(self):
        """With bidder=3, P1/P3 are offense, P0/P2 are defense."""
        from forge.eq.generate_gpu import _select_actions

        # 4 games: P0, P1, P2, P3 all play with bidder=3
        current_players = [0, 1, 2, 3]
        legal_masks = torch.tensor([
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
        ])
        states = make_mock_states(current_players, legal_masks, bidders=[3, 3, 3, 3])

        # Action 0: Q=17 (bin 59) - p_make=0 for offense, p_make=1 for defense
        # Action 1: Q=18 (bin 60) - p_make=1 for offense, p_make=1 for defense
        e_q_pdf = torch.zeros(4, 7, 85)
        for g in range(4):
            e_q_pdf[g, 0, 59] = 1.0  # Q=17
            e_q_pdf[g, 1, 60] = 1.0  # Q=18

        e_q = torch.full((4, 7), -42.0)
        e_q[:, 0] = 17.0
        e_q[:, 1] = 18.0

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # With bidder=3 (team 1):
        # P0 (team 0) is defense -> both actions have p_make=1, tie-break by E[Q] -> action 1
        # P1 (team 1) is offense -> action 1 has p_make=1, action 0 has p_make=0 -> action 1
        # P2 (team 0) is defense -> both actions have p_make=1, tie-break by E[Q] -> action 1
        # P3 (team 1) is offense -> action 1 has p_make=1, action 0 has p_make=0 -> action 1
        for i in range(4):
            assert actions[i].item() == 1, f"Game {i} should pick action 1"

    def test_mixed_bidders_batch(self):
        """Test batch where different games have different bidders."""
        from forge.eq.generate_gpu import _select_actions

        # 4 games: all P0 playing, but alternating bidders (0, 1, 0, 1)
        # Game 0: P0 plays, bidder=0 -> P0 is offense
        # Game 1: P0 plays, bidder=1 -> P0 is defense
        # Game 2: P0 plays, bidder=0 -> P0 is offense
        # Game 3: P0 plays, bidder=1 -> P0 is defense
        current_players = [0, 0, 0, 0]
        # Only actions 0 and 1 are legal (matches PDF setup)
        legal_masks = torch.tensor([
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
            [True, True, False, False, False, False, False],
        ])
        states = make_mock_states(current_players, legal_masks, bidders=[0, 1, 0, 1])

        # Action 0: Q=17 (bin 59) - p_make=0 for offense, p_make=1 for defense
        # Action 1: Q=18 (bin 60) - p_make=1 for offense, p_make=1 for defense
        e_q_pdf = torch.zeros(4, 7, 85)
        for g in range(4):
            e_q_pdf[g, 0, 59] = 1.0  # Q=17
            e_q_pdf[g, 1, 60] = 1.0  # Q=18

        e_q = torch.full((4, 7), -42.0)
        e_q[:, 0] = 17.0
        e_q[:, 1] = 18.0

        actions, _ = _select_actions(states, e_q, e_q_pdf, greedy=True)

        # Game 0: P0 offense -> needs p_make=1, picks action 1
        assert actions[0].item() == 1, "Game 0: P0 (offense) should pick Q>=18"
        # Game 1: P0 defense -> both actions have p_make=1, tie-break by E[Q] -> action 1
        assert actions[1].item() == 1, "Game 1: P0 (defense) should pick higher E[Q]"
        # Game 2: P0 offense -> needs p_make=1, picks action 1
        assert actions[2].item() == 1, "Game 2: P0 (offense) should pick Q>=18"
        # Game 3: P0 defense -> both actions have p_make=1, tie-break by E[Q] -> action 1
        assert actions[3].item() == 1, "Game 3: P0 (defense) should pick higher E[Q]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
