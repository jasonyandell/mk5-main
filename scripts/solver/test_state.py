"""
Tests for state packing/unpacking.

Run with: python -m pytest scripts/solver/test_state.py -v
"""

import torch
import pytest
from state import (
    POPCOUNT,
    pack_state,
    unpack_remaining,
    unpack_score,
    unpack_leader,
    unpack_trick_len,
    unpack_plays,
    compute_level,
    compute_team,
    compute_terminal_value,
)


class TestPopcount:
    """Test the POPCOUNT lookup table."""

    def test_popcount_shape(self):
        assert POPCOUNT.shape == (128,)

    def test_popcount_values(self):
        # Test specific known values
        assert POPCOUNT[0] == 0  # 0b0000000
        assert POPCOUNT[1] == 1  # 0b0000001
        assert POPCOUNT[127] == 7  # 0b1111111
        assert POPCOUNT[0b1010101] == 4  # 85
        assert POPCOUNT[0b0110110] == 4  # 54

    def test_popcount_all_values(self):
        for i in range(128):
            expected = bin(i).count("1")
            assert POPCOUNT[i] == expected, f"POPCOUNT[{i}] = {POPCOUNT[i]}, expected {expected}"


class TestRoundTrip:
    """Test pack/unpack round trips."""

    def test_single_state_roundtrip(self):
        # Create a single test state
        remaining = torch.tensor([[0b1111111, 0b1111111, 0b1111111, 0b1111111]], dtype=torch.int64)
        score = torch.tensor([21], dtype=torch.int64)
        leader = torch.tensor([2], dtype=torch.int64)
        trick_len = torch.tensor([1], dtype=torch.int64)
        p0 = torch.tensor([3], dtype=torch.int64)
        p1 = torch.tensor([7], dtype=torch.int64)  # N/A
        p2 = torch.tensor([7], dtype=torch.int64)  # N/A

        # Pack
        packed = pack_state(remaining, score, leader, trick_len, p0, p1, p2)

        # Unpack and verify
        assert torch.equal(unpack_remaining(packed), remaining)
        assert torch.equal(unpack_score(packed), score)
        assert torch.equal(unpack_leader(packed), leader)
        assert torch.equal(unpack_trick_len(packed), trick_len)
        up0, up1, up2 = unpack_plays(packed)
        assert torch.equal(up0, p0)
        assert torch.equal(up1, p1)
        assert torch.equal(up2, p2)

    def test_batched_roundtrip(self):
        N = 100
        torch.manual_seed(42)

        # Generate random valid states
        remaining = torch.randint(0, 128, (N, 4), dtype=torch.int64)
        score = torch.randint(0, 43, (N,), dtype=torch.int64)
        leader = torch.randint(0, 4, (N,), dtype=torch.int64)
        trick_len = torch.randint(0, 4, (N,), dtype=torch.int64)
        p0 = torch.randint(0, 8, (N,), dtype=torch.int64)
        p1 = torch.randint(0, 8, (N,), dtype=torch.int64)
        p2 = torch.randint(0, 8, (N,), dtype=torch.int64)

        # Pack
        packed = pack_state(remaining, score, leader, trick_len, p0, p1, p2)

        # Unpack and verify
        assert torch.equal(unpack_remaining(packed), remaining)
        assert torch.equal(unpack_score(packed), score)
        assert torch.equal(unpack_leader(packed), leader)
        assert torch.equal(unpack_trick_len(packed), trick_len)
        up0, up1, up2 = unpack_plays(packed)
        assert torch.equal(up0, p0)
        assert torch.equal(up1, p1)
        assert torch.equal(up2, p2)

    def test_boundary_values_roundtrip(self):
        """Test with boundary values for each field."""
        # Min values
        remaining_min = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        score_min = torch.tensor([0], dtype=torch.int64)
        leader_min = torch.tensor([0], dtype=torch.int64)
        trick_len_min = torch.tensor([0], dtype=torch.int64)
        p_min = torch.tensor([0], dtype=torch.int64)

        packed_min = pack_state(remaining_min, score_min, leader_min, trick_len_min, p_min, p_min, p_min)
        assert torch.equal(unpack_remaining(packed_min), remaining_min)
        assert torch.equal(unpack_score(packed_min), score_min)

        # Max values
        remaining_max = torch.tensor([[127, 127, 127, 127]], dtype=torch.int64)
        score_max = torch.tensor([42], dtype=torch.int64)
        leader_max = torch.tensor([3], dtype=torch.int64)
        trick_len_max = torch.tensor([3], dtype=torch.int64)
        p_max = torch.tensor([7], dtype=torch.int64)

        packed_max = pack_state(remaining_max, score_max, leader_max, trick_len_max, p_max, p_max, p_max)
        assert torch.equal(unpack_remaining(packed_max), remaining_max)
        assert torch.equal(unpack_score(packed_max), score_max)
        assert torch.equal(unpack_leader(packed_max), leader_max)
        assert torch.equal(unpack_trick_len(packed_max), trick_len_max)
        up0, up1, up2 = unpack_plays(packed_max)
        assert torch.equal(up0, p_max)
        assert torch.equal(up1, p_max)
        assert torch.equal(up2, p_max)


class TestBitAlignment:
    """Test that each field occupies the expected bit positions."""

    def test_remaining0_bits_0_to_6(self):
        remaining = torch.tensor([[0b1111111, 0, 0, 0]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, zeros)
        assert packed[0] == 0b1111111  # bits 0-6

    def test_remaining1_bits_7_to_13(self):
        remaining = torch.tensor([[0, 0b1111111, 0, 0]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, zeros)
        assert packed[0] == 0b1111111 << 7  # bits 7-13

    def test_remaining2_bits_14_to_20(self):
        remaining = torch.tensor([[0, 0, 0b1111111, 0]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, zeros)
        assert packed[0] == 0b1111111 << 14  # bits 14-20

    def test_remaining3_bits_21_to_27(self):
        remaining = torch.tensor([[0, 0, 0, 0b1111111]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, zeros)
        assert packed[0] == 0b1111111 << 21  # bits 21-27

    def test_score_bits_28_to_33(self):
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        score = torch.tensor([0b111111], dtype=torch.int64)  # 63 max for 6 bits, but valid is 0-42
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, score, zeros, zeros, zeros, zeros, zeros)
        assert packed[0] == 0b111111 << 28  # bits 28-33

    def test_leader_bits_34_to_35(self):
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        leader = torch.tensor([0b11], dtype=torch.int64)  # 3
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, leader, zeros, zeros, zeros, zeros)
        assert packed[0] == 0b11 << 34  # bits 34-35

    def test_trick_len_bits_36_to_37(self):
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        trick_len = torch.tensor([0b11], dtype=torch.int64)  # 3
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, trick_len, zeros, zeros, zeros)
        assert packed[0] == 0b11 << 36  # bits 36-37

    def test_p0_bits_38_to_40(self):
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        p0 = torch.tensor([0b111], dtype=torch.int64)  # 7
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, p0, zeros, zeros)
        assert packed[0] == 0b111 << 38  # bits 38-40

    def test_p1_bits_41_to_43(self):
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        p1 = torch.tensor([0b111], dtype=torch.int64)  # 7
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, p1, zeros)
        assert packed[0] == 0b111 << 41  # bits 41-43

    def test_p2_bits_44_to_46(self):
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        p2 = torch.tensor([0b111], dtype=torch.int64)  # 7
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, p2)
        assert packed[0] == 0b111 << 44  # bits 44-46

    def test_no_bit_overlap(self):
        """Verify that setting each field independently doesn't affect others."""
        remaining = torch.tensor([[0b1010101, 0b0101010, 0b1100110, 0b0011001]], dtype=torch.int64)
        score = torch.tensor([35], dtype=torch.int64)
        leader = torch.tensor([2], dtype=torch.int64)
        trick_len = torch.tensor([1], dtype=torch.int64)
        p0 = torch.tensor([5], dtype=torch.int64)
        p1 = torch.tensor([3], dtype=torch.int64)
        p2 = torch.tensor([6], dtype=torch.int64)

        packed = pack_state(remaining, score, leader, trick_len, p0, p1, p2)

        # Verify each field independently
        assert (packed[0] >> 0) & 0x7F == 0b1010101
        assert (packed[0] >> 7) & 0x7F == 0b0101010
        assert (packed[0] >> 14) & 0x7F == 0b1100110
        assert (packed[0] >> 21) & 0x7F == 0b0011001
        assert (packed[0] >> 28) & 0x3F == 35
        assert (packed[0] >> 34) & 0x3 == 2
        assert (packed[0] >> 36) & 0x3 == 1
        assert (packed[0] >> 38) & 0x7 == 5
        assert (packed[0] >> 41) & 0x7 == 3
        assert (packed[0] >> 44) & 0x7 == 6


class TestComputeLevel:
    """Test compute_level function."""

    def test_full_hands(self):
        """28 dominoes (7 per player) = level 28."""
        remaining = torch.tensor([[0b1111111, 0b1111111, 0b1111111, 0b1111111]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, zeros)
        assert compute_level(packed)[0] == 28

    def test_empty_hands(self):
        """0 dominoes = level 0."""
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, zeros)
        assert compute_level(packed)[0] == 0

    def test_mixed_hands(self):
        """Test with various hand sizes."""
        # Player 0: 5 dominoes (0b11111 = 5)
        # Player 1: 3 dominoes (0b111 = 3)
        # Player 2: 4 dominoes (0b1111 = 4)
        # Player 3: 2 dominoes (0b11 = 2)
        # Total: 14
        remaining = torch.tensor([[0b0011111, 0b0000111, 0b0001111, 0b0000011]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, zeros)
        assert compute_level(packed)[0] == 14

    def test_batched_level(self):
        """Test batched computation."""
        remaining = torch.tensor(
            [
                [0b1111111, 0b1111111, 0b1111111, 0b1111111],  # 28
                [0b0000001, 0b0000001, 0b0000001, 0b0000001],  # 4
                [0, 0, 0, 0],  # 0
            ],
            dtype=torch.int64,
        )
        zeros = torch.tensor([0, 0, 0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, zeros)
        levels = compute_level(packed)
        assert levels[0] == 28
        assert levels[1] == 4
        assert levels[2] == 0


class TestComputeTeam:
    """Test compute_team function."""

    def test_leader_0_trick_len_0(self):
        """Leader=0, trick_len=0 -> player=0 -> team 0."""
        remaining = torch.tensor([[127, 127, 127, 127]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, zeros, zeros, zeros, zeros)
        assert compute_team(packed)[0] == True

    def test_leader_0_trick_len_1(self):
        """Leader=0, trick_len=1 -> player=1 -> team 1."""
        remaining = torch.tensor([[127, 127, 127, 127]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        trick_len = torch.tensor([1], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, trick_len, zeros, zeros, zeros)
        assert compute_team(packed)[0] == False

    def test_leader_0_trick_len_2(self):
        """Leader=0, trick_len=2 -> player=2 -> team 0."""
        remaining = torch.tensor([[127, 127, 127, 127]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        trick_len = torch.tensor([2], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, trick_len, zeros, zeros, zeros)
        assert compute_team(packed)[0] == True

    def test_leader_0_trick_len_3(self):
        """Leader=0, trick_len=3 -> player=3 -> team 1."""
        remaining = torch.tensor([[127, 127, 127, 127]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        trick_len = torch.tensor([3], dtype=torch.int64)
        packed = pack_state(remaining, zeros, zeros, trick_len, zeros, zeros, zeros)
        assert compute_team(packed)[0] == False

    def test_leader_1_trick_len_0(self):
        """Leader=1, trick_len=0 -> player=1 -> team 1."""
        remaining = torch.tensor([[127, 127, 127, 127]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        leader = torch.tensor([1], dtype=torch.int64)
        packed = pack_state(remaining, zeros, leader, zeros, zeros, zeros, zeros)
        assert compute_team(packed)[0] == False

    def test_leader_2_trick_len_1(self):
        """Leader=2, trick_len=1 -> player=3 -> team 1."""
        remaining = torch.tensor([[127, 127, 127, 127]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        leader = torch.tensor([2], dtype=torch.int64)
        trick_len = torch.tensor([1], dtype=torch.int64)
        packed = pack_state(remaining, zeros, leader, trick_len, zeros, zeros, zeros)
        assert compute_team(packed)[0] == False

    def test_leader_3_trick_len_1(self):
        """Leader=3, trick_len=1 -> player=0 -> team 0."""
        remaining = torch.tensor([[127, 127, 127, 127]], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        leader = torch.tensor([3], dtype=torch.int64)
        trick_len = torch.tensor([1], dtype=torch.int64)
        packed = pack_state(remaining, zeros, leader, trick_len, zeros, zeros, zeros)
        assert compute_team(packed)[0] == True

    def test_batched_team(self):
        """Test batched team computation."""
        remaining = torch.tensor([[127, 127, 127, 127]] * 4, dtype=torch.int64)
        zeros = torch.tensor([0, 0, 0, 0], dtype=torch.int64)
        leader = torch.tensor([0, 0, 1, 3], dtype=torch.int64)
        trick_len = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        packed = pack_state(remaining, zeros, leader, trick_len, zeros, zeros, zeros)
        teams = compute_team(packed)
        # player 0 -> team 0
        assert teams[0] == True
        # player 1 -> team 1
        assert teams[1] == False
        # player 1 -> team 1
        assert teams[2] == False
        # player 0 -> team 0
        assert teams[3] == True


class TestComputeTerminalValue:
    """Test compute_terminal_value function."""

    def test_score_0(self):
        """Score=0 -> terminal_value = -42."""
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        score = torch.tensor([0], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, score, zeros, zeros, zeros, zeros, zeros)
        assert compute_terminal_value(packed)[0] == -42

    def test_score_21(self):
        """Score=21 -> terminal_value = 0."""
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        score = torch.tensor([21], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, score, zeros, zeros, zeros, zeros, zeros)
        assert compute_terminal_value(packed)[0] == 0

    def test_score_42(self):
        """Score=42 -> terminal_value = +42."""
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        score = torch.tensor([42], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, score, zeros, zeros, zeros, zeros, zeros)
        assert compute_terminal_value(packed)[0] == 42

    def test_batched_terminal_value(self):
        """Test batched terminal value computation."""
        remaining = torch.tensor([[0, 0, 0, 0]] * 5, dtype=torch.int64)
        scores = torch.tensor([0, 10, 21, 30, 42], dtype=torch.int64)
        zeros = torch.tensor([0, 0, 0, 0, 0], dtype=torch.int64)
        packed = pack_state(remaining, scores, zeros, zeros, zeros, zeros, zeros)
        terminal_values = compute_terminal_value(packed)
        assert terminal_values[0] == -42  # 2*0 - 42
        assert terminal_values[1] == -22  # 2*10 - 42
        assert terminal_values[2] == 0  # 2*21 - 42
        assert terminal_values[3] == 18  # 2*30 - 42
        assert terminal_values[4] == 42  # 2*42 - 42

    def test_terminal_value_dtype(self):
        """Verify terminal value is int8."""
        remaining = torch.tensor([[0, 0, 0, 0]], dtype=torch.int64)
        score = torch.tensor([21], dtype=torch.int64)
        zeros = torch.tensor([0], dtype=torch.int64)
        packed = pack_state(remaining, score, zeros, zeros, zeros, zeros, zeros)
        terminal_value = compute_terminal_value(packed)
        assert terminal_value.dtype == torch.int8


class TestScalarOperations:
    """Test that operations work with single-element tensors (scalar-like)."""

    def test_scalar_pack_unpack(self):
        """Test pack/unpack with N=1."""
        remaining = torch.tensor([[85, 42, 21, 10]], dtype=torch.int64)
        score = torch.tensor([15], dtype=torch.int64)
        leader = torch.tensor([1], dtype=torch.int64)
        trick_len = torch.tensor([2], dtype=torch.int64)
        p0 = torch.tensor([4], dtype=torch.int64)
        p1 = torch.tensor([2], dtype=torch.int64)
        p2 = torch.tensor([7], dtype=torch.int64)

        packed = pack_state(remaining, score, leader, trick_len, p0, p1, p2)
        assert packed.shape == (1,)

        assert unpack_remaining(packed).shape == (1, 4)
        assert unpack_score(packed).shape == (1,)
        assert compute_level(packed).shape == (1,)
        assert compute_team(packed).shape == (1,)
        assert compute_terminal_value(packed).shape == (1,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
