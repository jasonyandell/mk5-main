"""GPU-native historical state reconstruction for posterior weighting.

Reconstructs game state at each of the last K plays, enabling efficient
batched posterior weight computation entirely on GPU.

Key functions:
- reconstruct_past_states_gpu: Reconstruct game states for last K plays
- compute_legal_masks_gpu: Compute legal actions for each world at each past step
- compute_posterior_weights_gpu: Compute posterior weights using Q-values and observed actions

Phase 4(D) of GPU-native E[Q] pipeline:
- Phase 1: GameStateTensor - GPU game state representation
- Phase 2: WorldSampler - GPU world sampling
- Phase 3: GPUTokenizer - GPU tokenization
- Phase 4(A-C): Past state reconstruction + legal mask computation
- Phase 4(D): GPU weight computation (this module)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from forge.oracle.declarations import DOUBLES_SUIT, DOUBLES_TRUMP, NOTRUMP, PIP_TRUMP_IDS, N_DECLS
from forge.oracle.tables import DOMINO_HIGH, DOMINO_IS_DOUBLE



@dataclass
class PastStatesGPU:
    """Reconstructed game states for K past steps, for N games.

    All tensors are in batch-first format with shape (N, K, ...).
    For games with fewer than K plays, earlier steps have valid_mask=False.
    """
    played_before: Tensor      # [N, K, 28] bool - was domino played before this step?
    trick_plays: Tensor        # [N, K, 3, 2] int - up to 3 (player, domino) per step
    trick_lens: Tensor         # [N, K] int - number of plays in current trick
    leaders: Tensor            # [N, K] int - who led this trick
    actors: Tensor             # [N, K] int - who played at this step
    observed_actions: Tensor   # [N, K] int - domino ID played at this step
    step_indices: Tensor       # [N, K] int - absolute step index in history
    valid_mask: Tensor         # [N, K] bool - is this step valid (history long enough)?


def reconstruct_past_states_gpu(
    history: Tensor,           # [N, max_len, 3] (player, domino, lead_domino)
    history_len: Tensor,       # [N] actual length per game
    window_k: int,
    device: torch.device | None = None,
    vectorized: bool = True,
) -> PastStatesGPU:
    """Reconstruct game states for the last K plays of each game.

    For each game, extracts state information for steps [len-K, len-K+1, ..., len-1].
    Games with fewer than K plays will have valid_mask=False for earlier steps.

    Args:
        history: Play history tensor [N, max_len, 3] where each entry is
                 (player_id, domino_id, lead_domino_id). Unused entries are -1.
        history_len: Actual history length for each game [N].
        window_k: Number of past steps to reconstruct.
        device: Device to place output tensors on (defaults to history.device).

    Returns:
        PastStatesGPU containing reconstructed state for each of K steps per game.

    Algorithm:
        1. For each step index i in the window:
           - played_before[i] = dominoes played in steps 0..i-1
           - Scan backward from i to find trick start (same lead_domino)
           - Extract trick_plays from trick_start..i-1
           - Leader is who played at trick_start

    Args:
        history: Play history tensor [N, max_len, 3] where each entry is
                 (player_id, domino_id, lead_domino_id). Unused entries are -1.
        history_len: Actual history length for each game [N].
        window_k: Number of past steps to reconstruct.
        device: Device to place output tensors on (defaults to history.device).
        vectorized: If True (default), use fully vectorized implementation.
                    If False, use original loop-based implementation for comparison.
    """
    if device is None:
        device = history.device

    N = history.shape[0]
    max_len = history.shape[1]
    K = window_k

    # Ensure history is on the correct device
    history = history.to(device)
    history_len = history_len.to(device)

    # Compute step indices for each game: [len-K, len-K+1, ..., len-1]
    # Clamp to [0, len-1] and mark invalid steps
    step_offsets = torch.arange(K, device=device).unsqueeze(0)  # [1, K]
    step_indices = history_len.unsqueeze(1) - K + step_offsets  # [N, K]
    valid_mask = (step_indices >= 0) & (step_indices < history_len.unsqueeze(1))  # [N, K]
    step_indices_clamped = step_indices.clamp(0, max_len - 1)  # [N, K]

    # Extract actors, observed_actions, and lead_dominoes for each step
    # history[n, step_indices_clamped[n, k]] gives the history entry for game n, step k
    batch_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, K)  # [N, K]
    step_idx_flat = step_indices_clamped.reshape(-1)  # [N*K]
    batch_idx_flat = batch_idx.reshape(-1)  # [N*K]

    history_entries = history[batch_idx_flat, step_idx_flat]  # [N*K, 3]
    actors = history_entries[:, 0].reshape(N, K)  # [N, K]
    observed_actions = history_entries[:, 1].reshape(N, K)  # [N, K]
    lead_dominoes = history_entries[:, 2].reshape(N, K)  # [N, K]

    # Compute played_before[n, k] = dominoes played in steps 0..step_indices[n,k]-1
    # Strategy: Create one-hot encoding of all plays, compute cumsum, then slice
    # For each (game, step), we need dominoes from history[game, 0:step_idx]

    # Create one-hot encoding of all dominoes in history [N, max_len, 28]
    domino_ids = history[:, :, 1]  # [N, max_len]
    valid_entries = domino_ids >= 0  # [N, max_len]
    domino_ids_safe = torch.where(valid_entries, domino_ids, torch.zeros_like(domino_ids))

    # One-hot encode [N, max_len, 28]
    domino_one_hot = torch.nn.functional.one_hot(
        domino_ids_safe.long(), num_classes=28
    ).float()  # [N, max_len, 28]
    domino_one_hot = domino_one_hot * valid_entries.unsqueeze(-1).float()  # Mask invalid entries

    # Cumulative OR: cumsum + clamp
    played_cumsum = torch.cumsum(domino_one_hot, dim=1)  # [N, max_len, 28]

    # For each (n, k), extract played_before at step_indices[n, k]
    # We need the cumsum just before this step, so use step_indices - 1 (clamped to 0)
    prev_step_indices = (step_indices_clamped - 1).clamp(0, max_len - 1)  # [N, K]
    played_before = played_cumsum[batch_idx_flat, prev_step_indices.reshape(-1)]  # [N*K, 28]
    played_before = played_before.reshape(N, K, 28)  # [N, K, 28]
    played_before = (played_before > 0.5)  # Convert to bool

    # For step_indices that are 0, nothing should be played before
    first_step_mask = step_indices_clamped == 0  # [N, K]
    played_before = played_before & ~first_step_mask.unsqueeze(-1)  # [N, K, 28]

    # Find trick boundaries by scanning backward to find where lead_domino changes
    # This is the trickiest part - we need to find trick_start for each (n, k)

    if vectorized:
        # =====================================================================
        # VECTORIZED IMPLEMENTATION: Process all K steps in parallel
        # =====================================================================
        # Key insight: All K steps are independent, so we can process them
        # simultaneously using tensor operations instead of a Python loop.

        # Pre-compute batch indices for [N, K] operations (reuse throughout)
        batch_idx_nk = torch.arange(N, device=device).unsqueeze(1).expand(N, K)  # [N, K]

        # -----------------------------------------------------------------
        # Step 1: Vectorized backward scan to find trick boundaries
        # For each (n, k), check if steps current_step-1, -2, -3 have same lead_domino
        # -----------------------------------------------------------------

        # Offsets for backward scan: [3] -> check steps at -1, -2, -3
        offsets = torch.arange(1, 4, device=device)  # [3]

        # Compute previous step indices for all (n, k, offset) at once
        # step_indices_clamped: [N, K] -> expand to [N, K, 3]
        prev_steps = step_indices_clamped.unsqueeze(-1) - offsets  # [N, K, 3]
        prev_valid = prev_steps >= 0  # [N, K, 3]
        prev_steps_safe = prev_steps.clamp(0, max_len - 1)  # [N, K, 3]

        # Gather lead_domino for all prev_steps at once
        # We need history[n, prev_steps_safe[n, k, o], 2] for all (n, k, o)
        # Flatten to [N*K*3] for advanced indexing, then reshape
        batch_idx_nko = batch_idx_nk.unsqueeze(-1).expand(N, K, 3)  # [N, K, 3]
        prev_leads = history[
            batch_idx_nko.reshape(-1),
            prev_steps_safe.reshape(-1).long(),
            2
        ].reshape(N, K, 3)  # [N, K, 3]

        # Check if previous steps are in the same trick (same lead_domino)
        # lead_dominoes: [N, K] -> expand to [N, K, 3] for comparison
        same_trick = prev_valid & (prev_leads == lead_dominoes.unsqueeze(-1))  # [N, K, 3]

        # Compute trick_start: find the earliest step still in same trick
        # Start from current_step and go backward while same_trick is True
        # We need cumulative AND from right to left, then find first True
        # Trick: if same_trick = [T, T, F], trick started at current-2
        #        if same_trick = [T, F, F], trick started at current-1
        #        if same_trick = [F, F, F], trick started at current (leading)

        # Compute cumulative product from left (offset 1, 2, 3)
        # If any offset breaks the chain, all further offsets are invalid
        cumulative_same = same_trick.clone()
        cumulative_same[:, :, 1] = cumulative_same[:, :, 0] & same_trick[:, :, 1]
        cumulative_same[:, :, 2] = cumulative_same[:, :, 1] & same_trick[:, :, 2]

        # Count how many consecutive steps back are in same trick
        # This is sum of cumulative_same along offset dimension
        steps_back = cumulative_same.sum(dim=-1)  # [N, K] values in {0, 1, 2, 3}

        # trick_start = current_step - steps_back
        trick_start = step_indices_clamped - steps_back  # [N, K]

        # -----------------------------------------------------------------
        # Step 2: Vectorized trick_plays extraction
        # For each (n, k), extract plays from trick_start to current_step-1
        # At most 3 plays before current (offsets 0, 1, 2 from trick_start)
        # -----------------------------------------------------------------

        # Compute play step indices for all trick positions at once
        # trick_offsets: [3] = [0, 1, 2]
        trick_offsets = torch.arange(3, device=device)  # [3]

        # play_steps[n, k, o] = trick_start[n, k] + o
        play_steps = trick_start.unsqueeze(-1) + trick_offsets  # [N, K, 3]
        play_steps_safe = play_steps.clamp(0, max_len - 1)  # [N, K, 3]

        # Check which plays are before current step (valid for extraction)
        before_current = play_steps < step_indices_clamped.unsqueeze(-1)  # [N, K, 3]

        # Gather player and domino for all play_steps at once
        batch_idx_nk3 = batch_idx_nk.unsqueeze(-1).expand(N, K, 3)  # [N, K, 3]

        players = history[
            batch_idx_nk3.reshape(-1),
            play_steps_safe.reshape(-1).long(),
            0
        ].reshape(N, K, 3)  # [N, K, 3]

        dominoes = history[
            batch_idx_nk3.reshape(-1),
            play_steps_safe.reshape(-1).long(),
            1
        ].reshape(N, K, 3)  # [N, K, 3]

        # Build trick_plays tensor: [N, K, 3, 2]
        trick_plays = torch.full((N, K, 3, 2), -1, dtype=torch.int32, device=device)
        trick_plays[:, :, :, 0] = torch.where(before_current, players, torch.tensor(-1, device=device))
        trick_plays[:, :, :, 1] = torch.where(before_current, dominoes, torch.tensor(-1, device=device))

        # Compute trick_lens: count of valid plays before current
        trick_lens = before_current.sum(dim=-1).int()  # [N, K]

        # -----------------------------------------------------------------
        # Step 3: Compute leaders
        # Leader is who played at trick_start (or current actor if leading)
        # -----------------------------------------------------------------

        # Get leader at trick_start for all (n, k)
        leader_at_start = history[
            batch_idx_nk.reshape(-1),
            trick_start.clamp(0, max_len - 1).reshape(-1).long(),
            0
        ].reshape(N, K)  # [N, K]

        # If trick_lens == 0, current player is leading
        is_leading = trick_lens == 0  # [N, K]
        leaders = torch.where(is_leading, actors, leader_at_start).int()  # [N, K]

    else:
        # =====================================================================
        # ORIGINAL LOOP-BASED IMPLEMENTATION (for correctness comparison)
        # =====================================================================
        # Strategy: For each step, scan backward through history to find trick start
        # A trick continues while lead_domino matches
        # We'll do this with a loop over K (unavoidable, but K is small, typically 4-8)

        trick_plays = torch.full((N, K, 3, 2), -1, dtype=torch.int32, device=device)
        trick_lens = torch.zeros(N, K, dtype=torch.int32, device=device)
        leaders = torch.zeros(N, K, dtype=torch.int32, device=device)

        for k in range(K):
            # For each game, find trick start for step k
            current_step = step_indices_clamped[:, k]  # [N]
            current_lead = lead_dominoes[:, k]  # [N]

            # Find trick start by scanning backward
            # Start from current_step and go backward while lead_domino matches
            trick_start = current_step.clone()  # [N]

            # Scan backward up to 3 steps (max trick length before current play)
            for offset in range(1, 4):
                prev_step = current_step - offset  # [N]
                prev_valid = prev_step >= 0  # [N]

                # Get lead_domino at prev_step
                prev_step_safe = prev_step.clamp(0, max_len - 1)
                prev_lead = history[torch.arange(N, device=device), prev_step_safe, 2]  # [N]

                # Continue trick if lead matches and step is valid
                same_trick = prev_valid & (prev_lead == current_lead)
                trick_start = torch.where(same_trick, prev_step, trick_start)

            # Extract trick plays from trick_start to current_step-1
            # We have at most 3 plays before current (since current is 4th play at most)
            for trick_offset in range(3):
                play_step = trick_start + trick_offset  # [N]
                before_current = play_step < current_step  # [N]
                play_step_safe = play_step.clamp(0, max_len - 1)

                # Extract player and domino
                player = history[torch.arange(N, device=device), play_step_safe, 0]  # [N]
                domino = history[torch.arange(N, device=device), play_step_safe, 1]  # [N]

                # Store if valid
                trick_plays[:, k, trick_offset, 0] = torch.where(
                    before_current, player, torch.tensor(-1, device=device)
                )
                trick_plays[:, k, trick_offset, 1] = torch.where(
                    before_current, domino, torch.tensor(-1, device=device)
                )

                # Update trick_lens
                trick_lens[:, k] += before_current.int()

            # Leader is who played at trick_start (or current actor if leading)
            # If trick_lens == 0, current player is leading
            is_leading = trick_lens[:, k] == 0
            leader_at_start = history[torch.arange(N, device=device), trick_start.clamp(0, max_len - 1), 0]
            leaders[:, k] = torch.where(is_leading, actors[:, k], leader_at_start)

    return PastStatesGPU(
        played_before=played_before,
        trick_plays=trick_plays,
        trick_lens=trick_lens,
        leaders=leaders,
        actors=actors,
        observed_actions=observed_actions,
        step_indices=step_indices_clamped,
        valid_mask=valid_mask,
    )


def _build_led_suit_table() -> torch.Tensor:
    """Build lookup tensor for led suit determination.

    Returns:
        Tensor of shape (28, 10) where LED_SUIT[lead_domino_id, decl_id]
        gives the led suit (0-7, where 7 = called suit).
    """
    led_suit = torch.zeros(28, N_DECLS, dtype=torch.int64)

    for lead_domino_id in range(28):
        # Decode domino_id to (high, low) pips
        h = 0
        temp_id = lead_domino_id
        while temp_id >= (h + 1):
            temp_id -= (h + 1)
            h += 1
        l = temp_id
        high = h
        low = l
        is_double = (high == low)

        for decl_id in range(N_DECLS):
            # Determine if lead domino is in called suit
            if decl_id in PIP_TRUMP_IDS:
                in_called = (decl_id == high) or (decl_id == low)
            elif decl_id in (DOUBLES_TRUMP, DOUBLES_SUIT):
                in_called = is_double
            elif decl_id == NOTRUMP:
                in_called = False
            else:
                raise ValueError(f"Unknown decl_id: {decl_id}")

            if decl_id == NOTRUMP:
                # No-trump: led suit is always high pip
                led_suit[lead_domino_id, decl_id] = high
            elif in_called:
                # Lead is in called suit → led suit is 7
                led_suit[lead_domino_id, decl_id] = 7
            else:
                # Lead is off-suit → led suit is high pip
                led_suit[lead_domino_id, decl_id] = high

    return led_suit


def _build_can_follow_table() -> torch.Tensor:
    """Build lookup tensor for following suit rules.

    Returns:
        Tensor of shape (28, 8, 10) where CAN_FOLLOW[domino_id, led_suit, decl_id]
        is True if the domino can follow that suit under that declaration.

        led_suit in {0..6, 7=called suit}
    """
    can_follow = torch.zeros(28, 8, N_DECLS, dtype=torch.bool)

    for domino_id in range(28):
        # Decode domino_id to (high, low) pips
        h = 0
        temp_id = domino_id
        while temp_id >= (h + 1):
            temp_id -= (h + 1)
            h += 1
        l = temp_id
        high = h
        low = l
        is_double = (high == low)

        for decl_id in range(N_DECLS):
            # Determine if domino is in called suit
            if decl_id in PIP_TRUMP_IDS:
                in_called = (decl_id == high) or (decl_id == low)
            elif decl_id in (DOUBLES_TRUMP, DOUBLES_SUIT):
                in_called = is_double
            elif decl_id == NOTRUMP:
                in_called = False
            else:
                raise ValueError(f"Unknown decl_id: {decl_id}")

            # For each possible led suit
            for led_suit in range(8):
                if led_suit == 7:  # Called suit led
                    # Can follow if in called suit
                    can_follow[domino_id, led_suit, decl_id] = in_called
                else:  # Pip suit led (0-6)
                    # Can follow if: (1) contains that pip AND (2) not in called suit
                    has_pip = (led_suit == high) or (led_suit == low)
                    can_follow[domino_id, led_suit, decl_id] = has_pip and not in_called

    return can_follow


# Pre-compute lookup tables at module load
LED_SUIT = _build_led_suit_table()  # [28, 10] - led suit for each (lead_domino, decl)
CAN_FOLLOW = _build_can_follow_table()  # [28, 8, 10] - can domino follow (domino, led_suit, decl)


def compute_legal_masks_gpu(
    worlds: Tensor,            # [N, M, 4, 7] hands - domino IDs per slot
    past_states: PastStatesGPU,
    decl_id: int,
    device: torch.device | None = None,
) -> Tensor:
    """Compute legal actions for each world at each past step.

    An action (slot index 0-6) is legal if:
    1. The domino in that slot hasn't been played yet (not in played_before)
    2. If following (not leading), the domino can follow the led suit

    Args:
        worlds: [N, M, 4, 7] domino IDs for each player's hand
        past_states: Reconstructed states with played_before, leaders, actors, etc.
        decl_id: Declaration ID for suit rules
        device: Device for computation (defaults to worlds.device)

    Returns:
        legal_mask: [N, M, K, 7] bool - is slot i legal for game g, sample m, step k?
    """
    if device is None:
        device = worlds.device

    # Move lookup tables to device if needed
    led_suit_table = LED_SUIT.to(device)
    can_follow_table = CAN_FOLLOW.to(device)

    N, M, _, _ = worlds.shape
    K = past_states.actors.shape[1]

    # Extract actor's hand for each step: [N, M, K, 7]
    # We need to gather hands based on past_states.actors
    # actors: [N, K] -> expand to [N, M, K] for broadcasting
    actors_expanded = past_states.actors.long().unsqueeze(1).expand(N, M, K)  # [N, M, K]

    # Gather actor hands: worlds[n, m, actors[n, k], :] -> [N, M, K, 7]
    # We need to expand indices for gather operation
    actors_idx = actors_expanded.unsqueeze(-1).expand(N, M, K, 7)  # [N, M, K, 7]
    actor_hands = torch.gather(
        worlds.unsqueeze(2).expand(N, M, K, 4, 7),  # [N, M, K, 4, 7]
        dim=3,
        index=actors_idx.unsqueeze(3)  # [N, M, K, 1, 7]
    ).squeeze(3)  # [N, M, K, 7]

    # 1. Remaining filter: domino not yet played
    # played_before: [N, K, 28] -> check if actor_hands[..., slot] is in played set
    # actor_hands: [N, M, K, 7] contains domino IDs (0-27)

    # Expand played_before to [N, M, K, 28] for broadcasting
    played_expanded = past_states.played_before.unsqueeze(1).expand(N, M, K, 28)  # [N, M, K, 28]

    # Gather played status for each domino in actor's hand
    # actor_hands: [N, M, K, 7] contains domino IDs
    domino_ids = actor_hands.long()  # Ensure int64 for indexing

    # Use advanced indexing to check if each domino was played
    # played_expanded[n, m, k, domino_ids[n, m, k, slot]]
    batch_idx = torch.arange(N, device=device).view(N, 1, 1, 1).expand(N, M, K, 7)
    sample_idx = torch.arange(M, device=device).view(1, M, 1, 1).expand(N, M, K, 7)
    step_idx = torch.arange(K, device=device).view(1, 1, K, 1).expand(N, M, K, 7)

    was_played = played_expanded[batch_idx, sample_idx, step_idx, domino_ids]  # [N, M, K, 7]
    remaining_mask = ~was_played  # [N, M, K, 7] - True if domino is still available

    # 2. Following rules
    # Determine if actor is leading: trick_lens[n, k] == 0
    is_leading = (past_states.trick_lens == 0)  # [N, K]
    is_leading_expanded = is_leading.unsqueeze(1).unsqueeze(-1).expand(N, M, K, 7)  # [N, M, K, 7]

    # Get led suit for each step
    # observed_actions: [N, K] contains lead domino IDs
    lead_dominoes = past_states.observed_actions  # [N, K]
    led_suits = led_suit_table[lead_dominoes.long(), decl_id]  # [N, K]

    # Check which dominoes can follow: [N, M, K, 7]
    # can_follow_table[domino_id, led_suit, decl_id] -> bool
    led_suits_expanded = led_suits.unsqueeze(1).unsqueeze(-1).expand(N, M, K, 7)  # [N, M, K, 7]
    can_follow_mask = can_follow_table[domino_ids.long(), led_suits_expanded.long(), decl_id]  # [N, M, K, 7]

    # Check if any domino in hand can follow
    # If actor must follow, only followers are legal
    # If no followers exist, all remaining are legal (can't follow = must play something)
    has_any_follower = (can_follow_mask & remaining_mask).any(dim=-1, keepdim=True)  # [N, M, K, 1]

    # Following logic:
    # - If leading: all remaining are legal
    # - If following and has_follower: only followers are legal
    # - If following and no_follower: all remaining are legal (can't follow)
    following_mask = torch.where(
        has_any_follower,
        can_follow_mask,  # Must follow suit
        torch.ones_like(can_follow_mask)  # No followers → can play anything
    )

    # Combine: legal = remaining AND (leading OR following_rules)
    legal_mask = remaining_mask & (is_leading_expanded | following_mask)

    # Apply valid_mask: invalid steps should have all False
    valid_expanded = past_states.valid_mask.unsqueeze(1).unsqueeze(-1).expand(N, M, K, 7)
    legal_mask = legal_mask & valid_expanded

    return legal_mask


def compute_posterior_weights_gpu(
    q_past: Tensor,            # [N, M, K, 7] Q-values at past states
    legal_masks: Tensor,       # [N, M, K, 7] legal actions
    observed_actions: Tensor,  # [N, K] domino IDs played
    worlds: Tensor,            # [N, M, 4, 7] to find local indices
    actors: Tensor,            # [N, K] who played
    tau: float = 0.1,
    uniform_mix: float = 0.1,
) -> tuple[Tensor, dict]:
    """Compute posterior weights per world using GPU tensor operations.

    Algorithm (all tensor ops, no loops over N or M):
    1. Map observed_actions [N, K] domino IDs to local hand indices [N, M, K]
    2. Compute advantage: (Q - mean(Q_legal)) / tau for each step
    3. Softmax + uniform mix: (1 - uniform_mix) * softmax(adv) + uniform_mix * uniform_over_legal
    4. Gather observed action probabilities using local indices
    5. Sum log-probs across K steps (with valid_mask)
    6. Softmax to normalize weights across M worlds
    7. Compute ESS = 1 / sum(w^2) per game

    Args:
        q_past: [N, M, K, 7] Q-values at past states
        legal_masks: [N, M, K, 7] legal actions mask
        observed_actions: [N, K] domino IDs played at each step
        worlds: [N, M, 4, 7] sampled world hands (to map domino IDs to local indices)
        actors: [N, K] who played at each step
        tau: Temperature for advantage-softmax (default: 0.1)
        uniform_mix: Uniform mixture coefficient (beta) for robustness (default: 0.1)

    Returns:
        weights: [N, M] normalized weights per world
        diagnostics: dict with keys:
            - ess: [N] effective sample size per game
            - max_w: [N] maximum weight per game
            - entropy: [N] entropy of weight distribution per game
            - k_eff: [N] k_eff = exp(entropy) per game
    """
    N, M, K, _ = q_past.shape
    device = q_past.device

    # =========================================================================
    # Step 1: Map observed_actions domino IDs to local hand indices
    # =========================================================================
    # For each (n, k), find which slot in worlds[n, m, actors[n, k], :] contains observed_actions[n, k]
    # Result: obs_local_idx[n, m, k] = local index (0-6) or -1 if not found

    # Extract actor hands: [N, M, K, 7]
    actors_expanded = actors.long().unsqueeze(1).expand(N, M, K)  # [N, M, K]
    actors_idx = actors_expanded.unsqueeze(-1).expand(N, M, K, 7)  # [N, M, K, 7]
    actor_hands = torch.gather(
        worlds.unsqueeze(2).expand(N, M, K, 4, 7),
        dim=3,
        index=actors_idx.unsqueeze(3)
    ).squeeze(3)  # [N, M, K, 7]

    # Check which slots match observed_actions
    # observed_actions: [N, K] -> [N, 1, K, 1] for broadcasting
    observed_expanded = observed_actions.unsqueeze(1).unsqueeze(-1)  # [N, 1, K, 1]
    matches = (actor_hands == observed_expanded)  # [N, M, K, 7]

    # Find first matching index (argmax on bool returns first True)
    # Note: If no match, argmax returns 0, so we need to check if any match exists
    has_match = matches.any(dim=-1)  # [N, M, K]
    obs_local_idx = matches.long().argmax(dim=-1)  # [N, M, K]
    obs_local_idx = torch.where(has_match, obs_local_idx, torch.tensor(-1, device=device))  # [N, M, K]

    # =========================================================================
    # Step 2: Compute advantage for each action
    # =========================================================================
    # advantage = (Q - mean(Q_legal)) / tau
    # legal_masks: [N, M, K, 7] bool

    legal_f = legal_masks.float()  # [N, M, K, 7]
    legal_count = legal_f.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [N, M, K, 1]
    mean_q_legal = (q_past * legal_f).sum(dim=-1, keepdim=True) / legal_count  # [N, M, K, 1]

    advantage = (q_past - mean_q_legal) / tau  # [N, M, K, 7]
    advantage = advantage.masked_fill(~legal_masks, float('-inf'))  # [N, M, K, 7]

    # =========================================================================
    # Step 3: Softmax + uniform mix
    # =========================================================================
    p_soft = torch.nn.functional.softmax(advantage, dim=-1)  # [N, M, K, 7]
    uniform = legal_f / legal_count  # [N, M, K, 7]
    p_mixed = (1 - uniform_mix) * p_soft + uniform_mix * uniform  # [N, M, K, 7]

    # =========================================================================
    # Step 4: Gather observed action probabilities
    # =========================================================================
    # Use obs_local_idx to gather probabilities
    # obs_local_idx: [N, M, K], p_mixed: [N, M, K, 7]

    # Clamp obs_local_idx to valid range for gather
    obs_local_idx_safe = obs_local_idx.clamp(min=0)  # [N, M, K]
    obs_prob = p_mixed.gather(dim=-1, index=obs_local_idx_safe.unsqueeze(-1)).squeeze(-1)  # [N, M, K]

    # Mark invalid observations (domino not in hand)
    invalid_obs = (obs_local_idx < 0)  # [N, M, K]

    # =========================================================================
    # Step 5: Sum log-probs across K steps
    # =========================================================================
    # Compute log probabilities with numerical stability
    log_prob = torch.log(obs_prob + 1e-30)  # [N, M, K]

    # Mark invalid observations with very low log-prob
    log_prob = torch.where(invalid_obs, torch.tensor(-1e9, device=device), log_prob)  # [N, M, K]

    # Check if no legal actions (shouldn't happen with correct legal masks)
    has_legal = legal_count.squeeze(-1) > 0  # [N, M, K]
    log_prob = torch.where(has_legal, log_prob, torch.tensor(-1e9, device=device))  # [N, M, K]

    # Sum across steps K
    logw = log_prob.sum(dim=-1)  # [N, M]

    # =========================================================================
    # Step 6: Normalize weights using softmax (with log-sum-exp trick)
    # =========================================================================
    # For each game, normalize weights across M worlds
    max_logw = logw.max(dim=1, keepdim=True)[0]  # [N, 1]
    logw_stable = (logw - max_logw).clamp(min=-30.0)  # [N, M] - clamp at -30 for stability
    weights = torch.nn.functional.softmax(logw_stable, dim=1)  # [N, M]

    # =========================================================================
    # Step 7: Compute diagnostics
    # =========================================================================
    # ESS = 1 / sum(w^2) per game
    ess = 1.0 / (weights * weights).sum(dim=1)  # [N]

    # Maximum weight per game
    max_w = weights.max(dim=1)[0]  # [N]

    # Entropy per game
    log_weights = torch.log(weights + 1e-30)  # [N, M]
    entropy = -(weights * log_weights).sum(dim=1)  # [N]

    # k_eff = exp(entropy)
    k_eff = torch.exp(entropy)  # [N]

    diagnostics = {
        'ess': ess,
        'max_w': max_w,
        'entropy': entropy,
        'k_eff': k_eff,
    }

    return weights, diagnostics
