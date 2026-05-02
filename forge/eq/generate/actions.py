"""Action selection and decision recording for GPU E[Q] pipeline."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from forge.eq.game_tensor import GameStateTensor
from forge.eq.types import ExplorationPolicy, PosteriorDiagnostics

from .types import DecisionRecordGPU, GameRecordGPU

DEFAULT_BID_VALUE = 30
EQ_BIN_COUNT = 85


def _bid_values_tensor(
    bid_values: list[int] | Tensor | None,
    *,
    n_games: int,
    device: torch.device | str,
) -> Tensor:
    """Return one contract value per game."""
    if bid_values is None:
        return torch.full((n_games,), DEFAULT_BID_VALUE, dtype=torch.long, device=device)
    if isinstance(bid_values, Tensor):
        values = bid_values.to(device=device, dtype=torch.long).flatten()
    else:
        values = torch.tensor(bid_values, dtype=torch.long, device=device).flatten()
    if values.numel() == 1 and n_games != 1:
        values = values.expand(n_games)
    if values.numel() != n_games:
        raise ValueError(f"Expected {n_games} bid values, got {values.numel()}")
    return values


def _contract_points_from_bid_values(bid_values: Tensor) -> Tensor:
    """Map recorded bid values to contract point targets."""
    return torch.where(bid_values == 84, torch.full_like(bid_values, 42), bid_values)


def contract_threshold_bins(
    bid_values: list[int] | Tensor | None,
    *,
    n_games: int,
    device: torch.device | str,
) -> tuple[Tensor, Tensor]:
    """Compute PDF start bins for offense make and defense set thresholds.

    The PDF has bins 0..84 for Q values -42..+42. For an offense contract B,
    the bidder team needs Q >= 2B - 42, so the offense bin is 2B. The defense
    team needs the bidder team to fall short, matching the historical bid-30
    convention of Q >= -17 / bin 25, so the defense bin is 85 - 2B.
    """
    contract_points = _contract_points_from_bid_values(
        _bid_values_tensor(bid_values, n_games=n_games, device=device)
    )
    offense_bins = (2 * contract_points).clamp(min=0, max=EQ_BIN_COUNT - 1)
    defense_bins = (EQ_BIN_COUNT - 2 * contract_points).clamp(min=0, max=EQ_BIN_COUNT - 1)
    return offense_bins.long(), defense_bins.long()


def _p_make_from_pdf(
    e_q_pdf: Tensor,
    bidder: Tensor,
    current_players: Tensor,
    bid_values: list[int] | Tensor | None = None,
) -> Tensor:
    """Compute p_make (probability of making contract) per seat from PDF.

    Args:
        e_q_pdf: [n_games, 7, 85] PDF per action per game
        bidder: [n_games] bidder seat (int8)
        current_players: [n_games] seat whose perspective we're computing for
        bid_values: Optional [n_games] contract values. Defaults to 30.

    Returns:
        [n_games, 7] p_make values
    """
    n_games = e_q_pdf.shape[0]
    is_offense = ((current_players % 2) == (bidder % 2)).unsqueeze(1)  # [n_games, 1]
    offense_bins, defense_bins = contract_threshold_bins(
        bid_values,
        n_games=n_games,
        device=e_q_pdf.device,
    )
    bins = torch.arange(EQ_BIN_COUNT, device=e_q_pdf.device).view(1, 1, EQ_BIN_COUNT)
    p_make_offense = (e_q_pdf * (bins >= offense_bins.view(n_games, 1, 1))).sum(dim=2)
    p_make_defense = (e_q_pdf * (bins >= defense_bins.view(n_games, 1, 1))).sum(dim=2)
    return torch.where(is_offense, p_make_offense, p_make_defense)


def compute_per_seat_data(
    states: GameStateTensor,
    world_hands: Tensor,
    q_per_world: Tensor,
    device: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute per-seat oracle softmax, legal masks, and voids from joint-world tensors.

    For each of the 4 seats, marginalizes the oracle Q-values over the sampled worlds
    (treating each world as if that seat were the acting player) and produces a
    p_make-based normalized softmax over legal actions.

    Args:
        states: GameStateTensor with n_games games
        world_hands: [n_games, M, 3, 7] sampled opponent hands per world (relative to acting player)
        q_per_world: [n_games, M, 7] oracle Q-values from the acting player's perspective
        device: compute device

    Returns:
        Tuple of:
        - oracle_softmax_per_seat: [n_games, 4, 7] softmax over actions per seat
        - legal_mask_per_seat: [n_games, 4, 7] boolean legal mask per seat
        - voids_per_seat: [n_games, 4, 3, 8] void inferences per seat
    """
    n_games = states.n_games
    n_actions = 7
    current_player = states.current_player.long()  # [n_games]
    bidder = states.bidder.long()
    batch_idx = torch.arange(n_games, device=device)

    # Ensure world_hands and q_per_world are on the compute device
    if world_hands.device.type != device.split(':')[0]:
        world_hands = world_hands.to(device)
    if q_per_world.device.type != device.split(':')[0]:
        q_per_world = q_per_world.to(device)

    # Per-seat legal mask: for each seat, which hand slots are non-empty (have a domino)?
    # Shape: [n_games, 4, 7] — True if domino slot is non-empty (>= 0)
    # Use the actual current-player hand from states, and world-averaged hands for opponents.
    # We use the first world as representative (all worlds have the same current player's hand;
    # opponent hands vary but legal mask is based on whether slot has a domino >= 0).
    # For opponents, we use the world-average presence: slot is legal if present in >50% of worlds.
    legal_mask_per_seat = torch.zeros(n_games, 4, n_actions, dtype=torch.bool, device=device)

    # Current player's legal mask from states (exact)
    legal_mask_per_seat[batch_idx, current_player] = states.legal_actions()

    # Opponent seats: use world-based hand presence
    for rel in range(3):
        abs_seats = (current_player + rel + 1) % 4  # [n_games]
        # world_hands[:, :, rel, :]: [n_games, M, 7]
        opp_hands = world_hands[:, :, rel, :]  # [n_games, M, 7]
        # A slot is present if domino >= 0 in majority of worlds
        slot_present = (opp_hands >= 0).float().mean(dim=1) > 0.5  # [n_games, 7]
        legal_mask_per_seat[batch_idx, abs_seats] = slot_present

    # Per-seat oracle softmax: for each seat s, compute E[Q | s is acting player]
    # using the same worlds (q_per_world is from current player's perspective).
    # For other seats: q_per_world is the correct oracle signal (same worlds, same trick state)
    # since Q is a symmetric function of the full deal—just the POV differs for action legality.
    # We marginalise: e_q_per_seat[s] = mean over M worlds of q_per_world (same Q values,
    # different legal mask for seat s).
    e_q = q_per_world.mean(dim=1)  # [n_games, 7] marginal E[Q] from current player perspective

    # For computing p_make per seat, we use the marginal e_q for all seats (same game outcome)
    # but with seat-specific offense/defense threshold and seat-specific legal mask.
    oracle_softmax_per_seat = torch.zeros(n_games, 4, n_actions, device=device)

    for seat_offset in range(4):
        # Absolute seat index for each game
        abs_seats = (current_player + seat_offset) % 4  # [n_games]

        # Compute p_make from current e_q perspective (Q is symmetric: same outcome regardless of who asks)
        # Use the seat's offense/defense status
        is_offense = (abs_seats % 2 == bidder % 2)  # [n_games]
        # p_make is meaningful from e_q_pdf, but we don't have per-seat PDFs.
        # Use e_q normalized by legal mask as proxy softmax: softmax(e_q) over legal slots.
        seat_legal = legal_mask_per_seat[batch_idx, abs_seats]  # [n_games, 7]

        # Mask e_q to legal actions only
        e_q_masked = e_q.clone()
        e_q_masked[~seat_legal] = float('-inf')

        # p_make proxy: for offense, prefer higher Q; for defense, prefer lower Q
        # Sign-flip Q for defense so softmax over (-Q) = prefer lower Q
        sign = torch.where(is_offense, torch.ones(n_games, device=device), -torch.ones(n_games, device=device))
        signed_q = e_q_masked * sign.unsqueeze(1)  # [n_games, 7]

        # Compute softmax (temperature=1 for now; -inf positions get ~0 weight)
        # Replace -inf with large negative for softmax stability
        signed_q_clipped = torch.where(
            seat_legal,
            signed_q,
            torch.full_like(signed_q, -1e9),
        )
        softmax = torch.softmax(signed_q_clipped, dim=1)  # [n_games, 7]
        # Zero out illegal slots explicitly
        softmax = softmax * seat_legal.float()

        oracle_softmax_per_seat[batch_idx, abs_seats] = softmax

    # Per-seat voids: [n_games, 4, 3, 8] — void inferences from each seat's perspective
    # Compute using the Python-level infer_voids function (CPU, per game)
    from gus.model.voids import infer_voids

    voids_per_seat = torch.zeros(n_games, 4, 3, 8, dtype=torch.bool, device='cpu')

    # Reconstruct prior plays from states.history for each game
    # states.history: [n_games, 28, 3] -> (player, domino, ?)
    for g in range(n_games):
        decl_id = int(states.decl_ids[g].item())
        # Reconstruct play sequence from history
        prior_plays: list[tuple[int, int]] = []
        for step in range(28):
            entry = states.history[g, step]
            if entry[0] < 0:
                break
            p = int(entry[0].item())
            d = int(entry[1].item())
            prior_plays.append((p, d))

        for seat in range(4):
            voids = infer_voids(prior_plays, decl_id, seat)  # [3, 8]
            voids_per_seat[g, seat] = voids

    if device != 'cpu':
        voids_per_seat = voids_per_seat.to(device)

    return oracle_softmax_per_seat, legal_mask_per_seat, voids_per_seat


def select_actions(
    states: GameStateTensor,
    e_q: Tensor,
    e_q_pdf: Tensor,
    greedy: bool,
    exploration_policy: ExplorationPolicy | None = None,
    rng: np.random.Generator | None = None,
    bid_values: list[int] | Tensor | None = None,
) -> tuple[Tensor, list | None]:
    """Select actions by probability of making the contract (p_make).

    Texas 42 has a threshold-based payoff: the bidding team (P0/P2) needs >= 30 points
    to make the contract. E[Q] = 0 is NOT neutral - it's a 21-21 split, which is a
    LOSS for offense. This function optimizes p_make = P(Q >= threshold | action)
    instead of E[Q], with E[Q] as tie-breaker.

    Win thresholds:
        - Offense at bid 30: win when Q >= 18 (team scored >= 30) -> bin 60+
        - Defense at bid 30: win when Q >= -17 (bidder scored <30) -> bin 25+
        - Higher bids move those bins per bid value.

    Args:
        states: GameStateTensor
        e_q: [n_games, 7] E[Q] values (for tie-breaking)
        e_q_pdf: [n_games, 7, 85] P(Q=q|action) for q in [-42, +42], bin i -> Q = i - 42
        greedy: If True, argmax. If False, softmax sample.
        exploration_policy: Optional exploration policy (overrides greedy if provided)
        rng: NumPy RNG for exploration
        bid_values: Optional [n_games] contract values. Defaults to 30.

    Returns:
        Tuple of (actions, exploration_stats):
            - actions: [n_games] action indices (0-6)
            - exploration_stats: List of ExplorationStats (one per game), or None if no exploration
    """
    from forge.eq.exploration import _select_action_with_exploration

    n_games = states.n_games

    # Get legal actions: [n_games, 7]
    legal_mask = states.legal_actions()

    p_make = _p_make_from_pdf(e_q_pdf, states.bidder, states.current_player, bid_values)

    # If exploration policy provided, use it (per-game selection)
    # Note: exploration still uses E[Q] for now (separate concern)
    if exploration_policy is not None:
        from forge.eq.types import ExplorationStats

        actions = []
        exploration_stats = []
        for g in range(n_games):
            action_idx, selection_mode, action_entropy = _select_action_with_exploration(
                e_q_mean=e_q[g].cpu(),  # Move to CPU for numpy conversion
                legal_mask=legal_mask[g].cpu(),
                policy=exploration_policy,
                rng=rng,
            )
            actions.append(action_idx)

            # Compute greedy action (by p_make) and q_gap
            legal = legal_mask[g]
            masked_p_make = p_make[g].clone()
            masked_p_make[~legal] = float('-inf')
            greedy_action = masked_p_make.argmax().item()

            q_greedy = e_q[g][greedy_action].item()
            q_taken = e_q[g][action_idx].item()
            q_gap = q_greedy - q_taken

            stats = ExplorationStats(
                greedy_action=greedy_action,
                action_taken=action_idx,
                was_greedy=(action_idx == greedy_action),
                selection_mode=selection_mode,
                q_gap=q_gap,
                action_entropy=action_entropy,
            )
            exploration_stats.append(stats)
        return torch.tensor(actions, dtype=torch.long, device=e_q.device), exploration_stats

    # Mask illegal actions
    p_make_masked = p_make.clone()
    p_make_masked[~legal_mask] = float('-inf')

    if greedy:
        # Tie-break by E[Q]: add tiny normalized E[Q] term
        # This ensures "lose gracefully" (smaller margin) or "win big" (larger margin)

        # Normalize E[Q] to [0, 1] using only LEGAL action values
        # Use extreme values for illegal so they don't affect min/max
        e_q_for_min = e_q.clone()
        e_q_for_min[~legal_mask] = float('inf')  # Won't be the min
        e_q_for_max = e_q.clone()
        e_q_for_max[~legal_mask] = float('-inf')  # Won't be the max

        e_q_min = e_q_for_min.min(dim=1, keepdim=True).values
        e_q_max = e_q_for_max.max(dim=1, keepdim=True).values
        e_q_range = (e_q_max - e_q_min).clamp(min=1e-10)

        # Normalize E[Q] to [0, 1], zero out illegal actions
        e_q_normalized = (e_q - e_q_min) / e_q_range
        e_q_normalized[~legal_mask] = 0.0  # Don't affect score for illegal

        # p_make is in [0, 1], tie-break term is negligible (1e-6 scale)
        # Note: 1e-9 is too small for float32 precision, gets absorbed
        score = p_make_masked + 1e-6 * e_q_normalized
        actions = score.argmax(dim=1)
    else:
        # Softmax sample over p_make
        probs = torch.softmax(p_make_masked, dim=1)
        actions = torch.multinomial(probs, num_samples=1).squeeze(1)

    return actions, None


def record_decisions(
    states: GameStateTensor,
    e_q: Tensor,
    e_q_var: Tensor,
    e_q_pdf: Tensor,
    actions: Tensor,
    all_decisions: list[list[DecisionRecordGPU]],
    diagnostics: PosteriorDiagnostics | None = None,
    exploration_stats: list | None = None,
    n_samples_used: int | None = None,
    did_converge: bool | None = None,
    world_hands: Tensor | None = None,
    q_per_world: Tensor | None = None,
    bid_values: list[int] | None = None,
    oracle_softmax_per_seat: Tensor | None = None,
    legal_mask_per_seat: Tensor | None = None,
    voids_per_seat: Tensor | None = None,
):
    """Record decisions for each game (in-place).

    Args:
        states: GameStateTensor
        e_q: [n_games, 7] E[Q] mean values
        e_q_var: [n_games, 7] E[Q] variance values
        e_q_pdf: [n_games, 7, 85] full PDF P(Q=q|action) for q in [-42, +42]
        actions: [n_games] action indices
        all_decisions: List of decision lists (one per game)
        diagnostics: Optional posterior diagnostics (aggregated across games)
        exploration_stats: Optional exploration stats (one per game)
        n_samples_used: Optional number of samples used (for adaptive mode)
        did_converge: Optional convergence status (for adaptive mode)
        world_hands: Optional [n_games, M, 3, 7] sampled opponent hands per world
        q_per_world: Optional [n_games, M, 7] oracle Q-values per action per world
        bid_values: Optional list of bid values per game (Schema v2)
        oracle_softmax_per_seat: Optional [n_games, 4, 7] softmax per seat (Schema v2)
        legal_mask_per_seat: Optional [n_games, 4, 7] legal mask per seat (Schema v2)
        voids_per_seat: Optional [n_games, 4, 3, 8] voids per seat (Schema v2)
    """
    n_games = states.n_games
    legal_mask = states.legal_actions()
    current_players = states.current_player

    # Mode mapping for exploration (CPU pipeline convention)
    mode_to_int = {"greedy": 0, "boltzmann": 1, "epsilon": 2, "blunder": 3}

    for g in range(n_games):
        # Only record if game is active
        if not states.active_games()[g]:
            continue

        # Compute state uncertainty from variance
        sigma = torch.sqrt(e_q_var[g])  # [7]
        legal = legal_mask[g]
        if legal.any():
            u_mean = sigma[legal].mean().item()
            u_max = sigma[legal].max().item()
        else:
            u_mean = 0.0
            u_max = 0.0

        # Extract exploration stats if available
        exploration_mode = None
        q_gap = None
        greedy_action = None
        if exploration_stats is not None and g < len(exploration_stats):
            stats = exploration_stats[g]
            exploration_mode = mode_to_int.get(stats.selection_mode, 0)
            q_gap = stats.q_gap
            greedy_action = stats.greedy_action

        record = DecisionRecordGPU(
            player=current_players[g].item(),
            e_q=e_q[g].cpu(),
            action_taken=actions[g].item(),
            legal_mask=legal_mask[g].cpu(),
            e_q_var=e_q_var[g].cpu(),
            e_q_pdf=e_q_pdf[g].cpu(),
            u_mean=u_mean,
            u_max=u_max,
            ess=diagnostics.ess if diagnostics else None,
            max_w=diagnostics.max_w if diagnostics else None,
            exploration_mode=exploration_mode,
            q_gap=q_gap,
            greedy_action=greedy_action,
            n_samples=n_samples_used,
            converged=did_converge,
            world_hands=world_hands[g].cpu() if world_hands is not None else None,
            q_per_world=q_per_world[g].cpu() if q_per_world is not None else None,
            bid_value=bid_values[g] if bid_values is not None else None,
            oracle_softmax_per_seat=oracle_softmax_per_seat[g].cpu() if oracle_softmax_per_seat is not None else None,
            legal_mask_per_seat=legal_mask_per_seat[g].cpu() if legal_mask_per_seat is not None else None,
            voids_per_seat=voids_per_seat[g].cpu() if voids_per_seat is not None else None,
        )
        all_decisions[g].append(record)


def collate_records(
    hands: list[list[list[int]]],
    decl_ids: list[int],
    all_decisions: list[list[DecisionRecordGPU]],
    bid_values: list[int] | None = None,
) -> list[GameRecordGPU]:
    """Collate decision records into GameRecordGPU.

    Args:
        hands: Initial deals
        decl_ids: Declaration IDs
        all_decisions: Decision records per game
        bid_values: Optional bid values per game (Schema v2)

    Returns:
        List of GameRecordGPU
    """
    return [
        GameRecordGPU(
            decisions=decisions,
            hands=hands[g],
            decl_id=decl_ids[g],
            bid_value=bid_values[g] if bid_values is not None else None,
        )
        for g, decisions in enumerate(all_decisions)
    ]
