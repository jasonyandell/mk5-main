"""Sample opponent hands from belief_head output.

Two modes, intended for LAMIR-1 leaf evaluation:

  argmax_world(belief_logits, belief_mask)
    → [B, 28, 3] float world_assignment

    For each unseen domino, place it in the seat with highest belief
    probability. Fast, deterministic, zero-garbage — good enough for a
    first prototype. May produce slight hand-size imbalances when ties
    occur, but never produces collisions (each domino goes to exactly one
    seat).

  sample_worlds(belief_logits, belief_mask, K, rng, hand_sizes=None)
    → [B, K, 28, 3] float world_assignment

    Sample K collision-free, hand-size-consistent worlds per decision.
    Each world is a valid assignment: every unseen domino goes to exactly
    one seat, and each seat receives at most `hand_sizes[b, s]` dominoes.

    Algorithm (per batch item, per world):
      1. Derive target hand sizes from hand_sizes tensor or infer from the
         number of unseen dominoes split as evenly as possible.
      2. Shuffle the unseen dominoes.
      3. Assign each one by sampling from belief probabilities restricted
         to seats that still have capacity.
      4. If all seats are full for a domino (shouldn't happen with correct
         sizes), fall back to argmax over remaining capacity.

    This is O(B * K * D) sequential Python but D≤28 is tiny; K=10-50
    worlds per decision is the expected use case.

    `hand_sizes`: [B, 3] int tensor of (left_opp, partner, right_opp)
    remaining hand sizes. If None, inferred from belief_mask by splitting
    unseen count as evenly as possible (floor/ceiling).

Input shapes:
    belief_logits : [B, 28, 3] — raw logits from belief_head
    belief_mask   : [B, 28] bool — True for dominoes that are "unseen"
                    (not in current player's hand and not yet played).
                    These are the dominoes the belief head is predicting.
"""

from __future__ import annotations

import torch
from torch import Tensor


N_DOMINOES = 28
N_SEATS = 3  # left_opp, partner, right_opp


def argmax_world(
    belief_logits: Tensor,  # [B, 28, 3]
    belief_mask: Tensor,    # [B, 28] bool
) -> Tensor:
    """Place each unseen domino in its highest-probability seat.

    Returns [B, 28, 3] float world_assignment. Masked (seen) dominoes get
    the zero vector. Each unseen domino gets exactly one seat set to 1.0.
    """
    B = belief_logits.shape[0]
    probs = torch.softmax(belief_logits, dim=-1)       # [B, 28, 3]
    seat_pick = probs.argmax(dim=-1)                   # [B, 28]
    assignment = torch.nn.functional.one_hot(seat_pick, num_classes=N_SEATS).float()  # [B, 28, 3]
    # Zero out seen/played dominoes
    assignment = assignment * belief_mask.unsqueeze(-1).float()
    return assignment


def _infer_hand_sizes(belief_mask: Tensor) -> Tensor:
    """Infer [B, 3] remaining hand sizes from the number of unseen dominoes.

    Distributes floor(n/3) to each seat, then gives the +1 remainder to
    seats 0, 1, ... (left_opp first).  This matches the expected distribution
    at most game states where all three opponents started with equal hands
    and have played the same number of tricks.
    """
    B = belief_mask.shape[0]
    n_unseen = belief_mask.sum(dim=-1)  # [B] int
    base = n_unseen // 3               # [B]
    rem = n_unseen % 3                 # [B]
    sizes = base.unsqueeze(-1).expand(B, N_SEATS).clone()  # [B, 3]
    # Give +1 to seats 0..rem-1
    for s in range(N_SEATS):
        sizes[:, s] += (rem > s).long()
    return sizes  # [B, 3]


def sample_worlds(
    belief_logits: Tensor,       # [B, 28, 3]
    belief_mask: Tensor,         # [B, 28] bool
    K: int,
    rng: torch.Generator,
    hand_sizes: Tensor | None = None,  # [B, 3] int, optional
) -> Tensor:
    """Sample K collision-free, hand-size-consistent worlds per decision.

    Returns [B, K, 28, 3] float world_assignment.

    Each of the K worlds is independently sampled. For prototype speed,
    K=1 with argmax_world is the fastest path — use this function when
    diversity across worlds is needed for averaging.
    """
    B = belief_logits.shape[0]
    device = belief_logits.device
    probs = torch.softmax(belief_logits, dim=-1)  # [B, 28, 3]

    if hand_sizes is None:
        hand_sizes = _infer_hand_sizes(belief_mask).to(device)  # [B, 3]

    # Output tensor
    out = torch.zeros(B, K, N_DOMINOES, N_SEATS, dtype=torch.float32, device=device)

    # Inner loop runs on CPU regardless of input device; D=28 so this is fast.
    # Derive a CPU seed from rng.initial_seed() so the caller's rng state
    # advances deterministically without requiring a same-device randint.
    cpu_rng = torch.Generator(device="cpu")
    cpu_rng.manual_seed(rng.initial_seed() & 0xFFFFFFFF)

    probs_cpu = probs.cpu()
    mask_cpu = belief_mask.cpu()
    sizes_cpu = hand_sizes.cpu()

    for b in range(B):
        p_b = probs_cpu[b]         # [28, 3]
        m_b = mask_cpu[b]          # [28] bool
        s_b = sizes_cpu[b].clone() # [3] remaining capacity template

        unseen_idx = m_b.nonzero(as_tuple=False).squeeze(-1).tolist()  # list of domino ids

        for k in range(K):
            # Fresh capacity for each world
            capacity = s_b.clone()  # [3]
            assign_k = torch.zeros(N_DOMINOES, N_SEATS, dtype=torch.float32)

            # Shuffle domino order so bias doesn't accumulate for later doms
            order = torch.randperm(len(unseen_idx), generator=cpu_rng).tolist()
            shuffled = [unseen_idx[i] for i in order]

            for d in shuffled:
                # Restrict to seats with remaining capacity
                p_d = p_b[d].clone()  # [3]
                feasible = (capacity > 0)
                if not feasible.any():
                    # Degenerate: no capacity left (hand_sizes undercount)
                    # Fall back to argmax over all seats
                    seat = int(p_d.argmax().item())
                else:
                    p_d = p_d * feasible.float()
                    total = p_d.sum()
                    if total <= 0:
                        # All feasible seats have zero probability: uniform
                        p_d = feasible.float()
                        total = p_d.sum()
                    p_d = p_d / total
                    seat = int(torch.multinomial(p_d, 1, generator=cpu_rng).item())
                assign_k[d, seat] = 1.0
                capacity[seat] -= 1

            out[b, k] = assign_k.to(device)

    return out  # [B, K, 28, 3]
