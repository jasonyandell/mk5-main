"""Zeb belief-head exposed as a stateless tool for Burl.

Wraps the belief head of a `ZebModel` checkpoint so Burl can ask:

    "Given what I can see (my hand + play history), what is the probability
     that a *specific hidden domino* is held by each of my three opponents?"

Design:

- Stateless at the call boundary. Every call takes the full `ZebGameState`.
- The model is loaded once and cached (module-level singleton keyed by
  `(checkpoint_path, device)`).
- A forward pass produces the full `[28, 3]` belief matrix for the perspective
  seat; `get_belief` slices it. `get_belief_batch` amortizes one forward pass
  across many `(player, domino)` queries.

Seat / index conventions (these matter — be careful):

Zeb's observation encoder uses *relative* seat IDs where the perspective seat
is 0:

    0 = me (perspective)
    1 = left  opponent  (seat to my left,  i.e. the next player)
    2 = partner
    3 = right opponent  (seat on my right, i.e. the previous player)

The belief head outputs a `[28, 3]` tensor of logits over the three possible
opponents for each domino. The class axis matches the training target built
in `forge/zeb/eq_player.py::_belief_targets_from_owner`:

    class 0  -> relative seat 1  -> left opponent   (P_L)
    class 1  -> relative seat 2  -> partner         (P_partner)
    class 2  -> relative seat 3  -> right opponent  (P_R)

`get_belief(..., player_seat=k)` expects `k in {1, 2, 3}` *relative to the
perspective seat*. Passing `0` (yourself) raises — your own hand is visible
state, not a belief. The perspective seat defaults to the current player to
act; pass `me_seat=` explicitly to override.

Dominoes that are already visible (in my hand or already played) have no
well-defined belief: the model was trained only on hidden dominoes. For these,
`get_belief` returns a one-hot degenerate distribution over the true holder
when the true holder is me (so P_L=P_partner=P_R=0 sums to 0 -- handled by
masking) or raises. To keep the sum-to-one invariant we raise
`DominoVisibleError` instead, and document that callers should filter using
`forge/zeb/observation`'s play history before asking.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from forge.zeb import load_model
from forge.zeb.game import current_player
from forge.zeb.model import ZebModel
from forge.zeb.observation import observe
from forge.zeb.types import ZebGameState


# Default checkpoint. Chosen because it's the strongest *belief-head* snapshot
# committed to the repo: the "large" architecture (3.3M params, 256/8/6/512)
# with a bootstrapped belief head trained on EQ ownership targets.
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "forge/zeb/checkpoints/lb-v-eq-3740-bootstrap.pt"

# 28 dominoes, 3 opponents.
N_DOMINOES = 28
N_OPPONENTS = 3


class DominoVisibleError(ValueError):
    """Raised when a belief is requested for a domino that isn't hidden.

    A domino is "visible" to the perspective seat if it is in that seat's hand
    or already present in the play history. The belief head was not trained
    for these positions and no three-way distribution is meaningful.
    """


@dataclass(frozen=True)
class ZebBeliefModel:
    """Wrapper around a belief-head `ZebModel` with caller-friendly metadata."""

    model: ZebModel
    device: torch.device
    checkpoint_path: Path
    epoch: int | None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ZebBeliefModel(checkpoint={self.checkpoint_path.name}, "
            f"epoch={self.epoch}, device={self.device})"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _resolve_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=4)
def _load_cached(path_str: str, device_str: str) -> ZebBeliefModel:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Zeb checkpoint not found: {path}")

    model, ckpt = load_model(str(path), device=device_str, eval_mode=True)
    if not getattr(model, "has_belief_head", False):
        raise ValueError(
            f"Checkpoint {path} has no belief head. "
            "Pick one of the *-belief-*.pt snapshots."
        )
    return ZebBeliefModel(
        model=model,
        device=torch.device(device_str),
        checkpoint_path=path,
        epoch=ckpt.get("epoch"),
    )


def load_belief_model(
    checkpoint_path: str | Path | None = None,
    device: str | None = None,
) -> ZebBeliefModel:
    """Load (or reuse) a belief-head Zeb model.

    Subsequent calls with the same `(checkpoint_path, device)` return the
    already-loaded instance. The model is set to eval mode and moved to
    `device`; weights are frozen for inference.
    """
    path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_CHECKPOINT
    dev = _resolve_device(device)
    return _load_cached(str(path), str(dev))


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------


def _resolve_me_seat(game_state: ZebGameState, me_seat: int | None) -> int:
    if me_seat is None:
        return current_player(game_state)
    if not 0 <= me_seat < 4:
        raise ValueError(f"me_seat must be in [0, 3], got {me_seat}")
    return me_seat


def _visible_dominoes(game_state: ZebGameState, me_seat: int) -> frozenset[int]:
    """Dominoes whose owner is fully determined from `me_seat`'s view."""
    return frozenset(game_state.hands[me_seat]) | frozenset(game_state.played)


def _belief_matrix(
    bm: ZebBeliefModel,
    game_state: ZebGameState,
    me_seat: int,
) -> Tensor:
    """Forward-pass to get the full [28, 3] softmax belief matrix.

    Shape: `[28, 3]` float32 on CPU, already normalized along the class axis.
    Row `d` is `(P_L, P_partner, P_R)` for domino `d`.
    """
    tokens, mask, hand_indices = observe(game_state, me_seat)
    # All 7 hand slots are valid targets; the masking logic inside the model's
    # policy head uses `hand_mask`, but the belief head is mean-pooled over
    # valid sequence tokens and doesn't depend on it. We pass True to keep
    # the contract simple.
    hand_mask = torch.ones(7, dtype=torch.bool)

    tokens_b = tokens.unsqueeze(0).to(bm.device)
    mask_b = mask.unsqueeze(0).to(bm.device)
    hand_indices_b = hand_indices.unsqueeze(0).to(bm.device)
    hand_mask_b = hand_mask.unsqueeze(0).to(bm.device)

    with torch.no_grad():
        _policy, _value, belief_logits = bm.model(
            tokens_b, mask_b, hand_indices_b, hand_mask_b
        )
    if belief_logits is None:
        raise RuntimeError("Model returned no belief logits")

    probs = F.softmax(belief_logits.float(), dim=-1)  # [1, 28, 3]
    return probs.squeeze(0).cpu()


def get_belief_matrix(
    game_state: ZebGameState,
    me_seat: int | None = None,
    *,
    model: ZebBeliefModel | None = None,
) -> Tensor:
    """Return the full `[28, 3]` belief matrix for `me_seat`'s perspective.

    Useful when Burl wants to reason over many dominoes without 28 separate
    forward passes. Rows corresponding to visible dominoes (in my hand or
    already played) are still populated by the model but are not meaningful;
    use `get_belief` for the per-domino checked accessor.
    """
    bm = model if model is not None else load_belief_model()
    seat = _resolve_me_seat(game_state, me_seat)
    return _belief_matrix(bm, game_state, seat)


def get_belief(
    game_state: ZebGameState,
    player_seat: int,
    domino_id: int,
    *,
    me_seat: int | None = None,
    model: ZebBeliefModel | None = None,
) -> tuple[float, float, float]:
    """P(opponent holds `domino_id`) as `(P_left, P_partner, P_right)`.

    Args:
        game_state: Full Zeb `ZebGameState`. The tool is stateless: pass the
            current state on every call.
        player_seat: *Relative* seat of the opponent being asked about, using
            Zeb's convention: 1=left, 2=partner, 3=right. Passing 0 (self) is
            an error. The triple returned is redundant with `player_seat` --
            the tool always returns the full three-way distribution; callers
            typically pass `player_seat` when they want a single probability
            and index into the returned tuple themselves (see README).
        domino_id: Global domino id in `[0, 28)`.
        me_seat: Absolute seat (0..3) defining the perspective. Defaults to
            `current_player(game_state)` so Burl can just ask "about my play".
        model: Optional pre-loaded `ZebBeliefModel`. Defaults to the cached
            default checkpoint.

    Returns:
        `(P_L, P_partner, P_R)` -- three floats in `[0, 1]` summing to
        `1.0 +/- 1e-5`.

    Raises:
        DominoVisibleError: `domino_id` is in my hand or already played.
        ValueError: `player_seat == 0`, indices out of range, etc.

    Note:
        The returned tuple does not depend on `player_seat` -- the three
        probabilities are ordered (L, partner, R) regardless. `player_seat`
        is accepted for API symmetry with the OVERVIEW spec; pass it to
        indicate which slot the caller cares about. If you want only one
        probability, index: `get_belief(s, 2, d)[1]` for P(partner holds d).
    """
    if not 0 <= domino_id < N_DOMINOES:
        raise ValueError(f"domino_id must be in [0, {N_DOMINOES}), got {domino_id}")
    if player_seat == 0:
        raise ValueError(
            "player_seat=0 is yourself; own hand is visible, not a belief. "
            "Use 1 (left), 2 (partner), or 3 (right)."
        )
    if player_seat not in (1, 2, 3):
        raise ValueError(f"player_seat must be 1, 2, or 3; got {player_seat}")

    seat = _resolve_me_seat(game_state, me_seat)
    if domino_id in _visible_dominoes(game_state, seat):
        raise DominoVisibleError(
            f"Domino {domino_id} is visible to seat {seat} (in hand or played); "
            "no hidden-owner belief exists."
        )

    matrix = get_belief_matrix(game_state, me_seat=seat, model=model)
    row = matrix[domino_id]
    return float(row[0]), float(row[1]), float(row[2])


def get_belief_batch(
    game_state: ZebGameState,
    queries: Iterable[tuple[int, int]],
    *,
    me_seat: int | None = None,
    model: ZebBeliefModel | None = None,
) -> list[tuple[float, float, float]]:
    """Batched `get_belief`. One forward pass; all queries share the matrix.

    Args:
        queries: iterable of `(player_seat, domino_id)` pairs.

    Returns:
        List of `(P_L, P_partner, P_R)` triples, one per query, in order.
    """
    queries = list(queries)
    seat = _resolve_me_seat(game_state, me_seat)
    visible = _visible_dominoes(game_state, seat)
    matrix = get_belief_matrix(game_state, me_seat=seat, model=model)

    out: list[tuple[float, float, float]] = []
    for player_seat, domino_id in queries:
        if not 0 <= domino_id < N_DOMINOES:
            raise ValueError(f"domino_id must be in [0, {N_DOMINOES}), got {domino_id}")
        if player_seat not in (1, 2, 3):
            raise ValueError(
                f"player_seat must be 1, 2, or 3; got {player_seat}"
            )
        if domino_id in visible:
            raise DominoVisibleError(
                f"Domino {domino_id} is visible to seat {seat}; no belief."
            )
        row = matrix[domino_id]
        out.append((float(row[0]), float(row[1]), float(row[2])))
    return out


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _self_test() -> None:
    from forge.oracle.declarations import DECL_ID_TO_NAME
    from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW
    from forge.zeb.game import apply_action, legal_actions, new_game

    bm = load_belief_model()
    print(f"Loaded: {bm}")

    # Deal a game and play a few tricks so the play history is non-empty.
    state = new_game(seed=42, skip_bidding=True)
    for _ in range(6):
        legal = legal_actions(state)
        if not legal:
            break
        state = apply_action(state, legal[0])

    me_seat = current_player(state)
    visible = _visible_dominoes(state, me_seat)
    hidden = [d for d in range(N_DOMINOES) if d not in visible]

    print(
        f"seed=42, trump='{DECL_ID_TO_NAME[state.decl_id]}' (decl_id={state.decl_id}), "
        f"me={me_seat}, my_hand={sorted(state.hands[me_seat])}, "
        f"played={sorted(state.played)}, hidden_count={len(hidden)}"
    )

    # Single query.
    dom = hidden[0]
    p_l, p_p, p_r = get_belief(state, player_seat=2, domino_id=dom, model=bm)
    pip_hi, pip_lo = DOMINO_HIGH[dom], DOMINO_LOW[dom]
    total = p_l + p_p + p_r
    assert abs(total - 1.0) < 1e-5, f"sum != 1 ({total})"
    assert all(0.0 <= x <= 1.0 for x in (p_l, p_p, p_r))
    assert all(torch.isfinite(torch.tensor(x)) for x in (p_l, p_p, p_r))
    print(
        f"domino {dom} ({pip_hi}-{pip_lo}): "
        f"P_L={p_l:.3f}  P_partner={p_p:.3f}  P_R={p_r:.3f}  sum={total:.6f}"
    )

    # Batched query -- check it agrees with per-call queries.
    batch_queries = [(1, d) for d in hidden[:5]]
    batch = get_belief_batch(state, batch_queries, model=bm)
    for (_, d), (a, b, c) in zip(batch_queries, batch):
        single = get_belief(state, player_seat=1, domino_id=d, model=bm)
        assert all(abs(x - y) < 1e-6 for x, y in zip((a, b, c), single))
        s = a + b + c
        assert abs(s - 1.0) < 1e-5
    print(f"batched {len(batch)} queries -- all sum to 1, all agree with single-call")

    # Visible-domino guard.
    my_dom = next(iter(state.hands[me_seat]))
    try:
        get_belief(state, player_seat=1, domino_id=my_dom, model=bm)
    except DominoVisibleError:
        print(f"correctly rejected belief query for my own domino {my_dom}")
    else:  # pragma: no cover
        raise AssertionError("expected DominoVisibleError for own domino")

    # Full matrix path.
    matrix = get_belief_matrix(state, model=bm)
    assert matrix.shape == (N_DOMINOES, N_OPPONENTS)
    rows = matrix.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5), rows
    print(f"full [28, 3] matrix: shape OK, all 28 rows sum to 1 (+/- 1e-5)")

    # Show three hidden dominoes with their belief entropies.
    print("three hidden dominoes, belief entropy (nats):")
    for d in hidden[:3]:
        row = matrix[d]
        ent = -(row * torch.log(row.clamp_min(1e-12))).sum().item()
        hi, lo = DOMINO_HIGH[d], DOMINO_LOW[d]
        print(
            f"  {hi}-{lo} (id={d}): L={row[0]:.3f} P={row[1]:.3f} R={row[2]:.3f}  "
            f"H={ent:.3f}"
        )

    print("self-test OK")


if __name__ == "__main__":
    _self_test()
