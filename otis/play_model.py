"""OtisPlayNet — the play-time sibling of ``otis.model.OtisNet`` (issue #53, V2).

Where ``OtisNet`` is bid-time only (the 91-dim declarer-hand ⊕ auction row from
``champion.margin_net``), this net conditions on a mid-hand INFORMATION STATE from
the acting seat's perspective and predicts, off one trunk:

  * ``fate``  — each of the five count tiles' 8-class fate (capture × mode), with
                the capture axis RELABELED so class 0..3 = "the actor's team
                captures this tile" (``otis.play_data`` does the relabel).
  * ``trick`` — the actor-team trick count (0..7), an 8-way head.

Fate-class layout matches ``otis.model`` (``class = capture_idx*4 + mode_idx``,
mode order led=0, followed=1, trumped_in=2, sloughed=3), and the five fate heads
are ordered by ``otis.model.TILE_PIPS``.

CRITICAL invariant of :func:`featurize_play_state`: it is INFO-STATE ONLY. It reads
the perspective seat's own remaining hand, the public played set, the public
current trick, the auction, the running score, and engine-inferred voids — and
NEVER any other seat's unplayed tiles. Permuting the opponents' concealed hands
must leave the feature row bit-identical (``test_play_model`` asserts this).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState
from gus.model.voids import voids_feature_vector
from otis.model import N_FATE_CLASSES, N_TILES, N_TRICKS, TILE_PIPS

# --------------------------------------------------------------------------- #
# Feature schema                                                                #
# --------------------------------------------------------------------------- #

N_DOMINOES = 28
N_DECLS = 10           # decl_id 0..9 (9 = notrump)
N_TRICK_SLOTS = 3      # up to 3 tiles already in the trick before the actor plays
N_VOIDS = 24           # gus voids_feature_vector: 3 opponents × 8 suits

FEATURE_SCHEMA_VERSION = "otis-play-v0"

# Block sizes, in construction order (documented so FEATURE_DIM is auditable):
#   remaining hand ......... 28   perspective's own unplayed tiles (multi-hot)
#   played ................. 28   public played set (multi-hot)
#   current trick slots .... 84   3 ordered [28] one-hots, in play order, pad-to-3
#   led domino ............. 28   trick[0] one-hot, zeros when leading
#   trick position ......... 4    one-hot len(current_trick) in {0,1,2,3}
#   decl ................... 10   one-hot decl_id 0..9
#   seat rel to bidder ..... 4    one-hot (perspective - bidder) % 4
#   on bidding team ........ 1    perspective%2 == bidder%2
#   bid value normalized ... 1    clamp((v-30)/12, 0, 1); marks clamp to 1
#   team points ............ 2    (mine, theirs) / 42 from perspective's team
#   trick index ............ 1    completed tricks / 7
#   voids .................. 24   gus voids_feature_vector
FEATURE_DIM = (
    N_DOMINOES            # remaining hand
    + N_DOMINOES          # played
    + N_TRICK_SLOTS * N_DOMINOES  # current-trick ordered slots
    + N_DOMINOES          # led domino
    + 4                   # trick position one-hot
    + N_DECLS             # decl one-hot
    + 4                   # seat rel to bidder one-hot
    + 1                   # on bidding team
    + 1                   # bid value normalized
    + 2                   # team points
    + 1                   # trick index
    + N_VOIDS             # voids
)  # == 215
HIDDEN = 256


def _onehot(idx: int, n: int) -> Tensor:
    v = torch.zeros(n, dtype=torch.float32)
    if 0 <= idx < n:
        v[idx] = 1.0
    return v


def _multihot(domino_ids, n: int = N_DOMINOES) -> Tensor:
    v = torch.zeros(n, dtype=torch.float32)
    for d in domino_ids:
        v[int(d)] = 1.0
    return v


def featurize_play_state(state: ZebGameState, perspective: int) -> Tensor:
    """Build the [FEATURE_DIM] info-state row for seat ``perspective``.

    ``perspective`` is the acting seat (``forge.zeb.game.current_player(state)`` at
    a decision point). Reads only public state + the perspective seat's own hand;
    never any other seat's concealed tiles.
    """
    bidder = int(state.bidder)
    p_team = perspective % 2
    o_team = 1 - p_team

    # Perspective's remaining hand (own tiles minus the public played set).
    remaining = [d for d in state.hands[perspective] if d not in state.played]
    f_hand = _multihot(remaining)

    # Public played set.
    f_played = _multihot(state.played)

    # Current-trick tiles in play order, three ordered [28] one-hot slots.
    trick = state.current_trick
    slots = []
    for i in range(N_TRICK_SLOTS):
        if i < len(trick):
            slots.append(_onehot(int(trick[i]), N_DOMINOES))
        else:
            slots.append(torch.zeros(N_DOMINOES, dtype=torch.float32))
    f_trick_slots = torch.cat(slots)

    # Led domino (trick[0]); zeros when the perspective seat is leading.
    f_led = _onehot(int(trick[0]), N_DOMINOES) if len(trick) > 0 else torch.zeros(
        N_DOMINOES, dtype=torch.float32
    )

    # Trick position (how many tiles already down this trick).
    f_pos = _onehot(len(trick), 4)

    # Declaration.
    f_decl = _onehot(int(state.decl_id), N_DECLS)

    # Seat relative to bidder; on-bidding-team flag.
    f_seat_rel = _onehot((perspective - bidder) % 4, 4)
    f_on_bid = torch.tensor([1.0 if p_team == bidder % 2 else 0.0], dtype=torch.float32)

    # Bid value normalized: point bids 30..42 map to 0..1, marks clamp to 1.
    bid_norm = (float(state.bid_state.high_bid) - 30.0) / 12.0
    f_bid = torch.tensor([min(max(bid_norm, 0.0), 1.0)], dtype=torch.float32)

    # Running score from the perspective team's point of view.
    f_pts = torch.tensor(
        [state.team_points[p_team] / 42.0, state.team_points[o_team] / 42.0],
        dtype=torch.float32,
    )

    # Completed-trick index.
    f_trick_idx = torch.tensor([len(state.play_history) // 4 / 7.0], dtype=torch.float32)

    # Engine-inferred voids (prior_plays == play_history, already (seat, dom)).
    f_voids = voids_feature_vector(list(state.play_history), int(state.decl_id), perspective)

    row = torch.cat(
        [
            f_hand, f_played, f_trick_slots, f_led, f_pos, f_decl,
            f_seat_rel, f_on_bid, f_bid, f_pts, f_trick_idx, f_voids,
        ]
    )
    assert row.numel() == FEATURE_DIM, f"expected {FEATURE_DIM}, got {row.numel()}"
    return row


# --------------------------------------------------------------------------- #
# Network                                                                       #
# --------------------------------------------------------------------------- #


class OtisPlayNet(nn.Module):
    """Play-state trunk with five fate heads (order = ``TILE_PIPS``) + a trick head.

    ``forward`` returns ``{"fate": [B,5,8] logits, "trick": [B,8] logits}``.
    """

    def __init__(self, in_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        self.fate_heads = nn.ModuleList(
            [nn.Linear(HIDDEN, N_FATE_CLASSES) for _ in range(N_TILES)]
        )
        self.trick_head = nn.Linear(HIDDEN, N_TRICKS)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        h = self.trunk(x)
        fate = torch.stack([head(h) for head in self.fate_heads], dim=1)  # [B,5,8]
        return {"fate": fate, "trick": self.trick_head(h)}  # trick [B,8]

    # -- persistence ------------------------------------------------------- #

    def save(self, path) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "feature_dim": self.in_dim,
                "feature_schema": FEATURE_SCHEMA_VERSION,
                "tile_pips": list(TILE_PIPS),
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location="cpu") -> "OtisPlayNet":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if blob.get("feature_schema") != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"feature schema mismatch: checkpoint {blob.get('feature_schema')!r} "
                f"!= current {FEATURE_SCHEMA_VERSION!r}"
            )
        net = cls(in_dim=int(blob["feature_dim"]))
        net.load_state_dict(blob["state_dict"])
        return net


def fate_capture_probs(out: dict[str, Tensor]) -> Tensor:
    """P(perspective's team captures each tile) — ``[B,5]``.

    Labels are relabeled so capture axis 0 == the actor's team, so the four
    "my team captures" classes are indices 0..3; sum their softmax mass.
    """
    p = torch.softmax(out["fate"], dim=-1)  # [B,5,8]
    return p[..., 0:4].sum(dim=-1)  # [B,5]
