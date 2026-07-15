"""OtisNet — the fate-ledger-native value net (otis v0, issue #49).

Two arms behind one construction, so the control arm is *bit-exactly* the jud v0
`champion.margin_net.MarginNet` on the pricing path:

    control    = trunk (256→256) + pricing head (→43)          [MarginNet shape]
    treatment  = control + five 8-class fate heads + one 8-class trick head
                 + a mean-consistency penalty tying the fate/trick decomposition
                 back to the pricing pdf's mean.

The trunk is identical in shape *and layer order* to `MarginNet.net`'s first two
`Linear+ReLU` blocks; the pricing head is identical to `MarginNet.net`'s final
`Linear(256, 43)`. Construction order is trunk → pricing → (treatment) fate heads
→ trick head, so under one `torch.manual_seed` the trunk+pricing weights are drawn
identically in both arms — the treatment's extra heads only consume RNG *after* the
shared parameters exist. That is what makes ``export_state_dict`` a 1:1 lift into a
vanilla ``MarginNet`` checkpoint (`champion.value_bidder.load_margin_net` loads it
STRICT).

The featurization is IMPORTED from ``champion.margin_net`` verbatim — this module
never re-derives the 91-dim (declarer-hand ⊕ canonical-auction) row.

Fate class layout (8 classes, matching ``scratch/otis-night/fate_base_rates.json``):

    class = capture_idx * 4 + mode_idx
    capture_idx: bidding_team(=my team, per MarginDataset's y perspective) = 0,
                 opp_of_bidder = 1
    mode_idx:    led=0, followed=1, trumped_in=2, sloughed=3

so the four "my team captures this tile" classes are indices 0..3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# Import the proven featurization + shape constants verbatim — never re-derive.
from champion.margin_net import FEATURE_DIM, N_POINTS, mean_points  # noqa: F401

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #

TILE_PIPS: tuple[str, ...] = ("5-5", "6-4", "5-0", "4-1", "3-2")
TILE_VALUES: tuple[int, ...] = (10, 10, 5, 5, 5)  # per TILE_PIPS
N_TILES = len(TILE_PIPS)                            # 5
N_FATE_CLASSES = 8                                  # (capture 2) × (mode 4)
N_TRICKS = 8                                        # bidder-team tricks 0..7
HIDDEN = 256

# Fate-class axis encodings (must match fate_base_rates.json).
CAPTURE_IDX = {"bidding_team": 0, "opp_of_bidder": 1}
MODE_IDX = {"led": 0, "followed": 1, "trumped_in": 2, "sloughed": 3}
# The four "my team (bidding team) captures" fate classes.
MY_TEAM_CLASSES = tuple(range(4))  # capture_idx == 0 → classes 0,1,2,3


def fate_class(capture_bidding: str, played_mode: str) -> int:
    """8-class fate label from the two label axes (matches base-rate ordering)."""
    return CAPTURE_IDX[capture_bidding] * 4 + MODE_IDX[played_mode]


# --------------------------------------------------------------------------- #
# Network                                                                       #
# --------------------------------------------------------------------------- #


class OtisNet(nn.Module):
    """Shared trunk over info-states with a jud-v0 pricing head (control) and,
    under ``treatment=True``, five fate heads + a trick head off the same trunk.

    ``forward`` returns a dict: always ``{"pricing": [B,43]}``; under treatment
    also ``{"fate": [B,5,8], "trick": [B,8]}``.
    """

    def __init__(self, in_dim: int = FEATURE_DIM, treatment: bool = False) -> None:
        super().__init__()
        self.treatment = treatment
        # Trunk — identical shape AND order to MarginNet.net[0:4].
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
        )
        # Pricing head — identical to MarginNet.net[4] (Linear(256, 43)).
        self.pricing = nn.Linear(HIDDEN, N_POINTS)
        # Treatment organs — constructed AFTER the shared params so RNG parity of
        # trunk+pricing holds across arms under one seed.
        if treatment:
            self.fate_heads = nn.ModuleList(
                [nn.Linear(HIDDEN, N_FATE_CLASSES) for _ in range(N_TILES)]
            )
            self.trick_head = nn.Linear(HIDDEN, N_TRICKS)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        h = self.trunk(x)
        out: dict[str, Tensor] = {"pricing": self.pricing(h)}
        if self.treatment:
            out["fate"] = torch.stack([head(h) for head in self.fate_heads], dim=1)  # [B,5,8]
            out["trick"] = self.trick_head(h)  # [B,8]
        return out

    # -- pricing convenience (bit-exact with MarginNet.forward on the same x) -- #

    def pricing_logits(self, x: Tensor) -> Tensor:
        """[B,43] pricing logits — the pure MarginNet path (trunk→pricing)."""
        return self.pricing(self.trunk(x))

    # -- export: trunk+pricing → a vanilla MarginNet state_dict ---------------- #

    def export_margin_state_dict(self) -> dict[str, Tensor]:
        """Weights laid out as ``MarginNet().state_dict()`` (``net.{0,2,4}``).

        MarginNet.net = Sequential(Linear, ReLU, Linear, ReLU, Linear):
          net.0 ← trunk[0], net.2 ← trunk[2], net.4 ← pricing.
        """
        t = self.trunk
        return {
            "net.0.weight": t[0].weight.detach().clone(),
            "net.0.bias": t[0].bias.detach().clone(),
            "net.2.weight": t[2].weight.detach().clone(),
            "net.2.bias": t[2].bias.detach().clone(),
            "net.4.weight": self.pricing.weight.detach().clone(),
            "net.4.bias": self.pricing.bias.detach().clone(),
        }


# --------------------------------------------------------------------------- #
# Consistency penalty (treatment)                                              #
# --------------------------------------------------------------------------- #


def team_capture_probs(fate_logits: Tensor) -> Tensor:
    """P(my/bidding team captures each tile) from fate logits — ``[B,5]``.

    Sums the four ``capture_idx==0`` (bidding-team) fate classes per tile.
    """
    p = torch.softmax(fate_logits, dim=-1)  # [B,5,8]
    return p[..., list(MY_TEAM_CLASSES)].sum(dim=-1)  # [B,5]


def expected_tricks(trick_logits: Tensor) -> Tensor:
    """E[bidder-team tricks] under the trick head — ``[B]``."""
    p = torch.softmax(trick_logits, dim=-1)  # [B,8]
    k = torch.arange(N_TRICKS, dtype=p.dtype, device=p.device)
    return (p * k).sum(dim=-1)


def consistency_terms(
    pricing_logits: Tensor, fate_logits: Tensor, trick_logits: Tensor
) -> Tensor:
    """Per-row ``|E[pricing pts] − (Σ_t value_t·P(capture_t) + E[tricks])|`` — ``[B]``.

    The mean of this vector is the v0 mean-consistency penalty.
    """
    e_price = mean_points(pricing_logits)  # [B]
    p_cap = team_capture_probs(fate_logits)  # [B,5]
    values = torch.tensor(TILE_VALUES, dtype=p_cap.dtype, device=p_cap.device)
    e_count = (p_cap * values).sum(dim=-1)  # [B]
    e_tricks = expected_tricks(trick_logits)  # [B]
    return (e_price - (e_count + e_tricks)).abs()


def consistency_penalty(
    pricing_logits: Tensor, fate_logits: Tensor, trick_logits: Tensor
) -> Tensor:
    """Scalar mean-consistency penalty over the batch."""
    return consistency_terms(pricing_logits, fate_logits, trick_logits).mean()
