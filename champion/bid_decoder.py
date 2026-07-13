"""Auction decoder v0 (Lane A, GitHub milestone Champion): an INFERENCE
INSTRUMENT that models the probability of each seat's one-shot bid decision:

    pi(action | actor hand, seat position in bidding order, bid prefix, [population])

Where `margin_net` predicts realized value and `bid_net` predicts double-dummy
P(make), this head predicts the ACTION a bidder voluntarily took — a likelihood
that reweights consistent worlds by what the auction reveals
(`wiki/topics/auction-decoder.md`). It is graded ONLY as an instrument:
held-out NLL / accuracy / calibration. No marks or bidding-policy claims.

Featurization — the leakage defense
-----------------------------------
One training sample per bid DECISION (four per hand). For the actor at bidding
position `p` (order = dealer+1, dealer+2, dealer+3, dealer):

* Hand: `champion.bid_net.featurize_hand` (63-dim), reused verbatim. Dropped in
  the hand-independent ablation.
* Seat position one-hot (4): where in the auction the actor speaks.
* Bid PREFIX (9): for each relative prior position j in {0,1,2}, a triple
  [spoke_flag, bid_value/42, is_partner]. Only positions STRICTLY BEFORE the
  actor (j < p) are filled — the only bids the actor heard. The actor's own
  action (the target) and every later seat's bid are NEVER in the features.
* Running high bid so far / 42 (1).
* Optional population one-hot (3): margin:wp / net:wp / random — the policy-type
  label, in its crudest observed form (population-conditioned variant only).

Target
------
The actor's own action class: pass, or the bid level. The class set is derived
from the data (`build_class_map`); at this corpus it is
{pass, 30..42, 84} with most mass on pass/30/31.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from arena.auction import ONE_MARK
from champion.bid_net import FEATURE_DIM as HAND_DIM  # 63
from champion.bid_net import featurize_hand
from champion.margin_net import split_of

# --------------------------------------------------------------------- #
#  Populations                                                           #
# --------------------------------------------------------------------- #

# Mission-specified file → population map (by arena metadata.team_a).
POPULATION_FILES: dict[str, list[str]] = {
    "margin:wp": ["snaps_m1", "snaps_m2", "snaps_m3", "snaps_00", "snaps_01", "snaps_02"],
    "net:wp": ["snaps_n1", "snaps_n2"],
    "random": ["snaps_r1", "snaps_r2"],
}
POPULATIONS: list[str] = list(POPULATION_FILES)
POP_INDEX: dict[str, int] = {p: i for i, p in enumerate(POPULATIONS)}
N_POP = len(POPULATIONS)

# Feature-block dimensions.
POS_DIM = 4
PREFIX_DIM = 9   # 3 relative prior positions × [spoke, value/42, is_partner]
HIGH_DIM = 1


def bidding_order(dealer: int) -> tuple[int, int, int, int]:
    """Seats in bid order: left of shaker first, shaker (dealer) last."""
    return tuple((dealer + i) % 4 for i in (1, 2, 3, 0))


def feature_dim(*, include_hand: bool, include_pop: bool) -> int:
    dim = POS_DIM + PREFIX_DIM + HIGH_DIM
    if include_hand:
        dim += HAND_DIM
    if include_pop:
        dim += N_POP
    return dim


# --------------------------------------------------------------------- #
#  Per-decision featurization (leakage-disciplined)                      #
# --------------------------------------------------------------------- #

def decision_features(
    hand: Sequence[int],
    bids: Sequence[int],
    dealer: int,
    position: int,
    pop: str | None,
    *,
    include_hand: bool,
    include_pop: bool,
) -> Tensor:
    """Feature vector for the actor at bidding `position`.

    `bids` is the seat-indexed final bid vector (0 = pass). Only bids of seats
    strictly BEFORE `position` in bidding order are read; the actor's own bid and
    all later bids are ignored — the decision-time information set.
    """
    order = bidding_order(int(dealer))
    actor_seat = order[position]
    partner_seat = (actor_seat + 2) % 4

    parts: list[Tensor] = []
    if include_hand:
        parts.append(featurize_hand(tuple(int(t) for t in hand)))

    pos_oh = torch.zeros(POS_DIM)
    pos_oh[position] = 1.0
    parts.append(pos_oh)

    prefix = torch.zeros(PREFIX_DIM)  # 3 slots × [spoke, value/42, is_partner]
    high = 0
    for j in range(3):
        if j < position:
            prior_seat = order[j]
            val = int(bids[prior_seat])
            prefix[3 * j + 0] = 1.0
            prefix[3 * j + 1] = val / float(ONE_MARK)
            prefix[3 * j + 2] = 1.0 if prior_seat == partner_seat else 0.0
            high = max(high, val)
    parts.append(prefix)
    parts.append(torch.tensor([high / float(ONE_MARK)]))

    if include_pop:
        pop_oh = torch.zeros(N_POP)
        if pop is not None:
            pop_oh[POP_INDEX[pop]] = 1.0
        parts.append(pop_oh)

    return torch.cat(parts)


# --------------------------------------------------------------------- #
#  Corpus loading + decision expansion                                  #
# --------------------------------------------------------------------- #

def _dedup_key(snap: Mapping) -> tuple:
    """Collapse paired-half replays (self-play → byte-identical auctions)."""
    return (
        int(snap["seed"]), int(snap["hand_idx"]),
        tuple(int(b) for b in snap["bids"]),
        int(snap["bidder"]), int(snap["decl_id"]),
    )


def load_population_hands(corpus_dir: Path, pop: str) -> list[dict]:
    """Deduped hands for one population, tagged with `_pop`."""
    seen: set = set()
    out: list[dict] = []
    for stem in POPULATION_FILES[pop]:
        payload = json.loads((corpus_dir / f"{stem}.json").read_text())
        snaps = payload["snapshots"] if isinstance(payload, dict) else payload
        for s in snaps:
            k = _dedup_key(s)
            if k in seen:
                continue
            seen.add(k)
            s = dict(s)
            s["_pop"] = pop
            out.append(s)
    return out


def load_all_hands(corpus_dir: Path) -> list[dict]:
    hands: list[dict] = []
    for pop in POPULATIONS:
        hands.extend(load_population_hands(corpus_dir, pop))
    return hands


def hand_decisions(snap: Mapping) -> list[tuple[int, int]]:
    """(position, action) for the four seat decisions of one hand."""
    order = bidding_order(int(snap["dealer"]))
    bids = snap["bids"]
    return [(p, int(bids[order[p]])) for p in range(4)]


def build_class_map(hands: Iterable[Mapping]) -> dict[int, int]:
    """action value → class index, derived from the data. pass (0) is class 0;
    the remaining bid levels follow in ascending order."""
    actions: set[int] = set()
    for h in hands:
        for _pos, a in hand_decisions(h):
            actions.add(a)
    ordered = [0] + sorted(a for a in actions if a != 0)
    return {a: i for i, a in enumerate(ordered)}


# --------------------------------------------------------------------- #
#  Dataset                                                               #
# --------------------------------------------------------------------- #

class DecisionDataset(torch.utils.data.Dataset):
    """One sample per bid decision, filtered to a split by deal hash.

    Carries per-sample population index so metrics can be broken out by
    population on a shared held-out set.
    """

    def __init__(
        self,
        hands: Sequence[Mapping],
        class_map: dict[int, int],
        split: str,
        *,
        include_hand: bool,
        include_pop: bool,
    ) -> None:
        self.include_hand = include_hand
        self.include_pop = include_pop
        self.X: list[Tensor] = []
        self.y: list[int] = []
        self.pop_idx: list[int] = []
        for h in hands:
            if split != "all" and split_of(h["seed"], h["hand_idx"]) != split:
                continue
            pop = h.get("_pop")
            for position, action in hand_decisions(h):
                self.X.append(decision_features(
                    h["hands"][bidding_order(int(h["dealer"]))[position]],
                    h["bids"], int(h["dealer"]), position, pop,
                    include_hand=include_hand, include_pop=include_pop,
                ))
                self.y.append(class_map[action])
                self.pop_idx.append(POP_INDEX[pop] if pop is not None else -1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        return self.X[idx], self.y[idx]

    def tensors(self) -> tuple[Tensor, Tensor, Tensor]:
        if not self.y:
            d = feature_dim(include_hand=self.include_hand, include_pop=self.include_pop)
            return (torch.empty(0, d), torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long))
        return (torch.stack(self.X),
                torch.tensor(self.y, dtype=torch.long),
                torch.tensor(self.pop_idx, dtype=torch.long))


# --------------------------------------------------------------------- #
#  Network                                                               #
# --------------------------------------------------------------------- #

class BidDecoderNet(nn.Module):
    """MLP: in → 128 → 128 → n_classes logits."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# --------------------------------------------------------------------- #
#  Metrics                                                               #
# --------------------------------------------------------------------- #

def nll_acc_ece(logits: Tensor, ys: Tensor, n_bins: int = 10) -> dict:
    """Mean NLL (natural log CE), top-1 accuracy, and confidence-ECE."""
    if logits.numel() == 0:
        return {"nll": float("nan"), "acc": float("nan"), "ece": float("nan"), "n": 0}
    logp = torch.log_softmax(logits, dim=-1)
    nll = float(nn.functional.nll_loss(logp, ys).item())
    probs = logp.exp()
    conf, pred = probs.max(dim=-1)
    correct = (pred == ys).float()
    acc = float(correct.mean().item())

    conf_np = conf.numpy()
    corr_np = correct.numpy()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(ys)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == n_bins - 1
        m = (conf_np >= lo) & (conf_np <= hi) if last else (conf_np >= lo) & (conf_np < hi)
        if m.sum() == 0:
            continue
        ece += m.sum() / n * abs(conf_np[m].mean() - corr_np[m].mean())
    return {"nll": nll, "acc": acc, "ece": float(ece), "n": int(n)}


def marginal_baseline(train_ys: Tensor, n_classes: int) -> Tensor:
    """Log-probabilities of the marginal action-frequency baseline (train freq,
    Laplace-smoothed) as a length-`n_classes` vector."""
    counts = torch.bincount(train_ys, minlength=n_classes).float() + 1.0
    return torch.log(counts / counts.sum())


# --------------------------------------------------------------------- #
#  Training                                                              #
# --------------------------------------------------------------------- #

def train_variant(
    hands: Sequence[Mapping],
    class_map: dict[int, int],
    *,
    include_hand: bool,
    include_pop: bool,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 10,
    seed: int = 0,
    device: str = "cpu",
    log: object = None,
) -> tuple[BidDecoderNet, dict]:
    """Train one decoder variant; early-stop on val NLL, return (model, splits)."""
    torch.manual_seed(seed)
    n_classes = len(class_map)
    train_ds = DecisionDataset(hands, class_map, "train",
                               include_hand=include_hand, include_pop=include_pop)
    val_ds = DecisionDataset(hands, class_map, "val",
                             include_hand=include_hand, include_pop=include_pop)
    in_dim = feature_dim(include_hand=include_hand, include_pop=include_pop)

    def _emit(msg: str) -> None:
        print(msg, flush=True)
        if log is not None:
            log.write(msg + "\n")
            log.flush()

    _emit(f"[variant hand={include_hand} pop={include_pop}] in_dim={in_dim} "
          f"classes={n_classes} train={len(train_ds)} val={len(val_ds)}")

    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = BidDecoderNet(in_dim, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    vx, vy, _ = val_ds.tensors()
    best_val = float("inf")
    best_state = None
    bad = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb.to(device)), yb.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vm = nll_acc_ece(model(vx.to(device)).cpu(), vy)
        if vm["nll"] < best_val - 1e-5:
            best_val = vm["nll"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if epoch % 5 == 0 or epoch == 1:
            _emit(f"  epoch {epoch:3d}  val_nll={vm['nll']:.4f} "
                  f"val_acc={vm['acc']:.4f} best={best_val:.4f} bad={bad}")
        if bad >= patience:
            _emit(f"  early stop at epoch {epoch} (best val_nll={best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_nll": best_val, "epochs_run": epoch}
