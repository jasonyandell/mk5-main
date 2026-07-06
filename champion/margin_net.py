"""Realized-value head (jud v0, GitHub #32): (declarer hand + decision-time
auction) → distribution over the DECLARING team's captured points (0..42).

Where `bid_net` predicts P(make) of a double-dummy oracle, this head predicts
the REALIZED points a declaring team actually captures under 4-seat play — the
empirical distribution, categorical over 0..42, trained by cross-entropy on
`bidder_team_pts` from the arena snapshot corpus (`arena.match.snapshot_rows`).

Featurization — the leakage defense
-----------------------------------
x = concat(declarer-hand 63-dim, canonical-auction 28-dim) = 91-dim.

* Hand: `champion.bid_net.featurize_hand` (28-dim multi-hot + 7×5 trump feats),
  reused verbatim — NOT duplicated.
* Auction: `gus.model.auction.auction_feature_vector`, but on a CANONICALIZED
  bid vector (`canonical_auction`) with two deliberate erasures:

    1. The winning bid LEVEL is replaced by a constant (``CANON_BID = 30``), for
       both the per-seat entry (``bids_canon[bidder] = 30``) and the global
       ``bid_value`` argument. Under `lens:ev` play the realized outcome is
       independent of how high the winner bid — the level in the corpus is PURE
       SELECTION SIGNAL. Feeding it would teach P(pts | I bid high) >
       P(pts | I bid low) and rebuild the #26 over-bidder *through the training
       data*. So the level is erased; the winner FLAG (``is_winner`` at the
       bidder's seat) and the declared trump stay — that is the belief channel.
    2. Every seat whose turn comes AFTER the bidder's in auction order is masked
       to a pass. The bidder cannot see those bids at decision time; masking them
       keeps train/serve consistent with the hypothetical-auction query the
       future ValueBidder issues ("suppose I win this contract and lead trick 1",
       mirroring `champion.belief_bidder.hypothetical_state`). Earlier opponents'
       real bids/passes are kept — the bidder *did* hear them.

`canonical_auction(bids, bidder, dealer)` is exposed for exactly that reuse: the
ValueBidder featurizes hypothetical roots through the same function, so training
and serving see byte-identical auction encodings.

Model
-----
MLP: 91 → 256 → 256 → 43 logits, categorical over declaring-team points 0..42.
CE loss on realized `bidder_team_pts`. `exceedance(logits)` turns the pmf into
P(pts ≥ t) (reversed cumsum), and `MarginNet.pmake_table` sweeps all 9 EVAL_DECLS
at a hypothetical root to yield {decl: {threshold: P(pts ≥ threshold)}} — the
value-native analogue of `bid_net`'s P(make) table.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from arena.auction import MIN_BID, ONE_MARK, contract_points
from champion.bid_net import FEATURE_DIM as HAND_DIM  # 63
from champion.bid_net import featurize_hand
from forge.bidding.schema import EVAL_DECLS
from gus.model.auction import N_AUCTION_FEATURES, auction_feature_vector

# --------------------------------------------------------------------- #
#  Constants                                                              #
# --------------------------------------------------------------------- #

AUCTION_DIM = N_AUCTION_FEATURES            # 28
FEATURE_DIM = HAND_DIM + AUCTION_DIM        # 91
N_POINTS = ONE_MARK + 1                     # 43 classes: points 0..42
THRESHOLDS = tuple(range(MIN_BID, ONE_MARK + 1))  # 30..42

# The constant bid LEVEL every canonical auction is stamped with — see module
# docstring. 30 is the minimum points bid, so it reads as "a live winner" without
# leaking the true magnitude.
CANON_BID = MIN_BID

_DEFAULT_MODEL = Path("champion/margin_net.pt")


# --------------------------------------------------------------------- #
#  Canonical auction + featurization                                     #
# --------------------------------------------------------------------- #

def canonical_auction(
    bids: Sequence[int], bidder: int, dealer: int
) -> list[int]:
    """Decision-time canonical bid vector for the winning seat.

    Auction order is ``dealer+1, dealer+2, dealer+3, dealer`` (the shaker bids
    last), each seat once. Returns a seat-ordered length-4 list where:

      * seats speaking BEFORE the bidder keep their real bid (pass = 0),
      * the bidder's own entry is the constant ``CANON_BID`` (level erased),
      * seats speaking AFTER the bidder are masked to a pass (0) — unseen at the
        bidder's decision time.

    This is the exact encoding the ValueBidder reuses on hypothetical roots, so
    training and serving stay consistent.
    """
    order = [(dealer + step) % 4 for step in (1, 2, 3, 0)]
    bidder_pos = order.index(int(bidder))
    canon = [0, 0, 0, 0]
    for pos, seat in enumerate(order):
        if pos < bidder_pos:
            real = int(bids[seat])
            canon[seat] = real if real > 0 else 0  # keep real bid; clamp pass/-1 → 0
        elif pos == bidder_pos:
            canon[seat] = CANON_BID
        # pos > bidder_pos: stays 0 (masked, unseen)
    return canon


def featurize_auction(
    bids: Sequence[int], bidder: int, dealer: int, decl_id: int
) -> Tensor:
    """28-dim auction feature from the canonical (level-blind, later-seat-masked)
    bid vector, current-player POV = the bidder (is_winner at relative seat 0)."""
    canon = canonical_auction(bids, bidder, dealer)
    return auction_feature_vector(
        canon,
        bidder=int(bidder),
        bid_value=CANON_BID,  # constant — the winning LEVEL is never fed
        decl_id=int(decl_id),
        current_player=int(bidder),
    )


def featurize(
    hand: Sequence[int], bids: Sequence[int], bidder: int, dealer: int, decl_id: int
) -> Tensor:
    """91-dim input: declarer hand (63) ⊕ canonical auction (28)."""
    return torch.cat([
        featurize_hand(tuple(hand)),
        featurize_auction(bids, bidder, dealer, decl_id),
    ])


def featurize_snapshot(snap: Mapping) -> Tensor:
    """91-dim input from a raw snapshot row (declarer = ``hands[bidder]``)."""
    bidder = int(snap["bidder"])
    hand = tuple(int(t) for t in snap["hands"][bidder])
    return featurize(
        hand,
        [int(b) for b in snap["bids"]],
        bidder,
        int(snap["dealer"]),
        int(snap["decl_id"]),
    )


# --------------------------------------------------------------------- #
#  Distribution helpers                                                   #
# --------------------------------------------------------------------- #

def exceedance(logits: Tensor) -> Tensor:
    """P(pts ≥ t) for t in 0..42 — reversed cumsum of the softmax pmf.

    Works on ``[43]`` or ``[..., 43]``. ``exc[..., 0] == 1`` and the curve is
    monotone non-increasing in t by construction.
    """
    probs = torch.softmax(logits, dim=-1)
    rev = torch.flip(probs, dims=[-1])
    return torch.flip(torch.cumsum(rev, dim=-1), dims=[-1])


def mean_points(logits: Tensor) -> Tensor:
    """E[pts] under the predicted pmf — ``[...]`` reduced over the class axis."""
    probs = torch.softmax(logits, dim=-1)
    pts = torch.arange(N_POINTS, dtype=probs.dtype, device=probs.device)
    return (probs * pts).sum(dim=-1)


# --------------------------------------------------------------------- #
#  Network                                                                #
# --------------------------------------------------------------------- #

class MarginNet(nn.Module):
    """MLP: in → 256 → 256 → 43 logits (categorical over declaring-team points)."""

    def __init__(self, in_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, N_POINTS),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, in_dim] → [B, 43] logits."""
        return self.net(x)

    def pmake_table(
        self, hand: Sequence[int], bids: Sequence[int], bidder: int, dealer: int
    ) -> dict[int, dict[int, float]]:
        """{decl: {threshold: P(pts ≥ threshold)}} over all 9 EVAL_DECLS at the
        hypothetical root — the value-native P(make) table.

        One forward over the distinct declarations; the auction feature is
        re-derived per decl (the declared trump joins the winner identity). The
        threshold demanded by a contract is ``contract_points`` (mirrors
        `champion.bidder`), which for point bids 30..42 is the bid itself.
        """
        decls = list(EVAL_DECLS)
        feats = torch.stack([
            featurize(hand, bids, bidder, dealer, d) for d in decls
        ])  # [9, 91]
        was_training = self.training
        self.eval()
        with torch.no_grad():
            exc = exceedance(self(feats))  # [9, 43]
        if was_training:
            self.train()
        return {
            d: {t: float(exc[i, contract_points(t)]) for t in THRESHOLDS}
            for i, d in enumerate(decls)
        }


# --------------------------------------------------------------------- #
#  Dataset                                                                #
# --------------------------------------------------------------------- #

def _resolve_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """Accept a glob string, a single path, or an iterable of paths."""
    if isinstance(paths, (str, Path)):
        s = str(paths)
        if any(c in s for c in "*?[]"):
            return sorted(Path(p) for p in glob.glob(s))
        return [Path(s)]
    return [Path(p) for p in paths]


def split_of(seed: int, hand_idx: int, *, train: float = 0.90, val: float = 0.05) -> str:
    """Deterministic 90/5/5 train/val/test split by hash of (seed, hand_idx).

    Splitting on the DEAL POSITION (not the individual sample) keeps every
    snapshot of the same deal — including the paired-half replays — in one
    split, so a near-duplicate can never straddle train and test. ``hashlib`` is
    used (not ``hash``) because the built-in is salted per process.
    """
    digest = hashlib.sha256(f"{int(seed)}:{int(hand_idx)}".encode()).hexdigest()
    v = int(digest[:16], 16) / float(1 << 64)  # [0, 1)
    if v < train:
        return "train"
    if v < train + val:
        return "val"
    return "test"


def _dedup_key(snap: Mapping) -> tuple:
    """Exact-duplicate key. Paired halves replay identical seeds, so a self-play
    A==B match yields byte-identical auctions across the two halves; those are
    true duplicates and collapse to one sample."""
    return (
        int(snap["seed"]),
        int(snap["hand_idx"]),
        tuple(int(b) for b in snap["bids"]),
        int(snap["bidder"]),
        int(snap["decl_id"]),
    )


class MarginDataset(torch.utils.data.Dataset):
    """Snapshot-JSON corpus → (91-dim x, realized-points y) samples.

    ``paths`` is a glob string, a single file, or a list of files. Each file is
    either an ``{"snapshots": [...], "metadata": {...}}`` payload
    (`arena.cli --emit-snapshots`) or a bare list of snapshot rows.

    Exact duplicates are dropped (``_dedup_key``); ``split`` in
    {"train","val","test","all"} filters by the deterministic deal-position hash.
    """

    def __init__(
        self,
        paths: str | Path | Iterable[str | Path],
        split: str = "all",
    ) -> None:
        self.split = split
        self.samples: list[tuple[Tensor, Tensor]] = []
        self.keys: list[tuple] = []
        seen: set[tuple] = set()

        for f in _resolve_paths(paths):
            payload = json.loads(Path(f).read_text())
            snaps = payload["snapshots"] if isinstance(payload, dict) else payload
            for snap in snaps:
                if split != "all" and split_of(snap["seed"], snap["hand_idx"]) != split:
                    continue
                key = _dedup_key(snap)
                if key in seen:
                    continue
                seen.add(key)
                x = featurize_snapshot(snap)
                y = int(snap["bidder_team_pts"])
                self.samples.append((x, torch.tensor(y, dtype=torch.long)))
                self.keys.append(key)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.samples[idx]

    def tensors(self) -> tuple[Tensor, Tensor]:
        """Stacked (X [N, 91], Y [N]) — for whole-split metric passes."""
        if not self.samples:
            return torch.empty(0, FEATURE_DIM), torch.empty(0, dtype=torch.long)
        xs = torch.stack([x for x, _ in self.samples])
        ys = torch.stack([y for _, y in self.samples])
        return xs, ys


# --------------------------------------------------------------------- #
#  Metrics                                                                #
# --------------------------------------------------------------------- #

def _reliability(pred: np.ndarray, hit: np.ndarray, n_bins: int = 10) -> dict:
    """Deciles of a predicted probability vs its empirical hit-rate, plus ECE."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bp: list[float] = []
    bt: list[float] = []
    bc: list[int] = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == n_bins - 1
        mask = (pred >= lo) & (pred <= hi) if last else (pred >= lo) & (pred < hi)
        if mask.sum() == 0:
            continue
        bp.append(float(pred[mask].mean()))
        bt.append(float(hit[mask].mean()))
        bc.append(int(mask.sum()))
    total = sum(bc)
    ece = (
        float(sum(c * abs(p - t) for p, t, c in zip(bp, bt, bc)) / total)
        if total
        else float("nan")
    )
    return {"bin_pred_mean": bp, "bin_true_mean": bt, "bin_counts": bc, "ece": ece}


def _metrics(logits: Tensor, ys: Tensor) -> dict:
    """CE, MAE of mean-pts, and the P(pts≥30) reliability/ECE for a split."""
    if logits.numel() == 0:
        return {"ce": float("nan"), "mae_mean_pts": float("nan"), "ece_p30": float("nan")}
    ce = float(nn.functional.cross_entropy(logits, ys).item())
    exc = exceedance(logits)
    p30 = exc[:, MIN_BID].cpu().numpy()
    hit30 = (ys.cpu().numpy() >= MIN_BID).astype(float)
    mae = float((mean_points(logits) - ys.float()).abs().mean().item())
    rel = _reliability(p30, hit30)
    return {"ce": ce, "mae_mean_pts": mae, "ece_p30": rel["ece"], "reliability_p30": rel}


def _forward_all(model: MarginNet, ds: MarginDataset, device: str) -> tuple[Tensor, Tensor]:
    xs, ys = ds.tensors()
    if xs.numel() == 0:
        return xs, ys
    model.eval()
    with torch.no_grad():
        logits = model(xs.to(device)).cpu()
    return logits, ys


# --------------------------------------------------------------------- #
#  Training                                                               #
# --------------------------------------------------------------------- #

def train(
    corpus: str | Path | Iterable[str | Path],
    out_model: Path = _DEFAULT_MODEL,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 8,
    device: str = "cpu",
) -> dict:
    """Train MarginNet on the snapshot corpus; save best-val weights, return metrics."""
    train_ds = MarginDataset(corpus, split="train")
    val_ds = MarginDataset(corpus, split="val")
    test_ds = MarginDataset(corpus, split="test")
    if not len(train_ds):
        raise FileNotFoundError(
            f"No training samples from {corpus!r} — generate arena snapshots first."
        )
    print(
        f"Dataset: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test "
        f"(deduped, 90/5/5 by deal hash)",
        flush=True,
    )

    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = MarginNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    val_logits, val_ys = _forward_all(model, val_ds, device)
    best_val = float("inf")
    best_state = None
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for x_b, y_b in loader:
            logits = model(x_b.to(device))
            loss = loss_fn(logits, y_b.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x_b)
        train_ce = total / len(train_ds)

        val_logits, _ = _forward_all(model, val_ds, device)
        vm = _metrics(val_logits, val_ys)
        print(
            f"  epoch {epoch:3d}/{epochs}  train_ce={train_ce:.4f}  "
            f"val_ce={vm['ce']:.4f}  val_mae={vm['mae_mean_pts']:.3f}  "
            f"val_ece30={vm['ece_p30']:.4f}",
            flush=True,
        )

        # Early-stop on val CE (falls through cleanly if there is no val split).
        if vm["ce"] == vm["ce"] and vm["ce"] < best_val - 1e-5:
            best_val = vm["ce"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if len(val_ds) and bad >= patience:
                print(f"  early stop at epoch {epoch} (no val improvement in {patience})", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    out_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "feature_dim": FEATURE_DIM}, out_model)
    print(f"Saved model → {out_model}", flush=True)

    test_logits, test_ys = _forward_all(model, test_ds, device)
    tm = _metrics(test_logits, test_ys)
    metrics = {
        "best_val_ce": best_val if best_val != float("inf") else float("nan"),
        "test_ce": tm["ce"],
        "test_mae_mean_pts": tm["mae_mean_pts"],
        "test_ece_p30": tm["ece_p30"],
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "test_rows": len(test_ds),
        "epochs_run": epoch,
        "feature_dim": FEATURE_DIM,
    }
    print(
        f"\nBest val CE: {metrics['best_val_ce']:.4f}   "
        f"Test CE: {tm['ce']:.4f}   MAE(mean-pts): {tm['mae_mean_pts']:.3f}   "
        f"ECE P(pts≥30): {tm['ece_p30']:.4f}",
        flush=True,
    )
    return metrics


# --------------------------------------------------------------------- #
#  Evaluation                                                             #
# --------------------------------------------------------------------- #

def evaluate(
    corpus: str | Path | Iterable[str | Path],
    model_path: Path = _DEFAULT_MODEL,
    out_json: Path = Path("scratch/jud-v0/margin_net_eval.json"),
    out_png: Path = Path("scratch/jud-v0/margin_net_reliability.png"),
    optimism_gap: Path = Path("champion/optimism_gap.json"),
    device: str = "cpu",
) -> dict:
    """Test-split reliability report + figure (registered prediction 2)."""
    test_ds = MarginDataset(corpus, split="test")
    if not len(test_ds):
        raise FileNotFoundError(f"No test samples from {corpus!r}.")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = MarginNet(in_dim=ckpt.get("feature_dim", FEATURE_DIM)).to(device)
    model.load_state_dict(ckpt["model_state"])

    logits, ys = _forward_all(model, test_ds, device)
    exc = exceedance(logits).cpu().numpy()          # [N, 43]
    y_np = ys.cpu().numpy()
    N = len(y_np)

    # (i) reliability of P(pts ≥ 30)
    rel = _reliability(exc[:, MIN_BID], (y_np >= MIN_BID).astype(float))

    # (ii) predicted vs empirical exceedance at thresholds 30..42
    thr = list(THRESHOLDS)
    pred_exc = [float(exc[:, t].mean()) for t in thr]
    emp_exc = [float((y_np >= t).mean()) for t in thr]

    # (iii) MAE of mean-pts
    mae = float((mean_points(logits) - ys.float()).abs().mean().item())

    result = {
        "n_test": N,
        "reliability_p30": rel,
        "ece_p30": rel["ece"],
        "exceedance_thresholds": thr,
        "predicted_exceedance": pred_exc,
        "empirical_exceedance": emp_exc,
        "mae_mean_pts": mae,
        "test_ce": float(nn.functional.cross_entropy(logits, ys).item()),
        "model_path": str(model_path),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_json}", flush=True)

    _plot(result, optimism_gap, out_png)
    return result


def _plot(result: dict, optimism_gap: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))

    # -- panel (i): reliability of P(pts ≥ 30) --
    rel = result["reliability_p30"]
    ax0.plot([0, 1], [0, 1], ls=":", color="gray", lw=1, label="perfect")
    ax0.plot(rel["bin_pred_mean"], rel["bin_true_mean"], "-o", color="#c1440e",
             label=f"P(pts≥30), ECE={result['ece_p30']:.3f}")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xlabel("predicted P(pts ≥ 30)")
    ax0.set_ylabel("empirical fraction (pts ≥ 30)")
    ax0.set_title(f"Reliability of P(pts ≥ 30)  (N={result['n_test']})")
    ax0.legend(loc="upper left")
    ax0.grid(alpha=0.2)

    # -- panel (ii): exceedance curves, overlaid with the optimism-gap curves --
    thr = result["exceedance_thresholds"]
    ax1.plot(thr, result["predicted_exceedance"], "-o", color="#2a7f4f",
             label="margin_net predicted E[P(pts≥t)]")
    ax1.plot(thr, result["empirical_exceedance"], "-s", color="#1b1b1b",
             label="test-set empirical P(pts≥t)")

    try:
        gap = json.loads(Path(optimism_gap).read_text())
        rc = gap.get("realized_curve", {})
        rb = sorted(int(b) for b in rc)
        ax1.plot(rb, [rc[str(b)]["make_rate"] for b in rb], "--", color="#c1440e",
                 alpha=0.8, label="realized best-decl make-rate (optimism_gap, N=604)")
        oc = gap.get("oracle_curve", {})
        ob = [int(b) for b in oc if oc[b]["reliable"] and int(b) <= ONE_MARK]
        if ob:
            ax1.plot(ob, [oc[str(b)]["p_make"] for b in sorted(ob)], "^",
                     color="#3b6ea5", ls="none", ms=8,
                     label="oracle double-dummy p_make (optimism_gap, N=50)")
    except (OSError, KeyError, ValueError) as e:
        print(f"(optimism_gap overlay skipped: {e})", flush=True)

    ax1.set_xlabel("threshold t (points)")
    ax1.set_ylabel("P(declaring team ≥ t points)")
    ax1.set_title(
        f"Exceedance: predicted vs realized  (MAE mean-pts = {result['mae_mean_pts']:.2f})"
    )
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}", flush=True)


# --------------------------------------------------------------------- #
#  CLI                                                                    #
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="MarginNet — realized-value head (jud v0)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train", help="train on the snapshot corpus")
    tp.add_argument("--corpus", required=True,
                    help="glob of snapshot JSON files, e.g. 'scratch/jud-v0/corpus/*.json'")
    tp.add_argument("--out-model", type=Path, default=_DEFAULT_MODEL)
    tp.add_argument("--epochs", type=int, default=60)
    tp.add_argument("--batch-size", type=int, default=256)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--patience", type=int, default=8)
    tp.add_argument("--device", type=str, default="cpu")

    ep = sub.add_parser("eval", help="test-split reliability report + figure")
    ep.add_argument("--corpus", required=True, help="glob of snapshot JSON files")
    ep.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    ep.add_argument("--out-json", type=Path, default=Path("scratch/jud-v0/margin_net_eval.json"))
    ep.add_argument("--out-png", type=Path, default=Path("scratch/jud-v0/margin_net_reliability.png"))
    ep.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()
    if args.cmd == "train":
        train(
            corpus=args.corpus, out_model=args.out_model, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr, patience=args.patience,
            device=args.device,
        )
    else:
        evaluate(
            corpus=args.corpus, model_path=args.model,
            out_json=args.out_json, out_png=args.out_png, device=args.device,
        )
