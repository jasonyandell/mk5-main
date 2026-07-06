"""jud_net (jud v1): the unified belief-conditioned value organ.

One net, two consumers. Info-state (own hand + canonical auction + play
history so far) → categorical distribution over the DECLARING team's realized
points (0..42) — the same target, loss, and readouts as `champion.margin_net`,
extended from bid-time to EVERY decision. A bid-time query is a play-time
query with an empty play history: ONE featurization, one net, no special
cases. Belief stays implicit, exactly as v0 made it for the auction: a net
conditioned on everyone's actions (bids AND plays) learns what the belief
head would say about the worlds those actions select.

Consumers
---------
* **Bidding** — `JudNet.pmake_table` has `MarginNet.pmake_table`'s exact
  signature, so `champion.value_bidder.ValueBidder` prices contracts through
  it unchanged (arena spec ``jud[:...]``). The hypothetical-completed-auction
  root is the empty-history info-state at seat = bidder.
* **Play** — `arena.jud_play.JudPlay` evaluates every legal move by querying
  this net on the info-state AFTER the move (still the mover's own POV) and
  picks argmax E[points] — sign-flipped for defenders, who minimize the
  declaring team's points. No world sampling, no oracle at runtime.

Featurization — x = [hand 63 | auction 28 | play 259] = 350 dims
----------------------------------------------------------------
* Hand: `champion.bid_net.featurize_hand` on the seat's ORIGINAL 7 dominoes —
  constant through the hand; which of them are gone is in the play block.
* Auction: `champion.margin_net.canonical_auction` reused verbatim — the
  level-blind, later-seat-masked encoding (bidder's own level := CANON_BID,
  seats after the bidder := pass) that kills selection leakage. The only
  difference from margin_net: `auction_feature_vector`'s POV is the DECISION
  seat, so relative-seat features (is_winner at seat r) tell the net where the
  declarer sits — the offense/defense channel. At seat = bidder this reduces
  byte-identically to margin_net's encoding (tested).
* Play history (the load-bearing block): a per-domino map + a small global
  summary, all POV-relative and derived from the (seat, domino) play list
  alone — so training rows (snapshot prefixes) and serving rows
  (ZebGameState) featurize byte-identically (tested).

    per domino d in 0..27, 9 dims at offset 9*d:
      [0:4] played-by relative seat one-hot  (r = (player - seat) % 4; zero if unplayed)
      [4:8] position-in-trick one-hot        (0 = led the trick)
      [8]   trick index / 6
    global tail, 7 dims:
      [252] declaring-team points so far / 42   (complete tricks, via resolve_trick)
      [253] defending-team points so far / 42
      [254] plays so far / 28
      [255:259] current-trick fill one-hot      (n_played % 4)

  Who played what, in what order (trick index × position), trick winners
  (position-0 seats), and the running score are all present; nothing else is.

Dataset
-------
Each snapshot hand expands to the exact query set the two consumers issue:
for every play step k (mover = plays[k][0]), the mover's info-state BEFORE
the move (step k — the decision) and AFTER it (step k+1 — the evaluation
JudPlay prices). Step 0's mover is the bidder, so the bid-time root is the
k = 0 decision row — no special case. Defense rows are ~half of everything,
which is where information-set value concentrates (jud v1 note). All rows of
a hand share its realized outcome: Monte Carlo targets, no bootstrapping —
at a 7-trick horizon TD machinery is pointless.

Model
-----
MLP: 350 → 512 → 512 → 43 logits (~470K params). The margin_net family
scaled up one notch for the 3.8x input; a transformer is unjustified when
the play block is already a fixed structured map rather than a sequence.
Start simple; the loss curve says when to reach for attention.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor

from arena.auction import MIN_BID, contract_points
from champion.bid_net import FEATURE_DIM as HAND_DIM  # 63
from champion.bid_net import featurize_hand
from champion.margin_net import (
    AUCTION_DIM,
    CANON_BID,
    N_POINTS,
    THRESHOLDS,
    _metrics,
    _reliability,
    _resolve_paths,
    canonical_auction,
    exceedance,
    mean_points,
    split_of,
)
from forge.bidding.schema import EVAL_DECLS
from forge.oracle.tables import resolve_trick
from forge.zeb.game import current_player
from forge.zeb.types import ZebGameState
from gus.model.auction import auction_feature_vector

# --------------------------------------------------------------------- #
#  Constants                                                              #
# --------------------------------------------------------------------- #

N_DOMINOES = 28
N_TRICKS = 7
PER_DOMINO = 9                                   # seat one-hot 4 + pos one-hot 4 + trick idx
GLOBAL_DIM = 7                                   # pts x2 + n_played + trick-fill one-hot 4
PLAY_DIM = N_DOMINOES * PER_DOMINO + GLOBAL_DIM  # 259
FEATURE_DIM = HAND_DIM + AUCTION_DIM + PLAY_DIM  # 350

_DEFAULT_MODEL = Path("champion/jud_net.pt")


# --------------------------------------------------------------------- #
#  Featurization                                                          #
# --------------------------------------------------------------------- #

def featurize_play(
    plays: Sequence[tuple[int, int]], seat: int, bidder: int, decl_id: int,
) -> Tensor:
    """[259] play-history feature from the (player, domino) list, POV ``seat``.

    Points-so-far are reconstructed from complete tricks with the engine's own
    `resolve_trick` — the emission contract (arena.test_snapshots) guarantees
    this replay matches the recorded outcome exactly.
    """
    feat = torch.zeros(PLAY_DIM, dtype=torch.float32)
    pts = [0, 0]  # (declaring team, defending team)
    bid_team = int(bidder) % 2
    for k, (player, domino) in enumerate(plays):
        base = PER_DOMINO * int(domino)
        feat[base + (int(player) - int(seat)) % 4] = 1.0
        feat[base + 4 + k % 4] = 1.0
        feat[base + 8] = (k // 4) / (N_TRICKS - 1)
        if k % 4 == 3:  # trick complete — resolve and score it
            trick = tuple(int(d) for _, d in plays[k - 3:k + 1])
            out = resolve_trick(trick[0], trick, int(decl_id))
            winner = (int(plays[k - 3][0]) + out.winner_offset) % 4
            pts[0 if winner % 2 == bid_team else 1] += out.points
    n = len(plays)
    g = N_DOMINOES * PER_DOMINO
    feat[g + 0] = pts[0] / 42.0
    feat[g + 1] = pts[1] / 42.0
    feat[g + 2] = n / float(N_DOMINOES)
    feat[g + 3 + n % 4] = 1.0
    return feat


def featurize(
    hand: Sequence[int],
    bids: Sequence[int],
    bidder: int,
    dealer: int,
    decl_id: int,
    plays: Sequence[tuple[int, int]] = (),
    seat: int | None = None,
) -> Tensor:
    """350-dim info-state: own hand (63) ⊕ canonical auction (28) ⊕ play (259).

    ``seat`` is the observing seat (whose hand, whose POV); ``None`` means the
    bidder — with the default empty ``plays`` that is exactly the bid-time
    hypothetical root, and the first 91 dims equal `margin_net.featurize`.
    """
    pov = int(bidder) if seat is None else int(seat)
    canon = canonical_auction(bids, bidder, dealer)
    return torch.cat([
        featurize_hand(tuple(hand)),
        auction_feature_vector(
            canon, bidder=int(bidder), bid_value=CANON_BID,
            decl_id=int(decl_id), current_player=pov,
        ),
        featurize_play(plays, pov, bidder, decl_id),
    ])


def featurize_snapshot(snap: Mapping, step: int = 0, seat: int | None = None) -> Tensor:
    """350-dim input from a snapshot row at play step ``step``, POV ``seat``
    (default: the bidder — step 0 then matches `margin_net.featurize_snapshot`
    on the shared 91 dims)."""
    pov = int(snap["bidder"]) if seat is None else int(seat)
    return featurize(
        tuple(int(t) for t in snap["hands"][pov]),
        [int(b) for b in snap["bids"]],
        int(snap["bidder"]),
        int(snap["dealer"]),
        int(snap["decl_id"]),
        [(int(p), int(d)) for p, d in snap["plays"][:step]],
        seat=pov,
    )


def featurize_state(state: ZebGameState, seat: int | None = None) -> Tensor:
    """350-dim input from a live engine state — the serving path.

    ``seat`` defaults to the player to act; pass it explicitly for post-move
    (or terminal) states, where the mover's POV is the one being priced. Only
    ``state.hands[seat]`` and public information enter the features.
    """
    pov = current_player(state) if seat is None else int(seat)
    return featurize(
        state.hands[pov],
        state.bid_state.bids,
        state.bidder,
        state.dealer,
        state.decl_id,
        state.play_history,
        seat=pov,
    )


# --------------------------------------------------------------------- #
#  Network                                                                #
# --------------------------------------------------------------------- #

class JudNet(nn.Module):
    """MLP: 350 → 512 → 512 → 43 logits (categorical over declaring-team points)."""

    def __init__(self, in_dim: int = FEATURE_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, N_POINTS),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, in_dim] → [B, 43] logits."""
        return self.net(x)

    def pmake_table(
        self, hand: Sequence[int], bids: Sequence[int], bidder: int, dealer: int
    ) -> dict[int, dict[int, float]]:
        """{decl: {threshold: P(pts ≥ threshold)}} at the empty-history root —
        `MarginNet.pmake_table`'s exact signature, so the ValueBidder consumes
        this net unchanged. The bid-time query IS the play-time query with no
        plays: one featurization, no special case."""
        decls = list(EVAL_DECLS)
        feats = torch.stack([
            featurize(hand, bids, bidder, dealer, d) for d in decls
        ])  # [9, 350]
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


def load_jud_net(model_path: str | Path = _DEFAULT_MODEL, device: str = "cpu") -> JudNet:
    """Load a trained JudNet checkpoint (mirrors ``value_bidder.load_margin_net``)."""
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = JudNet(in_dim=ckpt.get("feature_dim", FEATURE_DIM)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# --------------------------------------------------------------------- #
#  Dataset                                                                #
# --------------------------------------------------------------------- #

def hand_samples(snap: Mapping) -> Iterable[tuple[int, int]]:
    """(step, pov_seat) sample coordinates for one snapshot hand — the exact
    query set the consumers issue: for each play step k the mover's info-state
    before the move (a decision, incl. the k=0 bid-time root: step 0's mover
    is the bidder) and after it (the child state JudPlay prices)."""
    for k, (player, _) in enumerate(snap["plays"]):
        yield k, int(player)
        yield k + 1, int(player)


class JudDataset(torch.utils.data.Dataset):
    """Snapshot-JSON corpus → (350-dim x, realized-points y) per-decision samples.

    Same corpus files as `MarginDataset` (``arena.cli --emit-snapshots``), but
    each hand expands via `hand_samples` into ~56 info-state rows, every one
    labeled with the hand's realized ``bidder_team_pts`` (one label per hand —
    the Monte Carlo target). Rows are exact-deduped: paired halves replay
    identical seeds, so identical auctions collapse at step 0, and a
    deterministic A==B self-play collapses entirely.
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
                if "plays" not in snap:
                    raise ValueError(
                        f"{f}: snapshot rows lack 'plays' — regenerate the corpus "
                        "with a post-jud-v1 arena (--emit-snapshots)."
                    )
                if split != "all" and split_of(snap["seed"], snap["hand_idx"]) != split:
                    continue
                base = (
                    int(snap["seed"]),
                    int(snap["hand_idx"]),
                    tuple(int(b) for b in snap["bids"]),
                    int(snap["bidder"]),
                    int(snap["decl_id"]),
                )
                y = torch.tensor(int(snap["bidder_team_pts"]), dtype=torch.long)
                for step, pov in hand_samples(snap):
                    key = base + (
                        step, pov,
                        tuple((int(p), int(d)) for p, d in snap["plays"][:step]),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    self.samples.append((featurize_snapshot(snap, step, pov), y))
                    self.keys.append(key)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.samples[idx]

    def tensors(self) -> tuple[Tensor, Tensor]:
        """Stacked (X [N, 350], Y [N]) — for whole-split metric passes."""
        if not self.samples:
            return torch.empty(0, FEATURE_DIM), torch.empty(0, dtype=torch.long)
        xs = torch.stack([x for x, _ in self.samples])
        ys = torch.stack([y for _, y in self.samples])
        return xs, ys


# --------------------------------------------------------------------- #
#  Training                                                               #
# --------------------------------------------------------------------- #

def _forward_all(model: JudNet, ds: JudDataset, device: str) -> tuple[Tensor, Tensor]:
    xs, ys = ds.tensors()
    if xs.numel() == 0:
        return xs, ys
    model.eval()
    with torch.no_grad():
        logits = model(xs.to(device)).cpu()
    return logits, ys


def train(
    corpus: str | Path | Iterable[str | Path],
    out_model: Path = _DEFAULT_MODEL,
    epochs: int = 60,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 8,
    device: str = "cpu",
) -> dict:
    """Train JudNet on the snapshot corpus; save best-val weights, return metrics."""
    train_ds = JudDataset(corpus, split="train")
    val_ds = JudDataset(corpus, split="val")
    test_ds = JudDataset(corpus, split="test")
    if not len(train_ds):
        raise FileNotFoundError(
            f"No training samples from {corpus!r} — generate arena snapshots first."
        )
    print(
        f"Dataset: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test "
        f"decision rows (deduped, 90/5/5 by deal hash)",
        flush=True,
    )

    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    model = JudNet().to(device)
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
    out_json: Path = Path("scratch/jud-v1/jud_net_eval.json"),
    device: str = "cpu",
) -> dict:
    """Test-split report: overall + per-phase (by trick) reliability.

    Where `margin_net.evaluate` reads one root exceedance curve, the play-time
    head should sharpen as evidence accrues — so the report also slices MAE and
    ECE by trick index, the calibration-vs-depth curve jud v1 cares about.
    """
    test_ds = JudDataset(corpus, split="test")
    if not len(test_ds):
        raise FileNotFoundError(f"No test samples from {corpus!r}.")
    model = load_jud_net(model_path, device)

    logits, ys = _forward_all(model, test_ds, device)
    exc = exceedance(logits).cpu().numpy()          # [N, 43]
    y_np = ys.cpu().numpy()

    rel = _reliability(exc[:, MIN_BID], (y_np >= MIN_BID).astype(float))
    thr = list(THRESHOLDS)
    result = {
        "n_test": len(y_np),
        "reliability_p30": rel,
        "ece_p30": rel["ece"],
        "exceedance_thresholds": thr,
        "predicted_exceedance": [float(exc[:, t].mean()) for t in thr],
        "empirical_exceedance": [float((y_np >= t).mean()) for t in thr],
        "mae_mean_pts": float((mean_points(logits) - ys.float()).abs().mean().item()),
        "test_ce": float(nn.functional.cross_entropy(logits, ys).item()),
        "model_path": str(model_path),
    }

    # Per-trick slices: sample k's trick index is n_played // 4, read back from
    # the global n_played feature (dim g+2) — no need to re-walk the corpus.
    xs, _ = test_ds.tensors()
    n_played = (xs[:, HAND_DIM + AUCTION_DIM + N_DOMINOES * PER_DOMINO + 2]
                * N_DOMINOES).round().long().numpy()
    by_trick = {}
    for trick in range(N_TRICKS + 1):  # 7 = terminal rows (all 28 played)
        m = (n_played // 4 == trick) if trick < N_TRICKS else (n_played == 28)
        if not m.any():
            continue
        sl = _metrics(logits[m], ys[m])
        by_trick[str(trick)] = {
            "n": int(m.sum()), "ce": sl["ce"],
            "mae_mean_pts": sl["mae_mean_pts"], "ece_p30": sl["ece_p30"],
        }
    result["by_trick"] = by_trick

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out_json}", flush=True)
    print(
        "  trick   n      ce    mae   ece30\n" + "\n".join(
            f"  {t:>5} {v['n']:>5}  {v['ce']:.3f}  {v['mae_mean_pts']:5.2f}  "
            f"{v['ece_p30']:.3f}" for t, v in by_trick.items()
        ),
        flush=True,
    )
    return result


# --------------------------------------------------------------------- #
#  CLI                                                                    #
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="JudNet — unified realized-value organ (jud v1)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    tp = sub.add_parser("train", help="train on the snapshot corpus")
    tp.add_argument("--corpus", required=True,
                    help="glob of snapshot JSON files, e.g. 'scratch/jud-v1/corpus/*.json'")
    tp.add_argument("--out-model", type=Path, default=_DEFAULT_MODEL)
    tp.add_argument("--epochs", type=int, default=60)
    tp.add_argument("--batch-size", type=int, default=512)
    tp.add_argument("--lr", type=float, default=1e-3)
    tp.add_argument("--patience", type=int, default=8)
    tp.add_argument("--device", type=str, default="cpu")

    ep = sub.add_parser("eval", help="test-split reliability report (overall + by trick)")
    ep.add_argument("--corpus", required=True, help="glob of snapshot JSON files")
    ep.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    ep.add_argument("--out-json", type=Path, default=Path("scratch/jud-v1/jud_net_eval.json"))
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
            out_json=args.out_json, device=args.device,
        )
