"""Train a field net (`scenario.field`) on a pour (`scenario.atlas`).

Target: cross-entropy to softmax(sign·q/τ) over the legal moves, τ = 2 points
by default — a near-tie leaves the target flat (either move costs little), a
hinge makes it sharp (the wrong move costs a lot): advantage-weighted by
construction, and softmax ignores the group mean, so this is the paired
advantage's distribution. A float MLP in numpy (Adam), then quantized to the
integer net both the referee and the core run: power-of-two scales, int16
weights, the rounding folded into the biases. Validation holds out every
tenth deal; `regret` is the mean points the net's move gives up against the
teacher's best move by the teacher's own q, in the mover's units; a hinge is
a position whose best move beats the second by 10 points or more.

    python -u -m scenario.field_train --pour DIR --out NET.npz [--epochs E] [--tau T]
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scenario.atlas import Pour
from scenario.field import CANON, N_FEATURES, FieldNet, features
from scenario.rules import N_TILES, rules_for

NEG = np.float32(-1e9)


@dataclass(frozen=True)
class Data:
    """One row per planned position, tiles in canonical slots."""

    x: np.ndarray        # [N, F] uint8 features
    legal: np.ndarray    # [N, 28] bool
    adv: np.ndarray      # [N, 28] float64: sign·(q − mean q), the mover's advantage; 0 off the legal moves
    target: np.ndarray   # [N, 28] float32: softmax(adv/τ) over the legal moves
    hinge: np.ndarray    # [N] bool: the best move's q beats the second's by >= 10 points
    deal: np.ndarray     # [N] the position's deal


def dataset(p: Pour, tau: float) -> Data:
    n = len(p)
    x = np.zeros((n, N_FEATURES), np.uint8)
    legal = np.zeros((n, N_TILES), bool)
    adv = np.zeros((n, N_TILES))
    hinge = np.zeros(n, bool)
    for i, pos in enumerate(p.positions()):
        rules = rules_for(pos.decl_id)
        x[i, features(rules, pos.viewer, pos.own, pos.record, pos.bid, pos.bidder % 2)] = 1
        slots = [CANON[pos.decl_id][t] for t in pos.actions]
        legal[i, slots] = True
        adv[i, slots] = pos.sign * (pos.q - pos.q.mean())
        q = np.sort(pos.sign * pos.q)
        hinge[i] = q[-1] - q[-2] >= 10
    z = np.where(legal, adv / tau, -np.inf)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return Data(x, legal, adv, (e / e.sum(axis=1, keepdims=True)).astype(np.float32), hinge,
                p.deal.copy())


@dataclass
class Trained:
    net: FieldNet
    losses: list[float]          # training loss per epoch
    val_losses: list[float]
    metrics: dict


def _forward(params, x):
    w1, b1, w2, b2, w3, b3 = params
    z1 = x @ w1 + b1
    a1 = np.maximum(z1, 0)
    z2 = a1 @ w2 + b2
    a2 = np.maximum(z2, 0)
    return a1, a2, a2 @ w3 + b3


def _logp(logits, legal):
    """Log-softmax over the legal moves (an illegal move's log-probability is
    hugely negative and its target is zero)."""
    z = np.where(legal, logits, NEG)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def _loss_grad(params, x, legal, target):
    a1, a2, logits = _forward(params, x)
    logp = _logp(logits, legal)
    w2, w3 = params[2], params[4]
    d = (np.exp(logp) - target) / len(x)
    g3, gb3 = a2.T @ d, d.sum(axis=0)
    d = (d @ w3.T) * (a2 > 0)
    g2, gb2 = a1.T @ d, d.sum(axis=0)
    d = (d @ w2.T) * (a1 > 0)
    return -float((target * logp).sum()) / len(x), [x.T @ d, d.sum(axis=0), g2, gb2, g3, gb3]


def _loss(params, x, legal, target) -> float:
    return -float((target * _logp(_forward(params, x)[2], legal)).sum()) / len(x)


def train(p: Pour, epochs: int = 30, seed: int = 0, hidden: tuple[int, int] = (128, 64),
          tau: float = 2.0, lr: float = 1e-3, batch: int = 256, decay: float = 1e-4,
          log=None) -> Trained:
    data = dataset(p, tau)
    val = data.deal % 10 == 0
    tr = ~val
    x = data.x.astype(np.float32)
    rng = np.random.default_rng(seed)
    h1, h2 = hidden
    params = [
        (rng.standard_normal((N_FEATURES, h1)) * math.sqrt(2 / 40)).astype(np.float32), np.zeros(h1, np.float32),
        (rng.standard_normal((h1, h2)) * math.sqrt(2 / h1)).astype(np.float32), np.zeros(h2, np.float32),
        (rng.standard_normal((h2, N_TILES)) * math.sqrt(1 / h2)).astype(np.float32), np.zeros(N_TILES, np.float32),
    ]
    m = [np.zeros_like(q) for q in params]
    v = [np.zeros_like(q) for q in params]
    rows = np.flatnonzero(tr)
    losses, val_losses, step = [], [], 0
    for epoch in range(epochs):
        order = rng.permutation(rows)
        total = 0.0
        for at in range(0, len(order), batch):
            b = order[at:at + batch]
            loss, grads = _loss_grad(params, x[b], data.legal[b], data.target[b])
            total += loss * len(b)
            step += 1
            for i, g in enumerate(grads):
                m[i] = 0.9 * m[i] + 0.1 * g
                v[i] = 0.999 * v[i] + 0.001 * g * g
                mh, vh = m[i] / (1 - 0.9 ** step), v[i] / (1 - 0.999 ** step)
                params[i] = params[i] * (1 - lr * decay * (i % 2 == 0)) - lr * mh / (np.sqrt(vh) + 1e-8)
        losses.append(total / len(rows))
        val_losses.append(_loss(params, x[val], data.legal[val], data.target[val]) if val.any() else math.nan)
        if log is not None:
            log(f"  epoch {epoch + 1:3d}: train {losses[-1]:.4f}  val {val_losses[-1]:.4f}")
    net = quantize(params, x[tr])
    metrics = {
        "tau": tau, "hidden": list(hidden), "epochs": epochs, "seed": seed,
        "rows": {"train": int(tr.sum()), "val": int(val.sum())},
        "train": _grade(data, tr, params, net), "val": _grade(data, val, params, net),
    }
    return Trained(net, losses, val_losses, metrics)


def quantize(params, x: np.ndarray, act_bits: int = 12) -> FieldNet:
    """The integer net: weights at the largest power-of-two scale that fits
    int16, activations carried at about `act_bits` bits (set from the
    largest activation on `x`), floor shifts rounded through the biases."""
    w1, b1, w2, b2, w3, b3 = (q.astype(np.float64) for q in params)
    a1, a2, _ = _forward(params, x)

    def wexp(w):
        return math.floor(math.log2(32767 / max(np.abs(w).max(), 1e-12)))

    def aexp(a):
        return math.floor(math.log2(2 ** act_bits / max(float(a.max()), 1e-12)))

    e1, e2, e3 = wexp(w1), wexp(w2), wexp(w3)
    x1 = min(aexp(a1), e1)
    s1 = e1 - x1
    x2 = min(aexp(a2), e2 + x1)
    s2 = e2 + x1 - x2

    def r(a, e):
        return np.rint(a * 2.0 ** e).astype(np.int64)

    def half(s):
        return 1 << (s - 1) if s > 0 else 0

    return FieldNet(r(w1, e1), r(b1, e1) + half(s1), r(w2.T, e2), r(b2, e2 + x1) + half(s2),
                    r(w3.T, e3), r(b3, e3 + x2), s1, s2)


def int_scores(net: FieldNet, x: np.ndarray) -> np.ndarray:
    """`FieldNet.scores` for a batch of feature rows."""
    h = np.maximum((x.astype(np.int64) @ net.w1 + net.b1) >> net.s1, 0)
    h = np.maximum((h @ net.w2.T + net.b2) >> net.s2, 0)
    return h @ net.w3.T + net.b3


def _pick(scores: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Argmax over the legal canonical slots (the referee breaks ties by real
    tile id; float scores and these grades do not see exact ties)."""
    return np.where(legal, scores, -np.inf if scores.dtype.kind == "f" else np.iinfo(np.int64).min).argmax(axis=1)


def _grade(data: Data, rows: np.ndarray, params, net: FieldNet) -> dict:
    if not rows.any():
        return {}
    x, legal, adv = data.x[rows], data.legal[rows], data.adv[rows]
    best = np.where(legal, adv, -np.inf).max(axis=1)
    top = np.where(legal, adv, -np.inf).argmax(axis=1)
    hinge = data.hinge[rows]
    fp = _pick(_forward(params, x.astype(np.float32))[2].astype(np.float64), legal)
    ip = _pick(int_scores(net, x), legal)
    n = np.arange(len(x))

    def regret(pick, mask=slice(None)):
        return float((best - adv[n, pick])[mask].mean())

    return {
        "positions": len(x), "hinges": int(hinge.sum()),
        "agree": float((ip == top).mean()), "agree_float": float((fp == top).mean()),
        "int_float_same": float((ip == fp).mean()),
        "regret": regret(ip), "regret_float": regret(fp), "regret_hinges": regret(ip, hinge),
        "regret_random": float((best - 0.0).mean()),  # a uniform legal pick gives up best − mean
        "regret_random_hinges": float(best[hinge].mean()) if hinge.any() else math.nan,
    }


def _main() -> None:
    import argparse
    import time

    from scenario.atlas import load

    ap = argparse.ArgumentParser(prog="python -u -m scenario.field_train")
    ap.add_argument("--pour", required=True)
    ap.add_argument("--out", required=True, help="the net's .npz; metrics go beside it as .json")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tau", type=float, default=2.0, help="in points")
    ap.add_argument("--hidden", default="128,64")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    p = load(args.pour)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    print(f"field net: {len(p)} positions from {args.pour}, hidden {hidden}, tau {args.tau}", flush=True)
    t = train(p, args.epochs, args.seed, hidden, args.tau, args.lr, log=lambda s: print(s, flush=True))  # type: ignore[arg-type]
    t.net.save(args.out)
    t.metrics.update(pour=args.pour, losses=t.losses, val_losses=t.val_losses,
                     params=int(sum(a.size for a in (t.net.w1, t.net.b1, t.net.w2, t.net.b2, t.net.w3, t.net.b3))),
                     wall_seconds=round(time.time() - t0, 1))
    Path(args.out).with_suffix(".json").write_text(json.dumps(t.metrics, indent=2) + "\n")
    print(json.dumps({k: t.metrics[k] for k in ("train", "val", "params", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    _main()
