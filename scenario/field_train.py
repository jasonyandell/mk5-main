"""Train a field net (`scenario.field`) on a pour (`scenario.atlas`).

Two targets, both read off the margin splat (a pour planned under `make`
still keeps margins, #107):

- `margin`: cross-entropy to softmax(sign·m/τ) over the legal moves, m each
  move's belief-weighted mean margin and τ = 2 points by default — a
  near-tie leaves the target flat, a hinge makes it sharp; softmax ignores
  the group mean, so this is the paired advantage's distribution.
- `pmake`: texas-42 PR #102's faithful target. Per legal tile, the belief
  mass in which the mover's side gets its way at the pour's bid — the
  bidding side makes (margin ≥ 2·bid − 42), a defender sets — fitted by
  per-tile BCE on sigmoid outputs, illegal tiles masked out of the loss.
  The defender's sign is folded into the target (P(set) = 1 − P(make)), so
  the net's score is the mover's own success logit and the integer
  selector (argmax over the legal tiles) is unchanged.

A float MLP trained with Adam — in numpy (the reference, the default) or on
the GPU with MLX (`--device mlx`: the same init, batches, loss and update,
float32 on the GPU) — then quantized to the integer net both the referee and
the core run: power-of-two scales, int16 weights, the rounding folded into
the biases. Validation holds out every tenth deal. Grades, in the mover's
units: `regret` is the mean points the net's move gives up against the best
move by mean margin, `regret_pmake` the make probability it gives up against
the best move by the mover's pmake, `agree` how often it plays the
teacher's own move; a hinge is a position whose best move beats the second
by 10 points or more.

    python -u -m scenario.field_train --pour DIR --out NET.npz [--target margin|pmake]
        [--device numpy|mlx] [--hidden 512,256] [--epochs E] [--tau T]
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scenario.atlas import Pour, makes
from scenario.field import CANON, N_FEATURES, FieldNet, features
from scenario.rules import N_TILES, rules_for

NEG = np.float32(-1e9)
TARGETS = ("margin", "pmake")
CHUNK = 1 << 16      # rows per forward pass when grading


@dataclass(frozen=True)
class Data:
    """One row per planned position, tiles in canonical slots; every
    per-tile column is 0 off the legal moves."""

    x: np.ndarray        # [N, F] uint8 features
    legal: np.ndarray    # [N, 28] bool
    adv: np.ndarray      # [N, 28] float32: sign·(m − mean m), the mover's advantage in points
    win: np.ndarray      # [N, 28] float32: the mover's side makes (bidder) or sets (defender), belief mass
    chosen: np.ndarray   # [N] int64: the teacher's move's slot
    hinge: np.ndarray    # [N] bool: the best move's margin beats the second's by >= 10 points
    deal: np.ndarray     # [N] the position's deal


def dataset(p: Pour) -> Data:
    n = len(p)
    x = np.zeros((n, N_FEATURES), np.uint8)
    legal = np.zeros((n, N_TILES), bool)
    adv = np.zeros((n, N_TILES), np.float32)
    win = np.zeros((n, N_TILES), np.float32)
    chosen = np.zeros(n, np.int64)
    hinge = np.zeros(n, bool)
    for i, pos in enumerate(p.positions()):
        rules = rules_for(pos.decl_id)
        x[i, features(rules, pos.viewer, pos.own, pos.record, pos.bid, pos.bidder % 2)] = 1
        canon = CANON[pos.decl_id]
        slots = [canon[t] for t in pos.actions]
        legal[i, slots] = True
        w = pos.weights / pos.weights.sum()
        m = pos.outcomes.astype(np.float64) @ w   # mean margin per move
        adv[i, slots] = pos.sign * (m - m.mean())
        pm = np.array([w[makes(pos, t)].sum() for t in pos.actions])
        win[i, slots] = pm if pos.sign > 0 else 1 - pm
        chosen[i] = canon[pos.chosen]
        m = np.sort(pos.sign * m)
        hinge[i] = m[-1] - m[-2] >= 10
    return Data(x, legal, adv, win, chosen, hinge, p.deal.copy())


def targets(data: Data, target: str, tau: float) -> np.ndarray:
    """[N, 28] float32: softmax(adv/τ) over the legal moves (`margin`), or
    the mover's per-tile success probability (`pmake`)."""
    if target == "pmake":
        return data.win
    if target != "margin":
        raise ValueError(f"target is one of {TARGETS}, not {target!r}")
    z = np.where(data.legal, data.adv / np.float32(tau), -np.inf)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)


@dataclass
class Trained:
    net: FieldNet
    params: list[np.ndarray]     # the float net, before quantization
    losses: list[float]          # training loss per epoch
    val_losses: list[float]
    metrics: dict


# -- the float net in numpy: the reference -------------------------------------------


def _forward(params, x):
    w1, b1, w2, b2, w3, b3 = params
    a1 = np.maximum(x @ w1 + b1, 0)
    a2 = np.maximum(a1 @ w2 + b2, 0)
    return a1, a2, a2 @ w3 + b3


def _logits(params, x: np.ndarray) -> np.ndarray:
    """The float net's scores for uint8 feature rows, in chunks."""
    return np.concatenate([_forward(params, x[at:at + CHUNK].astype(np.float32))[2]
                           for at in range(0, len(x), CHUNK)] or [np.zeros((0, N_TILES), np.float32)])


def _loss_dlogits(logits, legal, target, kind: str):
    """The loss (mean over rows) and its gradient in the logits."""
    if kind == "margin":     # cross-entropy over the legal moves; an illegal move's target is zero
        z = np.where(legal, logits, NEG)
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return -float((target * logp).sum()) / len(logits), (np.exp(logp) - target) / len(logits)
    # per-tile BCE on sigmoid(logits), the legal tiles only
    sp = np.maximum(logits, 0) + np.log1p(np.exp(-np.abs(logits)))
    loss = float(np.where(legal, sp - target * logits, 0).sum()) / len(logits)
    sigmoid = 0.5 * (1 + np.tanh(0.5 * logits))
    return loss, np.where(legal, sigmoid - target, 0).astype(np.float32) / len(logits)


def _loss_grad(params, x, legal, target, kind: str):
    a1, a2, logits = _forward(params, x)
    loss, d = _loss_dlogits(logits, legal, target, kind)
    w2, w3 = params[2], params[4]
    g3, gb3 = a2.T @ d, d.sum(axis=0)
    d = (d @ w3.T) * (a2 > 0)
    g2, gb2 = a1.T @ d, d.sum(axis=0)
    d = (d @ w2.T) * (a1 > 0)
    return loss, [x.T @ d, d.sum(axis=0), g2, gb2, g3, gb3]


def _adam(params, grads, m, v, c1, c2, lr: float, decay: float, sqrt):
    """One Adam step with decoupled weight decay on the weights (not the
    biases); c1, c2 are the bias corrections 1 − β^t. Backend-agnostic."""
    out = ([], [], [])
    for i, (p, g, mi, vi) in enumerate(zip(params, grads, m, v)):
        mi = 0.9 * mi + 0.1 * g
        vi = 0.999 * vi + 0.001 * g * g
        p = p * (1 - lr * decay * (i % 2 == 0)) - lr * (mi / c1) / (sqrt(vi / c2) + 1e-8)
        for o, a in zip(out, (p, mi, vi)):
            o.append(a)
    return out


class _Numpy:
    def __init__(self, data: Data, target: np.ndarray, kind: str, lr: float, decay: float):
        self.data, self.target, self.kind, self.lr, self.decay = data, target, kind, lr, decay

    def put(self, params):
        return params

    def get(self, params):
        return params

    def step(self, params, m, v, t: int, rows: np.ndarray):
        loss, grads = _loss_grad(params, self.data.x[rows].astype(np.float32), self.data.legal[rows],
                                 self.target[rows], self.kind)
        return (loss, *_adam(params, grads, m, v, 1 - 0.9 ** t, 1 - 0.999 ** t, self.lr, self.decay, np.sqrt))

    def loss(self, params, rows: np.ndarray) -> float:
        logits = _logits(params, self.data.x[rows])
        return _loss_dlogits(logits, self.data.legal[rows], self.target[rows], self.kind)[0]


# -- the same float net on the GPU ----------------------------------------------------


class _MLX:
    """MLX: the data resident on the GPU, the loss differentiated by MLX, the
    step (forward, backward, `_adam`) compiled into one graph."""

    def __init__(self, data: Data, target: np.ndarray, kind: str, lr: float, decay: float):
        try:
            import mlx.core as mx
        except ImportError as e:
            raise SystemExit("--device mlx needs MLX (pip install mlx, Apple silicon); "
                             "--device numpy is the reference") from e
        self.mx = mx
        self.x, self.legal, self.target = mx.array(data.x), mx.array(data.legal), mx.array(target)

        def loss_fn(params, x, legal, target):
            w1, b1, w2, b2, w3, b3 = params
            a1 = mx.maximum(x.astype(mx.float32) @ w1 + b1, 0)
            logits = mx.maximum(a1 @ w2 + b2, 0) @ w3 + b3
            if kind == "margin":
                z = mx.where(legal, logits, float(NEG))
                logp = z - mx.logsumexp(z, axis=1, keepdims=True)
                return -(target * logp).sum() / x.shape[0]
            return mx.where(legal, mx.logaddexp(logits, 0) - target * logits, 0).sum() / x.shape[0]

        grad = mx.value_and_grad(loss_fn)

        def step(params, m, v, c1, c2, idx):
            loss, grads = grad(params, self.x[idx], self.legal[idx], self.target[idx])
            return (loss, *_adam(params, grads, m, v, c1, c2, lr, decay, mx.sqrt))

        self._step, self._loss = mx.compile(step), mx.compile(loss_fn)

    def put(self, params):
        return [self.mx.array(p) for p in params]

    def get(self, params):
        return [np.array(p) for p in params]

    def step(self, params, m, v, t: int, rows: np.ndarray):
        mx = self.mx
        c1, c2 = mx.array(1 - 0.9 ** t, mx.float32), mx.array(1 - 0.999 ** t, mx.float32)
        loss, params, m, v = self._step(params, m, v, c1, c2, mx.array(rows))
        mx.eval(loss, params, m, v)
        return loss.item(), params, m, v

    def loss(self, params, rows: np.ndarray) -> float:
        total = 0.0
        for at in range(0, len(rows), CHUNK):
            idx = self.mx.array(rows[at:at + CHUNK])
            total += self._loss(params, self.x[idx], self.legal[idx], self.target[idx]).item() * len(idx)
        return total / len(rows)


DEVICES = {"numpy": _Numpy, "mlx": _MLX}


def train(data: Data, target: str = "margin", epochs: int = 30, seed: int = 0,
          hidden: tuple[int, int] = (128, 64), tau: float = 2.0, lr: float = 1e-3, batch: int = 256,
          decay: float = 1e-4, device: str = "numpy", log=None) -> Trained:
    if device not in DEVICES:
        raise ValueError(f"device is one of {tuple(DEVICES)}, not {device!r}")
    val = data.deal % 10 == 0
    tr = ~val
    rng = np.random.default_rng(seed)
    h1, h2 = hidden
    init = [
        (rng.standard_normal((N_FEATURES, h1)) * math.sqrt(2 / 40)).astype(np.float32), np.zeros(h1, np.float32),
        (rng.standard_normal((h1, h2)) * math.sqrt(2 / h1)).astype(np.float32), np.zeros(h2, np.float32),
        (rng.standard_normal((h2, N_TILES)) * math.sqrt(1 / h2)).astype(np.float32), np.zeros(N_TILES, np.float32),
    ]
    dev = DEVICES[device](data, targets(data, target, tau), target, lr, decay)
    params = dev.put(init)
    m = dev.put([np.zeros_like(q) for q in init])
    v = dev.put([np.zeros_like(q) for q in init])
    rows, val_rows = np.flatnonzero(tr), np.flatnonzero(val)
    losses, val_losses, t = [], [], 0
    for epoch in range(epochs):
        order = rng.permutation(rows)
        total = 0.0
        for at in range(0, len(order), batch):
            b = order[at:at + batch]
            t += 1
            loss, params, m, v = dev.step(params, m, v, t, b)
            total += loss * len(b)
        losses.append(total / len(rows))
        val_losses.append(dev.loss(params, val_rows) if len(val_rows) else math.nan)
        if log is not None:
            log(f"  epoch {epoch + 1:3d}: train {losses[-1]:.4f}  val {val_losses[-1]:.4f}")
    params = dev.get(params)
    net = quantize(params, data.x[tr])
    metrics = {
        "target": target, "tau": tau, "hidden": list(hidden), "epochs": epochs, "seed": seed,
        "device": device, "batch": batch, "lr": lr,
        "rows": {"train": int(tr.sum()), "val": int(val.sum())},
        "train": _grade(data, tr, params, net), "val": _grade(data, val, params, net),
    }
    return Trained(net, params, losses, val_losses, metrics)


def quantize(params, x: np.ndarray, act_bits: int = 12) -> FieldNet:
    """The integer net: weights at the largest power-of-two scale that fits
    int16, activations carried at about `act_bits` bits (set from the
    largest activation on the uint8 feature rows `x`), floor shifts rounded
    through the biases."""
    w1, b1, w2, b2, w3, b3 = (q.astype(np.float64) for q in params)
    a1max = a2max = 0.0
    for at in range(0, len(x), CHUNK):
        a1, a2, _ = _forward(params, x[at:at + CHUNK].astype(np.float32))
        a1max, a2max = max(a1max, float(a1.max(initial=0))), max(a2max, float(a2.max(initial=0)))

    def wexp(w):
        return math.floor(math.log2(32767 / max(np.abs(w).max(), 1e-12)))

    def aexp(a):
        return math.floor(math.log2(2 ** act_bits / max(a, 1e-12)))

    e1, e2, e3 = wexp(w1), wexp(w2), wexp(w3)
    x1 = min(aexp(a1max), e1)
    s1 = e1 - x1
    x2 = min(aexp(a2max), e2 + x1)
    s2 = e2 + x1 - x2

    def r(a, e):
        return np.rint(a * 2.0 ** e).astype(np.int64)

    def half(s):
        return 1 << (s - 1) if s > 0 else 0

    return FieldNet(r(w1, e1), r(b1, e1) + half(s1), r(w2.T, e2), r(b2, e2 + x1) + half(s2),
                    r(w3.T, e3), r(b3, e3 + x2), s1, s2)


def int_scores(net: FieldNet, x: np.ndarray) -> np.ndarray:
    """`FieldNet.scores` for a batch of feature rows. The products run in
    float64 BLAS: every partial sum is an integer far below 2^53, so they
    are exact."""
    def mm(a, w):
        return np.rint(a.astype(np.float64) @ w.astype(np.float64)).astype(np.int64)

    h = np.maximum((mm(x, net.w1) + net.b1) >> net.s1, 0)
    h = np.maximum((mm(h, net.w2.T) + net.b2) >> net.s2, 0)
    return mm(h, net.w3.T) + net.b3


def _pick(scores: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """Argmax over the legal canonical slots (the referee breaks ties by real
    tile id; float scores and these grades do not see exact ties)."""
    return np.where(legal, scores, -np.inf if scores.dtype.kind == "f" else np.iinfo(np.int64).min).argmax(axis=1)


def _grade(data: Data, rows: np.ndarray, params, net: FieldNet) -> dict:
    if not rows.any():
        return {}
    x, legal, adv, win = data.x[rows], data.legal[rows], data.adv[rows], data.win[rows]
    hinge = data.hinge[rows]
    best = np.where(legal, adv, -np.inf).max(axis=1)
    best_win = np.where(legal, win, -np.inf).max(axis=1)
    fp = _pick(_logits(params, x), legal)
    ip = _pick(np.concatenate([int_scores(net, x[at:at + CHUNK]) for at in range(0, len(x), CHUNK)]), legal)
    n = np.arange(len(x))

    def regret(pick, mask=slice(None)):
        return float((best - adv[n, pick])[mask].mean())

    return {
        "positions": len(x), "hinges": int(hinge.sum()),
        "agree": float((ip == data.chosen[rows]).mean()), "int_float_same": float((ip == fp).mean()),
        "regret": regret(ip), "regret_float": regret(fp), "regret_hinges": regret(ip, hinge),
        "regret_pmake": float((best_win - win[n, ip]).mean()),
        # a uniform legal pick gives up best − mean (the advantage's mean is 0)
        "regret_random": float(best.mean()),
        "regret_random_hinges": float(best[hinge].mean()) if hinge.any() else math.nan,
        "regret_pmake_random": float((best_win - win.sum(axis=1) / legal.sum(axis=1)).mean()),
    }


def _main() -> None:
    import argparse
    import time

    from scenario.atlas import load

    ap = argparse.ArgumentParser(prog="python -u -m scenario.field_train")
    ap.add_argument("--pour", required=True)
    ap.add_argument("--out", required=True, help="the net's .npz; metrics go beside it as .json")
    ap.add_argument("--target", choices=TARGETS, default="margin")
    ap.add_argument("--device", choices=tuple(DEVICES), default="numpy")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tau", type=float, default=2.0, help="in points (the margin target)")
    ap.add_argument("--hidden", default="128,64")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    p = load(args.pour)
    hidden = tuple(int(h) for h in args.hidden.split(","))
    print(f"field net: {len(p)} positions from {args.pour}, target {args.target}, hidden {hidden}, "
          f"{args.device}", flush=True)
    data = dataset(p)
    del p
    t1 = time.time()
    print(f"  dataset in {t1 - t0:.0f}s", flush=True)
    t = train(data, args.target, args.epochs, args.seed, hidden, args.tau, args.lr, args.batch,  # type: ignore[arg-type]
              device=args.device, log=lambda s: print(s, flush=True))
    t.net.save(args.out)
    t.metrics.update(pour=args.pour, losses=t.losses, val_losses=t.val_losses,
                     params=int(sum(a.size for a in (t.net.w1, t.net.b1, t.net.w2, t.net.b2, t.net.w3, t.net.b3))),
                     dataset_seconds=round(t1 - t0, 1), train_seconds=round(time.time() - t1, 1),
                     wall_seconds=round(time.time() - t0, 1))
    Path(args.out).with_suffix(".json").write_text(json.dumps(t.metrics, indent=2) + "\n")
    print(json.dumps({k: t.metrics[k] for k in ("train", "val", "params", "train_seconds", "wall_seconds")},
                     indent=2))


if __name__ == "__main__":
    _main()
