"""Does the fly's wiring help read a 42 table? A reservoir pilot (2026-10-14).

The adult Drosophila connectome (FlyWire v783, as packaged by Shiu et al.
2024) is used as a FIXED recurrent network. A 42 position is written onto
its sensory neurons, the network runs a few steps, and only a linear readout
is trained. The question is whether the real wiring beats the same-size
networks with the wiring scrambled — the structure has to earn its keep.

The task: the viewer (seat 0, sixes trump) does not hold the 5-5 and it has
not been played; which of the other three seats holds it? Positions come
from random legal play from a random deal. The answer is scored against the
EXACT posterior under the uniform-given-record belief — Zeb's default belief
(g ≡ 0): every hidden completion that respects the hand sizes and the voids
the record proves is equally likely, counted by `scenario.fiber`. The
targets are that posterior (soft), never the deal's true holder and never a
value: no clairvoyant number enters (the strategy-fusion fence is not
touched). Metrics, on held-out positions:

- `acc`: expected accuracy under the posterior, E[p(argmax q)]; the Bayes
  ceiling is E[max p], chance is 1/3;
- `kl`: mean KL(p ‖ q) in millinats; the ceiling is 0.

Representations, all read out by the same L2 logistic regression on soft
targets (L2 chosen on a validation split, per representation):

- `raw`        the 164 input bits themselves (linear);
- `mlp`        a 256-unit ReLU MLP on the bits (the field net's kind of model);
- `flyhash`    a random sparse expansion: each of 5,177 units sums 7 random
               bits (a Kenyon cell has ~7 claws), ReLU after a per-unit threshold;
- `<arm>:<g>`  the connectome reservoir, graph g ∈ {real, rewired, er}:
               real    the non-optic connectome (41,692 neurons, 5.5M edges);
               rewired every edge keeps its presynaptic neuron and weight, the
                       postsynaptic ends are permuted (in- and out-degree and
                       each neuron's sign kept: a configuration-model null);
               er      endpoints uniform at random, weights shuffled.
  Arms: `mb` writes the bits on the 2,279 olfactory receptor neurons and
  reads the 5,177 Kenyon cells (the mushroom body, the fly's associative
  memory); `wb` writes on all 5,497 non-visual sensory neurons and reads
  the 1,299 descending neurons (the brain's output bus to the body).

Dynamics (rate model, not the paper's spiking LIF): x₀ = 0,
x_{t+1} = tanh(G·Ŵx_t + u), Ŵ = W with each row scaled to unit L1 norm,
u = 1 on the sensory neurons of every set bit, G = 2, T = 4 steps.

    python -u -m scenario.flybrain_pilot --data scratch/fly/data [--n 6000] [--seed 1]

The data are not in the repo: `Connectivity_783.parquet` and
`Completeness_783.csv` from github.com/philshiu/Drosophila_brain_model (MIT
code; FlyWire data CC BY 4.0), `neuron_annotations.tsv` =
`Supplemental_file1_neuron_annotations.tsv` from
github.com/flyconnectome/flywire_annotations. See wiki/topics/fly-connectome-42.md.
"""
from __future__ import annotations

import argparse
import random
import time

import numpy as np

from forge.oracle.tables import DOMINOES
from scenario.fiber import fiber_size
from scenario.record import empty_record
from scenario.rules import ALL_TILES, bit, mask_of, rules_for, tiles_of

DECL = 6                                # sixes trump
TARGET = DOMINOES.index((5, 5))         # the 5-5
N_BITS = 28 + 28 + 3 * 28 + 3 * 8       # own hand, gone, voids by seat, hand sizes
G, T = 2.0, 4
L2S = (1e-4, 1e-3, 1e-2, 1e-1)


# ---------------------------------------------------------------- positions

def position(rng: random.Random, rules):
    """A random-play position where seat 0 lacks an unplayed 5-5, or None."""
    tiles = list(range(28))
    rng.shuffle(tiles)
    hands = [mask_of(tiles[7 * s:7 * s + 7]) for s in range(4)]
    if hands[0] >> TARGET & 1:
        return None
    rec = empty_record(rng.randrange(4))
    for _ in range(rng.randrange(4, 24)):
        s = rec.seat
        legal = tiles_of(rules.legal(hands[s], rec.led))
        t = rng.choice(legal)
        hands[s] &= ~bit(t)
        rec = rec.advance(t, rules)
    if rec.played >> TARGET & 1:
        return None
    own = hands[0]
    pool = ALL_TILES & ~own & ~rec.played
    sizes = tuple(rec.hand_size(s) for s in (1, 2, 3))
    forbid = (rec.forbid[1], rec.forbid[2], rec.forbid[3])
    total = fiber_size(pool, sizes, forbid)
    p = np.zeros(3)
    for i in range(3):
        if sizes[i] and not forbid[i] >> TARGET & 1:
            sz = list(sizes)
            sz[i] -= 1
            p[i] = fiber_size(pool & ~bit(TARGET), tuple(sz), forbid)
    p /= total
    x = np.zeros(N_BITS, np.float32)
    x[tiles_of(own)] = 1
    x[[28 + t for t in tiles_of(rec.played)]] = 1
    for i in range(3):
        x[[56 + 28 * i + t for t in tiles_of(forbid[i])]] = 1
        x[140 + 8 * i + sizes[i]] = 1
    return x, p, np.array(sizes, np.float64)


def positions(n: int, seed: int):
    rng, rules = random.Random(seed), rules_for(DECL)
    xs, ps, ss = [], [], []
    while len(xs) < n:
        r = position(rng, rules)
        if r is not None:
            xs.append(r[0]); ps.append(r[1]); ss.append(r[2])
    return np.stack(xs), np.stack(ps), np.stack(ss)


# ---------------------------------------------------------------- the fly

def connectome(data: str):
    import pandas as pd
    d = pd.read_parquet(f"{data}/Connectivity_783.parquet",
                        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"])
    ids = pd.read_csv(f"{data}/Completeness_783.csv").iloc[:, 0].values
    a = pd.read_csv(f"{data}/neuron_annotations.tsv", sep="\t", low_memory=False,
                    usecols=["root_id", "super_class", "cell_class"])
    ann = a.drop_duplicates("root_id").set_index("root_id").reindex(ids)
    optic = (ann.super_class.isin(["optic", "visual_projection", "visual_centrifugal"])
             | (ann.cell_class == "visual")).values
    keep = ~optic
    new = -np.ones(len(ids), np.int64)
    new[keep] = np.arange(keep.sum())
    pre, post = d.Presynaptic_Index.values, d.Postsynaptic_Index.values
    e = keep[pre] & keep[post]
    pre, post, w = new[pre[e]], new[post[e]], d["Excitatory x Connectivity"].values[e].astype(np.float32)
    k = ann[keep]
    sets = {
        "orn": np.flatnonzero((k.cell_class == "olfactory").values),
        "kc": np.flatnonzero((k.cell_class == "Kenyon_Cell").values),
        "sens": np.flatnonzero(((k.super_class == "sensory") & (k.cell_class != "visual")).values),
        "dn": np.flatnonzero((k.super_class == "descending").values),
    }
    return int(keep.sum()), pre, post, w, sets


def variant(kind: str, n: int, pre, post, w, rng: np.random.Generator):
    if kind == "real":
        return pre, post, w
    if kind == "rewired":
        return pre, rng.permutation(post), w
    if kind == "er":
        return rng.integers(0, n, len(pre)), rng.integers(0, n, len(pre)), rng.permutation(w)
    raise ValueError(kind)


def reservoir(n, pre, post, w, inputs, readout, X, rng, batch=1024):
    import torch
    torch.set_num_threads(max(1, torch.get_num_threads()))
    row_l1 = np.bincount(post, weights=np.abs(w), minlength=n)
    wn = (w / np.maximum(row_l1[post], 1e-9)).astype(np.float32)
    W = torch.sparse_coo_tensor(torch.tensor(np.vstack([post, pre])), torch.tensor(wn), (n, n)
                                ).coalesce().to_sparse_csr()
    # each input bit drives a fixed, disjoint random share of the input neurons
    owner = rng.permutation(len(inputs)) % X.shape[1]
    U = np.zeros((n, X.shape[1]), np.float32)
    U[inputs, owner] = 1.0
    U = torch.tensor(U)
    out = []
    for b in range(0, len(X), batch):
        u = U @ torch.tensor(X[b:b + batch]).T
        x = torch.zeros_like(u)
        for _ in range(T):
            x = torch.tanh(G * (W @ x) + u)
        out.append(x[readout].T.numpy())
    return np.concatenate(out)


def flyhash(X, width, rng, claws=7):
    R = np.zeros((X.shape[1], width), np.float32)
    for j in range(width):
        R[rng.choice(X.shape[1], claws, replace=False), j] = 1
    H = X @ R
    return np.maximum(H - np.median(H, axis=0), 0)


# ---------------------------------------------------------------- readouts

def logistic(F, P, l2, iters=200):
    import torch
    Ft, Pt = torch.tensor(F, dtype=torch.float32), torch.tensor(P, dtype=torch.float32)
    W = torch.zeros(F.shape[1], 3, requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = -(Pt * torch.log_softmax(Ft @ W + b, 1)).sum(1).mean() + l2 * (W * W).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lambda G: torch.softmax(torch.tensor(G, dtype=torch.float32) @ W + b, 1).detach().numpy()


def mlp(F, P, l2, hidden=256, epochs=300, seed=0):
    import torch
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(F.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 3))
    opt = torch.optim.Adam(net.parameters(), 3e-3, weight_decay=l2)
    Ft, Pt = torch.tensor(F), torch.tensor(P, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = -(Pt * torch.log_softmax(net(Ft), 1)).sum(1).mean()
        loss.backward()
        opt.step()
    return lambda G: torch.softmax(net(torch.tensor(G)), 1).detach().numpy()


def scores(p, q):
    """Per-position expected accuracy and KL(p‖q) in millinats."""
    acc = p[np.arange(len(p)), q.argmax(1)]
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = np.where(p > 0, p * (np.log(p) - np.log(np.clip(q, 1e-12, 1))), 0).sum(1) * 1000
    return acc, kl


def fit(F, P, tr, va, te, kind="logistic"):
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-6
    Z = (F - mu) / sd
    trainer = logistic if kind == "logistic" else mlp
    best = min(L2S, key=lambda l2: scores(P[va], trainer(Z[tr], P[tr], l2)(Z[va]))[1].mean())
    fitted = trainer(Z[np.r_[tr, va]], P[np.r_[tr, va]], best)
    return scores(P[te], fitted(Z[te])), best


def boot(diff, rng, n=2000):
    m = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return diff.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    t0 = time.time()
    X, P, S = positions(args.n, args.seed)
    print(f"positions {len(X)} in {time.time() - t0:.1f}s", flush=True)
    idx = np.random.default_rng(args.seed).permutation(len(X))
    n_te, n_va = len(X) // 4, len(X) // 8
    te, va, tr = idx[:n_te], idx[n_te:n_te + n_va], idx[n_te + n_va:]

    res = {}
    res["uniform"] = scores(P[te], np.full((len(te), 3), 1 / 3))
    sz = np.maximum(S[te], 0)
    res["sizes"] = scores(P[te], sz / sz.sum(1, keepdims=True))
    res["bayes"] = scores(P[te], P[te])
    res["raw"], _ = fit(X, P, tr, va, te)
    res["mlp"], _ = fit(X, P, tr, va, te, kind="mlp")
    res["flyhash"], _ = fit(flyhash(X, 5177, np.random.default_rng(args.seed + 7)), P, tr, va, te)
    print(f"baselines done {time.time() - t0:.1f}s", flush=True)

    n, pre, post, w, sets = connectome(args.data)
    print(f"connectome: {n} neurons, {len(pre)} edges; " +
          ", ".join(f"{k} {len(v)}" for k, v in sets.items()), flush=True)
    for g in ("real", "rewired", "er"):
        gp, gq, gw = variant(g, n, pre, post, w, np.random.default_rng(args.seed + 11))
        for arm, (inp, out) in {"mb": ("orn", "kc"), "wb": ("sens", "dn")}.items():
            F = reservoir(n, gp, gq, gw, sets[inp], sets[out], X, np.random.default_rng(args.seed + 13))
            live = (F.std(0) > 1e-6).mean()
            res[f"{arm}:{g}"], l2 = fit(F, P, tr, va, te)
            print(f"  {arm}:{g} live readout {live:.2f} l2 {l2} {time.time() - t0:.1f}s", flush=True)

    print(f"\n{'representation':16s} {'acc':>7s} {'KL mnat':>8s}")
    for k, (acc, kl) in res.items():
        print(f"{k:16s} {acc.mean():7.4f} {kl.mean():8.1f}")
    rng = np.random.default_rng(0)
    print("\npaired differences on the test positions (mean [95% bootstrap]); KL: negative = first is better")
    for a, b in [("mb:real", "mb:rewired"), ("mb:real", "mb:er"), ("mb:real", "flyhash"),
                 ("wb:real", "wb:rewired"), ("wb:real", "wb:er"), ("mb:real", "raw"), ("mlp", "mb:real")]:
        m, lo, hi = boot(res[a][1] - res[b][1], rng)
        ma, loa, hia = boot(res[a][0] - res[b][0], rng)
        print(f"{a:>8s} - {b:<10s} KL {m:+7.1f} [{lo:+7.1f}, {hi:+7.1f}]   acc {ma:+.4f} [{loa:+.4f}, {hia:+.4f}]")
    print(f"\nwall {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
