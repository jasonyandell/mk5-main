# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""A unit paper's measurements for one layer-1 unit (#235; wiki/experiments/lem-unit-*).

The essay's loop, one unit at a time: label both poles beside the untrained twin, the circuit in
and out, formation across checkpoints, and steering read by where play goes. *Where play goes*
is two rates over the rows where the mover follows to an open trick and the choice matters:
**take** (the pick beats the trick's current winner) and **smear** (the pick puts count on the
trick), each steered minus base, with deal-clustered intervals and every other live layer-1 unit steered on
the same rows as the control.

Unit 5 of the 16×8 seed-0 net is close to one input bit, so the paper also flips that bit in the
input (partner made the winner) and compares: a relay moves play the way its input does.

    python -u -m lem.probe unit SAMPLE.npz LABELS.npz NET UNIT OUT.json --twin TWIN --audit AUDIT_DIR \
        --formation RUN_DIR
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lem.probe.audit import BASE_CAP, BASELINE_MARGIN, expected_from_strata, feature_names, stratum_rates
from lem.probe.causal import downstream
from lem.probe.net import forward, load_net, picks
from lem.probe.squeeze import FAMILIES
from scenario.rules import N_LED_SUITS, N_TILES, rules_for

SEED = 20261020
DOSES = (0.25, 0.5, 1.0, 2.0)
BOOT = 1000
TOP = 0.01
SEATS = ("lead", "second", "third", "last")
WINNERS = ("LHO", "partner", "RHO")     # the `winning` input block: left, partner, right
TRUMP_SUIT = 7                          # the called suit: a tile leading it is a trump

# Prediction 6's local neighbourhood: the trick as it stands (seat, who is winning, what is on it).
TRICK_STATE = (set(FAMILIES["position"]) | set(FAMILIES["trick"]) | {"can-take-it"}
               | {"last-partner-winning", "last-setter-winning", "third-partner-winning"})
TRICK_STATE_RAW = ("x:open:", "x:winning:", "x:trick:")


def is_trick_state(label: str) -> bool:
    return label in TRICK_STATE or label.startswith(TRICK_STATE_RAW)


def tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """[10, 8, 28] trick rank, [10, 28] count, [10, 28] led suit, per declaration."""
    rank = np.zeros((10, N_LED_SUITS, N_TILES), np.int64)
    count = np.zeros((10, N_TILES), np.int64)
    led = np.zeros((10, N_TILES), np.int64)
    for d in range(10):
        r = rules_for(d)
        rank[d] = np.array(r.rank)
        count[d] = np.array(r.count)
        led[d] = np.array(r.led_suit)
    return rank, count, led


class Trick:
    """The open trick per row, and what each real tile would do to it."""

    def __init__(self, sample):
        rank, count, led = tables()
        n = sample.n_plays.astype(np.int64)
        self.seat = n % 4
        self.decl = sample.decl.astype(np.int64)
        rows = np.arange(len(n))
        start = n - self.seat
        lead_tile = sample.plays[rows, np.where(self.seat > 0, start, 0), 1].astype(np.int64)
        self.suit = led[self.decl, np.maximum(lead_tile, 0)]
        best = np.full(len(n), -1, np.int64)
        self.winner = np.full(len(n), -1, np.int64)      # index into WINNERS, −1 on lead
        for i in range(3):
            on = self.seat > i
            t = sample.plays[rows, np.minimum(start + i, 27), 1].astype(np.int64)
            r = np.where(on, rank[self.decl, self.suit, np.maximum(t, 0)], -1)
            better = on & (r > best)
            best = np.where(better, r, best)
            self.winner = np.where(better, (i - self.seat) % 4 - 1, self.winner)
        self.best = best
        self._rank, self._count, self._led = rank, count, led

    def beats(self, tile: np.ndarray) -> np.ndarray:
        t = np.maximum(tile, 0)
        return (self.seat > 0) & (self._rank[self.decl, self.suit, t] > self.best)

    def count(self, tile: np.ndarray) -> np.ndarray:
        return self._count[self.decl, np.maximum(tile, 0)] > 0

    def trump(self, tile: np.ndarray) -> np.ndarray:
        return self._led[self.decl, np.maximum(tile, 0)] == TRUMP_SUIT


def eligible(trick: Trick, actions: np.ndarray) -> np.ndarray:
    """Following, more than one legal tile, and the choice changes the take or the count."""
    legal = actions >= 0
    b = np.stack([trick.beats(actions[:, k]) & legal[:, k] for k in range(actions.shape[1])], 1)
    c = np.stack([trick.count(actions[:, k]) & legal[:, k] for k in range(actions.shape[1])], 1)
    nb, nc, nl = b.sum(1), c.sum(1), legal.sum(1)
    return (trick.seat > 0) & (nl > 1) & (((nb > 0) & (nb < nl)) | ((nc > 0) & (nc < nl)))


def clustered(v: np.ndarray, deal: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean of v with a deal-clustered bootstrap 95% interval."""
    if len(v) == 0:
        return (float("nan"),) * 3
    ids, inv = np.unique(deal, return_inverse=True)
    s, c = np.bincount(inv, v), np.bincount(inv)
    boots = np.empty(BOOT)
    for b in range(BOOT):
        pick = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))
        boots[b] = (pick * s).sum() / (pick * c).sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def where_play_goes(trick: Trick, base: np.ndarray, new: np.ndarray, rows: np.ndarray, deal: np.ndarray,
                    rng: np.random.Generator) -> dict:
    tb, tn = trick.beats(base), trick.beats(new)
    sb, sn = trick.count(base), trick.count(new)
    rec = {"rows": int(rows.sum())}
    for name, m in [("pooled", rows)] + [(SEATS[s], rows & (trick.seat == s)) for s in (1, 2, 3)]:
        d = deal[m]
        rec[name] = {"rows": int(m.sum()),
                     "take": clustered((tn[m].astype(float) - tb[m]), d, rng),
                     "smear": clustered((sn[m].astype(float) - sb[m]), d, rng),
                     "changed": float((new[m] != base[m]).mean()) if m.any() else None,
                     "base_take": float(tb[m].mean()) if m.any() else None,
                     "base_smear": float(sb[m].mean()) if m.any() else None}
    return rec


def precision(mask: np.ndarray, top: np.ndarray) -> float:
    return float(mask[top].mean())


class Labeller:
    """The audit's labeller under read R (#233): highest precision on the top set among concepts and
    raw inputs with base rate ≤ BASE_CAP, counted only where it beats the play-count mix by
    BASELINE_MARGIN."""

    def __init__(self, labels: np.ndarray, names: list[str], x: np.ndarray, n_plays: np.ndarray):
        self.allc = np.concatenate([labels.astype(bool), x.astype(bool)], axis=1)
        self.names = list(names) + feature_names(x.shape[1])
        self.ok = self.allc.mean(0) <= BASE_CAP
        self.rates, self.idx = stratum_rates(self.allc, n_plays)

    def margin(self, top: np.ndarray) -> np.ndarray:
        """Each column's precision on the top set, and the part of it the play count explains."""
        return self.allc[top].mean(0), expected_from_strata(top, self.rates, self.idx)

    def best(self, top: np.ndarray) -> tuple[str, float]:
        prec, exp = self.margin(top)
        cand = np.where(self.ok & (prec - exp >= BASELINE_MARGIN), prec, -1.0)
        i = int(np.argmax(cand))
        return self.names[i], float(cand[i])

    def counts(self, top: np.ndarray, col: str) -> float:
        """A column's precision on the top set if the baseline grants it, else 0."""
        prec, exp = self.margin(top)
        i = self.names.index(col)
        return float(prec[i]) if prec[i] - exp[i] >= BASELINE_MARGIN else 0.0


def run(sample, labels: np.ndarray, names: list[str], net_path: str, j: int, twin_path: str, audit: dict,
        formation_dir: str | None, log=print) -> dict:
    x = sample.x
    fn = feature_names(x.shape[1])
    pw = labels[:, names.index("partner-winning")].astype(bool)
    rhs = x[:, fn.index("x:winning:2")].astype(bool)
    net = load_net(net_path)
    acts = forward(net, x)
    z, a = acts.z1[:, j], acts.a1[:, j]
    k = max(1, int(round(TOP * len(z))))
    top_p, top_n = np.argsort(-z, kind="stable")[:k], np.argsort(z, kind="stable")[:k]
    trick = Trick(sample)
    lab = Labeller(labels, names, x, sample.n_plays.astype(np.int64))
    best_label = lambda top, *_: lab.best(top)
    out: dict = {"unit": j, "net": net_path, "rows": len(z), "top_k": k}

    # 1. label: the seat × winner table, both poles, the twin
    cells = []
    for s in range(4):
        for w in ([-1] if s == 0 else range(3)):
            m = (trick.seat == s) & (trick.winner == w)
            if s and w < 3 - s:       # that seat plays after me in this trick
                continue
            cells.append({"seat": SEATS[s], "winner": "-" if w < 0 else WINNERS[w], "rows": int(m.sum()),
                          "mean_z": float(z[m].mean()) if m.any() else None,
                          "on": float((a[m] > 0).mean()) if m.any() else None})
    out["cells"] = cells
    out["max_z"] = float(z.max())
    late = trick.seat >= 2
    kl = max(1, int(round(TOP * late.sum())))
    top_nl = np.flatnonzero(late)[np.argsort(z[late], kind="stable")[:kl]]
    out["poles"] = {"+": {"label": best_label(top_p, labels, names, x), "partner-winning": precision(pw, top_p)},
                    "-": {"label": best_label(top_n, labels, names, x), "x:winning:2": precision(rhs, top_n)},
                    "- third/last": {"label": best_label(top_nl, labels, names, x),
                                     "x:winning:2": precision(rhs, top_nl), "rows": int(late.sum())}}
    # raw shape: dial or detector
    q = np.linspace(0, 100, 11)
    lo_pw = np.percentile(z[pw], 5)
    third_pw = pw & (trick.seat == 2)
    safe = labels[:, names.index("winner-cannot-be-beaten")].astype(bool)
    held = labels[:, names.index("holds-count")].astype(bool)
    decl_means = [float(z[pw & (sample.decl == d)].mean()) for d in range(7) if (pw & (sample.decl == d)).sum() >= 200]
    out["shape"] = {"deciles_pw": np.percentile(z[pw], q).tolist(), "deciles_not": np.percentile(z[~pw], q).tolist(),
                    "overlap": float((z[~pw] >= lo_pw).mean()),
                    "third_pw_safe": float(z[third_pw & safe].mean()), "third_pw_unsafe": float(z[third_pw & ~safe].mean()),
                    "pw_holds_count": float(z[pw & held].mean()), "pw_no_count": float(z[pw & ~held].mean()),
                    "pw_decl_spread": float(max(decl_means) - min(decl_means))}
    tw = forward(load_net(twin_path), x)
    twin = {"raw partner-winning": 0.0, "raw x:winning:2": 0.0, "R partner-winning": 0.0, "R x:winning:2": 0.0}
    for zz in (tw.z1, tw.z2):
        for jj in range(zz.shape[1]):
            for sgn in (1, -1):
                t = np.argsort(-sgn * zz[:, jj], kind="stable")[:k]
                for col, m in (("partner-winning", pw), ("x:winning:2", rhs)):
                    twin[f"raw {col}"] = max(twin[f"raw {col}"], precision(m, t))
                    twin[f"R {col}"] = max(twin[f"R {col}"], lab.counts(t, col))
    out["twin"] = twin
    log(f"label: + {out['poles']['+']}  − {out['poles']['-']}  twin {out['twin']}")

    # 2. the circuit in
    w = net.w1[:, j].astype(np.float64)
    allmean = x.mean(0)
    votes = {}
    for pole, top in (("+", top_p), ("-", top_n)):
        v = w * x[top].mean(0)
        order = np.argsort(-np.abs(v))[:10]
        votes[pole] = [{"input": fn[f], "w": int(w[f]), "vote": float(v[f]), "vote_all": float(w[f] * allmean[f])}
                       for f in order]
    out["votes"] = votes

    # 3. the circuit out
    reader_label = {r["name"]: r["label"] for r in audit["rows"]}
    col = net.w2[:, j]
    out["readers"] = [{"unit": int(i), "w2": int(col[i]),
                       "labels": [reader_label.get(f"L2+{i:03d}"), reader_label.get(f"L2-{i:03d}")]}
                      for i in np.argsort(-np.abs(col))]
    sigma = float(a[a > 0].std())
    out["sigma"] = sigma
    base_out = acts.out
    up = downstream(net, x, 1, j, None, add=int(round(sigma)))
    canon = sample.canon()
    off = (a == 0) & (trick.seat > 0)
    acts_ = sample.actions.astype(np.int64)
    push: dict[str, list[float]] = {}
    for kk in range(acts_.shape[1]):
        t = acts_[:, kk]
        m = off & (t >= 0)
        slot = canon[np.arange(len(t)), np.maximum(t, 0)]
        d = (up[np.arange(len(t)), slot] - base_out[np.arange(len(t)), slot]).astype(np.float64)
        for cls, cm in (("count", trick.count(t)), ("no count", ~trick.count(t)),
                        ("trump", trick.trump(t)), ("not trump", ~trick.trump(t)),
                        ("beats winner", trick.beats(t)), ("does not beat", ~trick.beats(t))):
            push.setdefault(cls, []).extend(d[m & cm].tolist())
    out["push"] = {c: {"mean": float(np.mean(v)), "n": len(v)} for c, v in push.items()}
    log(f"push per σ: { {c: round(v['mean'], 1) for c, v in out['push'].items()} }")

    # 5. steering, read by where play goes
    rng = np.random.default_rng(SEED)
    actions, deal = sample.actions, sample.deal
    base_pick = picks(base_out, actions, canon)
    el = eligible(trick, sample.actions.astype(np.int64))
    h1 = net.b1.shape[0]
    live = [u for u in range(h1) if u != j and (acts.a1[:, u] > 0).sum() >= 200]
    rand = [int(u) for u in live]
    out["random_units"] = rand
    on_rows, off_rows = el & (a > 0), el & (a == 0)
    steer = {"up": [], "down": {}}
    for kd in DOSES:
        rec = {"k": kd}
        for u in [j] + rand:
            au = acts.a1[:, u]
            su = float(au[au > 0].std())
            new = picks(downstream(net, x, 1, u, None, add=int(round(kd * su))), actions, canon)
            rec["own" if u == j else f"u{u}"] = where_play_goes(trick, base_pick, new, off_rows, deal, rng)
        steer["up"].append(rec)
        log(f"up k={kd}: take {rec['own']['pooled']['take']} smear {rec['own']['pooled']['smear']} "
            f"changed {rec['own']['pooled']['changed']:.3f}")
    for u in [j] + rand:
        new = picks(downstream(net, x, 1, u, 0), actions, canon)
        steer["down"]["own" if u == j else f"u{u}"] = where_play_goes(trick, base_pick, new, on_rows, deal, rng)
    d = steer["down"]["own"]["pooled"]
    log(f"down: take {d['take']} smear {d['smear']} changed {d['changed']:.3f}")
    # the input flip: partner made the winner, on the off rows where partner has played
    fl = off_rows & (trick.seat >= 2) & (trick.winner != 1)
    xf = x[fl].copy()
    w0 = fn.index("x:winning:0")
    xf[:, w0:w0 + 3] = 0
    xf[:, w0 + 1] = 1
    fo = forward(net, xf)
    flip_pick = base_pick.copy()
    flip_pick[fl] = picks(fo.out, actions[fl], canon[fl])
    steer["flip"] = where_play_goes(trick, base_pick, flip_pick, fl, deal, rng)
    steer["flip_unit_a"] = {"base": float(a[fl].mean()), "flipped": float(fo.a1[:, j].mean())}
    up1 = picks(downstream(net, x, 1, j, None, add=int(round(sigma))), actions, canon)
    steer["up1_on_flip_rows"] = where_play_goes(trick, base_pick, up1, fl, deal, rng)
    f = steer["flip"]["pooled"]
    log(f"flip: take {f['take']} smear {f['smear']} changed {f['changed']:.3f}")
    # a few steered picks in full, for reading
    ch = np.flatnonzero(off_rows & (up1 != base_pick))
    ex = []
    for r in rng.choice(ch, size=min(8, len(ch)), replace=False):
        nn = int(sample.n_plays[r])
        ex.append({"row": int(r), "deal": int(deal[r]), "decl": int(sample.decl[r]), "seat": SEATS[int(trick.seat[r])],
                   "winner": WINNERS[int(trick.winner[r])],
                   "trick": [int(t) for t in sample.plays[r, nn - nn % 4:nn, 1]],
                   "legal": [int(t) for t in actions[r] if t >= 0],
                   "base": int(base_pick[r]), "steered": int(up1[r])})
    steer["examples"] = ex
    out["steer"] = steer

    # 6. history: the top 1% against partner-winning rows where the unit sits below its median
    tp = np.zeros(len(z), bool)
    tp[top_p] = True
    low = pw & ~tp & (z < np.median(z[pw]))
    trick_no = sample.n_plays.astype(np.int64) // 4 + 1
    need = x[:, fn.index("x:need:0"):fn.index("x:need:0") + 21].sum(1)
    slack = x[:, fn.index("x:slack:0"):fn.index("x:slack:0") + 21].sum(1)
    out["history"] = {grp: {"rows": int(m.sum()), "trick": float(trick_no[m].mean()),
                            "need_steps": float(need[m].mean()), "slack_steps": float(slack[m].mean()),
                            "bid": float(sample.bid[m].mean()),
                            "seat_last": float((trick.seat[m] == 3).mean())}
                      for grp, m in (("top", tp), ("low", low))}

    out["pw_seats"] = {SEATS[s]: float((trick.seat[pw] == s).mean()) for s in range(4)}
    # 4. formation
    if formation_dir:
        form = []
        for ck in sorted(Path(formation_dir).glob("step_*.npz")):
            fz = forward(load_net(ck), x).z1[:, j]
            tp_, tn_ = np.argsort(-fz, kind="stable")[:k], np.argsort(fz, kind="stable")[:k]
            lp, ln = best_label(tp_, labels, names, x), best_label(tn_, labels, names, x)
            form.append({"step": int(ck.stem.split("_")[1]),
                         "top_seats": {SEATS[s]: float((trick.seat[tp_] == s).mean()) for s in range(4)},
                         "+": {"partner-winning": precision(pw, tp_), "best": lp, "local": is_trick_state(lp[0])},
                         "-": {"x:winning:2": precision(rhs, tn_), "best": ln, "local": is_trick_state(ln[0])}})
            log(f"  {ck.stem}: + {lp} pw {form[-1]['+']['partner-winning']:.3f} | − {ln}")
        out["formation"] = form
    return out


def r2(z: np.ndarray, cols: np.ndarray) -> float:
    """Variance of z explained by a least-squares fit on the given 0/1 columns."""
    X = np.c_[np.ones(len(z)), cols.astype(np.float64)]
    b = np.linalg.lstsq(X, z, rcond=None)[0]
    return float(1 - ((z - X @ b) ** 2).mean() / z.var())


def shifts(trick: Trick, base: np.ndarray, new: np.ndarray, slot_tile: np.ndarray | None,
           suit_real: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Per-row shifts, steered minus base: take, smear, and (where given) playing the gated tile
    and playing a non-double tile of its suit (`suit_real` [N, k] real tile ids)."""
    out = {"take": trick.beats(new).astype(float) - trick.beats(base),
           "smear": trick.count(new).astype(float) - trick.count(base)}
    if slot_tile is not None:
        out["plays-gated"] = (new == slot_tile).astype(float) - (base == slot_tile)
    if suit_real is not None:
        out["plays-its-suit"] = (new[:, None] == suit_real).any(1).astype(float) - (base[:, None] == suit_real).any(1)
    return out


def paired(own: dict, others: list[dict], rows: np.ndarray, deal: np.ndarray, rng: np.random.Generator,
           where: dict[str, np.ndarray] | None = None) -> dict:
    """Own shift, the layer's mean shift (the other units' mean, row by row) and their paired
    difference, each with a deal-clustered interval, per measure, on `rows` (or its own mask)."""
    rec = {}
    for name, v in own.items():
        m = where.get(name, rows) if where else rows
        layer = np.mean([o[name] for o in others], axis=0)
        rec[name] = {"rows": int(m.sum()), "own": clustered(v[m], deal[m], rng),
                     "layer": clustered(layer[m], deal[m], rng), "own-layer": clustered((v - layer)[m], deal[m], rng),
                     "others": [float(o[name][m].mean()) for o in others]}
    return rec


def pip_tiles(pip: int) -> list[int]:
    """Canonical slots of the non-double tiles carrying `pip`, trump (pip 6) excluded."""
    out = []
    for hi in range(7):
        for lo in range(hi + 1):
            if hi != lo and pip in (hi, lo) and 6 not in (hi, lo):
                out.append(hi * (hi + 1) // 2 + lo)
    return out


def double_pip(slot: int) -> int:
    for p in range(7):
        if p * (p + 1) // 2 + p == slot:
            return p
    raise ValueError(f"slot {slot} is not a double")


def run_gate(sample, labels: np.ndarray, names: list[str], net_path: str, j: int, gate: str, twin_path: str,
             audit: dict, formation_dir: str | None, log=print) -> dict:
    """A unit paper for a unit gated by one input bit (`gate`, e.g. `x:hand:14`: I hold canonical
    tile 14). The cells are the gated tile's state (held, gone, out) by seat."""
    x = sample.x
    fn = feature_names(x.shape[1])
    tile = int(gate.split(":")[2])
    g = x[:, fn.index(gate)].astype(bool)
    gone = x[:, fn.index(f"x:gone:{tile}")].astype(bool)
    ten, five = x[:, fn.index(f"x:count:{tile}")].astype(bool), x[:, fn.index(f"x:count:{N_TILES + tile}")].astype(bool)
    net = load_net(net_path)
    acts = forward(net, x)
    z, a = acts.z1[:, j].astype(np.float64), acts.a1[:, j]
    k = max(1, int(round(TOP * len(z))))
    top_p, top_n = np.argsort(-z, kind="stable")[:k], np.argsort(z, kind="stable")[:k]
    trick = Trick(sample)
    lab = Labeller(labels, names, x, sample.n_plays.astype(np.int64))
    out: dict = {"unit": j, "net": net_path, "gate": gate, "rows": len(z), "top_k": k,
                 "max_z": float(z.max()), "min_z": float(z.min())}

    # 1. label: the gated tile's state by seat, both poles raw and under read R, the twin
    state = np.where(g, 0, np.where(gone, 1, 2))
    out["cells"] = [{"tile": st, "seat": SEATS[s], "rows": int(m.sum()), "mean_z": float(z[m].mean()),
                     "on": float((a[m] > 0).mean())}
                    for i, st in enumerate(("held", "gone", "out")) for s in range(4)
                    for m in [(state == i) & (trick.seat == s)] if m.sum()]
    out["poles"] = {"+": {"label": lab.best(top_p), "not gate": float((~g[top_p]).mean())},
                    "-": {"label": lab.best(top_n), "gate": float(g[top_n].mean())}}
    out["r2"] = {"gate": r2(z, g[:, None]), "gate+gone": r2(z, np.c_[g, gone]), "all inputs": r2(z, x)}
    out["count_slot"] = {kind: {"rows": int(m.sum()), "held": float(z[m & g].mean()), "not held": float(z[m & ~g].mean())}
                         for kind, m in (("ten", ten), ("five", five), ("none", ~ten & ~five)) if (m & g).sum() >= 200}
    # the open side: what grades z where the gate is shut off (I do not hold the tile)
    allc = np.concatenate([labels, x], 1).astype(np.float64)
    nm = list(names) + fn
    zo = z[~g]
    c = allc[~g]
    r = ((c - c.mean(0)) / (c.std(0) + 1e-9) * ((zo - zo.mean()) / zo.std())[:, None]).mean(0)
    out["open_side"] = [(nm[i], float(r[i])) for i in np.argsort(-np.abs(r))[:12]]
    tw = forward(load_net(twin_path), x)
    best = 0.0
    for zz in (tw.z1, tw.z2):
        for jj in range(zz.shape[1]):
            for sgn in (1, -1):
                best = max(best, float(g[np.argsort(-sgn * zz[:, jj], kind="stable")[:k]].mean()))
    def raw_bit_clean(zz: np.ndarray) -> float:
        """Share of a layer's live poles whose top 1% is ≥ 0.9 one raw input bit (base ≤ BASE_CAP)."""
        ok = x.mean(0) <= BASE_CAP
        n = c = 0
        for jj in range(zz.shape[1]):
            if zz[:, jj].std() == 0:
                continue
            for sgn in (1, -1):
                t = np.argsort(-sgn * zz[:, jj], kind="stable")[:k]
                n += 1
                c += float(np.where(ok, x[t].mean(0), 0).max()) >= 0.9
        return c / max(n, 1)
    out["raw_bit_clean_layer1"] = {"trained": raw_bit_clean(acts.z1), "twin": raw_bit_clean(tw.z1)}
    is_double = tile in {p * (p + 1) // 2 + p for p in range(7)}
    pip = double_pip(tile) if is_double else None
    suit = pip_tiles(pip) if is_double else []
    held_any = lambda slots: x[:, [fn.index(f"x:hand:{t}") for t in slots]].any(1)
    gate_gap = float(z[~g].mean() - z[g].mean())
    near = {}
    trumps = [6 * 7 // 2 + q for q in range(7)]
    misses = ([(f"double {p}-{p}", [p * (p + 1) // 2 + p]) for p in range(6) if p != pip] + [(f"a {pip}-x (not the double)", suit)]
              if is_double else [(f"trump 6-{q}", [6 * 7 // 2 + q]) for q in range(7) if 6 * 7 // 2 + q != tile])
    for name, slots in misses:
        h = held_any(slots)
        near[name] = float(z[~g & ~h].mean() - z[~g & h].mean()) / gate_gap
    out["near_misses"] = {"gate gap": gate_gap, "gap / gate gap (among rows without the gated tile)": near}
    out["twin"] = {"best gate precision": best,
                   "best gate r2": max(r2(zz[:, jj].astype(np.float64), g[:, None]) for zz in (tw.z1, tw.z2)
                                       for jj in range(zz.shape[1]) if zz[:, jj].std() > 0)}
    log(f"label: {out['poles']}  r2 {out['r2']}  twin {out['twin']}")

    # 2. the circuit in
    w = net.w1[:, j].astype(np.float64)
    out["votes"] = {pole: [{"input": fn[f], "w": int(w[f]), "vote": float(v[f])}
                           for f in np.argsort(-np.abs(v))[:10]]
                    for pole, top in (("+", top_p), ("-", top_n)) for v in [w * x[top].mean(0)]}

    # 3. the circuit out: readers, and the push of +σ on each legal tile where the unit is off
    reader_label = {r["name"]: r["label"] for r in audit["rows"]}
    col = net.w2[:, j]
    out["readers"] = [{"unit": int(i), "w2": int(col[i]),
                       "labels": [reader_label.get(f"L2+{i:03d}"), reader_label.get(f"L2-{i:03d}")]}
                      for i in np.argsort(-np.abs(col))]
    sigma = float(a[a > 0].std())
    out["sigma"] = sigma
    up = downstream(net, x, 1, j, None, add=int(round(sigma)))
    canon = sample.canon()
    rows_i = np.arange(len(z))
    acts_ = sample.actions.astype(np.int64)
    push: dict[str, list[float]] = {}
    for kk in range(acts_.shape[1]):
        t = acts_[:, kk]
        m = (a == 0) & (t >= 0)
        slot = canon[rows_i, np.maximum(t, 0)]
        d = (up[rows_i, slot] - acts.out[rows_i, slot]).astype(np.float64)
        in_suit = np.isin(slot, suit)
        for cls, cm in (("gated tile", slot == tile), ("other tiles", slot != tile),
                        ("its suit (not the double)", in_suit), ("other non-trump, not its suit", ~in_suit & (slot != tile) & ~trick.trump(t)),
                        ("count", trick.count(t)), ("no count", ~trick.count(t)),
                        ("trump", trick.trump(t)), ("not trump", ~trick.trump(t)),
                        ("beats winner", trick.beats(t) & (trick.seat > 0)),
                        ("does not beat", ~trick.beats(t) & (trick.seat > 0))):
            push.setdefault(cls, []).extend(d[m & cm].tolist())
    out["push"] = {c: {"mean": float(np.mean(v)) if v else None, "n": len(v)} for c, v in push.items()}
    log(f"push per σ: { {c: v['mean'] for c, v in out['push'].items()} }")

    # 5. steering, read by where play goes, against the layer-mean null
    rng = np.random.default_rng(SEED)
    actions, deal = sample.actions, sample.deal
    base_pick = picks(acts.out, actions, canon)
    real_of_slot = np.argsort(canon, axis=1)            # [N, 28]: the real tile in each canonical slot
    gated_real = real_of_slot[rows_i, tile]
    suit_real = real_of_slot[:, suit] if suit else None
    legal_suit = (actions[:, :, None] == suit_real[:, None, :]).any((1, 2)) if suit else np.zeros(len(z), bool)
    legal_gated = (actions == gated_real[:, None]).any(1)
    el = eligible(trick, acts_)
    multi = (actions >= 0).sum(1) > 1
    h1 = net.b1.shape[0]
    live = [u for u in range(h1) if u != j and (acts.a1[:, u] > 0).sum() >= 200]
    out["null_units"] = live
    on_rows, off_rows = el & (a > 0), el & (a == 0)
    gated_rows = multi & legal_gated & (a == 0)
    suit_lead_rows = (trick.seat == 0) & g & legal_suit & (a == 0)
    out["rows"] = {"on": int(on_rows.sum()), "off": int(off_rows.sum()), "off, gated tile legal": int(gated_rows.sum()),
                   "base plays-gated": float((base_pick == gated_real)[gated_rows].mean()),
                   "lead, double held, its suit legal": int(suit_lead_rows.sum()),
                   "base plays-its-suit there": float((base_pick[:, None] == suit_real).any(1)[suit_lead_rows].mean()) if suit else None,
                   "lead, gated tile legal": int((gated_rows & (trick.seat == 0)).sum()),
                   "base plays-gated on lead": float((base_pick == gated_real)[gated_rows & (trick.seat == 0)].mean())}
    sig = {u: float(acts.a1[:, u][acts.a1[:, u] > 0].std()) for u in [j] + live}
    steer: dict = {"up": []}
    for kd in DOSES:
        sh = {u: shifts(trick, base_pick, picks(downstream(net, x, 1, u, None, add=int(round(kd * sig[u]))), actions, canon),
                        gated_real, suit_real) for u in [j] + live}
        for u in sh:
            sh[u]["plays-gated, lead"] = sh[u]["plays-gated"]
        rec = paired(sh[j], [sh[u] for u in live], off_rows, deal, rng,
                     where={"plays-gated": gated_rows, "plays-gated, lead": gated_rows & (trick.seat == 0),
                            "plays-its-suit": suit_lead_rows})
        rec["k"] = kd
        steer["up"].append(rec)
        log(f"up k={kd}: " + "  ".join(f"{m} own {v['own'][0]:+.3f} layer {v['layer'][0]:+.3f}" for m, v in rec.items() if m != "k"))
    sh = {u: shifts(trick, base_pick, picks(downstream(net, x, 1, u, 0), actions, canon), None) for u in [j] + live}
    steer["down"] = paired(sh[j], [sh[u] for u in live], on_rows, deal, rng)
    log("down: " + "  ".join(f"{m} own {v['own'][0]:+.3f} layer {v['layer'][0]:+.3f}" for m, v in steer["down"].items()))
    # the input flip: the gate bit turned off on rows where I hold the tile (the tile stays legal)
    fl = multi & g & legal_gated
    xf = x[fl].copy()
    xf[:, fn.index(gate)] = 0
    fo = forward(net, xf)
    flip_pick = base_pick.copy()
    flip_pick[fl] = picks(fo.out, actions[fl], canon[fl])
    up1 = picks(downstream(net, x, 1, j, None, add=int(round(sigma))), actions, canon)
    steer["flip"] = {"rows": int(fl.sum()), "unit a before": float(a[fl].mean()), "unit a after": float(fo.a1[:, j].mean()),
                     "flip": {m: clustered(v[fl], deal[fl], rng) for m, v in shifts(trick, base_pick, flip_pick, gated_real).items()},
                     "unit k=1": {m: clustered(v[fl], deal[fl], rng) for m, v in shifts(trick, base_pick, up1, gated_real).items()},
                     "flip, leads": {m: clustered(v[fl & (trick.seat == 0)], deal[fl & (trick.seat == 0)], rng)
                                     for m, v in shifts(trick, base_pick, flip_pick, gated_real, suit_real).items()},
                     "unit k=1, leads": {m: clustered(v[fl & (trick.seat == 0)], deal[fl & (trick.seat == 0)], rng)
                                         for m, v in shifts(trick, base_pick, up1, gated_real, suit_real).items()}}
    log(f"flip: {steer['flip']['flip']}  unit {steer['flip']['unit k=1']}")
    ch = np.flatnonzero(gated_rows & (up1 != base_pick))
    steer["examples"] = [{"row": int(r), "decl": int(sample.decl[r]), "seat": SEATS[int(trick.seat[r])],
                          "trick": [int(t) for t in sample.plays[r, int(sample.n_plays[r]) - int(trick.seat[r]):int(sample.n_plays[r]), 1]],
                          "legal": [int(t) for t in actions[r] if t >= 0], "gated": int(gated_real[r]),
                          "base": int(base_pick[r]), "steered": int(up1[r])}
                         for r in rng.choice(ch, size=min(8, len(ch)), replace=False)]
    out["steer"] = steer

    # 4. formation: when the gate formed, and what the open side was
    if formation_dir:
        form = []
        for ck in sorted(Path(formation_dir).glob("step_*.npz")):
            fa = forward(load_net(ck), x)
            fz, fa1 = fa.z1[:, j].astype(np.float64), fa.a1[:, j]
            tp_, tn_ = np.argsort(-fz, kind="stable")[:k], np.argsort(fz, kind="stable")[:k]
            seat1h = np.eye(4)[trick.seat[~g]]
            form.append({"step": int(ck.stem.split("_")[1]), "r2 gate": r2(fz, g[:, None]),
                         "r2 seat, open side": r2(fz[~g], seat1h[:, 1:]),
                         "on held": float((fa1[g] > 0).mean()), "on not held": float((fa1[~g] > 0).mean()),
                         "- gate": float(g[tn_].mean()), "+": lab.best(tp_), "-": lab.best(tn_)})
            f = form[-1]
            log(f"  {ck.stem}: r2 {f['r2 gate']:.3f} seat-open {f['r2 seat, open side']:.3f} on held {f['on held']:.3f} / not {f['on not held']:.3f} + {f['+']} − {f['-']}")
        out["formation"] = form
    return out


def route(sample, net_path: str, bit: str, tile: int, log=print, pair: tuple[int, int] = (13, 0)) -> dict:
    """Which layer-1 units carry an input bit's effect on playing one tile (#235, paper 3).

    On rows where the bit is on, the tile is legal and there is a choice, turn the bit off. The
    tile stays legal, so the flipped input is one training never saw: read it as an intervention,
    not a position. Then, per layer-1 unit u:
      - *share*: the flip's effect minus the effect with u held at its unflipped activation
        (what depends on u, given every other unit has moved);
      - *alone*: the effect of moving u only, every other unit unflipped.
    Layer 2 is a ReLU layer, so neither sums to the flip; both sums are reported. Each effect is
    read on the pick (does the net play the tile) and on the margin (the tile's score minus the best
    other legal score), over all rows, over leads, and per row on which u moved. `pair` is also held
    jointly, to read its interaction."""
    x = sample.x
    fn = feature_names(x.shape[1])
    net = load_net(net_path)
    f = fn.index(bit)
    canon = sample.canon()
    n_all = len(x)
    real = np.argsort(canon, axis=1)[np.arange(n_all), tile]
    actions = sample.actions
    rows = x[:, f].astype(bool) & (actions == real[:, None]).any(1) & ((actions >= 0).sum(1) > 1)
    xr, ar, cr, rr = x[rows].astype(np.int64), actions[rows], canon[rows], real[rows]
    n = len(xr)
    lead = (sample.n_plays[rows] % 4) == 0
    a1 = np.maximum((xr @ net.w1 + net.b1) >> net.s1, 0)
    xf = xr.copy()
    xf[:, f] = 0
    a1f = np.maximum((xf @ net.w1 + net.b1) >> net.s1, 0)
    other = (ar >= 0) & (ar != rr[:, None])
    oslot = cr[np.arange(n)[:, None], np.where(ar >= 0, ar, 0)]

    def read(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        a2 = np.maximum((a @ net.w2.T + net.b2) >> net.s2, 0)
        out = a2 @ net.w3.T + net.b3
        sc = out[np.arange(n)[:, None], oslot].astype(float)
        margin = out[np.arange(n), tile] - np.where(other, sc, -np.inf).max(1)
        return picks(out, ar, cr) == rr, margin

    bp, bm = read(a1)

    def eff(a: np.ndarray, m=slice(None)) -> dict:
        p, mg = read(a)
        return {"pick": float((p.astype(float) - bp)[m].mean()), "margin": float((mg - bm)[m].mean())}

    def both(a: np.ndarray) -> dict:
        return {"all": eff(a), "lead": eff(a, lead)}

    flip = both(a1f)
    out = {"rows": n, "lead rows": int(lead.sum()), "base": float(bp.mean()), "base, lead": float(bp[lead].mean()),
           "base margin": float(bm.mean()), "flip": flip, "units": []}
    diff = lambda e, g: {k: {m: e[k][m] - g[k][m] for m in e[k]} for k in e}
    for u in range(a1.shape[1]):
        held = a1f.copy()
        held[:, u] = a1[:, u]
        alone = a1.copy()
        alone[:, u] = a1f[:, u]
        moved = a1f[:, u] != a1[:, u]
        out["units"].append({"unit": u, "w": int(net.w1[f, u]), "moved": float(moved.mean()),
                             "moved, lead": float(moved[lead].mean()),
                             "share": diff(flip, both(held)), "alone": both(alone),
                             "alone per moved row": eff(alone, moved) if moved.any() else None})
    for kind in ("share", "alone"):
        out[f"sum of {kind}s"] = {k: {m: sum(r[kind][k][m] for r in out["units"]) for m in ("pick", "margin")}
                                  for k in ("all", "lead")}
    held = a1f.copy()
    held[:, list(pair)] = a1[:, list(pair)]
    joint = diff(flip, both(held))
    s0, s1 = (out["units"][u]["share"] for u in pair)
    out["pair"] = {"units": list(pair), "joint share": joint,
                   "interaction": {k: {m: joint[k][m] - s0[k][m] - s1[k][m] for m in ("pick", "margin")} for k in joint}}
    log(f"route {bit} -> tile {tile}: flip pick {flip['all']['pick']:+.3f} margin {flip['all']['margin']:+.0f}; shares "
        + " ".join(f"u{r['unit']}:{r['share']['all']['pick']:+.3f}" for r in out["units"] if abs(r["share"]["all"]["pick"]) >= 0.005))
    return out


OFF_DOUBLES = [p * (p + 1) // 2 + p for p in range(6)]   # canonical slots 0-0 … 5-5
TRUMPS = list(range(21, 28))                              # canonical slots 6-0 … 6-6


def run_reader(sample, labels: np.ndarray, names: list[str], net_path: str, j: int, twin_path: str,
               audit: dict, formation_dir: str | None, log=print) -> dict:
    """A unit paper for a layer-2 unit that acts on lead (#235, paper 4). Its pre-activation is a
    weighted sum of layer-1 units, so the label is read on inputs, its circuit in on layer-1
    contributions, and its write is its own column of w₃. Steering is read on leads: whether the
    pick is an off double, a trump, the 6-5. The down arm ablates it where it is naturally on."""
    x = sample.x
    fn = feature_names(x.shape[1])
    net = load_net(net_path)
    acts = forward(net, x)
    z, a = acts.z2[:, j].astype(np.float64), acts.a2[:, j]
    k = max(1, int(round(TOP * len(z))))
    top_p, top_n = np.argsort(-z, kind="stable")[:k], np.argsort(z, kind="stable")[:k]
    trick = Trick(sample)
    lab = Labeller(labels, names, x, sample.n_plays.astype(np.int64))
    lead = trick.seat == 0
    hand = x[:, [fn.index(f"x:hand:{t}") for t in range(N_TILES)]].astype(bool)
    n_trump = hand[:, TRUMPS].sum(1)
    n_dbl = hand[:, OFF_DOUBLES].sum(1)
    out: dict = {"unit": j, "layer": 2, "net": net_path, "rows": len(z), "top_k": k,
                 "max_z": float(z.max()), "min_z": float(z.min()), "on": float((a > 0).mean())}

    # 1. label
    out["cells"] = [{"seat": SEATS[s], "winner": "-" if w < 0 else WINNERS[w], "rows": int(m.sum()),
                     "mean_z": float(z[m].mean()), "on": float((a[m] > 0).mean())}
                    for s in range(4) for w in ([-1] if s == 0 else range(3 - s, 3))
                    for m in [(trick.seat == s) & (trick.winner == w)] if m.sum()]
    out["lead"] = {
        "by trumps held": {int(n): {"rows": int(m.sum()), "on": float((a[m] > 0).mean()), "mean_z": float(z[m].mean())}
                           for n in range(8) for m in [lead & (n_trump == n)] if m.sum() >= 200},
        "by off doubles held": {int(n): {"rows": int(m.sum()), "on": float((a[m] > 0).mean()), "mean_z": float(z[m].mean())}
                                for n in range(7) for m in [lead & (n_dbl == n)] if m.sum() >= 200},
        "by trump tile held": {f"6-{q}": {"on held": float((a[lead & hand[:, 21 + q]] > 0).mean()),
                                          "on not": float((a[lead & ~hand[:, 21 + q]] > 0).mean())} for q in range(7)}}
    if "is-bidder" in names:
        bidder = labels[:, list(names).index("is-bidder")].astype(bool)
        out["lead"]["by bidder"] = {who: {"rows": int(m.sum()), "on": float((a[m] > 0).mean())}
                                    for who, m in (("bidder", lead & bidder), ("not bidder", lead & ~bidder))}
    else:
        bidder = None
    out["poles"] = {"+": lab.best(top_p), "-": lab.best(top_n), "+ lead share": float(lead[top_p].mean()),
                    "- lead share": float(lead[top_n].mean())}
    zl = z[lead]
    out["r2 lead"] = {"trumps held (one-hot)": r2(zl, np.eye(8)[n_trump[lead]]),
                      "off doubles held (one-hot)": r2(zl, np.eye(7)[n_dbl[lead]]),
                      "both": r2(zl, np.c_[np.eye(8)[n_trump[lead]], np.eye(7)[n_dbl[lead]]]),
                      "hand bits": r2(zl, hand[lead]), "all inputs": r2(zl, x[lead])}
    out["r2 seat"] = r2(z, np.eye(4)[trick.seat][:, 1:])
    # the same, conditioned on the play count (#226): each block's increment over the trick number
    tl = np.eye(7)[sample.n_plays[lead].astype(np.int64) // 4]
    base_t = r2(zl, tl)
    out["r2 lead, over trick number"] = {
        "trick number": base_t,
        "+ trumps held": r2(zl, np.c_[tl, np.eye(8)[n_trump[lead]]]) - base_t,
        "+ off doubles held": r2(zl, np.c_[tl, np.eye(7)[n_dbl[lead]]]) - base_t,
        "+ off doubles, given trumps held": r2(zl, np.c_[tl, np.eye(8)[n_trump[lead]], np.eye(7)[n_dbl[lead]]])
                                            - r2(zl, np.c_[tl, np.eye(8)[n_trump[lead]]])}
    short_nd = lead & (n_trump <= 1) & (n_dbl == 0)
    out["lead"]["short in trump, no off double"] = {"rows": int(short_nd.sum()), "on": float((a[short_nd] > 0).mean())}
    log(f"label: on {out['on']:.3f}  poles {out['poles']}  r2 lead {out['r2 lead']}  r2 seat {out['r2 seat']:.3f}")

    # 2. circuit in: layer-1 contributions w₂[j, u]·a₁[u], on lead and off it
    w2 = net.w2[j].astype(np.float64)
    contrib = acts.a1.astype(np.float64) * w2
    out["inputs"] = [{"unit": int(u), "w2": int(w2[u]), "mean on lead": float(contrib[lead, u].mean()),
                      "mean off lead": float(contrib[~lead, u].mean()), "sd on lead": float(contrib[lead, u].std())}
                     for u in np.argsort(-contrib[lead].std(0))]

    # 3. circuit out: the write, w₃[:, j], by class
    w3 = net.w3[:, j].astype(np.float64)
    out["write"] = {"by slot": [int(v) for v in w3],
                    "off doubles": float(w3[OFF_DOUBLES].mean()), "trumps": float(w3[TRUMPS].mean()),
                    "other": float(np.delete(w3, OFF_DOUBLES + TRUMPS).mean())}
    dmask = np.isin(np.arange(N_TILES), OFF_DOUBLES)
    corr = lambda col: float(np.corrcoef(col, dmask)[0, 1]) if col.std() > 0 else 0.0
    out["write"]["corr with off doubles"] = corr(w3)
    tw = load_net(twin_path)
    out["twin"] = {"max |corr| of a layer-2 write with off doubles": max(abs(corr(tw.w3[:, u].astype(np.float64)))
                                                                          for u in range(tw.w3.shape[1]))}
    log(f"write: {({kk: v for kk, v in out['write'].items() if kk != 'by slot'})}  twin {out['twin']}")

    # 5. steering on leads, against the layer-mean null of the other live layer-2 units
    rng = np.random.default_rng(SEED)
    actions, deal = sample.actions, sample.deal
    canon = sample.canon()
    rows_i = np.arange(len(z))
    base_pick = picks(acts.out, actions, canon)
    acts_ = actions.astype(np.int64)
    legal_slots = np.where(acts_ >= 0, canon[rows_i[:, None], np.maximum(acts_, 0)], -1)
    has = lambda slots: np.isin(legal_slots, slots).any(1)
    has_other = lambda slots: ((legal_slots >= 0) & ~np.isin(legal_slots, slots)).any(1)
    choice_d = lead & has(OFF_DOUBLES) & has_other(OFF_DOUBLES)
    choice_t = lead & has(TRUMPS) & has_other(TRUMPS)
    choice_65 = lead & has([26]) & has_other([26])
    el = eligible(trick, acts_)

    def reads(pick: np.ndarray) -> dict[str, np.ndarray]:
        sl = canon[rows_i, np.maximum(pick, 0)]
        return {"leads off double": np.isin(sl, OFF_DOUBLES).astype(float), "leads trump": np.isin(sl, TRUMPS).astype(float),
                "leads 6-5": (sl == 26).astype(float), "take": trick.beats(pick).astype(float),
                "smear": trick.count(pick).astype(float)}

    base = reads(base_pick)
    where = {"leads off double": choice_d, "leads trump": choice_t, "leads 6-5": choice_65}
    h2 = net.b2.shape[0]
    live = [u for u in range(h2) if u != j and (acts.a2[:, u] > 0).sum() >= 200]
    out["null_units"] = live
    sig = {u: float(acts.a2[:, u][acts.a2[:, u] > 0].std()) for u in [j] + live}
    out["sigma"] = sig[j]
    out["rows"] = {"lead, off double and other legal": int(choice_d.sum()), "lead, trump and non-trump legal": int(choice_t.sum()),
                   "lead, 6-5 and other legal": int(choice_65.sum()), "following, choice": int((el & ~lead).sum()),
                   "on, of those lead rows (off double)": float((a[choice_d] > 0).mean())}
    out["base"] = {m: float(base[m][where.get(m, el & ~lead)].mean()) for m in base}

    def shift(pick: np.ndarray) -> dict[str, np.ndarray]:
        r = reads(pick)
        return {m: r[m] - base[m] for m in r}

    steer: dict = {"up": []}
    for kd in DOSES:
        sh = {u: shift(picks(downstream(net, x, 2, u, None, add=int(round(kd * sig[u]))), actions, canon)) for u in [j] + live}
        rec = paired(sh[j], [sh[u] for u in live], el & ~lead, deal, rng, where=where)
        rec["k"] = kd
        steer["up"].append(rec)
        log(f"up k={kd}: " + "  ".join(f"{m} own {v['own'][0]:+.3f} layer {v['layer'][0]:+.3f}" for m, v in rec.items() if m != "k"))
    # down: ablate each unit to 0 where it is on (the natural-use read)
    picks_dn = {u: picks(downstream(net, x, 2, u, 0), actions, canon) for u in [j] + live}
    sh = {u: shift(picks_dn[u]) for u in [j] + live}
    on = a > 0
    steer["down"] = paired(sh[j], [sh[u] for u in live], el & ~lead & on, deal, rng,
                           where={m: wm & on for m, wm in where.items()})
    log("down: " + "  ".join(f"{m} own {v['own'][0]:+.3f} layer {v['layer'][0]:+.3f}" for m, v in steer["down"].items()))
    # each unit ablated where *it* is on, on the same row classes: the spread across units
    steer["down, each unit where it is on"] = {
        int(u): {m: {"rows": int((wm & (acts.a2[:, u] > 0)).sum()),
                     "Δ": float(sh[u][m][wm & (acts.a2[:, u] > 0)].mean()) if (wm & (acts.a2[:, u] > 0)).sum() >= 200 else None,
                     "changed": float((picks_dn[u] != base_pick)[wm & (acts.a2[:, u] > 0)].mean())
                     if (wm & (acts.a2[:, u] > 0)).sum() >= 200 else None}
                 for m, wm in where.items()} for u in [j] + live}
    # the down arm split by trumps held, relative to the base rate, within trick number (headroom, #226)
    d = sh[j]["leads off double"]
    rel = {}
    tn = sample.n_plays.astype(np.int64) // 4
    for name, grp in (("0-1 trumps", n_trump <= 1), ("3+ trumps", n_trump >= 3)):
        per = []
        for t_ in range(7):
            m = choice_d & on & grp & (tn == t_)
            if m.sum() >= 100:
                per.append((int(m.sum()), t_, float(-d[m].mean() / max(base["leads off double"][m].mean(), 1e-9))))
        rel[name] = per
    both_t = sorted(set(t_ for _, t_, _ in rel["0-1 trumps"]) & set(t_ for _, t_, _ in rel["3+ trumps"]))
    wsum = lambda per: (sum(n * v for n, t_, v in per if t_ in both_t) / max(sum(n for n, t_, _ in per if t_ in both_t), 1))
    steer["down, share of off-double leads removed, within trick number"] = {
        "per trick": rel, "tricks with both": both_t, "0-1 trumps": wsum(rel["0-1 trumps"]), "3+ trumps": wsum(rel["3+ trumps"])}
    steer["down by trumps held"] = {int(n): {"rows": int(m.sum()), "Δ": clustered(d[m], deal[m], rng)}
                                    for n in range(8) for m in [choice_d & on & (n_trump == n)] if m.sum() >= 200}
    if bidder is not None:
        steer["down by bidder"] = {who: {"rows": int(m.sum()), "Δ": clustered(d[m], deal[m], rng)}
                                   for who, m in (("bidder", choice_d & on & bidder), ("not bidder", choice_d & on & ~bidder))}
    dn = picks_dn[j]
    chg = choice_d & on & (dn != base_pick)
    was_d = np.isin(canon[rows_i, np.maximum(base_pick, 0)], OFF_DOUBLES)
    to = canon[rows_i, np.maximum(dn, 0)]
    steer["down, where the pick goes"] = {
        "changed share": float(chg.sum() / max((choice_d & on).sum(), 1)),
        "changed from an off double": int((chg & was_d).sum()),
        "to trump": float(np.isin(to, TRUMPS)[chg & was_d].mean()) if (chg & was_d).any() else None,
        "to the 6-5": float((to == 26)[chg & was_d].mean()) if (chg & was_d).any() else None,
        "to another off double": float(np.isin(to, OFF_DOUBLES)[chg & was_d].mean()) if (chg & was_d).any() else None,
        "to other": float((~np.isin(to, TRUMPS + OFF_DOUBLES))[chg & was_d].mean()) if (chg & was_d).any() else None}
    ch = np.flatnonzero(choice_d & on & (dn != base_pick))
    steer["examples"] = [{"row": int(r), "decl": int(sample.decl[r]),
                          "hand": [int(t) for t in np.flatnonzero(hand[r])], "legal": [int(t) for t in actions[r] if t >= 0],
                          "base": int(base_pick[r]), "ablated": int(dn[r])}
                         for r in rng.choice(ch, size=min(8, len(ch)), replace=False)] if len(ch) else []
    out["steer"] = steer

    # 4. formation
    if formation_dir:
        form = []
        for ck in sorted(Path(formation_dir).glob("step_*.npz")):
            fnet = load_net(ck)
            fa = forward(fnet, x)
            fz = fa.z2[:, j].astype(np.float64)
            tp_, tn_ = np.argsort(-fz, kind="stable")[:k], np.argsort(fz, kind="stable")[:k]
            form.append({"step": int(ck.stem.split("_")[1]), "on": float((fa.a2[:, j] > 0).mean()),
                         "on lead": float((fa.a2[lead, j] > 0).mean()), "on off lead": float((fa.a2[~lead, j] > 0).mean()),
                         "r2 lead, trumps held": r2(fz[lead], np.eye(8)[n_trump[lead]]),
                         "write corr with off doubles": corr(fnet.w3[:, j].astype(np.float64)),
                         "+": lab.best(tp_), "-": lab.best(tn_)})
            f = form[-1]
            log(f"  {ck.stem}: on lead {f['on lead']:.3f} off {f['on off lead']:.3f} r2 trumps {f['r2 lead, trumps held']:.3f} "
                f"write-doubles {f['write corr with off doubles']:+.3f} + {f['+']} − {f['-']}")
        out["formation"] = form
    return out


def trumps_on_trick(sample, trick: Trick) -> np.ndarray:
    """[N, 3] bool: whether LHO, partner, RHO has played a trump to the open trick (real tiles,
    so doubles-trump and no-trump declarations count as the rules do)."""
    n = sample.n_plays.astype(np.int64)
    rows = np.arange(len(n))
    start = n - trick.seat
    out = np.zeros((len(n), 3), bool)
    for i in range(3):
        on = trick.seat > i
        t = sample.plays[rows, np.minimum(start + i, 27), 1].astype(np.int64)
        who = (i - trick.seat) % 4 - 1
        hit = on & trick.trump(t)
        out[rows[hit], who[hit]] = True
    return out


def run_trick(sample, labels: np.ndarray, names: list[str], net_path: str, j: int, twin_path: str,
              audit: dict, formation_dir: str | None, log=print) -> dict:
    """A unit paper for a layer-1 unit shut by trumps on the open trick (#235, paper 5). The
    concept, "someone has played a trump to this trick", is not in list v2; it is built here from
    the record (`trumps_on_trick`) and split by who played it and whether trump was led."""
    x = sample.x
    fn = feature_names(x.shape[1])
    net = load_net(net_path)
    acts = forward(net, x)
    z, a = acts.z1[:, j].astype(np.float64), acts.a1[:, j]
    k = max(1, int(round(TOP * len(z))))
    top_p, top_n = np.argsort(-z, kind="stable")[:k], np.argsort(z, kind="stable")[:k]
    trick = Trick(sample)
    lab = Labeller(labels, names, x, sample.n_plays.astype(np.int64))
    tt = trumps_on_trick(sample, trick)
    T = tt.any(1)
    led_t = (trick.seat > 0) & (trick.suit == TRUMP_SUIT)
    opp = tt[:, 0] | tt[:, 2]
    follow = trick.seat > 0
    out: dict = {"unit": j, "net": net_path, "rows": len(z), "top_k": k, "on": float((a > 0).mean()),
                 "max_z": float(z.max()), "min_z": float(z.min()), "concept base rate": float(T.mean())}

    # 1. label
    state = np.select([~T, led_t, T & ~led_t], ["none", "trump led", "trumped in"], "none")
    out["cells"] = [{"seat": SEATS[s], "trick": st, "rows": int(m.sum()), "mean_z": float(z[m].mean()),
                     "on": float((a[m] > 0).mean())}
                    for s in range(4) for st in ("none", "trump led", "trumped in")
                    for m in [(trick.seat == s) & (state == st)] if m.sum() >= 200]
    out["by who"] = {name: {"rows": int(m.sum()), "mean_z": float(z[m].mean()), "on": float((a[m] > 0).mean())}
                     for name, m in (("no trump", follow & ~T), ("opponent only", follow & opp & ~tt[:, 1]),
                                     ("partner only", follow & tt[:, 1] & ~opp), ("both", follow & opp & tt[:, 1]))
                     if m.sum() >= 200}
    out["poles"] = {"+": lab.best(top_p), "-": lab.best(top_n),
                    "- precision: trump on trick": float(T[top_n].mean()), "- precision: trump led": float(led_t[top_n].mean()),
                    "+ lead share": float((trick.seat[top_p] == 0).mean())}
    # on/off, what readers see: agreement of "on" with "no trump on the trick", and with "trump not led", per seat
    on_ = a > 0
    out["on agrees"] = {SEATS[s]: {"with no trump on trick": float((on_ == ~T)[m].mean()),
                                   "with trump not led": float((on_ == ~led_t)[m].mean()), "rows": int(m.sum())}
                        for s in (1, 2, 3) for m in [trick.seat == s]}
    zf = z[follow]
    out["r2 following"] = {"trump on trick": r2(zf, T[follow][:, None]),
                           "trump led": r2(zf, led_t[follow][:, None]),
                           "who trumped (3 bits)": r2(zf, tt[follow]),
                           "who trumped + seat": r2(zf, np.c_[tt[follow], np.eye(4)[trick.seat[follow]][:, 1:]]),
                           "trump on trick + seat": r2(zf, np.c_[T[follow], np.eye(4)[trick.seat[follow]][:, 1:]]),
                           "a: trump on trick": r2(a[follow].astype(np.float64), T[follow][:, None]),
                           "a: who trumped + seat": r2(a[follow].astype(np.float64), np.c_[tt[follow], np.eye(4)[trick.seat[follow]][:, 1:]]),
                           "a: trump on trick + seat": r2(a[follow].astype(np.float64), np.c_[T[follow], np.eye(4)[trick.seat[follow]][:, 1:]]),
                           "all inputs": r2(zf, x[follow])}
    log(f"label: on {out['on']:.3f} poles {out['poles']} r2 {out['r2 following']}")
    # the vocabulary check: the best list-v2 concept (no raw inputs) on the − pole, without the baseline cap
    v2 = labels.astype(bool)
    prec = v2[top_n].mean(0)
    out["v2 best on - pole"] = [(names[i], float(prec[i]), float(v2[:, i].mean())) for i in np.argsort(-prec)[:5]]

    # 2. circuit in: largest weights by block
    w = net.w1[:, j].astype(np.float64)
    out["weights"] = [{"input": fn[f], "w": int(w[f])} for f in np.argsort(-np.abs(w))[:24]]
    out["bias"] = int(net.b1[j])

    # 3. circuit out
    reader_label = {r["name"]: r["label"] for r in audit["rows"]}
    col = net.w2[:, j]
    out["readers"] = [{"unit": int(i), "w2": int(col[i]), "labels": [reader_label.get(f"L2+{i:03d}"), reader_label.get(f"L2-{i:03d}")]}
                      for i in np.argsort(-np.abs(col))]

    # 5. steering: the trick read off lead, against the layer
    rng = np.random.default_rng(SEED)
    actions, deal = sample.actions, sample.deal
    canon = sample.canon()
    acts_ = actions.astype(np.int64)
    base_pick = picks(acts.out, actions, canon)
    el = eligible(trick, acts_)
    legal_trump = (np.where(acts_ >= 0, trick._led[trick.decl[:, None], np.maximum(acts_, 0)], -1) == TRUMP_SUIT).any(1)
    opp_win = follow & (trick.winner != 1)
    opp_trumped_win = opp_win & ~led_t & tt[np.arange(len(z)), np.maximum(trick.winner, 0)]
    multi = (acts_ >= 0).sum(1) > 1
    # prediction 4's rows: no trump on the trick yet, an opponent winning, and I can trump in
    can_ruff = follow & ~T & opp_win & legal_trump & ~led_t & multi
    where = {"smear, opponent trumped and winning": el & opp_trumped_win,
             "plays trump, trump legal and not led": el & legal_trump & ~led_t,
             "take, opponent trumped and winning": el & opp_trumped_win}
    rows = {"following, choice": int(el.sum()), **{m: int(v.sum()) for m, v in where.items()},
            "on, of opponent-trumped rows": float((a[el & opp_trumped_win] > 0).mean())}
    rows["can trump in: no trump yet, opponent winning, trump legal"] = int(can_ruff.sum())
    rows["on, of those"] = float((a[can_ruff] > 0).mean())
    out["rows"] = rows

    def reads(pick: np.ndarray) -> dict[str, np.ndarray]:
        return {"take": trick.beats(pick).astype(float), "smear": trick.count(pick).astype(float),
                "plays trump": trick.trump(pick).astype(float),
                "smear, opponent trumped and winning": trick.count(pick).astype(float),
                "plays trump, trump legal and not led": trick.trump(pick).astype(float),
                "take, opponent trumped and winning": trick.beats(pick).astype(float)}

    base = reads(base_pick)
    out["base"] = {m: float(base[m][where.get(m, el)].mean()) for m in base}
    h1 = net.b1.shape[0]
    live = [u for u in range(h1) if u != j and (acts.a1[:, u] > 0).sum() >= 200]
    sig = {u: float(acts.a1[:, u][acts.a1[:, u] > 0].std()) for u in [j] + live}
    out["sigma"] = sig[j]
    shift = lambda pick: {m: v - base[m] for m, v in reads(pick).items()}
    steer: dict = {"up": []}
    for kd in DOSES:
        sh = {u: shift(picks(downstream(net, x, 1, u, None, add=int(round(kd * sig[u]))), actions, canon)) for u in [j] + live}
        rec = paired(sh[j], [sh[u] for u in live], el, deal, rng, where=where)
        rec["k"] = kd
        steer["up"].append(rec)
        log(f"up k={kd}: " + "  ".join(f"{m} own {v['own'][0]:+.3f} layer {v['layer'][0]:+.3f}" for m, v in rec.items() if m != "k"))
    pdn = {u: picks(downstream(net, x, 1, u, 0), actions, canon) for u in [j] + live}
    sh = {u: shift(pdn[u]) for u in [j] + live}
    steer["down, each unit where it is on"] = {
        int(u): {m: {"rows": int(mm.sum()), "Δ": float(sh[u][m][mm].mean()) if mm.sum() >= 200 else None,
                     "changed": float((pdn[u] != base_pick)[mm].mean()) if mm.sum() >= 200 else None}
                 for m in ("take", "smear", "plays trump") for mm in [el & (acts.a1[:, u] > 0)]}
        for u in [j] + live}
    base_ruff = float(trick.trump(base_pick)[can_ruff].mean())
    steer["down, can trump in (every unit ablated on the same rows)"] = {
        "rows": int(can_ruff.sum()), "base plays trump": base_ruff, "pool": [j] + live,
        "units": {int(u): {"Δ plays trump": clustered(sh[u]["plays trump"][can_ruff], deal[can_ruff], rng),
                           "on there": float((acts.a1[can_ruff, u] > 0).mean())} for u in [j] + live}}
    on = a > 0
    steer["down"] = {m: clustered(sh[j][m][el & on], deal[el & on], rng) for m in ("take", "smear", "plays trump")}
    steer["down by seat"] = {SEATS[s]: {m: clustered(sh[j][m][el & on & (trick.seat == s)], deal[el & on & (trick.seat == s)], rng)
                                        for m in ("take", "smear", "plays trump")}
                             for s in (1, 2, 3) if (el & on & (trick.seat == s)).sum() >= 200}
    log(f"down: {steer['down']}")
    out["steer"] = steer

    # 4. formation
    if formation_dir:
        form = []
        for ck in sorted(Path(formation_dir).glob("step_*.npz")):
            fa = forward(load_net(ck), x)
            fz = fa.z1[:, j].astype(np.float64)
            tp_, tn_ = np.argsort(-fz, kind="stable")[:k], np.argsort(fz, kind="stable")[:k]
            ffz = fz[follow]
            form.append({"step": int(ck.stem.split("_")[1]), "on": float((fa.a1[:, j] > 0).mean()),
                         "r2 trump on trick": r2(ffz, T[follow][:, None]), "r2 trump led": r2(ffz, led_t[follow][:, None]),
                         "r2 who trumped": r2(ffz, tt[follow]),
                         "- precision: trump on trick": float(T[tn_].mean()), "+": lab.best(tp_), "-": lab.best(tn_)})
            f = form[-1]
            log(f"  {ck.stem}: on {f['on']:.3f} r2 T {f['r2 trump on trick']:.3f} led {f['r2 trump led']:.3f} who {f['r2 who trumped']:.3f} "
                f"−prec T {f['- precision: trump on trick']:.3f} + {f['+']} − {f['-']}")
        out["formation"] = form
    return out


def winning_tile(sample, trick: Trick) -> np.ndarray:
    """The real tile currently winning the open trick, −1 on lead."""
    n = sample.n_plays.astype(np.int64)
    rows = np.arange(len(n))
    start = n - trick.seat
    best = np.full(len(n), -1, np.int64)
    win = np.full(len(n), -1, np.int64)
    for i in range(3):
        ok = trick.seat > i
        t = sample.plays[rows, np.minimum(start + i, 27), 1].astype(np.int64)
        r = np.where(ok, trick._rank[trick.decl, trick.suit, np.maximum(t, 0)], -1)
        b = ok & (r > best)
        best = np.where(b, r, best)
        win = np.where(b, t, win)
    return win


def run_winner(sample, labels: np.ndarray, names: list[str], net_path: str, j: int, twin_path: str,
               audit: dict, formation_dir: str | None, log=print) -> dict:
    """A unit paper for a layer-1 unit read by who is winning the open trick and with what (#235,
    paper 6): partner, or an opponent with an off double, a trump, or another tile."""
    x = sample.x
    net = load_net(net_path)
    acts = forward(net, x)
    z, a = acts.z1[:, j].astype(np.float64), acts.a1[:, j]
    k = max(1, int(round(TOP * len(z))))
    top_p, top_n = np.argsort(-z, kind="stable")[:k], np.argsort(z, kind="stable")[:k]
    trick = Trick(sample)
    lab = Labeller(labels, names, x, sample.n_plays.astype(np.int64))
    canon = sample.canon()
    rows_i = np.arange(len(z))
    win = winning_tile(sample, trick)
    wslot = canon[rows_i, np.maximum(win, 0)]
    follow = trick.seat > 0
    partner = follow & (trick.winner == 1)
    opp = follow & (trick.winner != 1)
    w_trump = trick.trump(win)
    w_dbl = np.isin(wslot, OFF_DOUBLES) & ~w_trump
    kind = np.select([~follow, partner, opp & w_dbl, opp & w_trump], ["lead", "partner", "opp: off double", "opp: trump"],
                     "opp: other")
    out: dict = {"unit": j, "net": net_path, "rows": len(z), "top_k": k, "on": float((a > 0).mean()),
                 "max_z": float(z.max()), "min_z": float(z.min())}
    on_ = a > 0

    # 1. label
    out["cells"] = [{"seat": SEATS[s], "winning": kd, "rows": int(m.sum()), "mean_z": float(z[m].mean()),
                     "on": float(on_[m].mean())}
                    for s in range(4) for kd in ("lead", "partner", "opp: off double", "opp: trump", "opp: other")
                    for m in [(trick.seat == s) & (kind == kd)] if m.sum() >= 200]
    led_dbl = follow & np.isin(canon[rows_i, np.maximum(sample.plays[rows_i, np.maximum(sample.n_plays.astype(np.int64) - trick.seat, 0), 1].astype(np.int64), 0)], OFF_DOUBLES) & (trick.suit != TRUMP_SUIT)
    out["opp: off double, led or not"] = {name: {"rows": int(m.sum()), "on": float(on_[m].mean()), "mean_z": float(z[m].mean())}
                                          for name, m in (("double led", opp & w_dbl & led_dbl), ("double played later", opp & w_dbl & ~led_dbl))
                                          if m.sum() >= 200}
    # an opponent's off double on the trick, whoever is winning: tells "theirs" from "a double is there"
    n_ = sample.n_plays.astype(np.int64)
    opp_dbl_on = np.zeros(len(z), bool)      # an opponent's off double in the led suit
    opp_dbl_disc = np.zeros(len(z), bool)    # an opponent's off double discarded (another suit)
    for i in range(3):
        t_ = sample.plays[rows_i, np.minimum(n_ - trick.seat + i, 27), 1].astype(np.int64)
        who = (i - trick.seat) % 4 - 1
        is_d = (trick.seat > i) & (who != 1) & np.isin(canon[rows_i, np.maximum(t_, 0)], OFF_DOUBLES) & ~trick.trump(t_)
        in_led = trick._led[trick.decl, np.maximum(t_, 0)] == trick.suit
        opp_dbl_on |= is_d & in_led
        opp_dbl_disc |= is_d & ~in_led
    rng0 = np.random.default_rng(SEED)
    deal0 = sample.deal
    cell = lambda m: {"rows": int(m.sum()), "on": float(on_[m].mean()), "mean z": clustered(z[m], deal0[m], rng0)} if m.sum() >= 50 else None
    out["opp off double on trick"] = {"in the led suit, an opponent winning": cell(opp_dbl_on & opp),
                                      "in the led suit, partner winning (partner trumped it)": cell(opp_dbl_on & partner),
                                      "discarded (off suit), partner winning": cell(opp_dbl_disc & partner & ~opp_dbl_on),
                                      "discarded (off suit), an opponent winning": cell(opp_dbl_disc & opp & ~opp_dbl_on),
                                      "none, partner winning": cell(partner & ~opp_dbl_on & ~opp_dbl_disc),
                                      "none, an opponent winning": cell(opp & ~opp_dbl_on & ~opp_dbl_disc),
                                      "lead": cell(~follow)}
    out["poles"] = {"+": lab.best(top_p), "-": lab.best(top_n),
                    "+ share opp winning with off double": float((opp & w_dbl)[top_p].mean()),
                    "- share partner winning": float(partner[top_n].mean())}
    out["on agrees, following"] = {"with opponent winning": float((on_ == opp)[follow].mean()),
                                   "with opponent winning with an off double": float((on_ == (opp & w_dbl))[follow].mean())}
    af = a[follow].astype(np.float64)
    zf = z[follow]
    kind1h = np.c_[partner, opp & w_dbl, opp & w_trump][follow].astype(float)
    out["r2 following"] = {"opp winning (z)": r2(zf, opp[follow][:, None]), "winning kind (z)": r2(zf, kind1h),
                           "opp winning (a)": r2(af, opp[follow][:, None]), "winning kind (a)": r2(af, kind1h),
                           "winning kind + seat (a)": r2(af, np.c_[kind1h, np.eye(4)[trick.seat[follow]][:, 1:]])}
    log(f"label: on {out['on']:.3f} poles {out['poles']} agrees {out['on agrees, following']} r2 {out['r2 following']}")

    # 2-3. circuit in and out
    fn = feature_names(x.shape[1])
    w = net.w1[:, j].astype(np.float64)
    out["weights"] = [{"input": fn[f], "w": int(w[f])} for f in np.argsort(-np.abs(w))[:24]]
    out["bias"] = int(net.b1[j])
    reader_label = {r["name"]: r["label"] for r in audit["rows"]}
    col = net.w2[:, j]
    out["readers"] = [{"unit": int(i), "w2": int(col[i]), "labels": [reader_label.get(f"L2+{i:03d}"), reader_label.get(f"L2-{i:03d}")]}
                      for i in np.argsort(-np.abs(col))]

    # 5. steering: every live layer-1 unit ablated on the same rows; up against the layer-mean null
    rng = np.random.default_rng(SEED)
    actions, deal = sample.actions, sample.deal
    acts_ = actions.astype(np.int64)
    base_pick = picks(acts.out, actions, canon)
    multi = (acts_ >= 0).sum(1) > 1
    led_of = np.where(acts_ >= 0, trick._led[trick.decl[:, None], np.maximum(acts_, 0)], -1)
    legal_trump = (led_of == TRUMP_SUIT).any(1)
    legal_count = (np.where(acts_ >= 0, trick._count[trick.decl[:, None], np.maximum(acts_, 0)], 0) > 0).any(1)
    legal_nocount = ((acts_ >= 0) & (trick._count[trick.decl[:, None], np.maximum(acts_, 0)] == 0)).any(1)
    legal_nontrump = ((led_of >= 0) & (led_of != TRUMP_SUIT)).any(1)
    can_ruff = opp & w_dbl & legal_trump & legal_nontrump & (trick.suit != TRUMP_SUIT) & multi
    count_choice = multi & legal_count & legal_nocount
    beats_legal = np.zeros(len(z), bool)
    for kk in range(acts_.shape[1]):
        beats_legal |= (acts_[:, kk] >= 0) & trick.beats(acts_[:, kk])
    classes = {"opp double, I can trump in: plays trump": (can_ruff, "trump"),
               "opp double, count choice, nothing of mine wins: count given away": (opp & w_dbl & count_choice & ~beats_legal, "given"),
               "partner winning, count choice: count given (not overtaking)": (partner & count_choice, "given")}

    def read(pick: np.ndarray, what: str) -> np.ndarray:
        return (trick.trump(pick) if what == "trump" else trick.count(pick) & ~trick.beats(pick)).astype(float)

    out["rows"] = {c: {"rows": int(m.sum()), "base": float(read(base_pick, wt)[m].mean()), "on": float(on_[m].mean())}
                   for c, (m, wt) in classes.items()}
    out["rows"]["can trump in, by seat"] = {SEATS[s_]: {"rows": int((can_ruff & (trick.seat == s_)).sum()),
                                                        "base": float(trick.trump(base_pick)[can_ruff & (trick.seat == s_)].mean())}
                                            for s_ in (1, 2, 3) if (can_ruff & (trick.seat == s_)).sum() >= 100}
    h1 = net.b1.shape[0]
    live = [u for u in range(h1) if (acts.a1[:, u] > 0).sum() >= 200]
    sig = {u: float(acts.a1[:, u][acts.a1[:, u] > 0].std()) for u in live}
    out["sigma"] = sig[j]
    pdn = {u: picks(downstream(net, x, 1, u, 0), actions, canon) for u in live}
    steer: dict = {"down, same rows": {}}
    for c, (m, wt) in classes.items():
        b = read(base_pick, wt)
        steer["down, same rows"][c] = {int(u): {"Δ": clustered((read(pdn[u], wt) - b)[m], deal[m], rng),
                                                "on there": float((acts.a1[m, u] > 0).mean())} for u in live}
    others = [u for u in live if u != j]
    steer["up"] = []
    for kd in DOSES:
        rec = {"k": kd}
        pu = {u: picks(downstream(net, x, 1, u, None, add=int(round(kd * sig[u]))), actions, canon) for u in live}
        for c, (m, wt) in classes.items():
            b = read(base_pick, wt)
            own = read(pu[j], wt) - b
            layer = np.mean([read(pu[u], wt) - b for u in others], axis=0)
            rec[c] = {"own": clustered(own[m], deal[m], rng), "own-layer": clustered((own - layer)[m], deal[m], rng)}
        steer["up"].append(rec)
        log(f"up k={kd}: " + "  ".join(f"{c}: own {v['own'][0]:+.3f} o-l {v['own-layer'][0]:+.3f}" for c, v in rec.items() if c != "k"))
    steer["down, can trump in, by seat"] = {SEATS[s_]: clustered((trick.trump(pdn[j]).astype(float) - trick.trump(base_pick))[m], deal[m], rng)
                                            for s_ in (1, 2, 3) for m in [can_ruff & (trick.seat == s_)] if m.sum() >= 100}
    for c in classes:
        log(f"down {c}: " + " ".join(f"u{u}:{v['Δ'][0]:+.3f}" for u, v in sorted(steer['down, same rows'][c].items(), key=lambda kv: -abs(kv[1]['Δ'][0]))[:4]))
    out["steer"] = steer

    # 4. formation
    if formation_dir:
        form = []
        for ck in sorted(Path(formation_dir).glob("step_*.npz")):
            fa = forward(load_net(ck), x)
            fz, fa1 = fa.z1[:, j].astype(np.float64), fa.a1[:, j]
            tp_, tn_ = np.argsort(-fz, kind="stable")[:k], np.argsort(fz, kind="stable")[:k]
            fon = fa1 > 0
            form.append({"step": int(ck.stem.split("_")[1]), "on": float(fon.mean()),
                         "on: partner": float(fon[partner].mean()), "on: opp double": float(fon[opp & w_dbl].mean()),
                         "on: opp trump": float(fon[opp & w_trump].mean()), "on: opp other": float(fon[opp & ~w_dbl & ~w_trump].mean()),
                         "on: lead": float(fon[~follow].mean()),
                         "r2 opp winning (a)": r2(fa1[follow].astype(np.float64), opp[follow][:, None]),
                         "partner term (z: opp other − partner)": float(fz[opp & ~w_dbl & ~w_trump].mean() - fz[partner].mean()),
                         "double term (z: opp double − opp other)": float(fz[opp & w_dbl].mean() - fz[opp & ~w_dbl & ~w_trump].mean()),
                         "+": lab.best(tp_), "-": lab.best(tn_)})
            f = form[-1]
            log(f"  {ck.stem}: on {f['on']:.3f} partner {f['on: partner']:.3f} dbl {f['on: opp double']:.3f} trump {f['on: opp trump']:.3f} "
                f"other {f['on: opp other']:.3f} lead {f['on: lead']:.3f} r2 {f['r2 opp winning (a)']:.3f} + {f['+']} − {f['-']}")
        out["formation"] = form
    return out



def run_lead(sample, labels: np.ndarray, names: list[str], net_path: str, j: int, twin_path: str,
             audit: dict, formation_dir: str | None, log=print) -> dict:
    """A unit paper for a layer-1 unit on almost everywhere and shut on many leads (#235, paper 7).
    Its v2 boundary is near chance (#202), so the label is read in trumps: held, gone, and still
    out in other hands; and its use in whether the net leads trump, and what it leads instead."""
    x = sample.x
    fn = feature_names(x.shape[1])
    net = load_net(net_path)
    acts = forward(net, x)
    z, a = acts.z1[:, j].astype(np.float64), acts.a1[:, j]
    k = max(1, int(round(TOP * len(z))))
    top_p, top_n = np.argsort(-z, kind="stable")[:k], np.argsort(z, kind="stable")[:k]
    trick = Trick(sample)
    lab = Labeller(labels, names, x, sample.n_plays.astype(np.int64))
    bit = lambda f: x[:, fn.index(f)] > 0
    nt = sum(bit(f"x:hand:{s}").astype(np.int64) for s in TRUMPS)
    ng = sum(bit(f"x:gone:{s}").astype(np.int64) for s in TRUMPS)
    nout = len(TRUMPS) - nt - ng                     # trumps still in the other three hands
    nd = sum(bit(f"x:hand:{s}").astype(np.int64) for s in OFF_DOUBLES)
    boss, five = bit("x:hand:27"), bit("x:hand:26")
    side = labels[:, names.index("bidding-side")].astype(bool)
    lead, off = trick.seat == 0, a <= 0
    sixes = trick.decl == 6
    tn = sample.n_plays.astype(np.int64) // 4
    out: dict = {"unit": j, "net": net_path, "rows total": len(z), "top_k": k, "on": float((a > 0).mean()),
                 "max_z": float(z.max()), "min_z": float(z.min())}

    # 1. label
    cell = lambda m: {"rows": int(m.sum()), "off": float(off[m].mean()), "mean_z": float(z[m].mean())} if m.sum() >= 200 else None
    out["off by seat"] = [float(off[trick.seat == s].mean()) for s in range(4)]
    out["lead by trumps"] = {int(t): cell(lead & (nt == t)) for t in range(8)}
    out["lead by trumps out"] = {int(t): cell(lead & (nout == t)) for t in range(8)}
    out["lead by held × out"] = {f"held {h} out {o}": cell(lead & (np.minimum(nt, 4) == h) & (np.minimum(nout, 4) == o))
                                 for h in range(5) for o in range(5)}
    out["lead, 0-1 trumps"] = {"by trick": {int(n): cell(lead & (nt <= 1) & (tn == n)) for n in range(7)},
                               "by trumps gone": {int(g): cell(lead & (nt <= 1) & (np.minimum(ng, 6) == g)) for g in range(7)},
                               "by declaration": {int(d): cell(lead & (nt <= 1) & (trick.decl == d)) for d in range(7)},
                               "share of off-leads": float((nt[lead & off] <= 1).mean())}
    # length within controls: non-sixes, without the boss, tricks 1-2 apart from the rest
    early = tn <= 1
    out["lead by trumps, no boss, not sixes"] = {
        when: {int(t): cell(lead & ~boss & ~sixes & m & (np.minimum(nt, 4) == t)) for t in range(1, 5)}
        for when, m in (("tricks 1-2", early), ("tricks 3+", ~early))}
    out["lead by trumps × boss"] = {f"{t} {'boss' if b else 'no boss'}": cell(lead & (np.minimum(nt, 4) == t) & (boss == b))
                                   for t in range(1, 5) for b in (False, True)}
    out["lead by trumps × 6-5, no boss"] = {f"{t} {'6-5' if f else 'no 6-5'}": cell(lead & ~boss & (np.minimum(nt, 4) == t) & (five == f))
                                           for t in range(1, 5) for f in (False, True)}
    out["lead by trumps × side"] = {f"{t} {'bidding side' if b else 'defenders'}": cell(lead & (np.minimum(nt, 4) == t) & (side == b))
                                   for t in range(5) for b in (True, False)}
    out["lead by trick, 3+ trumps"] = {int(n): cell(lead & (nt >= 3) & (tn == n)) for n in range(7)}
    out["lead by declaration"] = {int(d): cell(lead & (trick.decl == d)) for d in range(7)}
    rng = np.random.default_rng(SEED)
    deal = sample.deal
    dd = lambda m1, m0: {"a": cell(m1), "b": cell(m0),
                         "a - b": float(off[m1].mean() - off[m0].mean()) if m1.sum() >= 200 and m0.sum() >= 200 else None}
    out["near misses"] = {"following, 4+ trumps, by seat": {SEATS[s]: cell((trick.seat == s) & (nt >= 4)) for s in (1, 2, 3)},
                          "lead 0-1 trumps: 3+ off doubles (a) vs none (b)": dd(lead & (nt <= 1) & (nd >= 3), lead & (nt <= 1) & (nd == 0))}
    nl = ~lead & ~sixes
    zl, ls = z[lead & ~sixes], lead & ~sixes
    oh = lambda v, n: np.eye(n)[np.minimum(v, n - 1)][:, 1:]
    ctl = np.c_[oh(tn[ls], 7), boss[ls], five[ls]]
    out["r2 on lead, not sixes"] = {
        "trick + boss + 6-5": r2(zl, ctl),
        "+ trumps held": r2(zl, np.c_[ctl, oh(nt[ls], 6)]),
        "trumps held": r2(zl, oh(nt[ls], 6)),
        "trumps out": r2(zl, oh(nout[ls], 8)),
        "trumps gone": r2(zl, oh(ng[ls], 8)),
        "held + gone": r2(zl, np.c_[oh(nt[ls], 6), oh(ng[ls], 8)]),
        "off doubles held": r2(zl, oh(nd[ls], 5))}
    on_ = a > 0
    out["on agrees on lead"] = {f"fewer than {t} trumps held": float((on_ == (nt < t))[lead].mean()) for t in (2, 3, 4)}
    out["on agrees on lead"].update({f"more than {t} trumps out": float((on_ == (nout > t))[lead].mean()) for t in (0, 1, 2)})
    out["on agrees on lead"]["chance"] = float(max(on_[lead].mean(), 1 - on_[lead].mean()))
    out["poles"] = {"+": lab.best(top_p), "-": lab.best(top_n), "- lead share": float(lead[top_n].mean()),
                    "- mean trumps held": float(nt[top_n].mean()), "- mean trumps out": float(nout[top_n].mean()),
                    "- sixes share": float(sixes[top_n].mean())}
    log(f"label: on {out['on']:.3f} off by seat {out['off by seat']} r2 {out['r2 on lead, not sixes']}")

    # 2. circuit in, 3. circuit out
    w = net.w1[:, j].astype(np.float64)
    out["weights"] = [{"input": fn[f], "w": int(w[f])} for f in np.argsort(-np.abs(w))[:24]]
    out["bias"] = int(net.b1[j])
    out["weights, trumps"] = {blk: {s: int(w[fn.index(f"x:{blk}:{s}")]) for s in TRUMPS} for blk in ("hand", "gone")}
    reader_label = {r["name"]: r["label"] for r in audit["rows"]}
    col = net.w2[:, j]
    out["readers"] = [{"unit": int(i), "w2": int(col[i]), "labels": [reader_label.get(f"L2+{i:03d}"), reader_label.get(f"L2-{i:03d}")]}
                      for i in np.argsort(-np.abs(col))]

    # 5. steering: on leads with a trump and a non-trump, whether the net leads trump, and what instead;
    # on following rows void in the led suit, with a trump and a non-trump legal, whether it trumps in
    actions = sample.actions
    canon = sample.canon()
    acts_ = actions.astype(np.int64)
    base_pick = picks(acts.out, actions, canon)
    legal = acts_ >= 0
    is_t = np.where(legal, trick._led[trick.decl[:, None], np.maximum(acts_, 0)] == TRUMP_SUIT, False)
    assert (is_t.sum(1)[lead] == nt[lead]).all(), "canonical trumps held disagree with the rules' trumps on lead"
    dbl = np.isin(np.arange(N_TILES), [p * (p + 1) // 2 + p for p in range(7)])   # real ids share the triangular order
    choice = lead & is_t.any(1) & (legal & ~is_t).any(1)
    follows_suit = np.where(legal, trick._led[trick.decl[:, None], np.maximum(acts_, 0)] == trick.suit[:, None], False)
    ruff = (trick.seat > 0) & (trick.suit != TRUMP_SUIT) & ~follows_suit.any(1) & is_t.any(1) & (legal & ~is_t).any(1)
    where = {"leads trump": choice, "leads trump, unit off": choice & off, "leads trump, unit on": choice & ~off,
             "leads off double": choice, "leads count": choice, "trumps in (following, void, use near miss)": ruff}
    def reads(pick: np.ndarray) -> dict[str, np.ndarray]:
        t = trick.trump(pick).astype(float)
        return {"leads trump": t, "leads trump, unit off": t, "leads trump, unit on": t,
                "leads off double": (dbl[np.maximum(pick, 0)] & ~trick.trump(pick)).astype(float),
                "leads count": trick.count(pick).astype(float), "trumps in (following, void, use near miss)": t}
    base = reads(base_pick)
    out["rows"] = {m: int(v.sum()) for m, v in where.items()}
    out["base"] = {m: float(base[m][v].mean()) for m, v in where.items()}
    h1 = net.b1.shape[0]
    live = [u for u in range(h1) if u != j and (acts.a1[:, u] > 0).sum() >= 200]
    sig = {u: float(acts.a1[:, u][acts.a1[:, u] > 0].std()) for u in [j] + live}
    # the gate-closed value: the unit's median activation on leads with 0-1 trumps where it is on
    med = {u: int(np.median(acts.a1[lead & (nt <= 1) & (acts.a1[:, u] > 0), u])) if (lead & (nt <= 1) & (acts.a1[:, u] > 0)).sum() >= 200
           else int(np.median(acts.a1[acts.a1[:, u] > 0, u])) for u in [j] + live}
    out["sigma"], out["median on value, 0-1 trump leads"] = sig[j], med[j]
    shift = lambda pick: {m: v - base[m] for m, v in reads(pick).items()}
    steer: dict = {"up": []}
    for kd in DOSES:
        sh = {u: shift(picks(downstream(net, x, 1, u, None, add=int(round(kd * sig[u]))), actions, canon)) for u in [j] + live}
        rec = paired(sh[j], [sh[u] for u in live], choice, deal, rng, where=where)
        rec["k"] = kd
        steer["up"].append(rec)
        log(f"up k={kd}: " + "  ".join(f"{m} own {v['own'][0]:+.3f} layer {v['layer'][0]:+.3f}" for m, v in rec.items() if m != "k"))
    pset = {u: picks(downstream(net, x, 1, u, None, add=med[u]), actions, canon) for u in [j] + live}
    sh = {u: shift(pset[u]) for u in [j] + live}
    rec = paired(sh[j], [sh[u] for u in live], choice, deal, rng, where=where)
    rec["changed, unit off choice leads"] = float((pset[j] != base_pick)[choice & off].mean())
    steer["up, add the gate-closed value"] = rec
    log("up med: " + "  ".join(f"{m} own-layer {v['own-layer'][0]:+.3f}" for m, v in rec.items() if isinstance(v, dict)))
    pdn = {u: picks(downstream(net, x, 1, u, 0), actions, canon) for u in [j] + live}
    sh = {u: shift(pdn[u]) for u in [j] + live}
    steer["down, every unit on the same rows"] = {
        m: {"rows": int(where[m].sum()), "base": out["base"][m], "pool": [j] + live,
            "units": {int(u): {"Δ": clustered(sh[u][m][where[m]], deal[where[m]], rng),
                               "on there": float((acts.a1[where[m], u] > 0).mean())} for u in [j] + live},
            "rank of own Δ, largest first": 1 + int(sum(sh[u][m][where[m]].mean() > sh[j][m][where[m]].mean() for u in live))}
        for m in ("leads trump, unit on", "leads trump", "leads off double", "leads count")}
    log("down: " + "  ".join(f"{u} {v['Δ'][0]:+.3f}" for u, v in steer["down, every unit on the same rows"]["leads trump, unit on"]["units"].items()))
    out["steer"] = steer

    # 4. formation, in mean z, each term measured from its value at initialisation:
    # the lead term (following minus lead, on 0-1 trumps held), and the trump term (leads with 0-1
    # trumps minus leads with 3+, tricks 1-2)
    if formation_dir:
        form = []
        lo_f, lo_l = ~lead & (nt <= 1), lead & (nt <= 1)
        hi_e, lo_e = lead & (nt >= 3) & early, lead & (nt <= 1) & early
        for ck in sorted(Path(formation_dir).glob("step_*.npz")):
            fa = forward(load_net(ck), x)
            fz = fa.z1[:, j].astype(np.float64)
            tp_, tn_ = np.argsort(-fz, kind="stable")[:k], np.argsort(fz, kind="stable")[:k]
            form.append({"step": int(ck.stem.split("_")[1]), "on": float((fa.a1[:, j] > 0).mean()),
                         "lead term": float(fz[lo_f].mean() - fz[lo_l].mean()),
                         "trump term": float(fz[lo_e].mean() - fz[hi_e].mean()),
                         "off on lead": float((fa.a1[lead, j] <= 0).mean()),
                         "r2 trumps held on lead": r2(fz[lead], oh(nt[lead], 6)),
                         "r2 trumps out on lead": r2(fz[lead], oh(nout[lead], 8)), "+": lab.best(tp_), "-": lab.best(tn_)})
            f = form[-1]
            log(f"  {ck.stem}: on {f['on']:.3f} lead {f['lead term']:.0f} trump {f['trump term']:.0f} "
                f"off|lead {f['off on lead']:.3f} r2 held {f['r2 trumps held on lead']:.3f} out {f['r2 trumps out on lead']:.3f} + {f['+']} − {f['-']}")
        out["formation"] = form
    return out
