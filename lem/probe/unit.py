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
    pip = double_pip(tile)
    suit = pip_tiles(pip)
    held_any = lambda slots: x[:, [fn.index(f"x:hand:{t}") for t in slots]].any(1)
    gate_gap = float(z[~g].mean() - z[g].mean())
    near = {}
    for name, slots in [(f"double {p}-{p}", [p * (p + 1) // 2 + p]) for p in range(6) if p != pip] + \
                       [(f"a {pip}-x (not the double)", suit)]:
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
    suit_real = real_of_slot[:, suit]
    legal_suit = (actions[:, :, None] == suit_real[:, None, :]).any((1, 2))
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
                   "base plays-its-suit there": float((base_pick[:, None] == suit_real).any(1)[suit_lead_rows].mean())}
    sig = {u: float(acts.a1[:, u][acts.a1[:, u] > 0].std()) for u in [j] + live}
    steer: dict = {"up": []}
    for kd in DOSES:
        sh = {u: shifts(trick, base_pick, picks(downstream(net, x, 1, u, None, add=int(round(kd * sig[u]))), actions, canon),
                        gated_real, suit_real) for u in [j] + live}
        rec = paired(sh[j], [sh[u] for u in live], off_rows, deal, rng,
                     where={"plays-gated": gated_rows, "plays-its-suit": suit_lead_rows})
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
