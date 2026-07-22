"""roles/probes.py — the brainstorm's two measurable claims, measured.

Probe 1 (alphabet): how coarse is the role alphabet per decl column, and
how coarse is a HAND's role projection at the auction horizon? This is
the compression-vs-horizon question the equivalence census left open (it
measured H4, where distinctions are maximally live).

Probe 2 (factorization): do hoyt's banked H4 reference values factor
through the 10-feature role basis? Compares three nested bases under
ridge regression with k-fold CV on the frozen 200-root line:
  scalars  — public context only (banked points, me-declares, count out)
  roles    — scalars + the role basis (no pip identity anywhere)
  pips     — scalars + 28-dim hand one-hot (full pip identity)
If roles ~ pips >> scalars, the value function factors through roles at
H4 and the brainstorm's compression is functional, not stateful.

    python -u -m roles.probes            # both probes, receipts to stdout
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forge.oracle.declarations import DECL_ID_TO_NAME
from walt.tables import N_DOMINOES, get_luts, hand_to_mask
from roles.threat import (
    ALL_TILES, GAME_DECL_IDS, hand_features, threat_tensor,
)

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
#  probe 1 — the role alphabet                                                 #
# --------------------------------------------------------------------------- #

def _behavior_key(decl: int) -> np.ndarray:
    """int64 [28]: a tile's FULL behavioral identity under decl — its
    trick rank under every led suit plus its count. Two tiles with equal
    keys are exactly interchangeable in any position under that decl
    (rank encodes tier: trump/follow/off; rank 0 offs differ only by
    count). Strictly finer than the lead-power view."""
    luts = get_luts(decl)
    key = luts.count.astype(np.int64).copy()
    for ls in range(8):
        key = key * 64 + luts.rank[ls].astype(np.int64)
    return key


def probe_alphabet() -> dict:
    out = {"per_decl": {}, }
    for decl in GAME_DECL_IDS:
        key = _behavior_key(decl)
        classes: dict = {}
        for d, r in enumerate(key.tolist()):
            classes.setdefault(r, []).append(d)
        sizes = sorted((len(v) for v in classes.values()), reverse=True)
        out["per_decl"][DECL_ID_TO_NAME[decl]] = {
            "distinct_roles": len(classes),
            "class_sizes": sizes,
            "tiles_sharing_a_role": int(sum(s for s in sizes if s > 1)),
        }
    # hand-level coarseness at the auction horizon: random 7-tile hands,
    # per decl column — how often do >=2 held tiles share a full
    # behavioral role (exact interchangeability, cold)?
    rng = np.random.default_rng(0)
    B = 20_000
    member = np.zeros((B, N_DOMINOES), dtype=bool)
    for b in range(B):
        member[b, rng.choice(N_DOMINOES, 7, replace=False)] = True
    share = {}
    for decl in GAME_DECL_IDS:
        key = _behavior_key(decl)
        n_dup = np.array([7 - len(np.unique(key[member[b]]))
                          for b in range(B)], dtype=np.int64)
        share[DECL_ID_TO_NAME[decl]] = {
            "mean_within_hand_role_dups": float(n_dup.mean()),
            "p_hand_has_dup": float((n_dup > 0).mean()),
        }
    out["auction_hand_coarseness"] = share
    return out


# --------------------------------------------------------------------------- #
#  probe 2 — does the reference value factor through roles?                    #
# --------------------------------------------------------------------------- #

def _root_state(rec: dict) -> tuple[int, int, int, dict]:
    rd = rec["root"]
    hand = int(hand_to_mask(tuple(rd["my_hand"])))
    played = 0
    for _s, d in rd["play_history"]:
        played |= 1 << int(d)
    for d in rd["current_trick"]:
        played |= 1 << int(d)
    out = int(ALL_TILES) & ~hand & ~played
    return hand, out, int(rd["decl_id"]), rd


def _bases(recs: list[dict]) -> dict[str, np.ndarray]:
    scal, role, pip = [], [], []
    for rec in recs:
        hand, out, decl, rd = _root_state(rec)
        bid_team = int(rd["bidder"]) % 2
        banked = float(rd["team_points"][bid_team])
        banked_def = float(rd["team_points"][1 - bid_team])
        me_decl = float(int(rd["me"]) % 2 == bid_team)
        luts = get_luts(decl)
        count_out = float(luts.count[
            np.flatnonzero((out >> np.arange(28)) & 1)].sum())
        s = [banked, banked_def, me_decl, count_out, float(rd["bid_value"])]
        scal.append(s)
        role.append(s + hand_features(hand, out, decl).tolist())
        pip.append(s + [(hand >> d) & 1 for d in range(N_DOMINOES)])
    return {"scalars": np.array(scal), "roles": np.array(role),
            "pips": np.array(pip, dtype=float)}


def _ridge_cv(X: np.ndarray, y: np.ndarray, folds: int = 5,
              lams=(0.01, 0.1, 1.0, 10.0)) -> dict:
    n = len(y)
    rng = np.random.default_rng(42)
    order = rng.permutation(n)
    best = None
    for lam in lams:
        pred = np.zeros(n)
        for f in range(folds):
            te = order[f::folds]
            tr = np.setdiff1d(order, te)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
            ym = y[tr].mean()
            A = Xt.T @ Xt + lam * np.eye(X.shape[1])
            w = np.linalg.solve(A, Xt.T @ (y[tr] - ym))
            pred[te] = Xe @ w + ym
        mae = float(np.abs(pred - y).mean())
        r2 = float(1 - np.sum((pred - y) ** 2)
                   / np.sum((y - y.mean()) ** 2))
        if best is None or mae < best["mae"]:
            best = {"lam": lam, "mae": mae, "r2": r2}
    return best


def probe_factorization() -> dict:
    evalset = {r["seed"]: r for r in map(
        json.loads, open(REPO / "hoyt/evalset_h4_v1.jsonl"))}
    refs = [json.loads(l) for l in open(
        REPO / "hoyt/reference_h4_v1_cap256.jsonl")]
    recs = [evalset[r["seed"]] for r in refs]
    y_ref = np.array([r["cfr_reference_value"] for r in refs])
    y_walt = np.array([r["walt_vs_jud_value"] for r in refs])
    X = _bases(recs)
    out = {}
    for tname, y in (("cfr_reference_value", y_ref),
                     ("walt_vs_jud_value", y_walt)):
        out[tname] = {
            "target_sd": float(y.std()),
            **{name: _ridge_cv(Xb, y) for name, Xb in X.items()},
        }
    return out


# --------------------------------------------------------------------------- #
#  probe 3 — factorization with teeth: mass-generated exact H4 values          #
# --------------------------------------------------------------------------- #

def _gen_valued_roots(n: int, world_cap: int = 128,
                      seed0: int = 910_000) -> list[dict]:
    """n fresh H4 roots with EXACT best-response values vs the
    lowest-legal deterministic field — a well-defined, net-free game
    functional (not the CFR reference; any exact value functional tests
    factorization equally). ~sub-second per root, all hoyt machinery."""
    import hoyt as K
    from hoyt.toys import gen_engine_root, payoff_points

    pay = payoff_points()

    def lowest(seat, hand_mask, legal_mask, decl):
        return int(legal_mask & -legal_mask).bit_length() - 1

    rows, s = [], seed0
    while len(rows) < n:
        s += 1
        try:
            root, worlds = gen_engine_root(s, 4)
        except ValueError:
            continue
        if len(worlds) > world_cap:
            idx = np.random.default_rng(s).choice(
                len(worlds), world_cap, replace=False)
            worlds = worlds[np.sort(idx)]
        try:
            tab = K.compile_rule_sigma(root, worlds, lowest)
            sub = K.build_subgame(root, worlds,
                                  np.full(len(worlds), 1.0 / len(worlds)))
            br = K.br_solve(sub, tab, pay, want_strategy=False)
        except Exception:
            continue
        sign = 1.0 if root.me % 2 == root.bidder % 2 else -1.0
        rv = {int(m): float(v) for m, v in br.root_values.items()}
        hero = {m: sign * v for m, v in rv.items()}   # hero orientation
        rows.append({"root": {
            "decl_id": root.decl_id, "bidder": root.bidder,
            "bid_value": root.bid_value, "me": root.me,
            "my_hand": list(root.my_hand),
            "play_history": [list(x) for x in root.play_history],
            "current_trick": list(root.current_trick),
            "team_points": list(root.team_points)},
            "value": float(br.value),
            "best_move": int(br.best_move),
            "spread": float(max(hero.values()) - min(hero.values())),
            "root_values": rv})
    return rows


def probe_factor_mass(n: int = 4000) -> dict:
    cache = REPO / "scratch/roles_h4_brvals.json"
    if cache.exists():
        rows = json.loads(cache.read_text())[:n]
    else:
        rows = []
    if len(rows) < n:
        rows = _gen_valued_roots(n)
        cache.write_text(json.dumps(rows))
    y = np.array([r["value"] for r in rows])
    X = _bases(rows)
    res = {"n": len(rows), "target_sd": float(y.std()),
           **{name: _ridge_cv(Xb, y) for name, Xb in X.items()}}
    # paired bootstrap on CV-MAE deltas: does each basis beat scalars,
    # and does pips beat roles (i.e. does pip identity add anything)?
    res["bootstrap"] = _boot_deltas(X, y)
    # hand-sensitive target: the root-move value SPREAD (max-min over my
    # legal leads, hero orientation) — how much my choice matters. Public
    # context can't see this; it lives in hand structure.
    ysp = np.array([r["spread"] for r in rows])
    res["spread"] = {"target_sd": float(ysp.std()),
                     **{name: _ridge_cv(Xb, ysp)
                        for name, Xb in X.items()},
                     "bootstrap": _boot_deltas(X, ysp)}
    # policy diagnostics: is the exact best lead described by its role?
    from roles.threat import threat_counts, walker_mask
    n_wexists = n_wbest = n_minthr = 0
    chance = 0.0
    for r in rows:
        hand, out, decl, _rd = _root_state(r)
        best = r["best_move"]
        wm = walker_mask(hand, out, decl)
        if wm:
            n_wexists += 1
            n_wbest += (wm >> best) & 1
        tiles = np.flatnonzero((hand >> np.arange(28)) & 1)
        tc = threat_counts(hand, out, decl)
        n_minthr += int(tc[list(tiles).index(best)] == tc.min())
        chance += float(np.mean(tc == tc.min()))   # random-lead baseline
    res["policy"] = {
        "p_best_is_walker_given_walker": n_wbest / max(n_wexists, 1),
        "n_walker_roots": n_wexists,
        "p_best_is_min_threat": n_minthr / len(rows),
        "chance_min_threat": chance / len(rows),
    }
    return res


def _cv_abs_errors(X, y, lam=1.0, folds=5):
    n = len(y)
    order = np.random.default_rng(42).permutation(n)
    pred = np.zeros(n)
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xt, Xe = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean()
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]),
                            Xt.T @ (y[tr] - ym))
        pred[te] = Xe @ w + ym
    return np.abs(pred - y)


def _boot_deltas(X, y, B=2000):
    errs = {name: _cv_abs_errors(Xb, y) for name, Xb in X.items()}
    rng = np.random.default_rng(7)
    n = len(y)
    out = {}
    for a, b in (("roles", "scalars"), ("pips", "scalars"),
                 ("roles", "pips")):
        d = errs[a] - errs[b]          # negative = a better
        idx = rng.integers(0, n, (B, n))
        boots = d[idx].mean(1)
        out[f"{a}_minus_{b}"] = {
            "mean": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))]}
    return out


def main() -> int:
    a = probe_alphabet()
    print("== probe 1: role alphabet ==")
    for name, row in a["per_decl"].items():
        print(f"  {name:14s} distinct {row['distinct_roles']:2d}/28  "
              f"sharing {row['tiles_sharing_a_role']:2d}  "
              f"sizes {row['class_sizes'][:6]}")
    print("  auction-horizon hand coarseness (20k random hands):")
    for name, row in a["auction_hand_coarseness"].items():
        print(f"  {name:14s} mean within-hand role dups "
              f"{row['mean_within_hand_role_dups']:.2f}  "
              f"P(any dup) {row['p_hand_has_dup']:.2f}")
    f = probe_factorization()
    print("\n== probe 2: factorization on the frozen 200-root line ==")
    for tname, row in f.items():
        print(f"  target {tname} (sd {row['target_sd']:.2f} pts):")
        for basis in ("scalars", "roles", "pips"):
            b = row[basis]
            print(f"    {basis:8s} CV-MAE {b['mae']:.3f} pts  "
                  f"R2 {b['r2']:.3f}  (lam {b['lam']})")
    import sys
    n3 = 4000 if "--full" in sys.argv else int(
        next((a2.split("=")[1] for a2 in sys.argv if a2.startswith("--n=")),
             200))
    m = probe_factor_mass(n3)
    print(f"\n== probe 3: factorization with teeth "
          f"(n={m['n']} exact BR-vs-lowest-legal H4 values, "
          f"sd {m['target_sd']:.2f} pts) ==")
    for basis in ("scalars", "roles", "pips"):
        b = m[basis]
        print(f"    {basis:8s} CV-MAE {b['mae']:.3f} pts  "
              f"R2 {b['r2']:.3f}  (lam {b['lam']})")
    for k, row in m["bootstrap"].items():
        lo, hi = row["ci95"]
        verdict = "a wins" if hi < 0 else ("b wins" if lo > 0 else "tie")
        print(f"    {k:22s} dMAE {row['mean']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]  {verdict}")
    rec = {"alphabet": a, "factorization": f, "mass": m}
    outp = REPO / "scratch/roles_probes.json"
    outp.parent.mkdir(exist_ok=True)
    outp.write_text(json.dumps(rec, indent=1))
    print(f"\nreceipts -> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
