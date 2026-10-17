# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The squeeze (the paper's G2 width pilot): narrow the field net, retrain it
from scratch, and ask whether its neurons get coarser or get mixed.

Cammarata ("Neural Networks Are Already Clean", Goodfire) cut a vision layer
from 480 neurons to 10 and from 1,024 to 13 and saw neurons *broaden*, one
"curves & loops" unit where there had been ten curve detectors, not unrelated
concepts sharing a unit. Superposition predicts the opposite. Under a single
fine label a broad neuron scores partial, so the clean share alone cannot tell
the two apart. The family test can: each Silas concept belongs to one family of
related concepts (mastery, count, trump held, ...); a pole is *broadened* when
the union of two concepts labels its top set better than any one concept does;
coarse neurons broaden within a family, mixed ones across families. The
statistic is the share of broadened poles whose best union is within a family,
against a null of random families of the same sizes.

    python -u -m lem.probe squeeze-train POUR [POUR ...] OUT_DIR --widths 128x64,32x16 [--seeds 0] [--epochs 60]
    python -u -m lem.probe squeeze-families SAMPLE.npz LABELS.npz AUDIT_DIR OUT.json --net NET
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from lem.probe.audit import BASE_CAP, TOP, top_set
from lem.probe.net import Activations

SEED = 20261017
NULLS = 200
GAIN = 0.05            # a union must beat the best single concept by this much to call the pole broadened

FAMILIES = {
    "position": ("lead", "second", "third", "last"),
    "time": ("trick-1", "trick-2", "late", "trick-7", "hand-le-3"),
    "side": ("bidding-side", "is-bidder", "bidders-lead"),
    "trump-held": ("holds-trump", "no-trump-held", "trumps-3plus", "holds-boss-trump", "holds-sure-trump"),
    "trump-state": ("trump-out", "trump-led", "partner-out-of-trump", "setter-out-of-trump", "can-ruff"),
    "mastery": ("holds-master", "masters-2plus", "holds-walker", "holds-off", "holds-boss-of-led", "can-take-it"),
    "trick": ("void-in-led", "partner-winning", "bidder-winning", "safe-trick", "winner-cannot-be-beaten"),
    "count": ("count-on-trick", "ten-on-trick", "holds-count", "holds-ten", "holds-five", "count-out", "count-out-20plus"),
    "score": ("made", "set", "to-make-le-10", "to-set-le-5", "to-make-ge-20"),
    "doubles": ("holds-double", "doubles-2plus"),
    "forced": ("forced",),
}


def fold_to_v2(net):
    """A field-v3 net trained on pip deals as the field-v2 net it equals there:
    the pip-kind bit is on in every pip position, so its first-layer weights
    join the bias (integers, exact) and the three kind columns go."""
    from scenario.field import OFFSET, FieldNet
    if net.layout != "field-v3":
        raise ValueError(f"fold_to_v2 takes a field-v3 net, not {net.layout}")
    k = OFFSET["kind"]
    b1 = net.b1 + net.w1[k]
    return FieldNet(net.w1[:k].copy(), b1, net.w2, net.b2, net.w3, net.b3, net.s1, net.s2, "field-v2")


def train_family(pours: list[str], out_dir: str | Path, widths: list[tuple[int, int]], seeds: list[int],
                 epochs: int, device: str = "mlx", log=print) -> list[dict]:
    """Every (width, seed) from one dataset build; each net saved as v3, folded v2, and metrics."""
    from scenario.atlas import load
    from scenario.field_train import dataset, train
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    data = dataset(*[load(p) for p in pours])
    log(f"dataset: {len(data.x):,} rows in {time.time() - t0:.0f}s")
    done = []
    for h1, h2 in widths:
        for seed in seeds:
            name = f"field_h{h1}x{h2}_s{seed}"
            if (out_dir / f"{name}.json").exists():
                log(f"{name}: already trained")
                continue
            t1 = time.time()
            t = train(data, "margin", epochs, seed, (h1, h2), device=device, log=lambda s: log(f"  {name} {s.strip()}"))
            t.net.save(out_dir / f"{name}.v3.npz")
            fold_to_v2(t.net).save(out_dir / f"{name}.npz")
            m = dict(t.metrics, name=name, pours=pours, losses=t.losses, val_losses=t.val_losses, train_seconds=round(time.time() - t1, 1))
            (out_dir / f"{name}.json").write_text(json.dumps(m, indent=1) + "\n")
            log(f"{name}: val regret {m['val'].get('regret_q', float('nan')):.4f} in {m['train_seconds']}s")
            done.append(m)
    return done


EPOCH_MARKS = (1, 2, 3, 5, 10, 20, 40, 60)
EARLY_STEPS = (0, 30, 100, 300, 1_000, 3_000, 10_000)


def formation_steps(train_rows: int, batch: int = 256, epochs: int = 60) -> tuple[int, ...]:
    """The formation test's checkpoints (#194): early optimizer steps, then epoch ends."""
    per = -(-train_rows // batch)
    return tuple(sorted(set(EARLY_STEPS) | {e * per for e in EPOCH_MARKS if e <= epochs}))


def train_formation(pours: list[str], out_dir: str | Path, widths: list[tuple[int, int]], seeds: list[int],
                    epochs: int = 60, device: str = "mlx", log=print) -> None:
    """Every (width, seed) from one dataset build, each with its in-run checkpoints saved
    (folded to field-v2 like the squeeze's nets) as `<run>/step_<n>.npz`."""
    from scenario.atlas import load
    from scenario.field_train import dataset, train
    out_dir = Path(out_dir)
    t0 = time.time()
    data = dataset(*[load(p) for p in pours])
    steps = formation_steps(int((data.deal % 10 != 0).sum()), epochs=epochs)
    log(f"dataset: {len(data.x):,} rows in {time.time() - t0:.0f}s; checkpoints at steps {steps}")
    for h1, h2 in widths:
        for seed in seeds:
            run = out_dir / f"field_h{h1}x{h2}_s{seed}"
            if (run / "run.json").exists():
                log(f"{run.name}: already trained")
                continue
            run.mkdir(parents=True, exist_ok=True)
            save = lambda step, net: (fold_to_v2(net).save(run / f"step_{step:07d}.npz"), log(f"  {run.name} checkpoint step {step:,}"))
            t1 = time.time()
            t = train(data, "margin", epochs, seed, (h1, h2), device=device, checkpoints=steps, on_checkpoint=save)
            (run / "run.json").write_text(json.dumps(dict(t.metrics, steps=list(steps), losses=t.losses, val_losses=t.val_losses,
                                                          train_seconds=round(time.time() - t1, 1)), indent=1) + "\n")
            log(f"{run.name}: val regret {t.metrics['val'].get('regret_q', float('nan')):.4f} in {time.time() - t1:.0f}s")


def union_table(top_labels: np.ndarray, pair_base_ok: np.ndarray) -> np.ndarray:
    """Precision of every concept pair's union on one top set: P(a or b) = P(a) + P(b) − P(a and b);
    pairs whose union's base rate is above the cap are −1."""
    k = len(top_labels)
    L = top_labels.astype(np.float64)
    pa = L.mean(axis=0)
    pab = (L.T @ L) / k
    u = pa[:, None] + pa[None, :] - pab
    u[~pair_base_ok] = -1.0
    np.fill_diagonal(u, -1.0)
    return u


def families(sample_labels: np.ndarray, names: list[str], acts: Activations, audit: dict, log=print) -> dict:
    """The family test over the live poles of one net (Silas concepts only)."""
    fam_of = {c: f for f, cs in FAMILIES.items() for c in cs}
    missing = [n for n in names if n not in fam_of]
    if missing:
        raise ValueError(f"concepts with no family: {missing}")
    fam = np.array([list(FAMILIES).index(fam_of[n]) for n in names])
    n, c = sample_labels.shape
    k = max(1, int(round(TOP * n)))
    L = sample_labels.astype(np.float64)
    base = L.mean(axis=0)
    base_u = base[:, None] + base[None, :] - (L.T @ L) / n
    ok = base_u <= BASE_CAP
    single_ok = base <= BASE_CAP
    poles = []
    for r in audit["rows"]:
        if r["kind"] == "dead":
            continue
        top = top_set(acts.pole(r["pole"]), k)
        tl = sample_labels[top]
        prec = tl.mean(axis=0)
        p1 = float(np.where(single_ok, prec, -1).max())
        u = union_table(tl, ok)
        i, j = np.unravel_index(int(np.argmax(u)), u.shape)
        poles.append({"pole": r["pole"], "name": r["name"], "p1": p1, "best_single": names[int(np.argmax(np.where(single_ok, prec, -1)))],
                      "p2": float(u[i, j]), "pair": [names[i], names[j]], "i": int(i), "j": int(j),
                      "broadened": bool(u[i, j] - p1 >= GAIN), "u": u})
    broad = [p for p in poles if p["broadened"]]

    def within_share(fam_labels: np.ndarray) -> float:
        if not broad:
            return float("nan")
        hits = 0
        for p in broad:
            # the best union under these families' membership: recomputed only through the pair's family test,
            # since the union precisions do not depend on the families
            hits += fam_labels[p["i"]] == fam_labels[p["j"]]
        return hits / len(broad)

    obs = within_share(fam)
    rng = np.random.default_rng(SEED)
    null = [within_share(rng.permutation(fam)) for _ in range(NULLS)]
    # also: within each broadened pole, is the best *within-family* union close to the best overall?
    for p in poles:
        same = fam[:, None] == fam[None, :]
        p["p2_within"] = float(np.where(same, p["u"], -1).max())
        p["p2_across"] = float(np.where(~same, p["u"], -1).max())
        del p["u"]
    out = {"live": len(poles), "broadened": len(broad), "within_share": obs,
           "null_mean": float(np.mean(null)), "null_lo": float(np.percentile(null, 2.5)), "null_hi": float(np.percentile(null, 97.5)),
           "nulls": NULLS, "gain": GAIN, "families": {f: list(cs) for f, cs in FAMILIES.items()}, "poles": poles}
    log(f"live {len(poles)}, broadened {len(broad)}; within-family share {obs:.3f} vs random families "
        f"{out['null_mean']:.3f} [{out['null_lo']:.3f}, {out['null_hi']:.3f}]")
    return out
