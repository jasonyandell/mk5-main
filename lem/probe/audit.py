# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The pole table: what each pole's top positions have in common.

For every pole, the top `TOP` share of rows by its signal is its *top set*.
A concept's *precision* on a pole is the share of the top set where the
concept holds; its *base rate* is its share of all rows; its *lift* is the
ratio. The pole's label is the concept of highest precision among those
whose base rate is at most `BASE_CAP` (a concept true nearly everywhere
explains nothing). A pole is *clean* when its label's precision is at least
`CLEAN`, *partial* at `PARTIAL` or above, *mixed* below that, and *dead*
when its neuron is on (`+`) or off (`−`) in fewer than `DEAD` of rows.

Concepts are the Silas list (`concepts.py`) and, as the trivial baseline,
every raw input feature of the net (a pole that is one feature's detector
is clean in the least interesting way; the table says which).

Intervals: each precision carries its Wilson 95% interval over the top set;
the share of clean poles is reported with the shares obtained when every
pole's precision is replaced by its lower, then its upper, bound. Chance:
the labels of every concept are shuffled across rows (`SHUFFLES` fixed-seed
permutations) and the whole procedure re-run; the shuffled clean share is
the baseline every headline share is read against.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from lem.probe.net import Activations, pole_name
from lem.probe.sample import Sample
from scenario.field import BLOCKS, OFFSET

TOP = 0.01
BASE_CAP = 0.5
CLEAN = 0.9
PARTIAL = 0.7
DEAD = 0.001
SHUFFLES = 20
SEED = 20261015


def feature_names() -> list[str]:
    names = []
    for block, width in BLOCKS:
        for i in range(width):
            names.append(f"x:{block}:{i}")
    return names


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass(frozen=True)
class PoleRow:
    pole: int
    name: str
    alive: float
    kind: str                 # clean | partial | mixed | dead
    label: str
    precision: float
    lo: float
    hi: float
    base: float
    lift: float
    label_r: str              # the concept of highest |point-biserial r| over all rows
    r: float
    runner_up: str
    runner_precision: float
    top: list[int]            # the top set's row indices, highest signal first


def top_set(signal: np.ndarray, k: int) -> np.ndarray:
    """The k rows of highest signal; ties broken by row order (stable)."""
    order = np.argsort(-signal, kind="stable")
    return order[:k]


def score_pole(signal: np.ndarray, labels: np.ndarray, base: np.ndarray, k: int):
    """precision per concept on the top set, and the point-biserial r per concept."""
    top = top_set(signal, k)
    prec = labels[top].mean(axis=0)
    s = signal.astype(np.float64)
    s = (s - s.mean()) / (s.std() + 1e-12)
    r = (labels.astype(np.float64).T @ s) / len(s)
    r = r / np.sqrt(base * (1 - base) + 1e-12)
    return top, prec, r


def classify(prec: np.ndarray, base: np.ndarray, names: list[str], k: int):
    ok = base <= BASE_CAP
    cand = np.where(ok, prec, -1.0)
    best = int(np.argmax(cand))
    order = np.argsort(-cand)
    runner = int(order[1]) if len(order) > 1 else best
    p = float(cand[best])
    lo, hi = wilson(int(round(p * k)), k)
    kind = "clean" if p >= CLEAN else "partial" if p >= PARTIAL else "mixed"
    return best, p, lo, hi, kind, runner, float(cand[runner])


def run(sample: Sample, acts: Activations, labels: np.ndarray, concept_names: list[str],
        log=print) -> dict:
    x = sample.x.astype(bool)
    allc = np.concatenate([labels, x], axis=1)
    names = list(concept_names) + feature_names()
    n = len(allc)
    k = max(1, int(round(TOP * n)))
    base = allc.mean(axis=0)
    rows: list[PoleRow] = []
    for p in range(acts.n_poles):
        alive = acts.alive(p)
        signal = acts.pole(p)
        top, prec, r = score_pole(signal, allc, base, k)
        best, pr, lo, hi, kind, runner, rp = classify(prec, base, names, k)
        if alive < DEAD:
            kind = "dead"
        jr = int(np.argmax(np.abs(r)))
        rows.append(PoleRow(p, pole_name_of(acts, p), alive, kind, names[best], pr, lo, hi,
                            float(base[best]), float(pr / max(base[best], 1e-9)), names[jr], float(r[jr]),
                            names[runner], rp, top[:64].tolist()))
        if p % 32 == 0 and log is not None:
            log(f"  pole {p:3d} {rows[-1].name}: {kind} {names[best]} p={pr:.3f} [{lo:.3f}, {hi:.3f}] base={base[best]:.3f}")
    shares = summarize(rows)
    # the shuffled baseline: every concept's column permuted independently, SHUFFLES times
    rng = np.random.default_rng(SEED)
    shuffled = []
    for s in range(SHUFFLES):
        perm = np.empty_like(allc)
        for c in range(allc.shape[1]):
            perm[:, c] = allc[rng.permutation(n), c]
        kinds = []
        for p in range(acts.n_poles):
            if acts.alive(p) < DEAD:
                kinds.append("dead")
                continue
            top, prec, _ = score_pole(acts.pole(p), perm, base, k)
            kinds.append(classify(prec, base, names, k)[4])
        shuffled.append(kinds)
        if log is not None:
            log(f"  shuffle {s + 1}/{SHUFFLES}: clean {kinds.count('clean')} partial {kinds.count('partial')}")
    live = [r for r in rows if r.kind != "dead"]
    sh_clean = [ks.count("clean") / max(1, len([x for x in ks if x != "dead"])) for ks in shuffled]
    return {
        "rows": [asdict(r) for r in rows],
        "n_rows": n, "top_k": k, "n_poles": acts.n_poles, "n_live": len(live),
        "shares": shares,
        "shuffled": {"clean_share_mean": float(np.mean(sh_clean)), "clean_share_min": float(np.min(sh_clean)),
                     "clean_share_max": float(np.max(sh_clean)), "shuffles": SHUFFLES},
        "params": {"TOP": TOP, "BASE_CAP": BASE_CAP, "CLEAN": CLEAN, "PARTIAL": PARTIAL, "DEAD": DEAD, "SEED": SEED},
    }


def pole_name_of(acts: Activations, p: int) -> str:
    h1 = acts.z1.shape[1]
    layer, j = (1, p // 2) if p < 2 * h1 else (2, (p - 2 * h1) // 2)
    return f"L{layer}{'+' if p % 2 == 0 else '-'}{j:03d}"


def summarize(rows: list[PoleRow]) -> dict:
    live = [r for r in rows if r.kind != "dead"]
    n = max(1, len(live))
    out = {"live": len(live), "dead": len(rows) - len(live)}
    for kind in ("clean", "partial", "mixed"):
        out[kind] = sum(r.kind == kind for r in live)
    out["clean_share"] = out["clean"] / n
    out["clean_share_lo"] = sum(r.lo >= CLEAN for r in live) / n
    out["clean_share_hi"] = sum(r.hi >= CLEAN for r in live) / n
    silas = [r for r in live if r.kind == "clean" and not r.label.startswith("x:")]
    out["clean_by_silas"] = len(silas)
    out["clean_by_feature"] = out["clean"] - len(silas)
    for layer in (1, 2):
        rs = [r for r in live if r.name.startswith(f"L{layer}")]
        out[f"L{layer}_live"] = len(rs)
        out[f"L{layer}_clean"] = sum(r.kind == "clean" for r in rs)
    return out


def write(result: dict, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(result, indent=1))
    lines = ["| pole | alive | kind | label | precision [95%] | base | lift | runner-up | top r |", "|---|---:|---|---|---|---:|---:|---|---|"]
    for r in result["rows"]:
        lines.append(f"| {r['name']} | {r['alive']:.3f} | {r['kind']} | {r['label']} | {r['precision']:.3f} "
                     f"[{r['lo']:.3f}, {r['hi']:.3f}] | {r['base']:.3f} | {r['lift']:.1f} | {r['runner_up']} "
                     f"({r['runner_precision']:.2f}) | {r['label_r']} ({r['r']:+.2f}) |")
    s = result["shares"]
    sh = result["shuffled"]
    head = [f"Rows {result['n_rows']:,}, top set {result['top_k']:,} per pole, {result['n_poles']} poles, {s['live']} live.",
            f"Clean {s['clean']} of {s['live']} live poles = {s['clean_share']:.3f} [{s['clean_share_lo']:.3f}, {s['clean_share_hi']:.3f}] "
            f"(by a Silas concept {s['clean_by_silas']}, by a raw feature {s['clean_by_feature']}); partial {s['partial']}, mixed {s['mixed']}.",
            f"Shuffled-label baseline: clean share {sh['clean_share_mean']:.3f} [{sh['clean_share_min']:.3f}, {sh['clean_share_max']:.3f}] over {sh['shuffles']} shuffles.",
            f"Layer 1: {s['L1_clean']} of {s['L1_live']} live clean; layer 2: {s['L2_clean']} of {s['L2_live']}.", ""]
    (out_dir / "poles.md").write_text("\n".join(head + lines) + "\n")
