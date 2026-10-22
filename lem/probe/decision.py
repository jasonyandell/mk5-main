# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Does Gus's representation at a decision linearly encode the decision class?

Rows are the discretionary unsettled decisions of a contingent-teacher course, read by
`lem.quotient` (three-way menus occur only at roots; two-way menus at roots and second
decisions). Three targets, each per menu size:

  class     the weak order of Q over the menu slots (`lem.quotient.weak_order`; slots are
            menu tiles ranked by tile id): 13 realizable at three-way, 3 at two-way;
  contrast  the exact Q differences q[slot k] - q[slot 0], k >= 1, as floats;
  flat      every menu tile maximizes.

Features at the query token: both blocks' MLP pre-activations (`lem.probe.gus.forward`), the
residual stream after each block, and the tile logits at the menu slots, centred. The observer
does not expose its residual stream; it is recomputed here by calling the model's own submodules
in order (embeddings, then each block), and the final logits must equal `forward`'s exactly.
Baselines carry no net: the sequence length one-hot (the play count) and the legal-tile one-hot
(which tiles are on the menu).

Linear probes only: standardized ridge for contrasts, standardized L2 logistic for classes.
Each probe is fit on the course's TRAIN roots, its penalty chosen by 4-fold CV grouped by
physical root inside TRAIN, and scored once on the TEST roots. Intervals resample TEST roots
(clusters), the same resamples for every model, so differences are paired.

    python -u -m lem.probe.decision probe STUDY OUT_DIR NAME CKPT|untrained:SEED|baseline [--like CKPT]
    python -u -m lem.probe.decision summarize STUDY OUT_DIR [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from lem.quotient import load_course, weak_order

LOGISTIC_C = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
RIDGE_ALPHA = (1e-1, 1.0, 10.0, 1e2, 1e3, 1e4)
RESAMPLES = 1000
TASKS = tuple((size, target) for size in (3, 2) for target in ("class", "contrast", "flat"))


def decision_rows(study: Path, *, eligible_only=True) -> dict:
    """The course's discretionary unsettled decisions, in key order, as arrays."""
    course, _ = load_course(study, eligible_only=eligible_only)
    nodes = [course[k] for k in sorted(course) if len(course[k].legal) > 1 and not course[k].settled]
    if any(len(n.legal) > 3 for n in nodes):
        raise ValueError("menus larger than three")
    lengths = np.array([len(n.tokens) for n in nodes], np.int64)
    tokens = np.zeros((len(nodes), int(lengths.max())), np.int64)
    menu = np.full((len(nodes), 3), -1, np.int64)
    contrast = np.full((len(nodes), 2), np.nan)
    legal = np.zeros((len(nodes), 28), np.float64)
    for i, n in enumerate(nodes):
        tokens[i, :len(n.tokens)] = n.tokens
        menu[i, :len(n.legal)] = n.legal
        contrast[i, :len(n.legal) - 1] = [float(n.q[t] - n.q[n.legal[0]]) for t in n.legal[1:]]
        legal[i, list(n.legal)] = 1
    return {"tokens": tokens, "lengths": lengths, "menu": menu, "legal": legal, "contrast": contrast,
            "size": np.array([len(n.legal) for n in nodes]),
            "cls": np.array(["".join(map(str, weak_order(n))) for n in nodes]),
            "flat": np.array([len(n.best) == len(n.legal) for n in nodes]),
            "root": np.array([n.root_id for n in nodes]),
            "train": np.array([n.split == "train" for n in nodes])}


def net_features(model, rows) -> dict:
    """Pre-activations from `gus.forward`; residuals from the model's own submodules."""
    import torch
    from torch.nn import functional as F
    from lem.probe.gus import forward
    acts = forward(model, rows["tokens"], rows["lengths"])
    with torch.inference_mode():
        t = torch.from_numpy(rows["tokens"])
        x = model.token(t) + model.position(torch.arange(t.shape[1]))
        idx, last = torch.arange(len(t)), torch.from_numpy(rows["lengths"] - 1)
        residual = []
        for block in model.blocks:
            x, _ = block(x)
            residual.append(x[idx, last].float().numpy())
        logits = F.linear(model.norm(x), model.token.weight[:28])[idx, last].float().numpy()
    if not np.array_equal(logits, acts.out):
        raise ValueError("recomputed residual path does not reproduce the model's logits")
    menu = rows["menu"]
    at = np.take_along_axis(acts.out, np.maximum(menu, 0), 1)
    at = np.where(menu >= 0, at, np.nan)
    at = np.nan_to_num(at - np.nanmean(at, 1, keepdims=True))
    return {"pre1": acts.z1, "pre2": acts.z2, "res1": residual[0], "res2": residual[1], "menu_logits": at}


def baseline_features(rows) -> dict:
    lengths = rows["lengths"]
    return {"play_count": (lengths[:, None] == np.unique(lengths)[None]).astype(np.float64),
            "menu_tiles": rows["legal"]}


def task_data(rows, size, target):
    mask = rows["size"] == size
    if target == "class":
        y = rows["cls"]
    elif target == "flat":
        y = rows["flat"]
    else:
        y = rows["contrast"][:, :size - 1]
    return mask, y


def fit_probe(x, y, groups, target):
    """Penalty by grouped CV inside the fit rows; returns the refit estimator and its CV score."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if target == "contrast":
        pipe, grid, scoring = make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": RIDGE_ALPHA}, "r2"
    else:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        grid, scoring = {"logisticregression__C": LOGISTIC_C}, "balanced_accuracy"
    search = GridSearchCV(pipe, grid, cv=GroupKFold(4), scoring=scoring)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(x, y, groups=groups)
    return search.best_estimator_, float(search.best_score_), search.best_params_


def probe(study: Path, out_dir: Path, name: str, source: str, like: Path | None = None, log=print) -> dict:
    """Fit every task on every feature set of one model (or the baselines); save TEST predictions."""
    rows = decision_rows(study)
    if source == "baseline":
        feats, info = baseline_features(rows), {"baseline": True}
    else:
        from lem.probe.gus import load_model
        if source.startswith("untrained:"):
            if like is None:
                raise ValueError("an untrained twin needs --like CKPT for its architecture")
            config = load_model(like)[1]["config"]
            model, info = load_model(None, untrained_seed=int(source.split(":")[1]), config=config)
        else:
            model, info = load_model(source)
        feats = net_features(model, rows)
    saved, meta = {}, {"name": name, "model": info, "probes": {}}
    if "menu_logits" in feats:
        # The net's own choice: the menu slot with the highest logit (padding never wins).
        saved["choice"] = np.argmax(np.where(rows["menu"] >= 0, feats["menu_logits"], -np.inf), 1)
    for size, target in TASKS:
        mask, y = task_data(rows, size, target)
        tr, te = mask & rows["train"], mask & ~rows["train"]
        for fname, x in feats.items():
            est, cv, params = fit_probe(x[tr], y[tr], rows["root"][tr], target)
            key = f"{size}/{target}/{fname}"
            if target == "class":
                saved[key] = est.predict(x[te])
            elif target == "flat":
                saved[key] = est.predict_proba(x[te])[:, list(est.classes_).index(True)]
            else:
                saved[key] = est.predict(x[te]).reshape(int(te.sum()), -1)
            meta["probes"][key] = {"train_grouped_cv": cv, "penalty": {k.split("__")[1]: v for k, v in params.items()}}
            log(f"{name} {key}: train grouped CV {cv:.3f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{name}.npz", **{k.replace("/", "|"): v for k, v in saved.items()})
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


# ---- scoring with root-cluster resamples ------------------------------------

def balanced_accuracy(y, p, w):
    recalls = [np.sum(w * (p == c) * (y == c)) / np.sum(w * (y == c)) for c in np.unique(y) if np.sum(w * (y == c)) > 0]
    return float(np.mean(recalls))


def r2(y, p, w):
    w = w[:, None]
    mean = np.sum(w * y, 0) / np.sum(w)
    return float(np.mean(1 - np.sum(w * (y - p) ** 2, 0) / np.sum(w * (y - mean) ** 2, 0)))


def auc(y, s, w):
    """Weighted P(score of a flat row > score of a nonflat row), ties counted half."""
    order = np.argsort(s, kind="stable")
    s, y, w = s[order], y[order], w[order]
    pos, neg = w * y, w * ~y
    _, start = np.unique(s, return_index=True)
    bounds = np.append(start, len(s))
    below, total = 0.0, 0.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        p, n = pos[a:b].sum(), neg[a:b].sum()
        total += p * (below + n / 2)
        below += n
    return float(total / (pos.sum() * neg.sum()))


METRICS = {"class": {"balanced_accuracy": balanced_accuracy, "accuracy": lambda y, p, w: float(np.sum(w * (y == p)) / np.sum(w))},
           "contrast": {"r2": r2}, "flat": {"auc": auc}}


def summarize(study: Path, out_dir: Path) -> dict:
    rows = decision_rows(study)
    names = sorted(p.stem for p in out_dir.glob("*.npz"))
    preds = {n: dict(np.load(out_dir / f"{n}.npz")) for n in names}
    choices = {n: preds[n].pop("choice") for n in names if "choice" in preds[n]}
    metas = {n: json.loads((out_dir / f"{n}.json").read_text()) for n in names}
    rng = np.random.default_rng(20261021)
    report = {"study": str(study), "resamples": RESAMPLES, "tasks": {}}
    for size, target in TASKS:
        mask, y = task_data(rows, size, target)
        tr, te = mask & rows["train"], mask & ~rows["train"]
        y_tr, y_te, roots = y[tr], y[te], rows["root"][te]
        ids, inverse = np.unique(roots, return_inverse=True)
        draws = rng.integers(0, len(ids), (RESAMPLES, len(ids)))
        weights = np.stack([np.bincount(d, minlength=len(ids))[inverse].astype(float) for d in draws])
        one = np.ones(len(y_te))
        if target == "class":
            values, counts = np.unique(y_tr, return_counts=True)
            chance = np.full(len(y_te), values[np.argmax(counts)])
        elif target == "flat":
            chance = np.full(len(y_te), float(np.mean(y_tr)))
        else:
            chance = np.broadcast_to(y_tr.mean(0), y_te.shape)
        entry = {"train": {"rows": int(tr.sum()), "roots": int(len(np.unique(rows["root"][tr])))},
                 "test": {"rows": int(te.sum()), "roots": int(len(ids))},
                 "test_label_counts": ({str(k): int(v) for k, v in zip(*np.unique(y_te, return_counts=True))}
                                       if target != "contrast" else None),
                 "metrics": {}}
        for metric, fn in METRICS[target].items():
            def score(p):
                return fn(y_te, p, one), np.array([fn(y_te, p, w) for w in weights])
            point, boot = score(chance)
            table = {"chance": {"value": point, "ci95": list(np.percentile(boot, [2.5, 97.5]))}}
            series = {"chance": boot}
            for n in names:
                for key, p in preds[n].items():
                    s, t, fname = key.split("|")
                    if (int(s), t) != (size, target):
                        continue
                    point, boot = score(p)
                    label = f"{n}/{fname}"
                    series[label] = boot
                    table[label] = {"value": point, "ci95": list(np.percentile(boot, [2.5, 97.5])),
                                    "train_grouped_cv": metas[n]["probes"][f"{size}/{target}/{fname}"]["train_grouped_cv"]}
            twins = sorted({k.split("/")[0] for k in series if k.startswith("untrained")})
            for fname in {k.split("/")[1] for k in series if "/" in k}:
                runs = [f"{t}/{fname}" for t in twins if f"{t}/{fname}" in series]
                if runs:
                    series[f"untrained_mean/{fname}"] = np.mean([series[r] for r in runs], 0)
                    table[f"untrained_mean/{fname}"] = {"value": float(np.mean([table[r]["value"] for r in runs])),
                                                         "ci95": list(np.percentile(series[f"untrained_mean/{fname}"], [2.5, 97.5]))}
            pairs = {}
            for a, b in (("wide", "parent"), ("narrow", "parent"), ("wide", "narrow"), ("parent", "untrained_mean"),
                         ("wide", "untrained_mean"), ("narrow", "untrained_mean")):
                for fname in ("pre1", "pre2", "res1", "res2", "menu_logits"):
                    if f"{a}/{fname}" in series and f"{b}/{fname}" in series:
                        d = series[f"{a}/{fname}"] - series[f"{b}/{fname}"]
                        pairs[f"{a}-{b}/{fname}"] = {"value": table[f"{a}/{fname}"]["value"] - table[f"{b}/{fname}"]["value"],
                                                     "ci95": list(np.percentile(d, [2.5, 97.5]))}
            entry["metrics"][metric] = {"table": table, "paired": pairs}
        report["tasks"][f"{size}/{target}"] = entry
    report["policy_agreement"] = policy_agreement(rows, choices, rng)
    return report


def policy_agreement(rows, choices, rng) -> dict:
    """At nonflat decisions, the share where the net's own argmax lies in the maximizing set.
    Chance is a uniform pick from the menu: the mean of |best| / |menu|."""
    best = np.array([[c[k] == "0" if k < len(c) else False for k in range(3)] for c in rows["cls"]])
    out = {}
    for size in (3, 2):
        for split, part in (("test", ~rows["train"]), ("train", rows["train"])):
            m = (rows["size"] == size) & ~rows["flat"] & part
            ids, inverse = np.unique(rows["root"][m], return_inverse=True)
            weights = np.stack([np.bincount(d, minlength=len(ids))[inverse].astype(float)
                                for d in rng.integers(0, len(ids), (RESAMPLES, len(ids)))])
            hit = {n: best[m][np.arange(m.sum()), c[m]].astype(float) for n, c in choices.items()}
            hit["chance"] = best[m].sum(1) / size
            boot = {n: weights @ h / weights.sum(1) for n, h in hit.items()}
            twins = [n for n in hit if n.startswith("untrained")]
            if twins:
                hit["untrained_mean"] = np.mean([hit[t] for t in twins], 0)
                boot["untrained_mean"] = np.mean([boot[t] for t in twins], 0)
            table = {n: {"value": float(h.mean()), "ci95": list(np.percentile(boot[n], [2.5, 97.5]))} for n, h in hit.items()}
            pairs = {f"{a}-{b}": {"value": float(hit[a].mean() - hit[b].mean()),
                                  "ci95": list(np.percentile(boot[a] - boot[b], [2.5, 97.5]))}
                     for a, b in (("wide", "parent"), ("narrow", "parent"), ("wide", "narrow"),
                                  ("parent", "chance"), ("wide", "chance"), ("parent", "untrained_mean"))
                     if a in hit and b in hit}
            out[f"{size}/{split}"] = {"rows": int(m.sum()), "roots": int(len(ids)), "table": table, "paired": pairs}
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe")
    p.add_argument("study", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("name")
    p.add_argument("source", help="a checkpoint path, untrained:SEED, or baseline")
    p.add_argument("--like", type=Path, help="the checkpoint whose architecture an untrained twin copies")
    s = sub.add_parser("summarize")
    s.add_argument("study", type=Path)
    s.add_argument("out_dir", type=Path)
    s.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.command == "probe":
        probe(args.study, args.out_dir, args.name, args.source, args.like)
    else:
        report = summarize(args.study, args.out_dir)
        text = json.dumps(report, indent=1, default=float)
        if args.out:
            args.out.write_text(text)
        print(text)


if __name__ == "__main__":
    main()
