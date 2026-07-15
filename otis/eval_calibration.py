"""otis/eval_calibration.py — P2 fate calibration + pricing reliability.

P2 band (wiki/experiments/otis-v0.md): held-out per-tile fate NLL beats the
marginal base rate by ≥ 0.15 nats with a CI excluding zero, and top-1 accuracy
by ≥ 8 pp. Base rates are recomputed on TRAIN ONLY. The CI is a CLUSTER bootstrap
over deal groups ``(seed, hand_idx)`` (1000 resamples) — paired ``a_team`` halves
share a deal/auction, so the cluster is the deal, not the row (fable W2b gate).

Also reports pricing-head val NLL (= CE) + P(pts≥30) reliability/ECE for BOTH arms.

The fate metrics require fate heads, so they are computed on the TREATMENT model
only; the base rate is the shared no-features baseline both a control-with-a-
fate-probe and treatment must beat.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from champion.margin_net import MIN_BID, _reliability, exceedance
from otis.data import build_split
from otis.export_bidder import load_otis
from otis.model import N_FATE_CLASSES, N_TILES, TILE_PIPS

EPS = 1e-12


# --------------------------------------------------------------------------- #
# Base rates (train only)                                                        #
# --------------------------------------------------------------------------- #


def train_base_rates(train_split) -> np.ndarray:
    """[5,8] per-tile marginal fate probabilities from the train labels."""
    yf = train_split.y_fate.numpy()  # [N,5]
    probs = np.zeros((N_TILES, N_FATE_CLASSES), dtype=np.float64)
    for t in range(N_TILES):
        cnt = np.bincount(yf[:, t], minlength=N_FATE_CLASSES).astype(np.float64)
        probs[t] = cnt / cnt.sum()
    return probs


# --------------------------------------------------------------------------- #
# Cluster bootstrap                                                              #
# --------------------------------------------------------------------------- #


def _cluster_bootstrap(per_row: np.ndarray, groups: np.ndarray, n_boot: int, seed: int) -> dict:
    """Percentile CI of the pooled mean of ``per_row`` under a cluster bootstrap
    over ``groups`` (each row's deal-cluster id). Resamples whole clusters."""
    uniq, inv = np.unique(groups, axis=0, return_inverse=True)
    G = len(uniq)
    # per-cluster sum + count for O(1) pooled mean of any resample.
    gsum = np.zeros(G)
    gcnt = np.zeros(G)
    np.add.at(gsum, inv, per_row)
    np.add.at(gcnt, inv, 1.0)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, G, size=G)
        stats[b] = gsum[pick].sum() / gcnt[pick].sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    point = per_row.mean()
    return {
        "point": float(point),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_clusters": int(G),
        "n_boot": int(n_boot),
    }


# --------------------------------------------------------------------------- #
# Pricing reliability (both arms)                                                #
# --------------------------------------------------------------------------- #


def pricing_report(model, split, device: str) -> dict:
    with torch.no_grad():
        logits = model.pricing_logits(split.X.to(device)).cpu()
    y = split.y_price
    ce = float(nn.functional.cross_entropy(logits, y).item())
    exc = exceedance(logits).numpy()
    rel = _reliability(exc[:, MIN_BID], (y.numpy() >= MIN_BID).astype(float))
    return {"val_nll": ce, "ece_p30": rel["ece"], "reliability_p30": rel, "n": int(len(y))}


# --------------------------------------------------------------------------- #
# Fate P2 (treatment)                                                            #
# --------------------------------------------------------------------------- #


def fate_report(model, test_split, base_probs: np.ndarray, device: str,
                n_boot: int, boot_seed: int) -> dict:
    with torch.no_grad():
        out = model(test_split.X.to(device))
    fate_logits = out["fate"].cpu()                    # [N,5,8]
    logp = torch.log_softmax(fate_logits, dim=-1).numpy()  # [N,5,8]
    yf = test_split.y_fate.numpy()                     # [N,5]
    N = yf.shape[0]

    base_lp = np.log(np.clip(base_probs, EPS, 1.0))     # [5,8]
    base_arg = base_probs.argmax(axis=1)                # [5]

    rows = np.arange(N)
    # per-row, per-tile log-likelihoods + correctness
    model_ll = np.stack([logp[rows, t, yf[:, t]] for t in range(N_TILES)], axis=1)  # [N,5]
    base_ll = np.stack([base_lp[t, yf[:, t]] for t in range(N_TILES)], axis=1)       # [N,5]
    model_arg = logp.argmax(axis=2)                    # [N,5]
    model_correct = (model_arg == yf).astype(np.float64)
    base_correct = np.stack([(yf[:, t] == base_arg[t]).astype(np.float64)
                             for t in range(N_TILES)], axis=1)

    per_tile = {}
    for t, pip in enumerate(TILE_PIPS):
        per_tile[pip] = {
            "model_nll": float(-model_ll[:, t].mean()),
            "base_nll": float(-base_ll[:, t].mean()),
            "nll_improvement_nats": float((model_ll[:, t] - base_ll[:, t]).mean()),
            "model_top1_acc": float(model_correct[:, t].mean()),
            "base_top1_acc": float(base_correct[:, t].mean()),
            "top1_acc_delta_pp": float((model_correct[:, t] - base_correct[:, t]).mean() * 100),
        }

    # overall = mean over tiles (per row), then cluster-bootstrapped over deals.
    imp_row = (model_ll - base_ll).mean(axis=1)         # [N] nats improvement
    acc_row = (model_correct - base_correct).mean(axis=1)  # [N] acc delta
    groups = test_split.groups.numpy()

    imp_ci = _cluster_bootstrap(imp_row, groups, n_boot, boot_seed)
    acc_ci = _cluster_bootstrap(acc_row * 100.0, groups, n_boot, boot_seed + 1)  # in pp

    band_nll = imp_ci["point"] >= 0.15 and imp_ci["excludes_zero"]
    band_acc = acc_ci["point"] >= 8.0 and acc_ci["excludes_zero"]
    return {
        "n_test": int(N),
        "overall_model_nll": float(-model_ll.mean()),
        "overall_base_nll": float(-base_ll.mean()),
        "nll_improvement_nats": imp_ci,
        "top1_acc_delta_pp": acc_ci,
        "overall_model_top1_acc": float(model_correct.mean()),
        "overall_base_top1_acc": float(base_correct.mean()),
        "per_tile": per_tile,
        "P2_band_nll_pass": bool(band_nll),
        "P2_band_acc_pass": bool(band_acc),
        "P2_pass": bool(band_nll and band_acc),
    }


# --------------------------------------------------------------------------- #
# Driver                                                                         #
# --------------------------------------------------------------------------- #


def evaluate(control_ckpt: Path, treatment_ckpt: Path, out_json: Path, *,
             source: str = "selfplay", device: str = "cpu",
             n_boot: int = 1000, boot_seed: int = 0) -> dict:
    train_split = build_split("train", source)
    val_split = build_split("val", source)
    test_split = build_split("test", source)

    base_probs = train_base_rates(train_split)

    control, _ = load_otis(control_ckpt, device)
    treatment, tck = load_otis(treatment_ckpt, device)
    if not tck.get("treatment", False):
        raise ValueError(f"{treatment_ckpt} is not a treatment checkpoint")

    result = {
        "source": source,
        "control_ckpt": str(control_ckpt),
        "treatment_ckpt": str(treatment_ckpt),
        "pricing": {
            "control": pricing_report(control, val_split, device),
            "treatment": pricing_report(treatment, val_split, device),
        },
        "fate_P2": fate_report(treatment, test_split, base_probs, device, n_boot, boot_seed),
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n")

    f = result["fate_P2"]
    print(f"[eval] P2 fate NLL improvement = {f['nll_improvement_nats']['point']:.4f} nats "
          f"CI[{f['nll_improvement_nats']['ci_lo']:.4f},{f['nll_improvement_nats']['ci_hi']:.4f}] "
          f"(band ≥0.15, excl 0) → {'PASS' if f['P2_band_nll_pass'] else 'FAIL'}", flush=True)
    print(f"[eval] P2 top-1 acc delta = {f['top1_acc_delta_pp']['point']:.2f}pp "
          f"CI[{f['top1_acc_delta_pp']['ci_lo']:.2f},{f['top1_acc_delta_pp']['ci_hi']:.2f}] "
          f"(band ≥8pp, excl 0) → {'PASS' if f['P2_band_acc_pass'] else 'FAIL'}", flush=True)
    print(f"[eval] pricing val NLL  control={result['pricing']['control']['val_nll']:.4f}  "
          f"treatment={result['pricing']['treatment']['val_nll']:.4f}", flush=True)
    print(f"[eval] wrote {out_json}", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="otis v0 P2 calibration eval")
    ap.add_argument("--control", type=Path, required=True)
    ap.add_argument("--treatment", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default="selfplay")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()
    evaluate(args.control, args.treatment, args.out, source=args.source,
             device=args.device, n_boot=args.n_boot, boot_seed=args.boot_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
