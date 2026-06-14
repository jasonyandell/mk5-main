"""Optimism meter — the oracle-vs-realized make-rate gap, measured from existing data.

WHY THIS EXISTS
===============
Champion rung #26 over-bids because it prices contracts off the *double-dummy*
oracle E[Q] (perfect play by all four seats), whose P(make) runs optimistic vs
*realized* PIMC play. The fix shipped as a single global knob, `pmake_scale=0.70`,
justified in prose as "the measured 0.58/0.83 gap at bid 30". That gap had **no
computing script** — 0.58 and 0.83 lived only as comments and wiki prose, never as
a data artifact (and 0.83 numerically collided with an unrelated offense-share).

This module computes the real thing from data already on disk, and emits a committed
artifact so the number stops being folklore. It is an *instrument*, not a
recalibration of the bidder — it does not touch `belief_bidder.py`. The established
result EV > p_make (Lens-v1) stands; this measures an optimism gap, it does not
propose make-rate as the play/bid utility.

TWO CURVES, BOTH FROM EXISTING DATA
===================================
* realized(bid) — `data/bidding-results/{test,val,train}/*.parquet`,
  `pmake_{decl}_{bid} = count(realized Gus-4-seat points >= bid) / n_samples`
  (forge/bidding/schema.py:50). Per hand we take the best declaration
  (max over decl); N = 604 hands. This is the *achievable* side.
* oracle(bid) — `w42/book_validation_v1/wave2/bid_aware_atlas/bid_aware_actions.csv`,
  eq `p_make` = P(contract made) under the oracle playing *double-dummy* in each
  sampled consistent world, at the declarer's opening lead. Per hand we take the
  best lead (max over action_slot) then the best declaration (max over decl);
  N = 50 hands. This is the *optimistic* side.

The two corpora are different hand pools, so this is a *distributional* gap, not a
paired one — a first instrument. The paired refinement (run the oracle over the exact
parquet hands) is the obvious next step. Top bids (>= 42) read 0.000 on the oracle
side due to a known all-or-nothing / 84-threshold artifact in the atlas; the
trustworthy comparison range is bids 30-39.

WHAT IT SHOWS (2026-06-14)
==========================
* realized make-rate falls 0.52 @ bid30 -> 0.11 @ bid42 — a 5x swing. A single
  multiplicative `pmake_scale` cannot map oracle->realized across this range; the
  optimism correction is the *wrong shape*, not the wrong constant.
* at bid 30 the real numbers are oracle 0.64 / realized 0.52 (ratio ~0.81), NOT the
  prose's 0.83 / 0.58 (ratio 0.70). The prose gap does not reproduce.

Run: `.venv/bin/python -m champion.optimism_meter`  (needs pyarrow + matplotlib)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARQUET_GLOBS = ["data/bidding-results/test", "data/bidding-results/val", "data/bidding-results/train"]
ATLAS_CSV = ROOT / "w42/book_validation_v1/wave2/bid_aware_atlas/bid_aware_actions.csv"
OUT_JSON = ROOT / "champion/optimism_gap.json"
OUT_PNG = ROOT / "champion/optimism_gap.png"

# Bids at/above this read unreliably on the oracle side (atlas top-bid threshold
# artifact: all-or-nothing 42 / 84-vs-all-42). Trustworthy gap range is below it.
ORACLE_RELIABLE_MAX_BID = 39


def _wilson(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (n = number of hands averaged)."""
    if n == 0:
        return (float("nan"), float("nan"))
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def realized_curve() -> dict[int, dict]:
    """Best-declaration realized make-rate by bid, from the forge bidding parquet."""
    files = [f for g in PARQUET_GLOBS for f in (ROOT / g).glob("*.parquet")]
    if not files:
        raise FileNotFoundError(f"no parquet under {PARQUET_GLOBS} (realized side missing)")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    pmake_cols = [c for c in df.columns if c.startswith("pmake_")]
    bids = sorted({int(c.split("_")[2]) for c in pmake_cols})
    decls = sorted({int(c.split("_")[1]) for c in pmake_cols})
    out: dict[int, dict] = {}
    for b in bids:
        cols = [f"pmake_{d}_{b}" for d in decls if f"pmake_{d}_{b}" in df.columns]
        best = df[cols].max(axis=1)  # bidder evaluates its best contract
        p, n = float(best.mean()), int(best.shape[0])
        lo, hi = _wilson(p, n)
        out[b] = {"make_rate": round(p, 4), "n_hands": n, "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}
    return out


def oracle_curve() -> dict[int, dict]:
    """Best-decl, best-lead double-dummy P(make) by bid, from the bid-aware atlas."""
    if not ATLAS_CSV.exists():
        raise FileNotFoundError(f"missing atlas {ATLAS_CSV} (oracle side missing)")
    a = pd.read_csv(ATLAS_CSV)
    decision = a[a["decision_idx"] == 0]  # declarer's opening lead = the contract value
    out: dict[int, dict] = {}
    for b in sorted(int(x) for x in a["bid_value"].unique()):
        sub = decision[decision["bid_value"] == b]
        best_lead = sub.groupby(["seed", "decl_id"])["p_make"].max()       # oracle plays best lead
        best_decl = best_lead.groupby(level="seed").max()                  # bidder picks best decl
        p, n = float(best_decl.mean()), int(best_decl.shape[0])
        out[b] = {
            "p_make": round(p, 4),
            "n_hands": n,
            "std": round(float(best_decl.std()), 4),
            "reliable": b <= ORACLE_RELIABLE_MAX_BID,
        }
    return out


def build() -> dict:
    realized = realized_curve()
    oracle = oracle_curve()
    gap = {}
    for b in sorted(set(realized) & set(oracle)):
        if not oracle[b]["reliable"]:
            continue
        o, r = oracle[b]["p_make"], realized[b]["make_rate"]
        gap[b] = {
            "oracle": o,
            "realized": r,
            "gap_oracle_minus_realized": round(o - r, 4),
            "ratio_realized_over_oracle": round(r / o, 4) if o else None,
        }
    ratios = [g["ratio_realized_over_oracle"] for g in gap.values() if g["ratio_realized_over_oracle"]]
    realized_bids = sorted(realized)
    return {
        "what": "Oracle (double-dummy) vs realized (4-seat) best-declaration make-rate, by bid.",
        "method": {
            "realized": "data/bidding-results parquet; pmake_{decl}_{bid}=count(points>=bid)/n_samples; max over decl per hand; mean over hands",
            "oracle": "bid_aware_atlas eq p_make at decision_idx==0; max over action_slot (best lead) then max over decl; mean over hands",
            "caveat": "different hand pools => distributional gap, not paired. oracle N small. bids>=42 oracle unreliable (atlas top-bid threshold artifact); gap computed on 30-39 only.",
        },
        "realized_curve": realized,
        "oracle_curve": oracle,
        "optimism_gap_reliable_range": gap,
        "findings": {
            "realized_make_rate_is_bid_dependent": (
                f"{realized[realized_bids[0]]['make_rate']:.2f} @ bid{realized_bids[0]} "
                f"-> {realized[realized_bids[-1]]['make_rate']:.2f} @ bid{realized_bids[-1]} "
                f"(N={realized[realized_bids[0]]['n_hands']}); a single global pmake_scale cannot fit this shape"
            ),
            "ratio_is_not_constant": (
                f"realized/oracle ratio spans {min(ratios):.2f}-{max(ratios):.2f} over bids 30-39 "
                f"(not the single 0.70 the prose claimed)"
            ),
            "prose_0.83_0.58_does_not_reproduce": (
                f"closest real analog at bid 30: oracle {gap[30]['oracle']:.2f} / realized {gap[30]['realized']:.2f} "
                f"(ratio {gap[30]['ratio_realized_over_oracle']:.2f}), not 0.83/0.58 (ratio 0.70). "
                "The prose pair was never computed; pmake_scale=0.70 is a tuned knob, not a measured gap."
            ),
        },
    }


def plot(result: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc = result["realized_curve"]
    oc = result["oracle_curve"]
    rb = sorted(rc)
    ob = [b for b in sorted(oc) if oc[b]["reliable"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rb, [rc[b]["make_rate"] for b in rb], "-o", color="#c1440e", label="realized (4-seat PIMC), N=604")
    ax.fill_between(rb, [rc[b]["ci_lo"] for b in rb], [rc[b]["ci_hi"] for b in rb], color="#c1440e", alpha=0.15)
    ax.plot(ob, [oc[b]["p_make"] for b in ob], "--s", color="#3b6ea5", label="oracle (double-dummy), N=50")
    ax.axhline(0.70, color="gray", ls=":", lw=1)
    ax.text(rb[-1], 0.71, "the single pmake_scale=0.70 knob", ha="right", va="bottom", color="gray", fontsize=8)
    ax.set_xlabel("bid (points contracted)")
    ax.set_ylabel("P(make) of best declaration")
    ax.set_title("Optimism meter: oracle is optimistic, and the gap is bid-dependent\n"
                 "(realized falls 0.52→0.11 — a global scalar can't fit it)")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=130)
    print(f"wrote {OUT_PNG.relative_to(ROOT)}")


def main() -> None:
    result = build()
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}")
    try:
        plot(result)
    except Exception as e:  # plotting is optional; the JSON is the artifact
        print(f"(plot skipped: {e})")
    print("\n-- realized best-decl make-rate by bid --")
    for b, v in result["realized_curve"].items():
        print(f"  bid {b:>3}: {v['make_rate']:.3f}  [{v['ci_lo']:.3f},{v['ci_hi']:.3f}]  N={v['n_hands']}")
    print("\n-- optimism gap (reliable range) --")
    for b, g in result["optimism_gap_reliable_range"].items():
        print(f"  bid {b:>3}: oracle {g['oracle']:.3f}  realized {g['realized']:.3f}  "
              f"gap {g['gap_oracle_minus_realized']:+.3f}  ratio {g['ratio_realized_over_oracle']:.3f}")
    for k, v in result["findings"].items():
        print(f"\n[{k}]\n  {v}")


if __name__ == "__main__":
    main()
