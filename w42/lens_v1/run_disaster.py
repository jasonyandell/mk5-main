"""Quick experiment: 'disaster' utility (clipped EV, anything below make threshold = -42).

Sanity test + head-to-head: disaster vs ev (the current champion) and disaster vs p_make.

Same conventions as round_robin.py: bid=30 forced, paired-seed, N=10, mps, fp32.
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from forge.zeb.eval.loading import DEFAULT_ORACLE, load_oracle
from w42.lens_v1.lens import utility_scores, UTILITIES
from w42.lens_v1.parallel_match import run_lens_match


def sanity_disaster_vs_ev():
    """Synthetic PDF check: disaster <= ev when there's any sub-threshold mass.
    disaster == ev when 100% of mass is at/above threshold.
    """
    device = torch.device("cpu")
    # n_games=2, 7 actions, 85 bins
    pdf = torch.zeros(2, 7, 85, device=device)
    e_q = torch.zeros(2, 7, device=device)

    # Game 0: action 0 has all mass at bin 60 (Q=+18, exactly at offense threshold for bid=30)
    pdf[0, 0, 60] = 1.0
    e_q[0, 0] = 18.0
    # Game 0: action 1 has mass split: 0.5 at bin 0 (Q=-42, set), 0.5 at bin 80 (Q=+38, big make)
    pdf[0, 1, 0] = 0.5
    pdf[0, 1, 80] = 0.5
    e_q[0, 1] = -2.0  # mean
    # Game 1: action 0 has all mass at bin 60 again
    pdf[1, 0, 60] = 1.0
    e_q[1, 0] = 18.0

    bidder = torch.tensor([0, 0])
    current = torch.tensor([0, 0])  # offense

    ev_scores = utility_scores("ev", e_q, pdf, bidder, current, [30, 30])
    disaster_scores = utility_scores("disaster", e_q, pdf, bidder, current, [30, 30])

    print("[sanity] EV       game 0 actions:", ev_scores[0].tolist())
    print("[sanity] DISASTER game 0 actions:", disaster_scores[0].tolist())
    # Game 0 action 0: pure mass at threshold -> EV=+18, disaster=+18 (no clip)
    # Game 0 action 1: split between -42 (set, clip to -42) and +38 (make, keep)
    #   EV = 0.5 * -42 + 0.5 * 38 = -2.0
    #   disaster = 0.5 * -42 + 0.5 * 38 = -2.0  (same, because the set bin is already -42!)
    # Hmm, that's the trivial case. Let me check a less-trivial split.
    pdf2 = torch.zeros(1, 1, 85, device=device)
    pdf2[0, 0, 50] = 0.5  # Q=+8, this is BELOW threshold (60), so set → clip to -42
    pdf2[0, 0, 70] = 0.5  # Q=+28, above threshold, keep
    e_q2 = torch.tensor([[8.0]])
    bidder2 = torch.tensor([0])
    current2 = torch.tensor([0])
    ev2 = utility_scores("ev", e_q2, pdf2, bidder2, current2, [30])
    dis2 = utility_scores("disaster", e_q2, pdf2, bidder2, current2, [30])
    print(f"[sanity-2] EV       = {ev2.item():.3f}  (expect: 0.5*8 + 0.5*28 = +18 from pdf, but e_q says +8)")
    print(f"[sanity-2] DISASTER = {dis2.item():.3f}  (expect: 0.5*-42 + 0.5*28 = -7.0)")
    # Note: ev uses e_q, disaster uses pdf. The mismatch in the sanity-2 case shows e_q is the
    # caller-supplied mean and may differ from pdf-implied mean. In real use, e_q == sum(pdf*qvals).
    pdf_ev = (pdf2 * torch.arange(-42, 43, dtype=torch.float32).view(1, 1, 85)).sum(dim=2)
    print(f"[sanity-2] pdf-mean = {pdf_ev.item():.3f}  (this is what disaster baseline EV would be)")
    assert dis2.item() < pdf_ev.item(), "disaster should be < pure pdf-EV when sub-threshold mass exists"
    print("[sanity-2] PASS: disaster < pdf-EV when below-threshold mass exists")


def main():
    sanity_disaster_vs_ev()
    print()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[disaster] loading oracle on {device}", flush=True)
    model = load_oracle(DEFAULT_ORACLE, device)

    out_dir = Path("w42/lens_v1/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    matchups = [
        ("disaster", "ev"),
        ("disaster", "p_make"),
        ("disaster", "robust_q25"),
    ]
    rows = []
    for util_a, util_b in matchups:
        t0 = time.time()
        result = run_lens_match(
            model,
            utility_a=util_a, utility_b=util_b,
            n_hands=1000, n_samples=10,
            base_seed=10000, device=device,
            verbose=True,
        )
        margins = np.array([h.margin for h in result.hands], dtype=np.float64)
        mean = float(margins.mean())
        # bootstrap CI
        rng = np.random.default_rng(42)
        idx = rng.integers(0, margins.size, size=(2000, margins.size))
        boots = margins[idx].mean(axis=1)
        ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        decisive = float((margins != 0).mean())
        a_wr = float((margins > 0).mean())
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        wall = time.time() - t0
        rows.append({
            "utility_a": util_a, "utility_b": util_b,
            "n_hands": result.n_hands, "n_samples": result.n_samples,
            "mean_margin": round(mean, 3),
            "margin_ci_lo_95": round(ci_lo, 3),
            "margin_ci_hi_95": round(ci_hi, 3),
            "ci_excludes_zero": int(excludes_zero),
            "a_win_rate": round(a_wr, 3),
            "decisive_rate": round(decisive, 3),
            "wall_s": round(wall, 1),
        })
        print(f"[disaster] {util_a:10s} vs {util_b:10s}  "
              f"margin={mean:+.2f}  CI=[{ci_lo:+.2f}, {ci_hi:+.2f}]  "
              f"a_wr={a_wr:.1%}  decisive={decisive:.1%}  "
              f"{'***' if excludes_zero else '   '}  wall={wall:.1f}s", flush=True)

    csv_path = out_dir / "disaster_head_to_head.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[disaster] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
