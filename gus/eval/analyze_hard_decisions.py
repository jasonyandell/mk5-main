"""For each decision_idx in a held-out corpus, report the oracle's E[Q]
SPREAD (max - min across legal actions). Pair with student regret to
see where regret is ENFORCED by the oracle (high-spread decisions — missing
costs real Q-points) vs where regret is NEGLIGIBLE (low-spread: all legal
plays are near-equivalent).

Takeaway from typical runs: the student's highest-regret decisions are
also the highest-spread decisions. Honest mistakes on genuinely hard
positions, not easy ones.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch

from gus.model.dataset_seq_world import JointWorldFullDataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", required=True, nargs="+")
    args = parser.parse_args()

    ds = JointWorldFullDataset(args.eval)

    by_dec: dict[int, list[float]] = defaultdict(list)
    for i in range(len(ds)):
        s = ds[i]
        d_idx = int(s["decision_idx"].item())
        e_q = s["e_q"]
        legal = s["legal_mask"]
        if legal.sum() < 2:
            continue
        legal_eq = e_q[legal]
        spread = float((legal_eq.max() - legal_eq.min()).item())
        by_dec[d_idx].append(spread)

    print(f"{'dec':>3s}  {'mean_spread':>11s}  {'std_spread':>10s}  {'max_spread':>10s}  {'n':>3s}")
    for d in sorted(by_dec.keys()):
        t = torch.tensor(by_dec[d])
        print(f"{d:>3d}  {t.mean().item():>11.2f}  "
              f"{t.std(unbiased=False).item():>10.2f}  "
              f"{t.max().item():>10.2f}  {len(by_dec[d]):>3d}")

    ranked = sorted(
        ((d, torch.tensor(vals).mean().item()) for d, vals in by_dec.items() if vals),
        key=lambda x: -x[1],
    )
    print()
    print("Top-10 by strategic spread (hardest decisions):")
    for d, s in ranked[:10]:
        print(f"  dec {d:2d}: spread {s:5.2f}")
    print()
    print("Bottom-10 by spread (near-tie / low-stakes decisions):")
    for d, s in ranked[-10:]:
        print(f"  dec {d:2d}: spread {s:5.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
