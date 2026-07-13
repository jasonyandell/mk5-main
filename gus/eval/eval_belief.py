"""Evaluate a trained v0 belief adapter on a held-out corpus.

Usage:
    python -u -m gus.eval.eval_belief \
        --adapter gus/adapters/v0_belief.pt \
        --corpus gus/data/corpus_eval_20.pt
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import torch
from torch.utils.data import DataLoader

from gus.model.dataset import JointWorldDecisionDataset
from gus.model.student import StudentV0, belief_accuracy


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or _pick_device()
    ckpt = torch.load(args.adapter, weights_only=False, map_location=device)
    hidden_dim = ckpt["args"].get("hidden_dim", 256)
    model = StudentV0(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ds = JointWorldDecisionDataset(args.corpus)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    total_correct = 0
    total_slots = 0
    # Bucket accuracy by decision_idx to see how belief sharpens over the game.
    by_decision_idx: dict[int, list[int]] = {}

    for batch in loader:
        features = batch["features"].to(device)
        target = batch["belief_target"].to(device)
        mask = batch["belief_mask"].to(device)
        d_idx = batch["decision_idx"].tolist()

        with torch.no_grad():
            out = model(features)
        preds = out["belief_logits"].argmax(dim=-1)  # [B, 28]

        for b in range(features.shape[0]):
            correct = int(((preds[b] == target[b]) & mask[b]).sum().item())
            total = int(mask[b].sum().item())
            if total > 0:
                by_decision_idx.setdefault(d_idx[b], []).extend([1] * correct + [0] * (total - correct))
            total_correct += correct
            total_slots += total

    overall = total_correct / max(total_slots, 1)
    print(f"Overall top-1 accuracy: {overall:.3%}  (chance: 33.33%)")
    print(f"Total unseen-slot predictions: {total_slots:,}")
    print()
    print("Accuracy by decision index (where belief should sharpen over the game):")
    print(f"  {'dec':>3s}  {'acc':>6s}  {'n':>5s}")
    for d in sorted(by_decision_idx.keys()):
        bits = by_decision_idx[d]
        acc = sum(bits) / len(bits) if bits else 0.0
        print(f"  {d:>3d}  {acc:>6.2%}  {len(bits):>5d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
