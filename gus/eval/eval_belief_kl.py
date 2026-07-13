"""Belief-KL + belief-accuracy between two adapters on a held-out corpus.

The convergence signal for the self-play fixed point (champion rung #26): as the
policy<->belief loop iterates, the round-over-round *symmetric* belief-KL between
successive adapters should fall toward zero (the belief stops moving) while
held-out belief-accuracy plateaus. That is the honest "conventions stabilized"
read — and, per the design review, the PRIMARY signal, since the paired-match
arena is information-blind to belief value in play.

Belief logits depend only on the pooled state embedding (tokens + voids + the
auction feature), NOT the sampled world (the world only feeds the q head), so
this is a deterministic single pass over the corpus.

Usage:
  python -u -m gus.eval.eval_belief_kl --a A.pt --b B.pt --corpus eval.pt \
      [--device mps] [--batch-size 256]
"""
from __future__ import annotations

import argparse
import sys

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.load import load_student
from gus.model.student import belief_accuracy
from gus.train.train_belief_q_joint import belief_kl


def symmetric_belief_kl(logits_a: Tensor, logits_b: Tensor, mask: Tensor) -> Tensor:
    """Mean symmetric KL between two belief distributions over masked slots.

    0.5 * (KL(a||b) + KL(b||a)), averaged over the masked (unseen-domino) slots.
    Zero iff the two distributions agree on every masked slot; symmetric; >= 0.
    """
    soft_a = torch.softmax(logits_a, dim=-1)
    soft_b = torch.softmax(logits_b, dim=-1)
    kl_ab = belief_kl(logits_a, soft_b, mask)  # KL(b || softmax(a))
    kl_ba = belief_kl(logits_b, soft_a, mask)  # KL(a || softmax(b))
    return 0.5 * (kl_ab + kl_ba)


def _belief_logits(model, is_auction: bool, batch: dict) -> Tensor:
    if is_auction:
        out = model(batch["tokens"], batch["attention_mask"],
                    batch["world_assignment"], batch["voids"], batch["bids"])
    else:
        out = model(batch["tokens"], batch["attention_mask"],
                    batch["world_assignment"], batch["voids"])
    return out["belief_logits"]


def evaluate(adapter_a: str, adapter_b: str, corpus, *,
             device: str = "mps", batch_size: int = 256, seed: int = 0) -> dict:
    """Belief-acc of each adapter + their symmetric belief-KL on a held-out corpus.

    Deterministic: belief logits depend only on the pooled state embedding, not
    the per-item sampled world, so `seed` (which only drives world sampling) does
    not change the result — a==b therefore yields KL exactly 0.0 (the self-test).
    """
    torch.manual_seed(seed)
    model_a, _ = load_student(adapter_a, device)
    model_b, _ = load_student(adapter_b, device)
    model_a.eval()
    model_b.eval()
    is_auction_a = hasattr(model_a, "bids_encoder")
    is_auction_b = hasattr(model_b, "bids_encoder")

    ds = JointWorldFullDataset(corpus, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    ca = ta = cb = tb = 0
    kl_weighted = 0.0
    kl_slots = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            la = _belief_logits(model_a, is_auction_a, batch)
            lb = _belief_logits(model_b, is_auction_b, batch)
            tgt, mask = batch["belief_target"], batch["belief_mask"]
            c, t = belief_accuracy(la, tgt, mask); ca += c; ta += t
            c, t = belief_accuracy(lb, tgt, mask); cb += c; tb += t
            n = int(mask.sum().item())
            if n:
                kl_weighted += float(symmetric_belief_kl(la, lb, mask)) * n
                kl_slots += n

    return {
        "belief_acc_a": ca / max(ta, 1),
        "belief_acc_b": cb / max(tb, 1),
        "sym_kl": kl_weighted / max(kl_slots, 1),
        "n_decisions": len(ds),
        "n_slots": ta,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter-a", required=True, help="Adapter A (.pt)")
    p.add_argument("--adapter-b", required=True, help="Adapter B (.pt)")
    p.add_argument("--eval", required=True, nargs="+", help="Held-out belief corpus (.pt)")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    r = evaluate(args.adapter_a, args.adapter_b, args.eval,
                 device=device, batch_size=args.batch_size, seed=args.seed)
    print(
        f"belief-acc  A={r['belief_acc_a']:.4f}  B={r['belief_acc_b']:.4f}  "
        f"(Δ={r['belief_acc_b'] - r['belief_acc_a']:+.4f})\n"
        f"symmetric belief-KL(A,B) = {r['sym_kl']:.5f} nats/slot  "
        f"over {r['n_slots']:,} unseen-domino slots ({r['n_decisions']:,} decisions)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
