"""Regret-based eval: how much E[Q] does the student's chosen action LEAVE
on the table compared to the oracle's best legal action?

For each held-out decision:
  - oracle_best_eq = max_{a legal} e_q[a]  (from corpus)
  - student_action = argmax π_me_head (legal-masked)
  - student_eq     = e_q[student_action]
  - regret         = oracle_best_eq - student_eq

Mean regret is the decision-quality metric that matters: the student is
a better PLAYER if its mean regret is low, even when bot-match is imperfect
(because many decisions have near-ties where multiple actions are all good).

Also computes:
  - bot-match rate (same as before, for reference)
  - E[Q] headroom: oracle_best_eq - e_q[action_taken_in_game] — how much
    did the original E[Q] bot itself leave on the table relative to its own
    max? (should be ~0; sanity check)
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.load import load_student


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval", required=True, nargs="+")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Adapter: {args.adapter}  device: {device}", flush=True)

    model, is_voids = load_student(args.adapter, device)
    ds = JointWorldFullDataset(args.eval, seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    total_regret = 0.0
    total_items = 0
    bot_matches = 0
    negligible_regret = 0  # fraction where regret < 0.5 (effectively tied)
    bucket_regret: dict[int, list[float]] = defaultdict(list)
    bucket_match: dict[int, list[int]] = defaultdict(list)

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["tokens"].shape[0]

        with torch.no_grad():
            if is_voids:
                out = model(
                    batch["tokens"], batch["attention_mask"],
                    batch["world_assignment"], batch["voids"],
                )
            else:
                out = model(
                    batch["tokens"], batch["attention_mask"],
                    batch["world_assignment"],
                )

        # Legal-masked argmax over π_me
        pi = out["pi_me_logits"].masked_fill(~batch["legal_mask"], -1e9)
        student_action = pi.argmax(dim=-1)  # [B]

        # Oracle best = max e_q over legal actions
        e_q = batch["e_q"].clone()
        e_q_legal = e_q.masked_fill(~batch["legal_mask"], float("-inf"))
        oracle_best = e_q_legal.max(dim=-1).values
        oracle_best_action = e_q_legal.argmax(dim=-1)

        idx = torch.arange(B, device=device)
        student_eq = e_q[idx, student_action]

        regret = oracle_best - student_eq

        for b in range(B):
            r = float(regret[b].item())
            d_idx = int(batch["decision_idx"][b].item())
            total_regret += r
            total_items += 1
            bucket_regret[d_idx].append(r)
            hit = int(student_action[b] == oracle_best_action[b])
            bot_matches += hit
            bucket_match[d_idx].append(hit)
            if r < 0.5:
                negligible_regret += 1

    mean_regret = total_regret / max(total_items, 1)
    print()
    print(f"=== Summary over {total_items} decisions ===")
    print(f"  Bot-match rate:           {bot_matches/total_items:.3%}")
    print(f"  Mean regret (Q-points):   {mean_regret:.3f}")
    print(f"  Decisions with regret<0.5: {negligible_regret}/{total_items} "
          f"= {negligible_regret/total_items:.1%} (near-ties)")
    print()
    print("=== Per-decision regret + bot-match ===")
    print(f"{'dec':>3s}  {'bot':>6s}  {'regret':>8s}  {'near-tie':>8s}  {'n':>3s}")
    for d in sorted(bucket_regret.keys()):
        regrets = bucket_regret[d]
        matches = bucket_match[d]
        n = len(regrets)
        mean_r = sum(regrets) / n
        match_rate = sum(matches) / n
        near_ties = sum(1 for r in regrets if r < 0.5) / n
        print(f"{d:>3d}  {match_rate:>6.2%}  {mean_r:>8.2f}  {near_ties:>8.1%}  {n:>3d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
