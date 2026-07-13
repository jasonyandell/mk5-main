"""LAMIR-1 regret eval — collision-free belief worlds vs direct π_me.

Compares three inference policies on the 560-decision held-out set:

  direct      — legal-masked argmax over π_me_logits (v3-10k baseline: 0.551)
  lamir1-argmax — argmax over Q_head with a single collision-free argmax world
                  (task #3 argmax_world: deterministic, no sampling)
  lamir1-K    — argmax over Q_head averaged across K collision-free worlds
                  sampled via sample_worlds (task #3)

The "rotated π_me as π_opp" hypothesis says the tokenizer is seat-symmetric
(verified in task #2), so querying π_me from a rotated view is equivalent to
π_opp — and the world-conditioned Q_head already encodes per-world values
trained against the oracle. LAMIR-1 uses those Q estimates over belief-sampled
worlds as the leaf evaluator.

PRACTICALITIES §3 note: naïve PIMC loses to π_me due to strategy fusion (world-
conditioned best action averaged ≠ best committed action). argmax_world avoids
this by using a single representative world rather than averaging — the model
commits to one world hypothesis and picks the optimal action for it.

Reports:
  - Overall mean regret and bot-match for each mode
  - Fraction where LAMIR-1 disagrees with direct π_me
  - Regret on just those disagreement decisions (key diagnostic)
  - Per-decision-index breakdown

Success bar: LAMIR-1 mean regret < 0.55 (baseline). Stretch: < 0.49.
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
from gus.model.sample_worlds import argmax_world, sample_worlds


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _q_action(
    model,
    state_emb: torch.Tensor,    # [B, D]
    world_assign: torch.Tensor,  # [B, 28, 3]
    legal_mask: torch.Tensor,   # [B, 7]
) -> torch.Tensor:
    """Run Q_head on a single world assignment, return legal-masked argmax [B]."""
    with torch.no_grad():
        world_emb = model.world_encoder(world_assign)
        q = model.q_head(state_emb, world_emb)  # [B, 7]
    q = q.masked_fill(~legal_mask, float("-inf"))
    return q.argmax(dim=-1)


def _q_mean_action(
    model,
    state_emb: torch.Tensor,     # [B, D]
    worlds: torch.Tensor,        # [B, K, 28, 3]
    legal_mask: torch.Tensor,    # [B, 7]
) -> torch.Tensor:
    """Average Q_head over K worlds, return legal-masked argmax [B]."""
    B, K = worlds.shape[:2]
    state_rep = state_emb.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    worlds_flat = worlds.reshape(B * K, 28, 3)
    with torch.no_grad():
        world_emb = model.world_encoder(worlds_flat)
        q_flat = model.q_head(state_rep, world_emb)  # [B*K, 7]
    q_mean = q_flat.view(B, K, 7).mean(dim=1)  # [B, 7]
    q_mean = q_mean.masked_fill(~legal_mask, float("-inf"))
    return q_mean.argmax(dim=-1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True,
                        help="Path to student adapter .pt (v3-10k recommended)")
    parser.add_argument("--eval", required=True, nargs="+",
                        help="Eval corpus .pt files")
    parser.add_argument("--k", type=int, default=20,
                        help="Worlds to sample for lamir1-K mode (default 20)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emit-per-decision-json", type=str, default=None,
                        help="If set, write one JSON line per decision to this path "
                             "with choice/regret/peak for each policy (append-only flag; "
                             "does not alter the summary logic).")
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Adapter: {args.adapter}", flush=True)
    print(f"Device:  {device}  K={args.k}", flush=True)

    model, is_voids = load_student(args.adapter, device)
    ds = JointWorldFullDataset(args.eval, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Eval decisions: {len(ds)}", flush=True)

    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    MODES = ("direct", "lamir1-argmax", "lamir1-K")
    total_regret = {m: 0.0 for m in MODES}
    total_match = {m: 0 for m in MODES}
    total_items = 0

    # Disagreement tracking: each non-direct mode vs direct.
    disagree_stats = {
        mode: {"n": 0, "direct_regret": 0.0, "mode_regret": 0.0}
        for mode in MODES
        if mode != "direct"
    }

    bucket_regret: dict[str, dict[int, list[float]]] = {m: defaultdict(list) for m in MODES}
    bucket_match: dict[str, dict[int, list[int]]] = {m: defaultdict(list) for m in MODES}

    emit_path = args.emit_per_decision_json
    emit_fp = None
    if emit_path is not None:
        import json as _json
        emit_fp = open(emit_path, "w")

    global_row_idx = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["tokens"].shape[0]
        legal = batch["legal_mask"]

        # Oracle reference
        e_q = batch["e_q"].clone()
        e_q_legal = e_q.masked_fill(~legal, float("-inf"))
        oracle_best_eq = e_q_legal.max(dim=-1).values  # [B]
        oracle_best_action = e_q_legal.argmax(dim=-1)  # [B]
        idx = torch.arange(B, device=device)

        # Forward pass
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

        state_emb = out["state_emb"]   # [B, D]
        belief_logits = out["belief_logits"]  # [B, 28, 3]
        belief_mask = batch["belief_mask"]    # [B, 28]

        # --- direct ---
        pi = out["pi_me_logits"].masked_fill(~legal, float("-inf"))
        direct_action = pi.argmax(dim=-1)

        # --- lamir1-argmax: single collision-free world, argmax Q_head ---
        aw = argmax_world(belief_logits, belief_mask)  # [B, 28, 3]
        lamir_argmax_action = _q_action(model, state_emb, aw, legal)

        # --- lamir1-K: K sampled worlds, mean Q_head ---
        worlds_k = sample_worlds(belief_logits, belief_mask, args.k, rng)  # [B, K, 28, 3]
        lamir_k_action = _q_mean_action(model, state_emb, worlds_k, legal)

        actions = {
            "direct": direct_action,
            "lamir1-argmax": lamir_argmax_action,
            "lamir1-K": lamir_k_action,
        }

        for mode, action in actions.items():
            eq = e_q[idx, action]
            regret = oracle_best_eq - eq
            match = (action == oracle_best_action)

            for b in range(B):
                r = float(regret[b].item())
                d_idx = int(batch["decision_idx"][b].item())
                total_regret[mode] += r
                bucket_regret[mode][d_idx].append(r)
                h = int(match[b].item())
                total_match[mode] += h
                bucket_match[mode][d_idx].append(h)

        # Disagreement analysis: each candidate mode vs direct.
        for mode, action in actions.items():
            if mode == "direct":
                continue
            disagree_mask = (action != direct_action)
            stats = disagree_stats[mode]
            for b in range(B):
                if disagree_mask[b]:
                    stats["n"] += 1
                    stats["direct_regret"] += float((oracle_best_eq[b] - e_q[b, direct_action[b]]).item())
                    stats["mode_regret"] += float((oracle_best_eq[b] - e_q[b, action[b]]).item())

        # Optional per-decision JSON emission (append-only diagnostic; does not
        # change any headline stat).
        if emit_fp is not None:
            import json as _json
            pi_logits_full = out["pi_me_logits"].masked_fill(~legal, float("-inf"))
            pi_probs = torch.softmax(pi_logits_full, dim=-1)
            pi_peak_full = pi_probs.max(dim=-1).values
            # entropy over legal mass (log terms on masked slots -> -inf*0 -> nan; filter)
            ent = torch.where(
                pi_probs > 0,
                -pi_probs * torch.log(pi_probs.clamp_min(1e-12)),
                torch.zeros_like(pi_probs),
            ).sum(dim=-1)
            for b in range(B):
                row = {
                    "global_idx": global_row_idx + b,
                    "decision_idx": int(batch["decision_idx"][b].item()),
                    "player": int(batch["player"][b].item()),
                    "action_taken_bot": int(batch["action_taken"][b].item()),
                    "legal_mask": [bool(x) for x in legal[b].tolist()],
                    "oracle_best_action": int(oracle_best_action[b].item()),
                    "oracle_best_eq": float(oracle_best_eq[b].item()),
                    "e_q": [float(x) for x in e_q[b].tolist()],
                    "pi_probs": [float(x) for x in pi_probs[b].tolist()],
                    "pi_peak": float(pi_peak_full[b].item()),
                    "pi_entropy": float(ent[b].item()),
                    "direct_action": int(direct_action[b].item()),
                    "direct_regret": float((oracle_best_eq[b] - e_q[b, direct_action[b]]).item()),
                    "qmean_action": int(lamir_k_action[b].item()),
                    "qmean_regret": float((oracle_best_eq[b] - e_q[b, lamir_k_action[b]]).item()),
                    "argmax_action": int(lamir_argmax_action[b].item()),
                    "argmax_regret": float((oracle_best_eq[b] - e_q[b, lamir_argmax_action[b]]).item()),
                }
                emit_fp.write(_json.dumps(row) + "\n")

        global_row_idx += B
        total_items += B

    if emit_fp is not None:
        emit_fp.close()
        print(f"(wrote per-decision JSONL to {emit_path})", flush=True)

    N = total_items
    print()
    print(f"=== LAMIR-1 eval on {N} decisions ===")
    print(f"{'mode':>14s}  {'regret':>8s}  {'bot-match':>9s}")
    for mode in MODES:
        r = total_regret[mode] / N
        m = total_match[mode] / N
        tag = ""
        if mode == "direct":
            tag = "  ← baseline"
        print(f"{mode:>14s}  {r:>8.4f}  {m:>8.3%}{tag}")

    print()
    print("=== Disagreements vs direct ===")
    for mode, stats in disagree_stats.items():
        disagree_n = int(stats["n"])
        pct = disagree_n / N
        print(f"{mode:>14s}: {disagree_n}/{N} = {pct:.1%}")
        if disagree_n == 0:
            continue
        dr = stats["direct_regret"] / disagree_n
        mr = stats["mode_regret"] / disagree_n
        delta = mr - dr
        print(f"  on disagreements — direct regret: {dr:.4f}  {mode} regret: {mr:.4f}  Δ={delta:+.4f}")
        if delta < 0:
            print(f"  → {mode} wins on disagreements (lower regret)")
        else:
            print(f"  → direct wins on disagreements ({mode} diverges from oracle)")

    print()
    print(f"=== Per-decision-index breakdown ===")
    print(f"{'dec':>3s}  {'n':>3s}  {'direct':>7s}  {'l1-argmax':>9s}  {'l1-K':>6s}")
    all_dec = sorted(bucket_regret["direct"].keys())
    for d in all_dec:
        n = len(bucket_regret["direct"][d])
        rd = sum(bucket_regret["direct"][d]) / n
        rl = sum(bucket_regret["lamir1-argmax"][d]) / n
        rk = sum(bucket_regret["lamir1-K"][d]) / n
        print(f"{d:>3d}  {n:>3d}  {rd:>7.3f}  {rl:>9.3f}  {rk:>6.3f}")

    # Verdict
    print()
    baseline = total_regret["direct"] / N
    mode_regrets = {mode: total_regret[mode] / N for mode in MODES}
    best_mode = min(MODES, key=lambda mode: mode_regrets[mode])
    print(f"=== Verdict ===")
    print(f"  baseline (direct π_me): {baseline:.4f}")
    for mode in MODES:
        if mode == "direct":
            continue
        delta = mode_regrets[mode] - baseline
        print(f"  {mode}: {mode_regrets[mode]:.4f}  Δ={delta:+.4f}")
    print(f"  best mode: {best_mode} ({mode_regrets[best_mode]:.4f})")
    if mode_regrets[best_mode] < 0.49:
        print("  STRETCH GOAL MET: best mode regret < 0.49")
    elif mode_regrets[best_mode] < 0.55:
        print("  SUCCESS BAR MET: best mode regret < 0.55")
    else:
        delta = mode_regrets[best_mode] - baseline
        print(f"  BELOW SUCCESS BAR — best mode delta vs baseline: {delta:+.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
