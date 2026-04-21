"""Ensemble evaluation of multiple Gus adapters.

Each adapter is a four-head transformer student (belief / V / π_me / Q) with
different size/data/recipe. Hypothesis: individual errors may differ enough
that ensembling their outputs beats the best single adapter's regret.

For each held-out decision, we compute each adapter's legal-masked π_me
softmax and its V-head estimate. We then combine via:

  - majority         : vote on argmax, ties broken by summed softmax
  - softmax_avg      : mean of legal-softmax across adapters, argmax
  - v_weighted       : softmax-weight each adapter by |V − oracle_best| proxy
                       (smaller gap ⇒ stronger confidence)
  - best_per_decision: ORACLE CEILING — pick adapter whose chosen action has
                       the highest oracle e_q on each decision. Unreachable
                       in practice; quantifies the MAX win ensembling could
                       deliver.

We also surface:
  - Each adapter's individual bot-match + mean regret
  - Disagreement rate (decisions where adapters split)
  - Blunder counts (regret > 8) per strategy
  - Case studies of disagreement decisions
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.student import StudentTransformerFull, StudentTransformerFullVoids

BLUNDER_THRESHOLD = 8.0


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_student(path: str, device: str):
    """Auto-detect voids vs non-voids and load. Returns (model, is_voids) or
    raises on bad checkpoint."""
    ckpt = torch.load(path, weights_only=False, map_location=device)
    args = ckpt["args"]
    is_voids = "voids_hidden" in args
    cls = StudentTransformerFullVoids if is_voids else StudentTransformerFull
    kwargs = {
        "d_model": args["d_model"],
        "n_heads": args["n_heads"],
        "n_layers": args["n_layers"],
        "ff_dim": args.get("ff_dim", 256),
        "dropout": 0.0,
        "d_world": args.get("d_world", 64),
        "q_hidden": args.get("q_hidden", 256),
    }
    if is_voids:
        kwargs["voids_hidden"] = args.get("voids_hidden", 64)
    model = cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, is_voids


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapters", required=True, nargs="+")
    parser.add_argument("--eval", required=True, nargs="+")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--case-studies", type=int, default=5,
                        help="Number of disagreement case studies to print")
    args = parser.parse_args()

    device = args.device or _pick_device()
    print(f"Device: {device}", flush=True)

    # --- load adapters, skipping any that fail ---
    adapters: list[tuple[str, torch.nn.Module, bool]] = []
    for p in args.adapters:
        name = Path(p).stem
        try:
            model, is_voids = _load_student(p, device)
            adapters.append((name, model, is_voids))
            n_params = sum(x.numel() for x in model.parameters()) / 1e6
            print(f"  loaded {name}  voids={is_voids}  {n_params:.1f}M params", flush=True)
        except Exception as e:
            print(f"  SKIP  {name}: {e}", flush=True)

    if not adapters:
        print("No adapters loaded.", file=sys.stderr)
        return 1
    A = len(adapters)
    print(f"Loaded {A} adapters", flush=True)

    ds = JointWorldFullDataset(args.eval, seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Eval decisions: {len(ds)}", flush=True)

    # Accumulators (per-decision lists, collect across loader then aggregate)
    per_adapter_regret = [[] for _ in range(A)]
    per_adapter_match = [[] for _ in range(A)]

    # Per-strategy regret
    strategy_names = ["majority", "softmax_avg", "v_weighted", "best_per_decision"]
    strat_regret: dict[str, list[float]] = {s: [] for s in strategy_names}
    strat_match: dict[str, list[int]] = {s: [] for s in strategy_names}

    # Track adapter choices + oracle choice for case studies / disagreement matrix
    adapter_picks_all: list[list[int]] = [[] for _ in range(A)]
    oracle_best_actions_all: list[int] = []
    decision_idx_all: list[int] = []
    oracle_eq_all: list[list[float]] = []  # per-decision e_q [7]
    # Track decisions where strategies disagreed for case studies
    case_studies_saved: list[dict] = []

    # Per-adapter V-head prediction at the oracle best (for diagnostics)
    # Not used for weighting directly — weighting uses |V - chosen_eq|.

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["tokens"].shape[0]

        legal = batch["legal_mask"]  # [B, 7]
        e_q = batch["e_q"]  # [B, 7]
        e_q_legal = e_q.masked_fill(~legal, float("-inf"))
        oracle_best_eq = e_q_legal.max(dim=-1).values  # [B]
        oracle_best_action = e_q_legal.argmax(dim=-1)  # [B]

        # --- Run each adapter, collect softmax pi and V ---
        # pi_sm: [A, B, 7] legal-masked softmax
        # v_pred: [A, B]
        # adapter_action: [A, B] legal-masked argmax
        pi_sm = torch.zeros(A, B, 7, device=device)
        v_pred = torch.zeros(A, B, device=device)
        adapter_action = torch.zeros(A, B, dtype=torch.long, device=device)

        for i, (_name, model, is_voids) in enumerate(adapters):
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
            logits = out["pi_me_logits"].masked_fill(~legal, -1e9)
            sm = torch.softmax(logits, dim=-1)
            # Zero out illegal (softmax of -1e9 is ~0 but be explicit)
            sm = sm * legal.float()
            # Renormalize in case of numeric drift
            sm = sm / sm.sum(dim=-1, keepdim=True).clamp(min=1e-9)
            pi_sm[i] = sm
            v_pred[i] = out["v"]
            adapter_action[i] = sm.argmax(dim=-1)

        # --- Per-adapter regret/match ---
        idx = torch.arange(B, device=device)
        for i in range(A):
            student_eq = e_q[idx, adapter_action[i]]
            regret = oracle_best_eq - student_eq  # [B]
            hit = (adapter_action[i] == oracle_best_action).long()
            per_adapter_regret[i].extend(regret.tolist())
            per_adapter_match[i].extend(hit.tolist())

        # --- Strategy 1: majority vote ---
        # For each item, for each action, count how many adapters chose it.
        # Ties broken by sum of softmax probability on that action.
        adapter_action_onehot = torch.nn.functional.one_hot(
            adapter_action, num_classes=7
        ).float()  # [A, B, 7]
        vote_counts = adapter_action_onehot.sum(dim=0)  # [B, 7]
        # Tiebreak: add a small * sum-of-softmax so argmax picks higher-prob on ties
        softmax_sum = pi_sm.sum(dim=0)  # [B, 7]
        tiebreak = softmax_sum * 1e-3
        vote_scores = vote_counts + tiebreak
        vote_scores = vote_scores.masked_fill(~legal, -1e9)
        majority_action = vote_scores.argmax(dim=-1)

        # --- Strategy 2: softmax average ---
        sm_avg = pi_sm.mean(dim=0)  # [B, 7]
        sm_avg = sm_avg.masked_fill(~legal, -1e9)
        softmax_avg_action = sm_avg.argmax(dim=-1)

        # --- Strategy 3: V-weighted ensemble ---
        # weight_i = softmax(-|V_i - V_i(argmax)|) ... actually we want confidence.
        # Proxy: adapter whose V is HIGHEST gets more weight (it thinks this is
        # a good state).  Alternative: weight by (max legal softmax prob) —
        # peakier adapter is more confident.
        # Use peakiness (max sm): w_i(b) = max_a pi_sm[i, b, a]
        peakiness = pi_sm.max(dim=-1).values  # [A, B]
        # normalize over adapters
        w = peakiness / peakiness.sum(dim=0, keepdim=True).clamp(min=1e-9)  # [A, B]
        weighted_sm = (pi_sm * w.unsqueeze(-1)).sum(dim=0)  # [B, 7]
        weighted_sm = weighted_sm.masked_fill(~legal, -1e9)
        v_weighted_action = weighted_sm.argmax(dim=-1)

        # --- Strategy 4: best_per_decision (oracle ceiling) ---
        # Pick whichever adapter chose the action with highest e_q on this decision.
        # adapter_action[A, B]; e_q[B, 7] → adapter_eq[A, B]
        adapter_eq = torch.gather(
            e_q.unsqueeze(0).expand(A, B, 7), 2,
            adapter_action.unsqueeze(-1),
        ).squeeze(-1)  # [A, B]
        best_adapter_idx = adapter_eq.argmax(dim=0)  # [B]
        # action chosen by the best adapter for that decision
        best_per_decision_action = torch.gather(
            adapter_action, 0, best_adapter_idx.unsqueeze(0),
        ).squeeze(0)  # [B]

        for strat_name, chosen in (
            ("majority", majority_action),
            ("softmax_avg", softmax_avg_action),
            ("v_weighted", v_weighted_action),
            ("best_per_decision", best_per_decision_action),
        ):
            student_eq = e_q[idx, chosen]
            regret = oracle_best_eq - student_eq
            hit = (chosen == oracle_best_action).long()
            strat_regret[strat_name].extend(regret.tolist())
            strat_match[strat_name].extend(hit.tolist())

        # Record per-decision data
        for b in range(B):
            oracle_best_actions_all.append(int(oracle_best_action[b].item()))
            decision_idx_all.append(int(batch["decision_idx"][b].item()))
            oracle_eq_all.append(e_q[b].tolist())
            for i in range(A):
                adapter_picks_all[i].append(int(adapter_action[i, b].item()))

            # Capture interesting disagreements for case study
            picks = [int(adapter_action[i, b].item()) for i in range(A)]
            if len(set(picks)) >= min(A, 3) and len(case_studies_saved) < args.case_studies * 3:
                case_studies_saved.append({
                    "decision_idx": int(batch["decision_idx"][b].item()),
                    "picks": picks,
                    "oracle": int(oracle_best_action[b].item()),
                    "oracle_eq": e_q[b].tolist(),
                    "majority": int(majority_action[b].item()),
                    "softmax_avg": int(softmax_avg_action[b].item()),
                    "v_weighted": int(v_weighted_action[b].item()),
                    "best_per_decision": int(best_per_decision_action[b].item()),
                    "oracle_best_eq": float(oracle_best_eq[b].item()),
                    "pi_sm": pi_sm[:, b, :].tolist(),
                    "legal": legal[b].tolist(),
                })

    N = len(per_adapter_regret[0])

    # ======================================================================
    # Report
    # ======================================================================

    print()
    print(f"=== Per-adapter baselines ({N} decisions) ===")
    print(f"{'adapter':35s}  {'match':>7s}  {'regret':>8s}  {'blunders':>9s}")
    for i, (name, _m, _v) in enumerate(adapters):
        match = sum(per_adapter_match[i]) / N
        regret_mean = sum(per_adapter_regret[i]) / N
        blunders = sum(1 for r in per_adapter_regret[i] if r > BLUNDER_THRESHOLD)
        print(f"{name:35s}  {match:>7.3%}  {regret_mean:>8.3f}  {blunders:>9d}")

    # Best single for later comparison
    best_single_idx = min(range(A), key=lambda i: sum(per_adapter_regret[i]) / N)
    best_single_name = adapters[best_single_idx][0]
    best_single_regret = sum(per_adapter_regret[best_single_idx]) / N
    best_single_match = sum(per_adapter_match[best_single_idx]) / N
    best_single_blunders = sum(
        1 for r in per_adapter_regret[best_single_idx] if r > BLUNDER_THRESHOLD
    )

    print()
    print(f"=== Ensemble strategies ({A} adapters, {N} decisions) ===")
    print(f"{'strategy':20s}  {'match':>7s}  {'regret':>8s}  {'blunders':>9s}  {'Δregret':>9s}")
    print(f"{'-- best single --':20s}  {best_single_match:>7.3%}  "
          f"{best_single_regret:>8.3f}  {best_single_blunders:>9d}  "
          f"  ({best_single_name})")
    for s in strategy_names:
        match = sum(strat_match[s]) / N
        regret_mean = sum(strat_regret[s]) / N
        blunders = sum(1 for r in strat_regret[s] if r > BLUNDER_THRESHOLD)
        delta = regret_mean - best_single_regret
        print(f"{s:20s}  {match:>7.3%}  {regret_mean:>8.3f}  {blunders:>9d}  {delta:>+9.3f}")

    print()
    print("=== Oracle ceiling analysis ===")
    oracle_regret = sum(strat_regret["best_per_decision"]) / N
    headroom = best_single_regret - oracle_regret
    print(f"  Best single:             regret={best_single_regret:.3f}  ({best_single_name})")
    print(f"  best_per_decision:       regret={oracle_regret:.3f}  (oracle ceiling)")
    print(f"  Max improvement from picking the right adapter per decision: "
          f"{headroom:.3f} Q-pts ({100*headroom/max(best_single_regret, 1e-9):.1f}% of single's regret)")

    print()
    print("=== Disagreement statistics ===")
    n_full_agree = 0
    n_any_disagree = 0
    n_strong_disagree = 0  # >=3 different actions picked
    for b_idx in range(N):
        picks = [adapter_picks_all[i][b_idx] for i in range(A)]
        unique = len(set(picks))
        if unique == 1:
            n_full_agree += 1
        else:
            n_any_disagree += 1
        if unique >= 3:
            n_strong_disagree += 1
    print(f"  Full agreement:      {n_full_agree}/{N} = {n_full_agree/N:.1%}")
    print(f"  Any disagreement:    {n_any_disagree}/{N} = {n_any_disagree/N:.1%}")
    print(f"  Strong (≥3 actions): {n_strong_disagree}/{N} = {n_strong_disagree/N:.1%}")

    # On disagreement decisions, how often does each strategy outperform best single?
    print()
    print("=== On disagreement decisions only ===")
    disagree_idx = [
        b for b in range(N)
        if len(set(adapter_picks_all[i][b] for i in range(A))) > 1
    ]
    if disagree_idx:
        best_single_regret_dis = (
            sum(per_adapter_regret[best_single_idx][b] for b in disagree_idx)
            / len(disagree_idx)
        )
        print(f"  (N_disagree = {len(disagree_idx)})")
        print(f"  best single regret on these:  {best_single_regret_dis:.3f}")
        for s in strategy_names:
            r = sum(strat_regret[s][b] for b in disagree_idx) / len(disagree_idx)
            delta = r - best_single_regret_dis
            print(f"  {s:20s}  regret={r:.3f}  Δ={delta:+.3f}")

    # Blunder analysis
    print()
    print(f"=== Blunder analysis (regret > {BLUNDER_THRESHOLD}) ===")
    # Use the best single's blunders as the comparison set
    blunder_decision_idxs = [
        b for b in range(N)
        if per_adapter_regret[best_single_idx][b] > BLUNDER_THRESHOLD
    ]
    print(f"  Best single ({best_single_name}) has {len(blunder_decision_idxs)} blunders")
    print(f"  Strategy:              blunders_remaining  ({len(blunder_decision_idxs)} baseline)")
    for s in strategy_names:
        remaining = sum(
            1 for b in blunder_decision_idxs if strat_regret[s][b] > BLUNDER_THRESHOLD
        )
        # Also count blunders INTRODUCED (non-blunder for best single → blunder for strategy)
        introduced = sum(
            1 for b in range(N)
            if (per_adapter_regret[best_single_idx][b] <= BLUNDER_THRESHOLD
                and strat_regret[s][b] > BLUNDER_THRESHOLD)
        )
        print(f"  {s:20s}  {remaining:>3d}  (introduced {introduced})")

    # Case studies
    print()
    print(f"=== Case studies ({min(args.case_studies, len(case_studies_saved))} disagreements) ===")
    for cs in case_studies_saved[:args.case_studies]:
        print(f"\n  decision_idx={cs['decision_idx']}")
        print(f"    legal:            {cs['legal']}")
        print(f"    oracle e_q:       "
              f"{[f'{q:.2f}' for q in cs['oracle_eq']]}")
        print(f"    oracle best:      action {cs['oracle']}  "
              f"(e_q={cs['oracle_best_eq']:.2f})")
        for i, (name, _m, _v) in enumerate(adapters):
            pick = cs["picks"][i]
            sm = cs["pi_sm"][i]
            print(f"    {name:33s} → {pick}  "
                  f"sm={[f'{s:.2f}' for s in sm]}  "
                  f"e_q(pick)={cs['oracle_eq'][pick]:.2f}")
        print(f"    majority={cs['majority']}  "
              f"softmax_avg={cs['softmax_avg']}  "
              f"v_weighted={cs['v_weighted']}  "
              f"best_per_decision={cs['best_per_decision']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
