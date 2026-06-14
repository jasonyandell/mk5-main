"""Detect-and-route inference: first end-to-end wrapper for the Gus student.

Motivation (PRACTICALITIES.md sections 12-13): ensembling hurts regret but
latent diversity is huge (oracle-per-decision regret drops from 1.39 → 0.36).
A router — small blunder detector + fallback policy — should bridge the gap
while staying deployable.

This script measures the ACTUAL regret of the routed decision, not the
projection of "replace flagged decisions with oracle argmax".

Pipeline per held-out decision:
  1. Primary student (v2_voids_3000g_big by default) → π_me argmax
  2. Compute student-derivable features (reuse blunder_detector_student.py)
  3. Detector (GradientBoostingClassifier) → blunder probability
  4. If threshold-gated detector fires:
       fallback action (oracle / pimc-q-k50 / next-best-adapter)
     else:
       primary action
  5. Record regret

Reports:
  - For each (fallback, flag_pct) combo: mean regret + bot-match
  - Decision_idx breakdown (which tricks benefit most?)
  - Blunder bucket (regret > 8): how many of the 34 caught / introduced?

Usage:
  python -u -m gus.eval.router \\
      --adapter gus/adapters/v2_voids_3000g_big.pt \\
      --eval gus/data/corpus_eval_20.pt \\
      --fallback oracle pimc-q-k50 next-best-adapter \\
      --fallback-adapter gus/adapters/v1_full_1000g.pt \\
      --device cpu
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from gus.eval.blunder_detector_student import (
    _student_features_for_batch,
    collect_samples,
    FEATURE_NAMES,
)
from gus.model.load import load_student
from gus.model.dataset_seq_world import JointWorldFullDataset

BLUNDER_THRESHOLD = 8.0
FALLBACK_POLICIES = ("oracle", "pimc-q-k50", "next-best-adapter")


# ---------------------------------------------------------------------------
# Detector — train on the fly or load from cache
# ---------------------------------------------------------------------------


def get_detector(
    adapter_path: str,
    device: str,
    train_corpus: list[str],
    batch_size: int,
    cache_path: Path,
    retrain: bool,
    subsample_frac: float = 0.30,
    subsample_seed: int = 42,
    k_worlds: int = 20,
):
    """Load or train the student-feature blunder detector.

    Returns sklearn GradientBoostingClassifier.
    """
    from sklearn.ensemble import GradientBoostingClassifier

    if cache_path.exists() and not retrain:
        print(f"[detector] Loading cached detector from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            clf = pickle.load(f)
        return clf

    print(f"[detector] Training detector (cache miss at {cache_path})", flush=True)
    print(f"[detector]   adapter={adapter_path}", flush=True)
    print(f"[detector]   train_corpus={train_corpus}", flush=True)
    print(f"[detector]   subsample_frac={subsample_frac}  K_worlds={k_worlds}", flush=True)

    X_train, y_regret_train = collect_samples(
        adapter_path,
        train_corpus,
        device,
        batch_size=batch_size,
        K_worlds=k_worlds,
        subsample_frac=subsample_frac,
        subsample_seed=subsample_seed,
        max_decisions=None,
    )
    y_train = (y_regret_train > BLUNDER_THRESHOLD).astype(np.int32)
    print(f"[detector] Train: {len(y_train)} samples, {y_train.sum()} blunders "
          f"({y_train.mean():.2%})", flush=True)

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"[detector] Fit in {time.time() - t0:.1f}s", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"[detector] Cached to {cache_path}", flush=True)
    return clf


# ---------------------------------------------------------------------------
# PIMC-Q fallback: sample K belief-weighted worlds, run Q_head, average, argmax
# ---------------------------------------------------------------------------


def _pimc_q_argmax(
    batch: dict,
    model,
    is_voids: bool,
    device: str,
    K: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Return legal-masked argmax over avg-Q across K belief-sampled worlds.

    We don't have access to M corpus worlds at serve time; sample from the
    student's belief_head (same trick blunder_detector_student.py uses).
    """
    B = batch["tokens"].shape[0]
    legal = batch["legal_mask"]

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

    state_emb = out["state_emb"]
    belief_logits = out["belief_logits"]
    belief_mask = batch["belief_mask"]

    belief_probs = torch.softmax(belief_logits, dim=-1)  # [B, 28, 3]
    D, S = belief_probs.shape[1], belief_probs.shape[2]
    flat_probs = belief_probs.view(B * D, S)
    samples = torch.multinomial(flat_probs, K, replacement=True, generator=rng)
    samples = samples.view(B, D, K).permute(0, 2, 1)  # [B, K, 28]
    assignment = torch.nn.functional.one_hot(samples, num_classes=S).float()
    m_mask = belief_mask.view(B, 1, D, 1).float()
    assignment = assignment * m_mask  # [B, K, 28, 3]

    state_emb_rep = state_emb.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
    assignment_flat = assignment.reshape(B * K, D, S)

    with torch.no_grad():
        world_emb = model.world_encoder(assignment_flat)
        q_all = model.q_head(state_emb_rep, world_emb)  # [B*K, 7]
    q_all = q_all.view(B, K, 7)
    q_mean = q_all.mean(dim=1)  # [B, 7]
    q_mean_legal = q_mean.masked_fill(~legal, -1e9)
    return q_mean_legal.argmax(dim=-1)


def _next_best_adapter_argmax(
    batch: dict,
    fallback_model,
    fallback_is_voids: bool,
) -> torch.Tensor:
    """Run the fallback adapter's π_me, legal-masked argmax."""
    legal = batch["legal_mask"]
    with torch.no_grad():
        if fallback_is_voids:
            out = fallback_model(
                batch["tokens"], batch["attention_mask"],
                batch["world_assignment"], batch["voids"],
            )
        else:
            out = fallback_model(
                batch["tokens"], batch["attention_mask"],
                batch["world_assignment"],
            )
    logits = out["pi_me_logits"].masked_fill(~legal, -1e9)
    return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------


def run(args) -> int:
    device = args.device
    print(f"device: {device}", flush=True)
    print(f"primary: {args.adapter}", flush=True)
    print(f"eval:    {args.eval}", flush=True)
    print(f"fallbacks: {args.fallback}", flush=True)

    # --- Load primary student ---
    primary, primary_voids = load_student(args.adapter, device)

    # --- Load fallback adapter if next-best-adapter is requested ---
    fallback_model = None
    fallback_is_voids = False
    if "next-best-adapter" in args.fallback:
        if args.fallback_adapter is None:
            print("ERROR: --fallback-adapter required when 'next-best-adapter' is "
                  "in --fallback", file=sys.stderr)
            return 1
        print(f"next-best fallback adapter: {args.fallback_adapter}", flush=True)
        fallback_model, fallback_is_voids = load_student(args.fallback_adapter, device)

    # --- Train / load detector ---
    cache_path = Path(args.detector_cache)
    clf = get_detector(
        args.adapter,
        device,
        args.train_corpus,
        batch_size=args.batch_size,
        cache_path=cache_path,
        retrain=args.retrain_detector,
        subsample_frac=args.subsample_frac,
        subsample_seed=args.subsample_seed,
        k_worlds=args.k_worlds_detector,
    )

    # --- Iterate eval corpus once; collect everything per-decision ---
    ds = JointWorldFullDataset(args.eval, seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    N = len(ds)
    print(f"eval decisions: {N}", flush=True)

    rng_det = torch.Generator(device=device)
    rng_det.manual_seed(args.subsample_seed + 100)
    rng_pimc = torch.Generator(device=device)
    rng_pimc.manual_seed(args.subsample_seed + 200)

    # Per-decision records:
    all_primary_regret = []
    all_primary_match = []   # primary action == oracle argmax
    all_oracle_regret = []   # upper bound via oracle fallback
    all_pimc_regret = []
    all_nextbest_regret = []
    all_pimc_match = []
    all_nextbest_match = []
    all_oracle_match = []
    all_detector_prob = []
    all_decision_idx = []

    fallback_needed = {
        "oracle": "oracle" in args.fallback,
        "pimc-q-k50": "pimc-q-k50" in args.fallback,
        "next-best-adapter": "next-best-adapter" in args.fallback,
    }

    t0 = time.time()
    seen = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["tokens"].shape[0]
        legal = batch["legal_mask"]
        e_q = batch["e_q"]
        e_q_legal = e_q.masked_fill(~legal, float("-inf"))
        oracle_best_eq = e_q_legal.max(dim=-1).values
        oracle_best_action = e_q_legal.argmax(dim=-1)
        idx_t = torch.arange(B, device=device)

        # --- Primary student action + features ---
        with torch.no_grad():
            if primary_voids:
                primary_out = primary(
                    batch["tokens"], batch["attention_mask"],
                    batch["world_assignment"], batch["voids"],
                )
            else:
                primary_out = primary(
                    batch["tokens"], batch["attention_mask"],
                    batch["world_assignment"],
                )
        pi = primary_out["pi_me_logits"].masked_fill(~legal, -1e9)
        primary_action = pi.argmax(dim=-1)
        primary_eq = e_q[idx_t, primary_action]
        primary_regret = (oracle_best_eq - primary_eq).cpu().numpy()
        primary_match = (primary_action == oracle_best_action).cpu().numpy().astype(np.int32)

        # --- Student features → detector probability ---
        feats = _student_features_for_batch(
            batch, primary, primary_voids, device,
            args.k_worlds_detector, rng_det,
        )
        probs = clf.predict_proba(feats)[:, 1]

        # --- Oracle fallback ---
        if fallback_needed["oracle"]:
            oracle_eq = e_q[idx_t, oracle_best_action]
            oracle_regret = (oracle_best_eq - oracle_eq).cpu().numpy()  # always 0
            oracle_match = np.ones(B, dtype=np.int32)
        else:
            oracle_regret = np.zeros(B)
            oracle_match = np.zeros(B, dtype=np.int32)

        # --- PIMC-Q-K50 fallback ---
        if fallback_needed["pimc-q-k50"]:
            pimc_action = _pimc_q_argmax(
                batch, primary, primary_voids, device,
                args.k_worlds_pimc, rng_pimc,
            )
            pimc_eq = e_q[idx_t, pimc_action]
            pimc_regret = (oracle_best_eq - pimc_eq).cpu().numpy()
            pimc_match = (pimc_action == oracle_best_action).cpu().numpy().astype(np.int32)
        else:
            pimc_regret = np.zeros(B)
            pimc_match = np.zeros(B, dtype=np.int32)

        # --- Next-best-adapter fallback ---
        if fallback_needed["next-best-adapter"]:
            nb_action = _next_best_adapter_argmax(
                batch, fallback_model, fallback_is_voids,
            )
            nb_eq = e_q[idx_t, nb_action]
            nb_regret = (oracle_best_eq - nb_eq).cpu().numpy()
            nb_match = (nb_action == oracle_best_action).cpu().numpy().astype(np.int32)
        else:
            nb_regret = np.zeros(B)
            nb_match = np.zeros(B, dtype=np.int32)

        d_idx_np = batch["decision_idx"].cpu().numpy()

        all_primary_regret.extend(primary_regret.tolist())
        all_primary_match.extend(primary_match.tolist())
        all_oracle_regret.extend(oracle_regret.tolist())
        all_oracle_match.extend(oracle_match.tolist())
        all_pimc_regret.extend(pimc_regret.tolist())
        all_pimc_match.extend(pimc_match.tolist())
        all_nextbest_regret.extend(nb_regret.tolist())
        all_nextbest_match.extend(nb_match.tolist())
        all_detector_prob.extend(probs.tolist())
        all_decision_idx.extend(d_idx_np.tolist())

        seen += B
        if seen % (args.batch_size * 2) == 0 or seen == N:
            rate = seen / max(time.time() - t0, 0.001)
            print(f"  processed {seen}/{N}  rate={rate:.1f}/s", flush=True)

    # Convert to arrays
    primary_regret = np.array(all_primary_regret)
    primary_match = np.array(all_primary_match)
    oracle_regret = np.array(all_oracle_regret)
    oracle_match = np.array(all_oracle_match)
    pimc_regret = np.array(all_pimc_regret)
    pimc_match = np.array(all_pimc_match)
    nb_regret = np.array(all_nextbest_regret)
    nb_match = np.array(all_nextbest_match)
    det_prob = np.array(all_detector_prob)
    decision_idx = np.array(all_decision_idx)

    # ========================================================================
    # Report
    # ========================================================================
    print()
    print(f"=== Baseline (no router) ===")
    print(f"  decisions:         {N}")
    print(f"  mean regret:       {primary_regret.mean():.4f}")
    print(f"  bot-match:         {primary_match.mean():.3%}")
    baseline_blunders = int((primary_regret > BLUNDER_THRESHOLD).sum())
    print(f"  blunders (>8 Qpt): {baseline_blunders}")

    # Compute per-flag-rate-per-policy routed regret
    flag_pcts = [5, 10, 15, 20, 25]
    fallback_arr = {
        "oracle": (oracle_regret, oracle_match),
        "pimc-q-k50": (pimc_regret, pimc_match),
        "next-best-adapter": (nb_regret, nb_match),
    }

    # Sort decisions by detector probability (descending)
    order = np.argsort(-det_prob)

    # Collect rows for the results table
    rows = []  # dict per (fallback, flag_pct)
    for flag_pct in flag_pcts:
        k = max(1, int(N * flag_pct / 100))
        flagged = np.zeros(N, dtype=bool)
        flagged[order[:k]] = True
        thresh = float(det_prob[order[k - 1]])

        # Diagnostic: precision/recall against real blunders on eval corpus
        real_blunders = (primary_regret > BLUNDER_THRESHOLD).astype(np.int32)
        tp = int(((real_blunders == 1) & flagged).sum())
        fp = int(((real_blunders == 0) & flagged).sum())
        total_pos = int(real_blunders.sum())
        prec = tp / max(tp + fp, 1)
        recall = tp / max(total_pos, 1)

        for pol_name in args.fallback:
            fb_regret, fb_match = fallback_arr[pol_name]
            # Routed: flagged → fallback, else → primary
            chosen_regret = np.where(flagged, fb_regret, primary_regret)
            chosen_match = np.where(flagged, fb_match, primary_match)

            new_mean = float(chosen_regret.mean())
            new_match = float(chosen_match.mean())
            new_blunders = int((chosen_regret > BLUNDER_THRESHOLD).sum())

            # Blunder bucket analysis
            caught = int(
                ((primary_regret > BLUNDER_THRESHOLD)
                 & flagged
                 & (fb_regret <= BLUNDER_THRESHOLD)).sum()
            )
            introduced = int(
                ((primary_regret <= BLUNDER_THRESHOLD)
                 & flagged
                 & (fb_regret > BLUNDER_THRESHOLD)).sum()
            )
            missed = int(
                ((primary_regret > BLUNDER_THRESHOLD) & ~flagged).sum()
            )

            rows.append({
                "fallback": pol_name,
                "flag_pct": flag_pct,
                "k": k,
                "threshold": thresh,
                "precision": prec,
                "recall": recall,
                "new_mean_regret": new_mean,
                "new_bot_match": new_match,
                "new_blunders": new_blunders,
                "blunders_caught": caught,
                "blunders_introduced": introduced,
                "blunders_missed": missed,
            })

    # --- Main table ---
    print()
    print("=== Routed inference: measured mean regret by (fallback, flag%) ===")
    print(f"baseline mean regret: {primary_regret.mean():.4f}  "
          f"baseline bot-match: {primary_match.mean():.3%}  "
          f"baseline blunders: {baseline_blunders}")
    print()
    hdr = (
        f"{'fallback':>18s}  {'flag%':>5s}  {'thresh':>6s}  "
        f"{'prec':>5s}  {'recl':>5s}  "
        f"{'regret':>7s}  {'match':>6s}  "
        f"{'blund':>5s}  {'caught':>6s}  {'intro':>5s}  {'missed':>6s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['fallback']:>18s}  {r['flag_pct']:>4d}%  {r['threshold']:>6.3f}  "
            f"{r['precision']:>5.3f}  {r['recall']:>5.3f}  "
            f"{r['new_mean_regret']:>7.4f}  {r['new_bot_match']:>6.2%}  "
            f"{r['new_blunders']:>5d}  {r['blunders_caught']:>6d}  "
            f"{r['blunders_introduced']:>5d}  {r['blunders_missed']:>6d}"
        )

    # --- Breakdown by decision_idx ---
    print()
    print("=== Per-decision_idx breakdown (at 20% flag rate, best fallback) ===")
    # Pick the best fallback at flag_pct=20
    rows_20 = [r for r in rows if r["flag_pct"] == 20]
    if rows_20:
        best_row = min(rows_20, key=lambda r: r["new_mean_regret"])
        best_fb = best_row["fallback"]
        print(f"  best fallback @ 20%: {best_fb}  "
              f"(mean regret={best_row['new_mean_regret']:.4f})")
        fb_regret, _ = fallback_arr[best_fb]

        flag_pct = 20
        k = max(1, int(N * flag_pct / 100))
        flagged = np.zeros(N, dtype=bool)
        flagged[order[:k]] = True
        chosen_regret = np.where(flagged, fb_regret, primary_regret)

        print()
        print(f"  {'dec':>3s}  {'n':>3s}  {'base_reg':>8s}  "
              f"{'routed_reg':>10s}  {'Δ':>7s}  {'flag_%':>7s}")
        by_dec: dict[int, list[int]] = defaultdict(list)
        for i, d in enumerate(decision_idx):
            by_dec[int(d)].append(i)
        for d in sorted(by_dec.keys()):
            ids = by_dec[d]
            n = len(ids)
            base_r = primary_regret[ids].mean()
            routed_r = chosen_regret[ids].mean()
            delta = routed_r - base_r
            flag_rate = flagged[ids].mean()
            print(f"  {d:>3d}  {n:>3d}  {base_r:>8.3f}  "
                  f"{routed_r:>10.3f}  {delta:>+7.3f}  {flag_rate:>6.1%}")

    # --- Blunder forensics per policy (at 20%) ---
    print()
    print("=== Blunder forensics @ 20% flag rate ===")
    print(f"  baseline blunders: {baseline_blunders}")
    for r in rows_20:
        print(
            f"  {r['fallback']:>18s}: caught {r['blunders_caught']}/"
            f"{baseline_blunders}  introduced {r['blunders_introduced']}  "
            f"missed {r['blunders_missed']}  "
            f"→ final {r['new_blunders']}"
        )

    # --- Write summary markdown to scratch ---
    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_summary_md(out_path, rows, primary_regret, primary_match,
                      baseline_blunders, args)
    print()
    print(f"Wrote {out_path}")
    return 0


def _write_summary_md(
    out_path: Path,
    rows: list[dict],
    primary_regret: np.ndarray,
    primary_match: np.ndarray,
    baseline_blunders: int,
    args,
) -> None:
    lines = []
    lines.append("# Router eval summary")
    lines.append("")
    lines.append(f"- primary adapter: `{args.adapter}`")
    lines.append(f"- eval corpus: `{args.eval}`  (decisions: {len(primary_regret)})")
    lines.append(f"- detector: student-feature GBClassifier "
                 f"(trained on {args.train_corpus}, "
                 f"{int(args.subsample_frac*100)}% subsample, K_worlds={args.k_worlds_detector})")
    if "next-best-adapter" in args.fallback:
        lines.append(f"- next-best-adapter: `{args.fallback_adapter}`")
    if "pimc-q-k50" in args.fallback:
        lines.append(f"- pimc-q-k50: K={args.k_worlds_pimc} belief-sampled worlds")
    lines.append("")
    lines.append(f"**Baseline** (no router): mean regret = {primary_regret.mean():.4f}"
                 f", bot-match = {primary_match.mean():.2%}, "
                 f"blunders (>8) = {baseline_blunders}")
    lines.append("")
    lines.append("## Measured regret & bot-match by (fallback, flag%)")
    lines.append("")
    lines.append("| fallback | flag% | threshold | prec | recall | mean regret | bot-match | blunders | caught | intro | missed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['fallback']} | {r['flag_pct']}% | {r['threshold']:.3f} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | "
            f"**{r['new_mean_regret']:.4f}** | {r['new_bot_match']:.2%} | "
            f"{r['new_blunders']} | {r['blunders_caught']} | "
            f"{r['blunders_introduced']} | {r['blunders_missed']} |"
        )
    lines.append("")
    lines.append("Legend: caught = baseline blunder flagged & fixed, "
                 "intro = non-blunder flagged & fallback made it worse (>8 Q-pt), "
                 "missed = baseline blunder not flagged.")
    lines.append("")
    out_path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="gus/adapters/v2_voids_3000g_big.pt")
    parser.add_argument("--eval", nargs="+", default=["gus/data/corpus_eval_20.pt"])
    parser.add_argument(
        "--fallback", nargs="+", default=list(FALLBACK_POLICIES),
        choices=list(FALLBACK_POLICIES),
    )
    parser.add_argument(
        "--fallback-adapter",
        default="gus/adapters/v1_full_1000g.pt",
        help="Used when 'next-best-adapter' is in --fallback",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--k-worlds-detector", type=int, default=20)
    parser.add_argument("--k-worlds-pimc", type=int, default=50)
    parser.add_argument(
        "--train-corpus", nargs="+",
        default=[
            "gus/data/corpus_train_chunk_0-99.pt",
            "gus/data/corpus_train_chunk_100-199.pt",
            "gus/data/corpus_train_chunk_200-299.pt",
        ],
    )
    parser.add_argument("--subsample-frac", type=float, default=0.30)
    parser.add_argument("--subsample-seed", type=int, default=42)
    parser.add_argument(
        "--detector-cache", default="scratch/blunder_detector_student.pkl",
    )
    parser.add_argument("--retrain-detector", action="store_true")
    parser.add_argument("--out-md", default="scratch/router_eval_summary.md")
    args = parser.parse_args()

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
