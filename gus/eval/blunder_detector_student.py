"""Blunder detector v2 — STUDENT-DERIVABLE features only.

Follow-up to gus/eval/blunder_detector.py. The v1 detector hit 0.926 ROC-AUC
using oracle E[Q] summary stats (spread/std/min/max) as features — those are
NOT available at inference. This version trains the same
GradientBoostingClassifier but using ONLY features the student can produce
at serve-time:

  - Policy (pi_me_legal softmax): entropy, peak, argmax-margin
  - V_head scalar
  - Q_head spread across K sampled worlds (mean/std/min/max per-legal-action;
    spread-of-mean-Q across legal actions = student's analog of oracle_spread)
  - Belief: max-prob, entropy, high-confidence count
  - Meta: decision_idx, trick_num, trick_pos, player, legal_count,
    declaration one-hot, per-opponent void counts

Same labeling, same split semantics, same GBClassifier, same report layout
as v1. Compare headline AUC directly.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from gus.eval.eval_regret import _load_student
from gus.model.dataset_seq_world import JointWorldFullDataset

DECL_OFFSET = 30


# ---------------------------------------------------------------------------
# Student-feature extraction
# ---------------------------------------------------------------------------


def _student_features_for_batch(
    batch: dict,
    model,
    is_voids: bool,
    device: str,
    K_worlds: int,
    rng: torch.Generator,
) -> np.ndarray:
    """Run the student and compute ONLY student-derivable features.

    Feature layout (len = 53):
      0             decision_idx
      1             trick_num
      2             trick_pos
      3             player
      4..13         declaration one-hot (10 dim)
      14            legal_count
      15..17        voids count per relative opponent (3)
      18            V_head scalar
      19            pi entropy (over legal)
      20            pi peak (max prob over legal)
      21            pi margin (max1 - max2) over legal
      22            pi argmax action idx
      23..29        q_mean_across_worlds per-action [7]  (illegal→0)
      30..36        q_std_across_worlds  per-action [7]
      37            spread of q_mean across legal (max-min)  ← student oracle_spread analog
      38            std of q_mean across legal
      39            q_mean min across legal
      40            q_mean max across legal
      41            q_std_chosen (at pi-argmax)
      42            q_mean_chosen
      43            q_mean_rank (rank of chosen action by q_mean among legal)
      44            mean q_std across legal (how uncertain student is overall)
      45            belief max-prob (mean across unseen dominoes)
      46            belief entropy (mean across unseen dominoes)
      47            belief high-conf count (number of unseen with max_p > 0.8)
      48            spread of per-world-sampled Q (mean across worlds of max_a Q -
                    min across worlds of max_a Q) — "world disagreement on best action"
      49            std across worlds of which-action-wins (fraction of worlds
                    whose argmax differs from the mode)
      50            V_head - policy_expected_Q (consistency signal)
      51            V_head - q_mean_chosen (inter-head disagreement)
      52            |V_head| magnitude
    """
    B = batch["tokens"].shape[0]
    N_FEAT = 53
    feats = np.zeros((B, N_FEAT), dtype=np.float32)

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

    state_emb = out["state_emb"]  # [B, d_model]
    belief_logits = out["belief_logits"]  # [B, 28, 3]
    pi_logits = out["pi_me_logits"]  # [B, 7]
    v = out["v"]  # [B]
    legal_mask = batch["legal_mask"]  # [B, 7] bool

    # ---- Policy softmax over legal actions ----
    pi_masked = pi_logits.masked_fill(~legal_mask, -1e9)
    pi_probs = torch.softmax(pi_masked, dim=-1)  # [B, 7] (illegal ≈ 0)

    # ---- Sample K worlds per decision from the decision's own M corpus worlds.
    # We don't have direct access to M worlds here (dataset only yields 1), so
    # instead we sample K belief-weighted worlds from belief_head. This IS
    # realistic at inference time — it's exactly what pimc-belief mode does.
    # That way features are genuinely student-derivable.
    B_ = belief_logits.shape[0]
    D = belief_logits.shape[1]
    S = belief_logits.shape[2]
    belief_mask = batch["belief_mask"]  # [B, 28] bool

    belief_probs = torch.softmax(belief_logits, dim=-1)  # [B, 28, 3]
    flat_probs = belief_probs.view(B_ * D, S)
    # multinomial requires non-zero rows; for masked rows (not unseen), rows may be
    # legitimate probs too but we zero them after sampling.
    samples = torch.multinomial(flat_probs, K_worlds, replacement=True, generator=rng)
    samples = samples.view(B_, D, K_worlds).permute(0, 2, 1)  # [B, K, 28]
    assignment = torch.nn.functional.one_hot(samples, num_classes=S).float()  # [B, K, 28, 3]
    m_mask = belief_mask.view(B_, 1, D, 1).float()
    assignment = assignment * m_mask  # [B, K, 28, 3]

    # Q_head over these K worlds
    state_emb_rep = state_emb.unsqueeze(1).expand(B_, K_worlds, -1).reshape(B_ * K_worlds, -1)
    assignment_flat = assignment.reshape(B_ * K_worlds, 28, 3)
    with torch.no_grad():
        world_emb = model.world_encoder(assignment_flat)
        q_all = model.q_head(state_emb_rep, world_emb)  # [B*K, 7]
    q_all = q_all.view(B_, K_worlds, 7)  # [B, K, 7]

    # Per-action q stats across worlds
    q_mean = q_all.mean(dim=1)  # [B, 7]
    q_std = q_all.std(dim=1)    # [B, 7]
    # zero out illegal positions for stat features
    q_mean_legal = q_mean.masked_fill(~legal_mask, 0.0)
    q_std_legal = q_std.masked_fill(~legal_mask, 0.0)

    # World-disagreement: per-world argmax among legal, then count how many
    # different winning actions there are.
    q_all_legal = q_all.masked_fill(~legal_mask.unsqueeze(1), -1e9)
    per_world_best = q_all_legal.argmax(dim=-1)  # [B, K]

    # Pull CPU for numpy feature fill
    pi_probs_np = pi_probs.cpu().numpy()
    pi_argmax_np = pi_masked.argmax(dim=-1).cpu().numpy()
    v_np = v.cpu().numpy()
    q_mean_np = q_mean.cpu().numpy()
    q_std_np = q_std.cpu().numpy()
    q_mean_legal_np = q_mean_legal.cpu().numpy()
    legal_np = legal_mask.cpu().numpy().astype(bool)
    per_world_best_np = per_world_best.cpu().numpy()
    belief_probs_np = belief_probs.cpu().numpy()
    belief_mask_np = belief_mask.cpu().numpy().astype(bool)
    e_q_np = batch["e_q"].cpu().numpy()
    tokens_np = batch["tokens"].cpu().numpy()
    voids_np = batch["voids"].cpu().numpy()
    d_idx_np = batch["decision_idx"].cpu().numpy()
    player_np = batch["player"].cpu().numpy()

    for b in range(B):
        d_idx = int(d_idx_np[b])
        feats[b, 0] = float(d_idx)
        feats[b, 1] = float(d_idx // 4)
        feats[b, 2] = float(d_idx % 4)
        feats[b, 3] = float(int(player_np[b]))

        # decl one-hot
        decl_token = int(tokens_np[b, 1, 0])
        decl_id = decl_token - DECL_OFFSET if decl_token >= DECL_OFFSET else 0
        if 0 <= decl_id < 10:
            feats[b, 4 + decl_id] = 1.0

        legal_b = legal_np[b]
        n_legal = int(legal_b.sum())
        feats[b, 14] = float(n_legal)

        voids_b = voids_np[b].reshape(3, 8)
        feats[b, 15] = float(voids_b[0].sum())
        feats[b, 16] = float(voids_b[1].sum())
        feats[b, 17] = float(voids_b[2].sum())

        feats[b, 18] = float(v_np[b])

        # Policy features over legal actions
        pi_b = pi_probs_np[b]
        pi_legal_probs = pi_b[legal_b]
        if len(pi_legal_probs) > 0:
            # Entropy (clip to avoid log(0))
            pi_safe = np.clip(pi_legal_probs, 1e-10, 1.0)
            pi_ent = float(-(pi_safe * np.log(pi_safe)).sum())
            sorted_legal = np.sort(pi_legal_probs)[::-1]
            pi_peak = float(sorted_legal[0])
            pi_margin = float(sorted_legal[0] - (sorted_legal[1] if len(sorted_legal) > 1 else 0.0))
        else:
            pi_ent = 0.0
            pi_peak = 0.0
            pi_margin = 0.0
        feats[b, 19] = pi_ent
        feats[b, 20] = pi_peak
        feats[b, 21] = pi_margin

        pi_argmax = int(pi_argmax_np[b])
        feats[b, 22] = float(pi_argmax)

        # Per-action q_mean / q_std
        feats[b, 23:30] = q_mean_np[b]
        feats[b, 30:37] = q_std_np[b]

        # Legal-only Q stats
        if n_legal > 0:
            q_mean_legal_vals = q_mean_np[b][legal_b]
            q_std_legal_vals = q_std_np[b][legal_b]
            feats[b, 37] = float(q_mean_legal_vals.max() - q_mean_legal_vals.min())
            feats[b, 38] = float(q_mean_legal_vals.std()) if n_legal >= 2 else 0.0
            feats[b, 39] = float(q_mean_legal_vals.min())
            feats[b, 40] = float(q_mean_legal_vals.max())
            # chosen-action q stats
            if legal_b[pi_argmax]:
                feats[b, 41] = float(q_std_np[b, pi_argmax])
                feats[b, 42] = float(q_mean_np[b, pi_argmax])
                # rank (0 = best)
                rank = int((q_mean_legal_vals > q_mean_np[b, pi_argmax]).sum())
                feats[b, 43] = float(rank)
            else:
                feats[b, 41] = 0.0
                feats[b, 42] = 0.0
                feats[b, 43] = float(n_legal)
            feats[b, 44] = float(q_std_legal_vals.mean())

        # Belief features (unseen dominoes only)
        bp_b = belief_probs_np[b]  # [28, 3]
        bm_b = belief_mask_np[b]    # [28]
        if bm_b.any():
            bp_unseen = bp_b[bm_b]  # [n_unseen, 3]
            max_p = bp_unseen.max(axis=-1)  # [n_unseen]
            bp_safe = np.clip(bp_unseen, 1e-10, 1.0)
            belief_ent = -(bp_safe * np.log(bp_safe)).sum(axis=-1)
            feats[b, 45] = float(max_p.mean())
            feats[b, 46] = float(belief_ent.mean())
            feats[b, 47] = float((max_p > 0.8).sum())
        else:
            feats[b, 45] = 0.0
            feats[b, 46] = 0.0
            feats[b, 47] = 0.0

        # World disagreement
        pw_best = per_world_best_np[b]  # [K]
        # range of "which action won" spread: use max of max_a Q across worlds minus min of max_a Q
        worlds_max_q = q_all_legal[b].max(dim=-1).values.cpu().numpy()  # [K]
        feats[b, 48] = float(worlds_max_q.max() - worlds_max_q.min())
        # fraction of worlds whose winner disagrees with mode
        if len(pw_best) > 0:
            vals, counts = np.unique(pw_best, return_counts=True)
            mode_count = counts.max()
            feats[b, 49] = float(1.0 - mode_count / len(pw_best))
        else:
            feats[b, 49] = 0.0

        # Inter-head disagreement
        # policy-expected Q (over legal, using q_mean)
        if n_legal > 0:
            legal_q = q_mean_np[b][legal_b]
            legal_pi = pi_b[legal_b]
            pol_exp_q = float((legal_pi * legal_q).sum())
        else:
            pol_exp_q = 0.0
        feats[b, 50] = float(v_np[b] - pol_exp_q)
        feats[b, 51] = feats[b, 18] - feats[b, 42]
        feats[b, 52] = float(abs(v_np[b]))

    return feats


FEATURE_NAMES = [
    "decision_idx", "trick_num", "trick_pos", "player",
    *[f"decl_{i}" for i in range(10)],
    "legal_count",
    "voids_left_opp", "voids_partner", "voids_right_opp",
    "v_head",
    "pi_entropy", "pi_peak", "pi_margin", "pi_argmax_idx",
    *[f"q_mean_a{i}" for i in range(7)],
    *[f"q_std_a{i}" for i in range(7)],
    "q_mean_spread_legal", "q_mean_std_legal",
    "q_mean_min_legal", "q_mean_max_legal",
    "q_std_chosen", "q_mean_chosen", "q_mean_rank_chosen",
    "q_std_mean_legal",
    "belief_max_prob_mean", "belief_entropy_mean", "belief_high_conf_count",
    "world_max_q_spread", "world_disagreement_frac",
    "v_minus_polexpq", "v_minus_qmean_chosen", "v_abs",
]

assert len(FEATURE_NAMES) == 53, f"feature name/count mismatch: {len(FEATURE_NAMES)}"


# ---------------------------------------------------------------------------
# Collect (features, regret)
# ---------------------------------------------------------------------------


def collect_samples(
    adapter_path: str,
    corpus_paths: list[str],
    device: str,
    batch_size: int,
    K_worlds: int,
    subsample_frac: float | None,
    subsample_seed: int,
    max_decisions: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading adapter: {adapter_path}", flush=True)
    model, is_voids = _load_student(adapter_path, device)
    model.eval()

    print(f"Loading corpus: {corpus_paths}", flush=True)
    ds_full = JointWorldFullDataset(corpus_paths, seed=42)
    N_full = len(ds_full)

    if subsample_frac is not None and subsample_frac < 1.0:
        rng = np.random.default_rng(subsample_seed)
        keep = rng.random(N_full) < subsample_frac
        idxs = np.where(keep)[0].tolist()
        ds = Subset(ds_full, idxs)
        print(f"  Subsample: {len(ds)}/{N_full} decisions ({subsample_frac:.0%})", flush=True)
    else:
        ds = ds_full
        print(f"  Dataset size: {len(ds)}", flush=True)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    rng_torch = torch.Generator(device=device)
    rng_torch.manual_seed(subsample_seed)

    all_feats = []
    all_regret = []
    seen = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        B = batch["tokens"].shape[0]

        feats = _student_features_for_batch(
            batch, model, is_voids, device, K_worlds, rng_torch
        )

        # Regret: oracle_best - student_eq (same as v1)
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
        pi = out["pi_me_logits"].masked_fill(~batch["legal_mask"], -1e9)
        student_action = pi.argmax(dim=-1)
        e_q = batch["e_q"]
        e_q_legal = e_q.masked_fill(~batch["legal_mask"], float("-inf"))
        oracle_best = e_q_legal.max(dim=-1).values
        idx_t = torch.arange(B, device=device)
        student_eq = e_q[idx_t, student_action]
        regret = (oracle_best - student_eq).cpu().numpy()

        all_feats.append(feats)
        all_regret.append(regret)

        seen += B
        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0
            rate = seen / max(elapsed, 0.001)
            eta = (len(ds) - seen) / max(rate, 0.001)
            print(f"  processed {seen}/{len(ds)}  rate={rate:.1f}/s  eta={eta:.0f}s", flush=True)

        if max_decisions is not None and seen >= max_decisions:
            break

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_regret, axis=0)
    print(f"Collected {X.shape[0]} samples  features={X.shape[1]}  total_time={time.time()-t0:.1f}s",
          flush=True)
    return X, y


# ---------------------------------------------------------------------------
# Train & evaluate
# ---------------------------------------------------------------------------


def train_and_eval(
    X_train: np.ndarray,
    y_regret_train: np.ndarray,
    X_test: np.ndarray,
    y_regret_test: np.ndarray,
    blunder_threshold: float = 8.0,
    out_dir: Path | None = None,
):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    y_train = (y_regret_train > blunder_threshold).astype(np.int32)
    y_test = (y_regret_test > blunder_threshold).astype(np.int32)

    print(f"\nTrain: {len(y_train)} samples, {y_train.sum()} blunders ({y_train.mean():.2%})")
    print(f"Test:  {len(y_test)} samples, {y_test.sum()} blunders ({y_test.mean():.2%})")

    clf = GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42,
    )
    print("\nTraining GradientBoostingClassifier(n_estimators=300, max_depth=3)...")
    clf.fit(X_train, y_train)

    probs_test = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs_test)
    avg_prec = average_precision_score(y_test, probs_test)
    print(f"\nTest ROC-AUC: {auc:.4f}")
    print(f"Test Avg Precision (PR-AUC): {avg_prec:.4f}")

    importance = clf.feature_importances_
    ranked = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])
    print("\nTop-15 features by importance:")
    for name, imp in ranked[:15]:
        print(f"  {name:28s} {imp:.4f}")

    print("\nThreshold sweep — flag_rate → precision/recall + projected mean regret")
    print(f"{'flag%':>6s}  {'thresh':>6s}  {'prec':>6s}  {'recall':>6s}  "
          f"{'caught_regret':>13s}  {'missed_regret':>13s}  {'new_mean':>9s}")

    baseline_mean_regret = float(y_regret_test.mean())
    N = len(y_regret_test)

    sweep_rows = []
    for flag_pct in [5, 10, 15, 20, 25, 30, 40, 50]:
        k = max(1, int(N * flag_pct / 100))
        order = np.argsort(-probs_test)
        flagged = np.zeros(N, dtype=bool)
        flagged[order[:k]] = True
        thresh = float(probs_test[order[k - 1]])

        tp = int((flagged & (y_test == 1)).sum())
        fp = int((flagged & (y_test == 0)).sum())
        total_pos = int(y_test.sum())
        prec = tp / max(tp + fp, 1)
        recall = tp / max(total_pos, 1)

        missed_regret_sum = float(y_regret_test[~flagged].sum())
        caught_regret_sum = float(y_regret_test[flagged].sum())
        new_mean = missed_regret_sum / N

        sweep_rows.append({
            "flag_pct": flag_pct, "threshold": thresh,
            "precision": prec, "recall": recall,
            "caught_regret_sum": caught_regret_sum,
            "missed_regret_sum": missed_regret_sum,
            "new_mean_regret": new_mean,
        })
        print(f"{flag_pct:>5d}%  {thresh:>6.3f}  {prec:>6.3f}  {recall:>6.3f}  "
              f"{caught_regret_sum:>13.2f}  {missed_regret_sum:>13.2f}  {new_mean:>9.4f}")

    print(f"\nBaseline mean regret (no detector): {baseline_mean_regret:.4f}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "blunder_detector_student_importance.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature", "importance"])
            for name, imp in ranked:
                w.writerow([name, f"{imp:.6f}"])

        with open(out_dir / "blunder_detector_student_sweep.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "flag_pct", "threshold", "precision", "recall",
                "caught_regret_sum", "missed_regret_sum", "new_mean_regret",
            ])
            for r in sweep_rows:
                w.writerow([
                    r["flag_pct"], f"{r['threshold']:.6f}",
                    f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                    f"{r['caught_regret_sum']:.4f}",
                    f"{r['missed_regret_sum']:.4f}",
                    f"{r['new_mean_regret']:.4f}",
                ])

        with open(out_dir / "blunder_detector_student_summary.txt", "w") as f:
            f.write(f"ROC-AUC: {auc:.4f}\n")
            f.write(f"PR-AUC: {avg_prec:.4f}\n")
            f.write(f"Baseline mean regret: {baseline_mean_regret:.4f}\n")
            f.write(f"N_train: {len(y_train)}  N_test: {len(y_test)}\n")
            f.write(f"Blunder threshold (Q-pt): {blunder_threshold}\n")
            f.write(f"\nReference (oracle-feature v1): ROC-AUC=0.9260  PR-AUC=0.2915\n")
            f.write("\nTop-15 features:\n")
            for name, imp in ranked[:15]:
                f.write(f"  {name:28s} {imp:.6f}\n")
            f.write("\nSweep:\n")
            for r in sweep_rows:
                f.write(
                    f"  flag={r['flag_pct']:3d}%  "
                    f"thresh={r['threshold']:.3f}  "
                    f"prec={r['precision']:.3f}  "
                    f"recall={r['recall']:.3f}  "
                    f"new_mean={r['new_mean_regret']:.4f}\n"
                )
        print(f"\nArtifacts written to: {out_dir}")

    return clf, auc, avg_prec, ranked, sweep_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="gus/adapters/v2_voids_3000g_big.pt")
    parser.add_argument(
        "--train-corpus", nargs="+",
        default=[
            "gus/data/corpus_train_chunk_0-99.pt",
            "gus/data/corpus_train_chunk_100-199.pt",
            "gus/data/corpus_train_chunk_200-299.pt",
        ],
    )
    parser.add_argument(
        "--test-corpus", nargs="+",
        default=[
            "gus/data/corpus_train_chunk_900-999.pt",
            "gus/data/corpus_train_chunk_1000-1099.pt",
        ],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k-worlds", type=int, default=20)
    parser.add_argument("--subsample-frac", type=float, default=0.30)
    parser.add_argument("--subsample-seed", type=int, default=42)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--blunder-threshold", type=float, default=8.0)
    parser.add_argument("--out-dir", default="scratch")
    args = parser.parse_args()

    device = args.device
    print(f"device: {device}", flush=True)
    print(f"K_worlds per decision: {args.k_worlds}", flush=True)
    print(f"Subsample frac: {args.subsample_frac}", flush=True)

    print("\n=== Gathering TRAIN samples ===", flush=True)
    X_train, y_train_regret = collect_samples(
        args.adapter, args.train_corpus, device,
        batch_size=args.batch_size, K_worlds=args.k_worlds,
        subsample_frac=args.subsample_frac,
        subsample_seed=args.subsample_seed,
        max_decisions=args.max_train,
    )

    print("\n=== Gathering TEST samples ===", flush=True)
    X_test, y_test_regret = collect_samples(
        args.adapter, args.test_corpus, device,
        batch_size=args.batch_size, K_worlds=args.k_worlds,
        subsample_frac=args.subsample_frac,
        subsample_seed=args.subsample_seed + 1,
        max_decisions=args.max_test,
    )

    out_dir = Path(args.out_dir)
    train_and_eval(
        X_train, y_train_regret,
        X_test, y_test_regret,
        blunder_threshold=args.blunder_threshold,
        out_dir=out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
