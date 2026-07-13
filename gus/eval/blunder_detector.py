"""Blunder detector: tiny classifier predicting whether the Gus student
will blunder (regret > 8 Q-pt) at a given decision, from STATE-ONLY features.

Pipeline:
  1. Run student `v2_voids_3000g_big.pt` across a slice of the training
     corpus (seeds 0-899) to collect (state_features, regret) labels.
  2. Train a sklearn GradientBoostedClassifier on blunder=1{regret > 8}.
  3. Evaluate on a held-out test slice (seeds 900000+, or a disjoint
     portion of the available corpus).
  4. Report ROC-AUC, PR curve, feature importance, and the "expected
     mean-regret-after-replacement" curve vs flag rate.

Features are state-only so the detector doesn't need the student to run.
Output: console report + CSV artifacts in scratch/.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from gus.eval.eval_regret import _pick_device
from gus.model.load import load_student
from gus.model.dataset_seq_world import JointWorldFullDataset


# ---------------------------------------------------------------------------
# Feature extraction (STATE-ONLY — does NOT depend on student outputs)
# ---------------------------------------------------------------------------


def _state_features_for_batch(batch: dict) -> np.ndarray:
    """Compute state-only features for a batch of decisions.

    Feature layout (all numeric):
      0:          decision_idx (int 0..27)
      1:          player (int 0..3)
      2..11:      declaration one-hot (10 dim)
      12:         legal_action_count
      13..16:     oracle_eq min, max, range, std across LEGAL actions
      17:         trick_num (decision_idx // 4)
      18:         trick_pos (decision_idx % 4)
      19:         hand_size_now (7 - plays I have made)
      20:         unseen_count (28 - 7 - played_total)
      21..23:     voids observed per opponent (count of void suits, 3 opps)
      24..27:     played-count by rel seat (me, left, partner, right)

    NB: we do NOT compute anything from the student's own outputs. Oracle
    E[Q] comes from the corpus (oracle/solver output), not the student.
    """
    B = batch["tokens"].shape[0]
    N_FEAT = 28

    feats = np.zeros((B, N_FEAT), dtype=np.float32)

    decision_idx = batch["decision_idx"].cpu().numpy()
    player = batch["player"].cpu().numpy()
    legal_mask = batch["legal_mask"].cpu().numpy().astype(bool)  # [B, 7]
    e_q = batch["e_q"].cpu().numpy()  # [B, 7]
    voids = batch["voids"].cpu().numpy()  # [B, 24] = 3 opps × 8 suits
    tokens = batch["tokens"].cpu().numpy() if batch["tokens"].dim() == 2 else None

    for b in range(B):
        d_idx = int(decision_idx[b])
        feats[b, 0] = float(d_idx)
        feats[b, 1] = float(int(player[b]))

        # decl one-hot — reverse-engineer from e_q/legal/... actually the
        # decl token is encoded in tokens[b, 1] as DECL_OFFSET + decl_id.
        # tokens shape = [B, L, 5] (5 channels). Take channel 0, pos 1.
        decl_token = int(batch["tokens"][b, 1, 0].item())
        decl_id = decl_token - 30 if decl_token >= 30 else 0  # DECL_OFFSET=30
        if 0 <= decl_id < 10:
            feats[b, 2 + decl_id] = 1.0

        legal_b = legal_mask[b]
        n_legal = int(legal_b.sum())
        feats[b, 12] = float(n_legal)

        if n_legal >= 1:
            legal_eq = e_q[b, legal_b]
            feats[b, 13] = float(legal_eq.min())
            feats[b, 14] = float(legal_eq.max())
            feats[b, 15] = float(legal_eq.max() - legal_eq.min())
            feats[b, 16] = float(legal_eq.std()) if n_legal >= 2 else 0.0
        # else: zeros (shouldn't happen, there's always 1 legal action)

        trick_num = d_idx // 4
        trick_pos = d_idx % 4
        feats[b, 17] = float(trick_num)
        feats[b, 18] = float(trick_pos)
        feats[b, 19] = float(7 - trick_num)  # hand size after trick_num plays-by-me
        feats[b, 20] = float(28 - 7 - d_idx)  # unseen after d_idx plays total (approx)

        # voids: [24] = 3 opps × 8 suits. Count voids per opponent.
        v = voids[b].reshape(3, 8)
        feats[b, 21] = float(v[0].sum())
        feats[b, 22] = float(v[1].sum())
        feats[b, 23] = float(v[2].sum())

        # played-count per rel seat = number of completed plays by rel seat
        # up through decision d_idx. In 42, by decision k we have floor(k/4)
        # completed tricks (all 4 seats played) plus (k mod 4) seats in the
        # current trick that already played. Approximation: use d_idx/4 for
        # each seat, +1 for seats "before" the current player's turn.
        feats[b, 24] = float(trick_num + (1 if trick_pos >= 0 else 0))
        feats[b, 25] = float(trick_num + (1 if trick_pos >= 1 else 0))
        feats[b, 26] = float(trick_num + (1 if trick_pos >= 2 else 0))
        feats[b, 27] = float(trick_num + (1 if trick_pos >= 3 else 0))

    return feats


FEATURE_NAMES = [
    "decision_idx",
    "player",
    *[f"decl_{i}" for i in range(10)],
    "legal_count",
    "oracle_eq_min",
    "oracle_eq_max",
    "oracle_spread",
    "oracle_eq_std",
    "trick_num",
    "trick_pos",
    "hand_size",
    "unseen_approx",
    "voids_left_opp",
    "voids_partner",
    "voids_right_opp",
    "plays_rel_0",
    "plays_rel_1",
    "plays_rel_2",
    "plays_rel_3",
]


# ---------------------------------------------------------------------------
# Collect (features, regret) labels by running the student
# ---------------------------------------------------------------------------


def collect_samples(
    adapter_path: str,
    corpus_paths: list[str],
    device: str,
    batch_size: int = 64,
    max_decisions: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the student and return (features [N, F], regret [N], decision_idx [N])."""
    print(f"Loading adapter: {adapter_path}", flush=True)
    model, is_voids = load_student(adapter_path, device)
    model.eval()

    print(f"Loading corpus: {corpus_paths}", flush=True)
    ds = JointWorldFullDataset(corpus_paths, seed=42)
    print(f"  Dataset size: {len(ds)}", flush=True)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_feats = []
    all_regret = []
    all_d_idx = []

    seen = 0
    for i, batch in enumerate(loader):
        batch_dev = {k: v.to(device) for k, v in batch.items()}
        B = batch_dev["tokens"].shape[0]

        with torch.no_grad():
            if is_voids:
                out = model(
                    batch_dev["tokens"], batch_dev["attention_mask"],
                    batch_dev["world_assignment"], batch_dev["voids"],
                )
            else:
                out = model(
                    batch_dev["tokens"], batch_dev["attention_mask"],
                    batch_dev["world_assignment"],
                )

        pi = out["pi_me_logits"].masked_fill(~batch_dev["legal_mask"], -1e9)
        student_action = pi.argmax(dim=-1)  # [B]
        e_q = batch_dev["e_q"]
        e_q_legal = e_q.masked_fill(~batch_dev["legal_mask"], float("-inf"))
        oracle_best = e_q_legal.max(dim=-1).values
        idx = torch.arange(B, device=device)
        student_eq = e_q[idx, student_action]
        regret = (oracle_best - student_eq).cpu().numpy()  # [B]

        feats = _state_features_for_batch(batch)

        all_feats.append(feats)
        all_regret.append(regret)
        all_d_idx.append(batch["decision_idx"].cpu().numpy())

        seen += B
        if (i + 1) % 10 == 0:
            print(f"  processed {seen}/{len(ds)}", flush=True)

        if max_decisions is not None and seen >= max_decisions:
            break

    X = np.concatenate(all_feats, axis=0)
    y = np.concatenate(all_regret, axis=0)
    d_idx = np.concatenate(all_d_idx, axis=0)
    print(f"Collected {X.shape[0]} samples  features={X.shape[1]}", flush=True)
    return X, y, d_idx


# ---------------------------------------------------------------------------
# Train + evaluate classifier
# ---------------------------------------------------------------------------


def train_and_eval(
    X_train: np.ndarray,
    y_regret_train: np.ndarray,
    X_test: np.ndarray,
    y_regret_test: np.ndarray,
    blunder_threshold: float = 8.0,
    out_dir: Path = None,
):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

    y_train = (y_regret_train > blunder_threshold).astype(np.int32)
    y_test = (y_regret_test > blunder_threshold).astype(np.int32)

    print(f"\nTrain: {len(y_train)} samples, {y_train.sum()} blunders ({y_train.mean():.2%})")
    print(f"Test:  {len(y_test)} samples, {y_test.sum()} blunders ({y_test.mean():.2%})")

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )
    print("\nTraining GradientBoostingClassifier(n_estimators=300, max_depth=3)...")
    clf.fit(X_train, y_train)

    probs_test = clf.predict_proba(X_test)[:, 1]

    # Core metrics
    auc = roc_auc_score(y_test, probs_test)
    avg_prec = average_precision_score(y_test, probs_test)
    print(f"\nTest ROC-AUC: {auc:.4f}")
    print(f"Test Avg Precision (PR-AUC): {avg_prec:.4f}")

    # Feature importance
    importance = clf.feature_importances_
    ranked = sorted(
        zip(FEATURE_NAMES, importance), key=lambda x: -x[1]
    )
    print("\nTop-15 features by importance:")
    for name, imp in ranked[:15]:
        print(f"  {name:20s} {imp:.4f}")

    # Precision/recall at thresholds that flag X% of decisions
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
        fn = int((~flagged & (y_test == 1)).sum())
        total_pos = int(y_test.sum())
        prec = tp / max(tp + fp, 1)
        recall = tp / max(total_pos, 1)

        # Business case: if we replace flagged decisions with ORACLE argmax,
        # regret on flagged goes to 0. So projected mean regret = sum of
        # regret on un-flagged only.
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
    print(f"Oracle-replacement ceiling (flag 100%): 0.0000")

    # Save artifacts
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # Feature importance CSV
        with open(out_dir / "blunder_detector_importance.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature", "importance"])
            for name, imp in ranked:
                w.writerow([name, f"{imp:.6f}"])

        # Sweep CSV
        with open(out_dir / "blunder_detector_sweep.csv", "w", newline="") as f:
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

        # Summary JSON-ish
        with open(out_dir / "blunder_detector_summary.txt", "w") as f:
            f.write(f"ROC-AUC: {auc:.4f}\n")
            f.write(f"PR-AUC: {avg_prec:.4f}\n")
            f.write(f"Baseline mean regret: {baseline_mean_regret:.4f}\n")
            f.write(f"N_train: {len(y_train)}  N_test: {len(y_test)}\n")
            f.write(f"Blunder threshold (Q-pt): {blunder_threshold}\n")
            f.write("\nTop-15 features:\n")
            for name, imp in ranked[:15]:
                f.write(f"  {name:20s} {imp:.6f}\n")
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        default="gus/adapters/v2_voids_3000g_big.pt",
    )
    parser.add_argument(
        "--train-corpus",
        nargs="+",
        default=["gus/data/corpus_train_chunk_0-99.pt"],
        help="Corpus chunk(s) to use for detector TRAINING",
    )
    parser.add_argument(
        "--test-corpus",
        nargs="+",
        default=["gus/data/corpus_train_chunk_100-199.pt"],
        help="Corpus chunk(s) for detector TEST (must be DIFFERENT from train)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-train", type=int, default=2800)
    parser.add_argument("--max-test", type=int, default=1400)
    parser.add_argument("--blunder-threshold", type=float, default=8.0)
    parser.add_argument("--out-dir", default="scratch")
    args = parser.parse_args()

    device = args.device
    print(f"device: {device}", flush=True)

    # 1. TRAIN features + labels (student on held-out-to-adapter data)
    print("\n=== Gathering TRAIN samples ===", flush=True)
    X_train, y_train_regret, _ = collect_samples(
        args.adapter,
        args.train_corpus,
        device,
        batch_size=args.batch_size,
        max_decisions=args.max_train,
    )

    # 2. TEST features + labels
    print("\n=== Gathering TEST samples ===", flush=True)
    X_test, y_test_regret, _ = collect_samples(
        args.adapter,
        args.test_corpus,
        device,
        batch_size=args.batch_size,
        max_decisions=args.max_test,
    )

    # 3. Train + eval classifier
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
