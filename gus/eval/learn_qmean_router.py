"""Train a tiny learned router for the Q-mean second opinion.

This is a held-out sanity check for the hand-written Q-mean router gates.
It trains on one corpus and evaluates on another:

  direct π         -> default action
  Q-mean           -> second opinion from belief-sampled worlds
  learned router   -> route only the highest-scoring disagreements to Q-mean

The default target is deliberately narrow: "direct would blunder, and Q-mean
would not." That optimizes for blunder-rate reduction instead of generic regret
improvement.

Example:
  python -u gus/eval/learn_qmean_router.py \
    --adapter gus/adapters/v3_consistency_10000g.pt \
    --train gus/data/corpus_train_100.pt \
    --eval gus/data/corpus_eval_20.pt \
    --k 100 \
    --device cpu
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from gus.model.load import load_student
from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.sample_worlds import sample_worlds


FEATURE_NAMES = [
    "disagree",
    "pi_peak",
    "pi_entropy",
    "pi_margin",
    "qmean_margin",
    "qmean_minus_direct_q",
    "q_at_direct",
    "q_at_qmean",
    "v",
    "v_minus_q_at_direct",
    "v_minus_q_at_qmean",
    "decision_idx",
    "trick_num",
    "trick_pos",
    "legal_count",
    "direct_action",
    "qmean_action",
]


@dataclass
class Rows:
    direct_regret: np.ndarray
    qmean_regret: np.ndarray
    direct_match: np.ndarray
    qmean_match: np.ndarray
    disagree: np.ndarray


@dataclass
class Summary:
    name: str
    regret: float
    blunders: int
    big_misses: int
    bot_match: float
    routed_frac: float | None = None
    fixed_blunders: int | None = None
    new_blunders: int | None = None


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _collect(
    model,
    is_voids: bool,
    corpus_path: str,
    device: str,
    k: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, Rows]:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    ds = JointWorldFullDataset(corpus_path, seed=42)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    features: list[list[float]] = []
    direct_regrets: list[float] = []
    qmean_regrets: list[float] = []
    direct_matches: list[int] = []
    qmean_matches: list[int] = []
    disagreements: list[int] = []

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        B = batch["tokens"].shape[0]
        idx = torch.arange(B, device=device)
        legal = batch["legal_mask"]
        e_q = batch["e_q"]
        e_q_legal = e_q.masked_fill(~legal, float("-inf"))
        best = e_q_legal.max(dim=-1).values
        best_action = e_q_legal.argmax(dim=-1)

        with torch.no_grad():
            if is_voids:
                out = model(
                    batch["tokens"],
                    batch["attention_mask"],
                    batch["world_assignment"],
                    batch["voids"],
                )
            else:
                out = model(
                    batch["tokens"],
                    batch["attention_mask"],
                    batch["world_assignment"],
                )

        pi_logits = out["pi_me_logits"].masked_fill(~legal, float("-inf"))
        pi_probs = torch.softmax(pi_logits, dim=-1).masked_fill(~legal, 0.0)
        direct_action = pi_logits.argmax(dim=-1)

        worlds = sample_worlds(out["belief_logits"], batch["belief_mask"], k, rng)
        state_rep = out["state_emb"].unsqueeze(1).expand(B, k, -1).reshape(B * k, -1)
        worlds_flat = worlds.reshape(B * k, 28, 3)
        with torch.no_grad():
            q_flat = model.q_head(state_rep, model.world_encoder(worlds_flat))
        q_mean = q_flat.view(B, k, 7).mean(dim=1).masked_fill(~legal, float("-inf"))
        qmean_action = q_mean.argmax(dim=-1)
        q_sorted = q_mean.sort(dim=-1, descending=True).values

        direct_regret = best - e_q[idx, direct_action]
        qmean_regret = best - e_q[idx, qmean_action]

        for b in range(B):
            legal_probs = pi_probs[b][legal[b]]
            peak = float(legal_probs.max().item())
            entropy = float((-(legal_probs * torch.log(legal_probs.clamp_min(1e-12))).sum()).item())
            sorted_pi = torch.sort(legal_probs, descending=True).values
            pi_margin = float((sorted_pi[0] - (sorted_pi[1] if sorted_pi.numel() > 1 else 0.0)).item())

            d = int(direct_action[b].item())
            q = int(qmean_action[b].item())
            q_at_direct = float(q_mean[b, d].item())
            q_at_qmean = float(q_mean[b, q].item())
            qmean_margin = float((q_sorted[b, 0] - q_sorted[b, 1]).item()) if legal[b].sum() > 1 else 99.0
            v = float(out["v"][b].item())
            d_idx = int(batch["decision_idx"][b].item())

            features.append([
                1.0 if d != q else 0.0,
                peak,
                entropy,
                pi_margin,
                qmean_margin,
                q_at_qmean - q_at_direct,
                q_at_direct,
                q_at_qmean,
                v,
                v - q_at_direct,
                v - q_at_qmean,
                float(d_idx),
                float(d_idx // 4),
                float(d_idx % 4),
                float(legal[b].sum().item()),
                float(d),
                float(q),
            ])
            direct_regrets.append(float(direct_regret[b].item()))
            qmean_regrets.append(float(qmean_regret[b].item()))
            direct_matches.append(int(direct_action[b] == best_action[b]))
            qmean_matches.append(int(qmean_action[b] == best_action[b]))
            disagreements.append(int(d != q))

    X = np.asarray(features, dtype=np.float32)
    rows = Rows(
        direct_regret=np.asarray(direct_regrets, dtype=np.float32),
        qmean_regret=np.asarray(qmean_regrets, dtype=np.float32),
        direct_match=np.asarray(direct_matches, dtype=np.int32),
        qmean_match=np.asarray(qmean_matches, dtype=np.int32),
        disagree=np.asarray(disagreements, dtype=np.bool_),
    )
    return X, rows


def _summarize(
    name: str,
    regret: np.ndarray,
    match: np.ndarray,
    routed: np.ndarray | None = None,
    direct_regret: np.ndarray | None = None,
) -> Summary:
    summary = Summary(
        name=name,
        regret=float(regret.mean()),
        blunders=int((regret >= 8).sum()),
        big_misses=int((regret >= 4).sum()),
        bot_match=float(match.mean()),
    )
    msg = (
        f"{name:18s} regret={summary.regret:.4f} "
        f"bl={summary.blunders:3d} ({(regret >= 8).mean():.2%}) "
        f"big={summary.big_misses:3d} ({(regret >= 4).mean():.2%}) "
        f"bot={summary.bot_match:.2%}"
    )
    if routed is not None:
        summary.routed_frac = float(routed.mean())
        msg += f" route={summary.routed_frac:.2%}"
        if direct_regret is not None:
            fixed = ((direct_regret >= 8) & (regret < 8) & routed).sum()
            introduced = ((direct_regret < 8) & (regret >= 8) & routed).sum()
            summary.fixed_blunders = int(fixed)
            summary.new_blunders = int(introduced)
            msg += f" fix_bl={fixed} new_bl={introduced}"
    print(msg)
    return summary


def _parse_seed_list(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--eval-seeds", default=None, help="Comma-separated eval seeds; overrides --eval-seed.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--target", choices=("fix_blunder", "improve_0_5"), default="fix_blunder")
    parser.add_argument("--route-fracs", default="0.03,0.05,0.07,0.10,0.15")
    args = parser.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    device = args.device or _pick_device()
    route_fracs = [float(x) for x in args.route_fracs.split(",") if x.strip()]
    eval_seeds = _parse_seed_list(args.eval_seeds) if args.eval_seeds else [args.eval_seed]

    print(f"Adapter: {args.adapter}")
    print(f"Train:   {args.train}")
    print(f"Eval:    {args.eval}")
    print(f"Device:  {device}  K={args.k}  target={args.target}")
    print(f"Eval seeds: {','.join(str(s) for s in eval_seeds)}")

    model, is_voids = load_student(args.adapter, device)
    print("Collecting train features...", flush=True)
    X_train, train_rows = _collect(model, is_voids, args.train, device, args.k, args.train_seed, args.batch_size)

    train_delta = train_rows.direct_regret - train_rows.qmean_regret
    if args.target == "fix_blunder":
        y_train = ((train_rows.direct_regret >= 8.0) & (train_rows.qmean_regret < 8.0)).astype(np.int32)
    else:
        y_train = (train_delta > 0.5).astype(np.int32)

    print(f"Train positives: {y_train.sum()}/{len(y_train)} = {y_train.mean():.2%}")

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=2,
        learning_rate=0.05,
        random_state=0,
    )
    clf.fit(X_train, y_train)

    importances = sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda kv: -kv[1])
    print("Top feature importances:")
    for name, value in importances[:8]:
        print(f"  {name:20s} {value:.3f}")

    all_summaries: dict[str, list[Summary]] = {}

    def record(summary: Summary) -> None:
        all_summaries.setdefault(summary.name, []).append(summary)

    for eval_seed in eval_seeds:
        print(f"\nCollecting eval features for seed {eval_seed}...", flush=True)
        X_eval, eval_rows = _collect(model, is_voids, args.eval, device, args.k, eval_seed, args.batch_size)
        eval_delta = eval_rows.direct_regret - eval_rows.qmean_regret
        if args.target == "fix_blunder":
            y_eval = ((eval_rows.direct_regret >= 8.0) & (eval_rows.qmean_regret < 8.0)).astype(np.int32)
        else:
            y_eval = (eval_delta > 0.5).astype(np.int32)

        score = clf.predict_proba(X_eval)[:, 1]
        print(f"Eval positives:  {y_eval.sum()}/{len(y_eval)} = {y_eval.mean():.2%}")
        if len(np.unique(y_eval)) > 1:
            print(f"Eval ROC-AUC: {roc_auc_score(y_eval, score):.3f}")
            print(f"Eval PR-AUC:  {average_precision_score(y_eval, score):.3f}")
        print()

        record(_summarize("direct", eval_rows.direct_regret, eval_rows.direct_match))
        record(_summarize("qmean", eval_rows.qmean_regret, eval_rows.qmean_match))
        allowed = np.where(eval_rows.disagree)[0]
        order = allowed[np.argsort(score[allowed])[::-1]]
        for frac in route_fracs:
            n_route = min(len(order), max(1, int(round(len(score) * frac))))
            routed = np.zeros(len(score), dtype=np.bool_)
            routed[order[:n_route]] = True
            regret = np.where(routed, eval_rows.qmean_regret, eval_rows.direct_regret)
            match = np.where(routed, eval_rows.qmean_match, eval_rows.direct_match)
            record(_summarize(f"route top {frac:.0%}", regret, match, routed, eval_rows.direct_regret))

    if len(eval_seeds) > 1:
        print("\nSeed summary:")
        for name, summaries in all_summaries.items():
            regrets = np.asarray([s.regret for s in summaries])
            blunders = np.asarray([s.blunders for s in summaries])
            big_misses = np.asarray([s.big_misses for s in summaries])
            bot_matches = np.asarray([s.bot_match for s in summaries])
            msg = (
                f"{name:18s} regret={regrets.mean():.4f}"
                f" ({regrets.min():.4f}-{regrets.max():.4f}) "
                f"bl={blunders.mean():.2f} ({blunders.min()}-{blunders.max()}) "
                f"big={big_misses.mean():.2f} "
                f"bot={bot_matches.mean():.2%}"
            )
            if summaries[0].routed_frac is not None:
                routed = np.asarray([s.routed_frac for s in summaries])
                fixed = np.asarray([s.fixed_blunders for s in summaries])
                new = np.asarray([s.new_blunders for s in summaries])
                msg += (
                    f" route={routed.mean():.2%} "
                    f"fix_bl={fixed.mean():.2f} new_bl={new.mean():.2f}"
                )
            print(msg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
