"""Train a learned router over multiple sampled-world Q aggregators.

This generalizes learn_qmean_router.py. Instead of asking whether to route to
only Q-mean, it creates candidate rows for several deployable aggregators over
the same sampled worlds:

  mean, quantiles, lower/upper confidence bounds, per-world argmax vote

The classifier is trained on one corpus to score "direct would blunder and
this candidate would not", then evaluation chooses the highest-scoring
disagreeing candidate per decision and routes only the top-scored decisions.

Example:
  python -u gus/eval/learn_qworld_router.py \
    --adapter gus/adapters/v3_consistency_10000g.pt \
    --train gus/data/corpus_train_100.pt \
    --eval gus/data/corpus_eval_20.pt \
    --k 100 \
    --eval-seeds 0,1,2,42,99 \
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


VARIANT_NAMES = [
    "mean",
    "median",
    "q10",
    "q25",
    "q75",
    "q90",
    "lcb0.5",
    "lcb1.0",
    "ucb0.5",
    "ucb1.0",
    "vote",
]

BASE_FEATURE_NAMES = [
    "disagree",
    "pi_peak",
    "pi_entropy",
    "pi_margin",
    "candidate_margin",
    "candidate_minus_direct_score",
    "score_at_direct",
    "score_at_candidate",
    "v",
    "v_minus_score_at_direct",
    "v_minus_score_at_candidate",
    "decision_idx",
    "trick_num",
    "trick_pos",
    "legal_count",
    "direct_action",
    "candidate_action",
]

FEATURE_NAMES = BASE_FEATURE_NAMES + [f"is_{name}" for name in VARIANT_NAMES]


@dataclass
class DecisionRows:
    direct_regret: np.ndarray
    direct_match: np.ndarray
    oracle_pool_regret: np.ndarray
    oracle_pool_match: np.ndarray


@dataclass
class CandidateRows:
    X: np.ndarray
    regret: np.ndarray
    match: np.ndarray
    decision_id: np.ndarray
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


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_floats(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _variant_scores(q: torch.Tensor, legal: torch.Tensor) -> dict[str, torch.Tensor]:
    legal3 = legal.unsqueeze(1)
    q_legal = q.masked_fill(~legal3, float("-inf"))
    mean = q.mean(dim=1)
    std = q.std(dim=1, unbiased=False)

    variants = {
        "mean": mean,
        "median": q.median(dim=1).values,
        "q10": torch.quantile(q, 0.10, dim=1),
        "q25": torch.quantile(q, 0.25, dim=1),
        "q75": torch.quantile(q, 0.75, dim=1),
        "q90": torch.quantile(q, 0.90, dim=1),
        "lcb0.5": mean - 0.5 * std,
        "lcb1.0": mean - std,
        "ucb0.5": mean + 0.5 * std,
        "ucb1.0": mean + std,
    }
    per_world_best = q_legal.argmax(dim=-1)
    counts = torch.zeros(q.shape[0], 7, device=q.device)
    counts.scatter_add_(1, per_world_best, torch.ones_like(per_world_best, dtype=counts.dtype))
    variants["vote"] = counts + 1e-4 * mean

    return {name: scores.masked_fill(~legal, float("-inf")) for name, scores in variants.items()}


def _q_per_world(
    model,
    state_emb: torch.Tensor,
    belief_logits: torch.Tensor,
    belief_mask: torch.Tensor,
    k: int,
    rng: torch.Generator,
) -> torch.Tensor:
    worlds = sample_worlds(belief_logits, belief_mask, k, rng)
    B = state_emb.shape[0]
    state_rep = state_emb.unsqueeze(1).expand(B, k, -1).reshape(B * k, -1)
    worlds_flat = worlds.reshape(B * k, 28, 3)
    with torch.no_grad():
        q_flat = model.q_head(state_rep, model.world_encoder(worlds_flat))
    return q_flat.view(B, k, 7)


def _collect(
    model,
    is_voids: bool,
    corpus_path: str,
    device: str,
    k: int,
    seed: int,
    batch_size: int,
) -> tuple[DecisionRows, CandidateRows]:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    ds = JointWorldFullDataset(corpus_path, seed=42)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    direct_regrets: list[float] = []
    direct_matches: list[int] = []
    oracle_pool_regrets: list[float] = []
    oracle_pool_matches: list[int] = []

    features: list[list[float]] = []
    candidate_regrets: list[float] = []
    candidate_matches: list[int] = []
    candidate_decision_ids: list[int] = []
    candidate_disagrees: list[int] = []
    global_decision_id = 0

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
        direct_regret = best - e_q[idx, direct_action]
        direct_match = direct_action == best_action

        q = _q_per_world(model, out["state_emb"], out["belief_logits"], batch["belief_mask"], k, rng)
        variant_scores = _variant_scores(q, legal)
        variant_regrets = [direct_regret]
        variant_matches = [direct_match]
        variant_regret_by_name: dict[str, torch.Tensor] = {}
        variant_match_by_name: dict[str, torch.Tensor] = {}

        for name in VARIANT_NAMES:
            scores = variant_scores[name]
            action = scores.argmax(dim=-1)
            regret = best - e_q[idx, action]
            match = action == best_action
            variant_regrets.append(regret)
            variant_matches.append(match)
            variant_regret_by_name[name] = regret
            variant_match_by_name[name] = match

        for b in range(B):
            legal_probs = pi_probs[b][legal[b]]
            peak = float(legal_probs.max().item())
            entropy = float((-(legal_probs * torch.log(legal_probs.clamp_min(1e-12))).sum()).item())
            sorted_pi = torch.sort(legal_probs, descending=True).values
            pi_margin = float((sorted_pi[0] - (sorted_pi[1] if sorted_pi.numel() > 1 else 0.0)).item())
            d_action = int(direct_action[b].item())
            v = float(out["v"][b].item())
            d_idx = int(batch["decision_idx"][b].item())
            legal_count = int(legal[b].sum().item())

            for variant_idx, name in enumerate(VARIANT_NAMES):
                scores = variant_scores[name]
                action = int(scores[b].argmax().item())
                sorted_scores = scores[b].sort(descending=True).values
                margin = float((sorted_scores[0] - sorted_scores[1]).item()) if legal_count > 1 else 99.0
                score_at_direct = float(scores[b, d_action].item())
                score_at_candidate = float(scores[b, action].item())
                one_hot = [1.0 if i == variant_idx else 0.0 for i in range(len(VARIANT_NAMES))]
                row = [
                    1.0 if action != d_action else 0.0,
                    peak,
                    entropy,
                    pi_margin,
                    margin,
                    score_at_candidate - score_at_direct,
                    score_at_direct,
                    score_at_candidate,
                    v,
                    v - score_at_direct,
                    v - score_at_candidate,
                    float(d_idx),
                    float(d_idx // 4),
                    float(d_idx % 4),
                    float(legal_count),
                    float(d_action),
                    float(action),
                    *one_hot,
                ]
                features.append(row)
                candidate_decision_ids.append(global_decision_id + b)
                candidate_disagrees.append(int(action != d_action))
                candidate_regrets.append(float(variant_regret_by_name[name][b].item()))
                candidate_matches.append(int(variant_match_by_name[name][b].item()))

        regret_stack = torch.stack(variant_regrets, dim=0)
        best_variant_idx = regret_stack.argmin(dim=0)
        oracle_regret = regret_stack[best_variant_idx, idx]
        match_stack = torch.stack(variant_matches, dim=0)
        oracle_match = match_stack[best_variant_idx, idx]

        direct_regrets.extend(float(x) for x in direct_regret.detach().cpu())
        direct_matches.extend(int(x) for x in direct_match.detach().cpu())
        oracle_pool_regrets.extend(float(x) for x in oracle_regret.detach().cpu())
        oracle_pool_matches.extend(int(x) for x in oracle_match.detach().cpu())
        global_decision_id += B

    decisions = DecisionRows(
        direct_regret=np.asarray(direct_regrets, dtype=np.float32),
        direct_match=np.asarray(direct_matches, dtype=np.int32),
        oracle_pool_regret=np.asarray(oracle_pool_regrets, dtype=np.float32),
        oracle_pool_match=np.asarray(oracle_pool_matches, dtype=np.int32),
    )
    candidates = CandidateRows(
        X=np.asarray(features, dtype=np.float32),
        regret=np.asarray(candidate_regrets, dtype=np.float32),
        match=np.asarray(candidate_matches, dtype=np.int32),
        decision_id=np.asarray(candidate_decision_ids, dtype=np.int32),
        disagree=np.asarray(candidate_disagrees, dtype=np.bool_),
    )
    return decisions, candidates


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
        blunders=int((regret >= 8.0).sum()),
        big_misses=int((regret >= 4.0).sum()),
        bot_match=float(match.mean()),
    )
    msg = (
        f"{name:18s} regret={summary.regret:.4f} "
        f"bl={summary.blunders:3d} ({(regret >= 8.0).mean():.2%}) "
        f"big={summary.big_misses:3d} ({(regret >= 4.0).mean():.2%}) "
        f"bot={summary.bot_match:.2%}"
    )
    if routed is not None:
        summary.routed_frac = float(routed.mean())
        msg += f" route={summary.routed_frac:.2%}"
        if direct_regret is not None:
            fixed = ((direct_regret >= 8.0) & (regret < 8.0) & routed).sum()
            introduced = ((direct_regret < 8.0) & (regret >= 8.0) & routed).sum()
            summary.fixed_blunders = int(fixed)
            summary.new_blunders = int(introduced)
            msg += f" fix_bl={fixed} new_bl={introduced}"
    print(msg)
    return summary


def _best_candidate_per_decision(candidates: CandidateRows, score: np.ndarray, n_decisions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    best_score = np.full(n_decisions, -np.inf, dtype=np.float32)
    best_regret = np.full(n_decisions, np.nan, dtype=np.float32)
    best_match = np.zeros(n_decisions, dtype=np.int32)
    allowed = np.flatnonzero(candidates.disagree)
    for row_idx in allowed:
        decision_id = candidates.decision_id[row_idx]
        if score[row_idx] > best_score[decision_id]:
            best_score[decision_id] = score[row_idx]
            best_regret[decision_id] = candidates.regret[row_idx]
            best_match[decision_id] = candidates.match[row_idx]
    return best_score, best_regret, best_match


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--eval-seeds", default="0,1,2,42,99")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--route-fracs", default="0.03,0.05,0.07,0.10,0.15")
    parser.add_argument(
        "--objective",
        choices=("fix_classifier", "blunder_delta", "regret_delta"),
        default="blunder_delta",
        help=(
            "fix_classifier predicts fixes only; blunder_delta regresses +1 fix / -1 new blunder; "
            "regret_delta regresses direct_regret - candidate_regret."
        ),
    )
    args = parser.parse_args()

    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.metrics import average_precision_score, roc_auc_score

    device = args.device or _pick_device()
    eval_seeds = _parse_ints(args.eval_seeds)
    route_fracs = _parse_floats(args.route_fracs)

    print(f"Adapter: {args.adapter}")
    print(f"Train:   {args.train}")
    print(f"Eval:    {args.eval}")
    print(f"Device:  {device}  K={args.k}  objective={args.objective}")
    print(f"Eval seeds: {','.join(str(s) for s in eval_seeds)}")

    model, is_voids = load_student(args.adapter, device)
    print("Collecting train candidate features...", flush=True)
    train_decisions, train_candidates = _collect(
        model, is_voids, args.train, device, args.k, args.train_seed, args.batch_size
    )
    direct_by_candidate = train_decisions.direct_regret[train_candidates.decision_id]
    train_fix = (
        (direct_by_candidate >= 8.0)
        & (train_candidates.regret < 8.0)
        & train_candidates.disagree
    )
    train_new_blunder = (
        (direct_by_candidate < 8.0)
        & (train_candidates.regret >= 8.0)
        & train_candidates.disagree
    )
    if args.objective == "fix_classifier":
        y_train = train_fix.astype(np.int32)
        scorer = GradientBoostingClassifier(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.04,
            random_state=0,
        )
    elif args.objective == "blunder_delta":
        y_train = train_fix.astype(np.float32) - train_new_blunder.astype(np.float32)
        scorer = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.04,
            random_state=0,
        )
    else:
        y_train = direct_by_candidate - train_candidates.regret
        scorer = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.04,
            random_state=0,
        )
    positive_decisions = np.unique(train_candidates.decision_id[train_fix]).size
    print(
        f"Train fixes: {train_fix.sum()}/{len(train_fix)} candidate rows = {train_fix.mean():.2%}; "
        f"{positive_decisions} distinct decisions"
    )
    print(f"Train introduced-blunder candidates: {train_new_blunder.sum()} = {train_new_blunder.mean():.2%}")

    scorer.fit(train_candidates.X, y_train)

    importances = sorted(zip(FEATURE_NAMES, scorer.feature_importances_), key=lambda kv: -kv[1])
    print("Top feature importances:")
    for name, value in importances[:10]:
        print(f"  {name:24s} {value:.3f}")

    all_summaries: dict[str, list[Summary]] = {}

    def record(summary: Summary) -> None:
        all_summaries.setdefault(summary.name, []).append(summary)

    for eval_seed in eval_seeds:
        print(f"\nCollecting eval candidate features for seed {eval_seed}...", flush=True)
        eval_decisions, eval_candidates = _collect(
            model, is_voids, args.eval, device, args.k, eval_seed, args.batch_size
        )
        direct_by_eval_candidate = eval_decisions.direct_regret[eval_candidates.decision_id]
        eval_fix = (
            (direct_by_eval_candidate >= 8.0)
            & (eval_candidates.regret < 8.0)
            & eval_candidates.disagree
        )
        eval_new_blunder = (
            (direct_by_eval_candidate < 8.0)
            & (eval_candidates.regret >= 8.0)
            & eval_candidates.disagree
        )
        if args.objective == "fix_classifier":
            score = scorer.predict_proba(eval_candidates.X)[:, 1]
        else:
            score = scorer.predict(eval_candidates.X)
        positive_eval_decisions = np.unique(eval_candidates.decision_id[eval_fix]).size
        print(
            f"Eval fixes:  {eval_fix.sum()}/{len(eval_fix)} candidate rows = {eval_fix.mean():.2%}; "
            f"{positive_eval_decisions} distinct decisions"
        )
        print(f"Eval introduced-blunder candidates: {eval_new_blunder.sum()} = {eval_new_blunder.mean():.2%}")
        y_eval_classifier = eval_fix.astype(np.int32)
        if len(np.unique(y_eval_classifier)) > 1:
            print(f"Eval ROC-AUC for fixes: {roc_auc_score(y_eval_classifier, score):.3f}")
            print(f"Eval PR-AUC for fixes:  {average_precision_score(y_eval_classifier, score):.3f}")

        best_score, best_regret, best_match = _best_candidate_per_decision(
            eval_candidates, score, len(eval_decisions.direct_regret)
        )
        finite = np.isfinite(best_score)
        order = np.flatnonzero(finite)[np.argsort(best_score[finite])[::-1]]

        print()
        record(_summarize("direct", eval_decisions.direct_regret, eval_decisions.direct_match))
        record(_summarize("oracle pool", eval_decisions.oracle_pool_regret, eval_decisions.oracle_pool_match))
        for frac in route_fracs:
            n_route = min(len(order), max(1, int(round(len(eval_decisions.direct_regret) * frac))))
            routed = np.zeros(len(eval_decisions.direct_regret), dtype=np.bool_)
            routed[order[:n_route]] = True
            regret = np.where(routed, best_regret, eval_decisions.direct_regret)
            match = np.where(routed, best_match, eval_decisions.direct_match)
            record(_summarize(f"route top {frac:.0%}", regret, match, routed, eval_decisions.direct_regret))

        best_prefix: tuple[tuple[int, int, float], int, float, int, int, int] | None = None
        for n_route in range(1, len(order) + 1):
            routed = np.zeros(len(eval_decisions.direct_regret), dtype=np.bool_)
            routed[order[:n_route]] = True
            regret = np.where(routed, best_regret, eval_decisions.direct_regret)
            blunders = int((regret >= 8.0).sum())
            fixed = int(((eval_decisions.direct_regret >= 8.0) & (regret < 8.0) & routed).sum())
            introduced = int(((eval_decisions.direct_regret < 8.0) & (regret >= 8.0) & routed).sum())
            mean_regret = float(regret.mean())
            key = (blunders, introduced, mean_regret)
            if best_prefix is None or key < best_prefix[0]:
                best_prefix = (key, n_route, mean_regret, blunders, fixed, introduced)
        if best_prefix is not None:
            _, n_route, mean_regret, blunders, fixed, introduced = best_prefix
            print(
                f"best learned-prefix cutoff: n={n_route} ({n_route / len(eval_decisions.direct_regret):.2%}) "
                f"regret={mean_regret:.4f} bl={blunders} fix_bl={fixed} new_bl={introduced}"
            )

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
