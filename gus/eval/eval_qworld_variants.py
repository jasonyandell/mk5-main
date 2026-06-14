"""Evaluate alternate sampled-world Q aggregators for blunder reduction.

Q-mean is only one way to turn per-world Q estimates into a second opinion.
This script tests a small family of deployable aggregators and reports:

  - pure policy regret/blunders for each aggregator
  - how many direct-pi blunders each aggregator fixes
  - how many new blunders each aggregator introduces
  - an oracle-over-candidates ceiling for this second-opinion family

Example:
  python -u gus/eval/eval_qworld_variants.py \
    --adapter gus/adapters/v3_consistency_10000g.pt \
    --eval gus/data/corpus_eval_20.pt \
    --k 100 \
    --seeds 0,1,2,42,99 \
    --device cpu
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
from torch.utils.data import DataLoader

from gus.model.load import load_student
from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.sample_worlds import sample_worlds


@dataclass
class Accum:
    regret: float = 0.0
    matches: int = 0
    blunders: int = 0
    big_misses: int = 0
    fixed_blunders: int = 0
    introduced_blunders: int = 0
    count: int = 0

    def add(self, regret: torch.Tensor, match: torch.Tensor, direct_regret: torch.Tensor | None = None) -> None:
        self.regret += float(regret.sum().item())
        self.matches += int(match.sum().item())
        self.blunders += int((regret >= 8.0).sum().item())
        self.big_misses += int((regret >= 4.0).sum().item())
        self.count += int(regret.numel())
        if direct_regret is not None:
            self.fixed_blunders += int(((direct_regret >= 8.0) & (regret < 8.0)).sum().item())
            self.introduced_blunders += int(((direct_regret < 8.0) & (regret >= 8.0)).sum().item())

    @property
    def mean_regret(self) -> float:
        return self.regret / max(self.count, 1)

    @property
    def bot_match(self) -> float:
        return self.matches / max(self.count, 1)


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _aggregate_q(q: torch.Tensor, legal: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-action scores from q shaped [B, K, 7]."""
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval", required=True, nargs="+")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seeds", default="0,1,2,42,99")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or _pick_device()
    seeds = _parse_ints(args.seeds)

    print(f"Adapter: {args.adapter}", flush=True)
    print(f"Eval:    {args.eval}", flush=True)
    print(f"Device:  {device}  K={args.k}  seeds={seeds}", flush=True)

    model, is_voids = load_student(args.adapter, device)
    ds = JointWorldFullDataset(args.eval, seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Decisions: {len(ds)}", flush=True)

    totals: dict[str, Accum] = {}

    for seed in seeds:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        print(f"seed {seed}", flush=True)

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
            direct_action = pi_logits.argmax(dim=-1)
            direct_regret = best - e_q[idx, direct_action]
            direct_match = direct_action == best_action
            totals.setdefault("direct", Accum()).add(direct_regret, direct_match)

            q = _q_per_world(model, out["state_emb"], out["belief_logits"], batch["belief_mask"], args.k, rng)
            variant_regrets = [direct_regret]
            variant_matches = [direct_match]

            for name, scores in _aggregate_q(q, legal).items():
                action = scores.argmax(dim=-1)
                regret = best - e_q[idx, action]
                match = action == best_action
                totals.setdefault(name, Accum()).add(regret, match, direct_regret)
                variant_regrets.append(regret)
                variant_matches.append(match)

            regret_stack = torch.stack(variant_regrets, dim=0)
            best_variant_idx = regret_stack.argmin(dim=0)
            oracle_regret = regret_stack[best_variant_idx, idx]
            match_stack = torch.stack(variant_matches, dim=0)
            oracle_match = match_stack[best_variant_idx, idx]
            totals.setdefault("oracle_pool", Accum()).add(oracle_regret, oracle_match, direct_regret)

    print()
    print("=== sampled-world aggregator variants ===")
    print(
        f"{'policy':>12s} {'regret':>8s} {'bot':>7s} {'bl/seed':>8s} "
        f"{'big/seed':>8s} {'fix_bl':>7s} {'new_bl':>7s}"
    )
    ranked = sorted(
        totals.items(),
        key=lambda kv: (
            kv[1].blunders / len(seeds),
            kv[1].mean_regret,
            0 if kv[0] == "direct" else 1,
        ),
    )
    for name, acc in ranked:
        print(
            f"{name:>12s} {acc.mean_regret:8.4f} {acc.bot_match:7.2%} "
            f"{acc.blunders / len(seeds):8.2f} {acc.big_misses / len(seeds):8.2f} "
            f"{acc.fixed_blunders / len(seeds):7.2f} {acc.introduced_blunders / len(seeds):7.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
