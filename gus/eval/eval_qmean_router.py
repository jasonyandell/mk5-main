"""Evaluate belief-sampled Q-mean as a fallback router for Gus.

The policy under test:
  - direct: legal-masked argmax over pi_me
  - qmean: average Q_head over K worlds sampled from belief_head, then argmax
  - routers: use qmean only when it disagrees with direct and a cheap
    uncertainty gate fires

This is deliberately not full LAMIR. It is a detect-and-route probe for the
blunder tail: can Q_mean rescue low-confidence direct-pi decisions without
damaging easy direct-pi wins?

Example:
  python -u gus/eval/eval_qmean_router.py \
    --adapter gus/adapters/v3_consistency_10000g.pt \
    --eval gus/data/corpus_eval_20.pt \
    --k 200 \
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


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Totals:
    regret: float = 0.0
    matches: int = 0
    routed: int = 0
    count: int = 0
    reference_regret: float = 0.0
    blunders: int = 0
    big_misses: int = 0

    def add(self, regret: torch.Tensor, match: torch.Tensor, routed: torch.Tensor | None = None) -> None:
        self.regret += float(regret.sum().item())
        self.matches += int(match.sum().item())
        if routed is not None:
            self.routed += int(routed.sum().item())
        self.count += int(regret.numel())
        self.blunders += int((regret >= 8.0).sum().item())
        self.big_misses += int((regret >= 4.0).sum().item())

    @property
    def mean_regret(self) -> float:
        return self.regret / max(self.count, 1)

    @property
    def match_rate(self) -> float:
        return self.matches / max(self.count, 1)

    @property
    def route_rate(self) -> float:
        return self.routed / max(self.count, 1)

    @property
    def blunder_rate(self) -> float:
        return self.blunders / max(self.count, 1)

    @property
    def big_miss_rate(self) -> float:
        return self.big_misses / max(self.count, 1)


def _q_mean_values(
    model,
    state_emb: torch.Tensor,
    belief_logits: torch.Tensor,
    belief_mask: torch.Tensor,
    legal_mask: torch.Tensor,
    k: int,
    rng: torch.Generator,
) -> torch.Tensor:
    worlds = sample_worlds(belief_logits, belief_mask, k, rng)
    B = state_emb.shape[0]
    state_rep = state_emb.unsqueeze(1).expand(B, k, -1).reshape(B * k, -1)
    worlds_flat = worlds.reshape(B * k, 28, 3)
    with torch.no_grad():
        world_emb = model.world_encoder(worlds_flat)
        q_flat = model.q_head(state_rep, world_emb)
    q_mean = q_flat.view(B, k, 7).mean(dim=1)
    return q_mean.masked_fill(~legal_mask, float("-inf"))


def _summarize_seed(
    model,
    is_voids: bool,
    loader: DataLoader,
    device: str,
    k: int,
    seed: int,
    pi_peak_threshold: float,
    entropy_threshold: float,
) -> dict[str, Totals]:
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    totals = {
        "direct": Totals(),
        "qmean": Totals(),
        "router_peak": Totals(),
        "router_entropy": Totals(),
        "router_union": Totals(),
        "oracle_min": Totals(),
        "direct_blunder": Totals(),
        "direct_bigmiss": Totals(),
    }

    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        B = batch["tokens"].shape[0]
        idx = torch.arange(B, device=device)
        legal = batch["legal_mask"]
        e_q = batch["e_q"]
        e_q_legal = e_q.masked_fill(~legal, float("-inf"))
        oracle_best = e_q_legal.max(dim=-1).values
        oracle_best_action = e_q_legal.argmax(dim=-1)

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
        pi_probs = torch.softmax(pi_logits, dim=-1)
        direct_action = pi_logits.argmax(dim=-1)
        q_mean = _q_mean_values(
            model,
            out["state_emb"],
            out["belief_logits"],
            batch["belief_mask"],
            legal,
            k,
            rng,
        )
        qmean_action = q_mean.argmax(dim=-1)

        direct_regret = oracle_best - e_q[idx, direct_action]
        qmean_regret = oracle_best - e_q[idx, qmean_action]
        direct_match = direct_action == oracle_best_action
        qmean_match = qmean_action == oracle_best_action

        legal_probs = pi_probs.masked_fill(~legal, 0.0)
        pi_peak = legal_probs.max(dim=-1).values
        pi_entropy = -(legal_probs * torch.log(legal_probs.clamp_min(1e-12))).sum(dim=-1)
        disagree = direct_action != qmean_action

        route_peak = disagree & (pi_peak < pi_peak_threshold)
        route_entropy = disagree & (pi_entropy > entropy_threshold)
        route_union = route_peak | route_entropy

        peak_regret = torch.where(route_peak, qmean_regret, direct_regret)
        entropy_regret = torch.where(route_entropy, qmean_regret, direct_regret)
        union_regret = torch.where(route_union, qmean_regret, direct_regret)
        min_regret = torch.minimum(direct_regret, qmean_regret)
        min_match = (
            ((direct_regret <= qmean_regret) & direct_match)
            | ((qmean_regret < direct_regret) & qmean_match)
        )

        totals["direct"].add(direct_regret, direct_match)
        totals["qmean"].add(qmean_regret, qmean_match)
        totals["router_peak"].add(peak_regret, torch.where(route_peak, qmean_match, direct_match), route_peak)
        totals["router_entropy"].add(
            entropy_regret,
            torch.where(route_entropy, qmean_match, direct_match),
            route_entropy,
        )
        totals["router_union"].add(union_regret, torch.where(route_union, qmean_match, direct_match), route_union)
        totals["oracle_min"].add(min_regret, min_match)

        blunder = direct_regret >= 8.0
        bigmiss = direct_regret >= 4.0
        if blunder.any():
            totals["direct_blunder"].add(qmean_regret[blunder], qmean_match[blunder])
            totals["direct_blunder"].reference_regret += float(direct_regret[blunder].sum().item())
        if bigmiss.any():
            totals["direct_bigmiss"].add(qmean_regret[bigmiss], qmean_match[bigmiss])
            totals["direct_bigmiss"].reference_regret += float(direct_regret[bigmiss].sum().item())

    return totals


def _parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval", required=True, nargs="+")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2,42,99")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--pi-peak-threshold", type=float, default=0.5)
    parser.add_argument("--entropy-threshold", type=float, default=1.0)
    args = parser.parse_args()

    device = args.device or _pick_device()
    seeds = _parse_seeds(args.seeds)

    print(f"Adapter: {args.adapter}", flush=True)
    print(f"Eval:    {args.eval}", flush=True)
    print(f"Device:  {device}  K={args.k}  seeds={seeds}", flush=True)
    print(
        f"Gates:   pi_peak < {args.pi_peak_threshold:.3f}, "
        f"entropy > {args.entropy_threshold:.3f}",
        flush=True,
    )

    model, is_voids = load_student(args.adapter, device)
    ds = JointWorldFullDataset(args.eval, seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Decisions: {len(ds)}", flush=True)

    all_seed_totals: list[dict[str, Totals]] = []
    for seed in seeds:
        totals = _summarize_seed(
            model,
            is_voids,
            loader,
            device,
            args.k,
            seed,
            args.pi_peak_threshold,
            args.entropy_threshold,
        )
        all_seed_totals.append(totals)
        print()
        print(f"=== seed {seed} ===")
        for name in ("direct", "qmean", "router_peak", "router_entropy", "router_union", "oracle_min"):
            t = totals[name]
            route = f"  route={t.route_rate:6.2%}" if name.startswith("router") else ""
            print(
                f"{name:>14s}  regret={t.mean_regret:7.4f}  "
                f"bot={t.match_rate:7.3%}  "
                f"bl={t.blunders:2d} ({t.blunder_rate:5.2%})  "
                f"big={t.big_misses:2d} ({t.big_miss_rate:5.2%})"
                f"{route}"
            )

        bl = totals["direct_blunder"]
        bg = totals["direct_bigmiss"]
        if bl.count:
            direct_blunder_regret = bl.reference_regret / bl.count
            print(
                f"direct blunders >=8: n={bl.count}  "
                f"direct={direct_blunder_regret:.3f}  qmean={bl.mean_regret:.3f}"
            )
        if bg.count:
            direct_bigmiss_regret = bg.reference_regret / bg.count
            print(
                f"direct big misses >=4: n={bg.count}  "
                f"direct={direct_bigmiss_regret:.3f}  qmean={bg.mean_regret:.3f}"
            )

    if len(all_seed_totals) > 1:
        print()
        print("=== seed summary ===")
        for name in ("qmean", "router_peak", "router_entropy", "router_union"):
            vals = torch.tensor([totals[name].mean_regret for totals in all_seed_totals])
            routes = torch.tensor([totals[name].route_rate for totals in all_seed_totals])
            blunders = torch.tensor([totals[name].blunders for totals in all_seed_totals], dtype=torch.float32)
            big_misses = torch.tensor([totals[name].big_misses for totals in all_seed_totals], dtype=torch.float32)
            route_suffix = ""
            if name.startswith("router"):
                route_suffix = f"  route={routes.mean().item():.2%}"
            print(
                f"{name:>14s}  mean={vals.mean().item():.4f}  "
                f"min={vals.min().item():.4f}  max={vals.max().item():.4f}"
                f"  bl={blunders.mean().item():.1f}  big={big_misses.mean().item():.1f}"
                f"{route_suffix}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
