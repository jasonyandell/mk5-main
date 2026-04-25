"""Sweep Q-mean router gates for blunder reduction.

This script answers a narrower question than eval_qmean_router.py:

  If Q-mean is only a second opinion, when should Gus listen to it?

It computes direct π and Q-mean once per sampled-world seed, then evaluates
threshold gates of the form:

  route = (direct_action != qmean_action)
          AND (pi_peak < peak_threshold)
          AND (qmean_margin >= margin_threshold)

where qmean_margin is the top-1 minus top-2 Q-mean value among legal actions.

Example:
  python -u gus/eval/sweep_qmean_router.py \
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

from gus.eval.eval_regret import _load_student
from gus.model.dataset_seq_world import JointWorldFullDataset
from gus.model.sample_worlds import sample_worlds


@dataclass
class Accum:
    regret: float = 0.0
    matches: int = 0
    blunders: int = 0
    big_misses: int = 0
    routed: int = 0
    fixed_blunders: int = 0
    introduced_blunders: int = 0
    count: int = 0

    def add(
        self,
        regret: torch.Tensor,
        match: torch.Tensor,
        routed: torch.Tensor | None = None,
        direct_regret: torch.Tensor | None = None,
    ) -> None:
        self.regret += float(regret.sum().item())
        self.matches += int(match.sum().item())
        self.blunders += int((regret >= 8.0).sum().item())
        self.big_misses += int((regret >= 4.0).sum().item())
        self.count += int(regret.numel())
        if routed is not None:
            self.routed += int(routed.sum().item())
            if direct_regret is not None:
                self.fixed_blunders += int(((direct_regret >= 8.0) & (regret < 8.0)).sum().item())
                self.introduced_blunders += int(((direct_regret < 8.0) & (regret >= 8.0)).sum().item())

    @property
    def mean_regret(self) -> float:
        return self.regret / max(self.count, 1)

    @property
    def bot_match(self) -> float:
        return self.matches / max(self.count, 1)

    @property
    def route_rate(self) -> float:
        return self.routed / max(self.count, 1)


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


def _q_mean(
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
        q_flat = model.q_head(state_rep, model.world_encoder(worlds_flat))
    q = q_flat.view(B, k, 7).mean(dim=1)
    return q.masked_fill(~legal_mask, float("-inf"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--eval", required=True, nargs="+")
    parser.add_argument("--k", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2,42,99")
    parser.add_argument("--peak-grid", default="0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8")
    parser.add_argument("--margin-grid", default="0,0.25,0.5,1,2")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or _pick_device()
    seeds = _parse_ints(args.seeds)
    peak_grid = _parse_floats(args.peak_grid)
    margin_grid = _parse_floats(args.margin_grid)
    combos = [(p, m) for p in peak_grid for m in margin_grid]

    print(f"Adapter: {args.adapter}", flush=True)
    print(f"Eval:    {args.eval}", flush=True)
    print(f"Device:  {device}  K={args.k}  seeds={seeds}", flush=True)
    print(f"Peaks:   {peak_grid}", flush=True)
    print(f"Margins: {margin_grid}", flush=True)

    model, is_voids = _load_student(args.adapter, device)
    ds = JointWorldFullDataset(args.eval, seed=42)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    print(f"Decisions: {len(ds)}", flush=True)

    direct_total = Accum()
    qmean_total = Accum()
    sweep = {combo: Accum() for combo in combos}

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
            pi_probs = torch.softmax(pi_logits, dim=-1).masked_fill(~legal, 0.0)
            pi_peak = pi_probs.max(dim=-1).values
            direct_action = pi_logits.argmax(dim=-1)

            q_mean = _q_mean(model, out["state_emb"], out["belief_logits"], batch["belief_mask"], legal, args.k, rng)
            qmean_action = q_mean.argmax(dim=-1)
            q_sorted = q_mean.sort(dim=-1, descending=True).values
            q_margin = q_sorted[:, 0] - q_sorted[:, 1]

            direct_regret = best - e_q[idx, direct_action]
            qmean_regret = best - e_q[idx, qmean_action]
            direct_match = direct_action == best_action
            qmean_match = qmean_action == best_action
            disagree = direct_action != qmean_action

            direct_total.add(direct_regret, direct_match)
            qmean_total.add(qmean_regret, qmean_match)

            for combo in combos:
                peak_t, margin_t = combo
                routed = disagree & (pi_peak < peak_t) & (q_margin >= margin_t)
                regret = torch.where(routed, qmean_regret, direct_regret)
                match = torch.where(routed, qmean_match, direct_match)
                sweep[combo].add(regret, match, routed, direct_regret)

    print()
    print("=== baselines ===")
    for name, acc in (("direct", direct_total), ("qmean", qmean_total)):
        print(
            f"{name:>8s}  regret={acc.mean_regret:.4f}  bot={acc.bot_match:.2%}  "
            f"bl={acc.blunders / len(seeds):.1f}/seed  big={acc.big_misses / len(seeds):.1f}/seed"
        )

    print()
    print("=== router sweep sorted by blunders then regret ===")
    ranked = sorted(
        sweep.items(),
        key=lambda kv: (
            kv[1].blunders / len(seeds),
            kv[1].mean_regret,
            kv[1].route_rate,
        ),
    )
    print(
        f"{'peak<':>6s} {'qmargin>=':>9s} {'regret':>8s} {'bot':>7s} "
        f"{'route':>7s} {'bl/seed':>8s} {'big/seed':>8s} {'fix_bl':>7s} {'new_bl':>7s}"
    )
    for (peak_t, margin_t), acc in ranked[:30]:
        print(
            f"{peak_t:6.2f} {margin_t:9.2f} {acc.mean_regret:8.4f} "
            f"{acc.bot_match:7.2%} {acc.route_rate:7.2%} "
            f"{acc.blunders / len(seeds):8.2f} {acc.big_misses / len(seeds):8.2f} "
            f"{acc.fixed_blunders / len(seeds):7.2f} {acc.introduced_blunders / len(seeds):7.2f}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
