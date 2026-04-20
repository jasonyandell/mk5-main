"""Pre-render MINIMAL candlewax PNGs for the trick-6 decision dataset.

Sister tool to ``burl/candlewax_spike/snapshot.py``. Reads the existing v1
manifest at ``scratch/candlewax_spike/snapshots/index.jsonl`` (so decision
indices line up 1:1 with v1) and renders the minimal variant to
``scratch/candlewax_spike/snapshots_minimal/decision_{N:03d}.png`` with its own
manifest at ``snapshots_minimal/index.jsonl``.

Run:
    python -u -m burl.candlewax_spike.snapshot_minimal --n 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from burl.candlewax_spike.render import CandlewaxHeader, _domino_label
from burl.candlewax_spike.render_minimal import render_candlewax_minimal
from burl.eval.decision_dataset import load_dataset
from burl.tools.eq_distribution import (
    OutcomeDistribution,
    eq_outcome_distribution,
    load_eq_oracle,
)


DEFAULT_DATASET = Path("burl/eval/data/move4_decisions_n50.jsonl")
DEFAULT_V1_MANIFEST = Path("scratch/candlewax_spike/snapshots/index.jsonl")
DEFAULT_OUT_DIR = Path("scratch/candlewax_spike/snapshots_minimal")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _current_player_from_state(state) -> int:
    leader = getattr(state, "trick_leader", None)
    if leader is None:
        leader = getattr(state, "leader", 0)
    return (int(leader) + len(state.current_trick)) % 4


def _trump_name(state) -> str:
    from burl.tools.engine import trump_declared
    return trump_declared(state)


def render_decision_minimal(
    decision,
    *,
    oracle,
    device: str,
    n_samples: int = 10,
    decision_label: str | None = None,
) -> tuple[bytes, list[OutcomeDistribution]]:
    """Compute distributions for every legal play and render one minimal PNG.

    Mirrors ``snapshot.render_decision`` but calls the minimal renderer.
    """
    state = decision.game_state
    me_abs = _current_player_from_state(state)
    trick_no = len(state.play_history) // 4 + 1
    position_in_trick = len(state.current_trick) + 1

    dists: list[OutcomeDistribution] = []
    for play in decision.legal_plays:
        d = eq_outcome_distribution(
            state, int(play),
            n_samples=n_samples,
            oracle=oracle, device=device,
            enumerate="auto",
            suggest_counterfactuals=False,
            include_spike_drivers=False,
        )
        dists.append(d)

    is_offense = bool(dists[0].is_offense)
    hdr = CandlewaxHeader(
        trump=_trump_name(state),
        is_offense=is_offense,
        my_seat=me_abs,
        trick_no=trick_no,
        position_in_trick=position_in_trick,
        decision_label=decision_label,
    )
    png = render_candlewax_minimal(dists, hdr, sort_by_p_make=True)
    return png, dists


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n", type=int, default=3,
                        help="Number of decisions to render from the front of the dataset.")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--start", type=int, default=0,
                        help="Start index (0-based) into the dataset.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[snapshot-min] device={device}")
    log(f"[snapshot-min] dataset={args.dataset}")

    all_decisions = load_dataset(args.dataset)
    decisions = all_decisions[args.start : args.start + args.n]
    log(f"[snapshot-min] rendering {len(decisions)} decisions "
        f"(idx {args.start}..{args.start + len(decisions) - 1})")

    oracle = load_eq_oracle(device=device)

    manifest_path = args.out_dir / "index.jsonl"
    t_total = time.perf_counter()
    with manifest_path.open("w") as manifest:
        for i, dec in enumerate(decisions):
            idx_global = args.start + i
            state = dec.game_state
            label = f"seed {dec.seed} · decl {dec.declaration} · narrator {dec.narrator_seat}"
            t0 = time.perf_counter()
            png, dists = render_decision_minimal(
                dec, oracle=oracle, device=device,
                n_samples=args.n_samples, decision_label=label,
            )
            png_path = args.out_dir / f"decision_{idx_global:03d}.png"
            png_path.write_bytes(png)
            elapsed = time.perf_counter() - t0

            per_play = {
                int(d.play): {
                    "mean": round(float(d.mean), 3),
                    "p_make": round(float(d.p_make), 4),
                    "shape": str(d.distribution_shape),
                    "sampling_mode": str(d.sampling_mode),
                    "label": _domino_label(int(d.play)),
                }
                for d in dists
            }
            play_history_serializable = [
                [int(p), int(d)] for p, d in state.play_history
            ]
            entry = {
                "idx": idx_global,
                "png": str(png_path.relative_to(Path.cwd())) if png_path.is_absolute() else str(png_path),
                "seed": int(dec.seed),
                "declaration": int(dec.declaration),
                "narrator_seat": int(dec.narrator_seat),
                "bidder": int(state.bidder),
                "play_history": play_history_serializable,
                "legal_plays": [int(p) for p in dec.legal_plays],
                "legal_labels": [_domino_label(int(p)) for p in dec.legal_plays],
                "bot_play": int(dec.bot_play),
                "bot_eq": float(dec.bot_eq),
                "eq_gap": float(dec.eq_gap),
                "per_play_eq": {str(int(k)): float(v) for k, v in dec.per_play_eq.items()},
                "per_play_summary": per_play,
                "render_ms": round(elapsed * 1000, 1),
                "variant": "minimal",
            }
            manifest.write(json.dumps(entry) + "\n")
            log(f"[snapshot-min] d{idx_global:03d} seed={dec.seed} decl={dec.declaration} "
                f"n_legal={len(dec.legal_plays)} -> {png_path.name} "
                f"({len(png)/1024:.1f} KB, {elapsed*1000:.0f} ms)")

    log(f"[snapshot-min] total time: {time.perf_counter() - t_total:.1f}s")
    log(f"[snapshot-min] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
