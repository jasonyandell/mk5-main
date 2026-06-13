"""Arena CLI: full-game head-to-head between two (bidder + play) players.

    python -u -m arena.cli --team-a heuristic+lens:ev --team-b bid30+lens:ev \
        --n-games 64 --n-samples 10 --device mps

Player spec: <bidder>+<play>
    bidder: heuristic | heuristic:<min_trumps>,<caution> | bid30 | random[:<p_bid>]
    play:   lens:<utility> | random

Writes summary.json, per_hand.csv, per_game.csv under --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from arena.auction import BidPolicy
from arena.bidders import Bid30Bidder, HeuristicBidder, RandomBidder
from arena.engine import ArenaConfig
from arena.match import game_rows, hand_rows, run_match, summarize
from arena.play import PlayPolicy, RandomPlay


def parse_bidder(spec: str) -> BidPolicy:
    name, _, arg = spec.partition(":")
    if name == "heuristic":
        if arg:
            min_trumps, _, caution = arg.partition(",")
            return HeuristicBidder(int(min_trumps), int(caution or 0))
        return HeuristicBidder()
    if name == "bid30":
        return Bid30Bidder()
    if name == "random":
        return RandomBidder(float(arg)) if arg else RandomBidder()
    raise ValueError(f"Unknown bidder: {spec!r} (heuristic | bid30 | random)")


def parse_play(spec: str, *, model, n_samples: int, device: str, seed: int) -> PlayPolicy:
    name, _, arg = spec.partition(":")
    if name == "random":
        return RandomPlay(seed=seed)
    if name == "lens":
        from arena.lens_play import LensPlay
        return LensPlay(model, utility=arg or "ev", n_samples=n_samples, device=device)
    raise ValueError(f"Unknown play policy: {spec!r} (lens:<utility> | random)")


def needs_model(*specs: str) -> bool:
    return any(s.split("+", 1)[1].startswith("lens") for s in specs)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--team-a", required=True, help="e.g. heuristic+lens:ev")
    parser.add_argument("--team-b", required=True, help="e.g. bid30+lens:ev")
    parser.add_argument("--n-games", type=int, default=64,
                        help="Total games (split evenly across the two halves)")
    parser.add_argument("--marks-to-win", type=int, default=7)
    parser.add_argument("--max-redeals", type=int, default=3)
    parser.add_argument("--n-samples", type=int, default=10,
                        help="Worlds per Lens decision")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str,
                        default=str(Path(__file__).parent / "results"))
    args = parser.parse_args()

    for spec in (args.team_a, args.team_b):
        if "+" not in spec:
            parser.error(f"Player spec needs <bidder>+<play>, got {spec!r}")

    model = None
    device = args.device
    if needs_model(args.team_a, args.team_b):
        import torch
        from forge.zeb.eval.loading import DEFAULT_ORACLE, load_oracle

        if device == "mps" and not torch.backends.mps.is_available():
            print("MPS unavailable; falling back to CPU.", flush=True)
            device = "cpu"
        torch.manual_seed(args.base_seed)
        ckpt = Path(args.checkpoint or PROJECT_ROOT / DEFAULT_ORACLE)
        print(f"Loading oracle: {ckpt} on {device}", flush=True)
        model = load_oracle(str(ckpt), device)

    bid_a, bid_b = parse_bidder(args.team_a.split("+", 1)[0]), parse_bidder(args.team_b.split("+", 1)[0])
    play_a = parse_play(args.team_a.split("+", 1)[1], model=model,
                        n_samples=args.n_samples, device=device, seed=args.base_seed)
    play_b = parse_play(args.team_b.split("+", 1)[1], model=model,
                        n_samples=args.n_samples, device=device, seed=args.base_seed + 1)

    cfg = ArenaConfig(
        marks_to_win=args.marks_to_win,
        max_redeals=args.max_redeals,
        base_seed=args.base_seed,
    )
    print(
        f"Arena: {args.team_a} vs {args.team_b}  "
        f"games={args.n_games} to {cfg.marks_to_win} marks  seed={cfg.base_seed}",
        flush=True,
    )
    result = run_match(
        bid_a=bid_a, bid_b=bid_b, play_a=play_a, play_b=play_b,
        n_games=args.n_games, cfg=cfg,
        label_a=args.team_a, label_b=args.team_b, verbose=True,
    )

    s = summarize(result)
    print(
        f"\nA ({s['label_a']}) wins {s['a_wins']}/{s['n_games']} "
        f"({s['a_game_win_rate']:.1%}; halves {s['a_win_rate_half1']:.1%} / {s['a_win_rate_half2']:.1%})\n"
        f"mark margin {s['mean_mark_margin']:+.2f}/game "
        f"(95% CI [{s['mark_margin_ci_lo_95']:+.2f}, {s['mark_margin_ci_hi_95']:+.2f}])  "
        f"point margin {s['mean_hand_point_margin']:+.2f}/hand\n"
        f"hands/game {s['mean_hands_per_game']:.1f}  "
        f"A offense share {s['auction']['a_offense_share']:.1%}  "
        f"made: A {s['contracts']['a_offense']['made_rate']:.1%} "
        f"(bid {s['contracts']['a_offense']['mean_bid']:.1f}) / "
        f"B {s['contracts']['b_offense']['made_rate']:.1%} "
        f"(bid {s['contracts']['b_offense']['mean_bid']:.1f})\n"
        f"bids {s['auction']['bid_hist']}\n"
        f"decls {s['auction']['decl_hist']}\n"
        f"({s['elapsed_s']:.1f}s)",
        flush=True,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        **s,
        "cfg": {
            "marks_to_win": cfg.marks_to_win,
            "max_redeals": cfg.max_redeals,
            "base_seed": cfg.base_seed,
            "n_samples": args.n_samples,
            "device": device,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(out_dir / "per_hand.csv", hand_rows(result))
    write_csv(out_dir / "per_game.csv", game_rows(result))
    print(f"\nWrote {out_dir}/summary.json, per_hand.csv, per_game.csv", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
