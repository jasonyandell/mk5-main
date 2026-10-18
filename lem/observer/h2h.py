# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Frozen observers in the existing arena, bid 30, paired first-to-seven games."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from arena.bidders import Bid30Bidder
from arena.engine import ArenaConfig
from arena.match import run_match
from arena.play import RandomPlay
from scenario.record import empty_record
from scenario.rules import mask_of, rules_for

from .play import ObserverPlay


def paired_summary(games, draws=10000):
    pairs = {}
    declarations = {}
    for game in games:
        pair = pairs.setdefault(game.game_idx, {})
        if game.a_team in pair:
            raise ValueError("duplicate side of a pair")
        pair[game.a_team] = game
        for hand in game.hands:
            declarations[hand.decl_id] = declarations.get(hand.decl_id, 0) + 1
    if not pairs or any(set(pair) != {0, 1} for pair in pairs.values()):
        raise ValueError("every seed must have both team assignments")
    paired = list(pairs.values())
    values = {
        "game_win_rate": np.array([sum(g.a_won for g in p.values()) / 2 for p in paired]),
        "marks_per_game": np.array([sum(g.a_marks-g.b_marks for g in p.values()) / 2 for p in paired]),
    }
    # A single pair cannot estimate between-pair uncertainty. Repeating its
    # one value 10,000 times would print a misleading zero-width interval.
    idx = (np.random.default_rng(42).integers(0, len(paired), (draws, len(paired)))
           if len(paired) > 1 else None)
    return {"games": len(games), "pairs": len(paired),
            "hands": sum(len(g.hands) for g in games), "declarations": declarations,
            "method": "paired game-seed percentile bootstrap; 95%; seed 42", "draws": draws,
            "inference": "one-pair smoke; intervals unavailable" if idx is None else "fixed-policy paired sample",
            "chance_game_win_rate": 0.5, "neutral_marks_per_game": 0,
            **{name: {"estimate": float(v.mean()),
                       "ci95": None if idx is None else np.quantile(v[idx].mean(1), [.025, .975]).tolist()}
               for name, v in values.items()}}


def verify_replays(games, excluded):
    """Independent public-record replay and physical-deal overlap audit."""
    unique = set()
    for game in games:
        for hand in game.hands:
            hands = tuple(mask_of(h) for h in hand.hands)
            unique.add(hands)
            if hands in excluded:
                raise ValueError("evaluation contains a previously recorded deal")
            rules, rec = rules_for(hand.decl_id), empty_record(hand.bidder)
            for seat, tile in hand.plays:
                if seat != rec.seat or not rules.legal(hands[seat] & ~rec.played, rec.led) & (1 << tile):
                    raise ValueError("illegal arena replay")
                rec = rec.advance(tile, rules)
            if not rec.finished or rec.banked != hand.team_points or sum(rec.banked) != 42:
                raise ValueError("arena and public referee disagree")
    return {"unique_physical_deals": len(unique), "excluded_deals": len(excluded),
            "overlap": 0, "all_plays_legal_and_scores_replayed": True}


def run(args):
    if args.games < 2 or args.games % 2:
        raise ValueError("need a positive even number of games")
    torch.set_num_threads(2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    candidate = ObserverPlay(args.candidate, args.device)
    if args.opponent == "random":
        opponent = RandomPlay(seed=args.seed)
        opponent_id = {"policy": "uniform legal", "seed": args.seed}
    elif args.opponent.startswith("walt:"):
        from arena.walt_play import parse_walt_spec
        opponent = parse_walt_spec(args.opponent.removeprefix("walt:"))
        opponent_id = {"policy": args.opponent, "resolved": repr(opponent)}
        for token in args.opponent.split(","):
            if token.startswith("F="):
                path = Path(token[2:])
                opponent_id["field_file"] = {"path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        opponent = ObserverPlay(args.opponent, args.device)
        opponent_id = opponent.identity
    excluded, exclusion_files = set(), []
    for source in args.exclude_deals:
        path = Path(source)
        raw = path.read_bytes()
        excluded.update(tuple(r["hands"]) for r in json.loads(raw))
        exclusion_files.append({"path": source, "sha256": hashlib.sha256(raw).hexdigest()})
    manifest = {"candidate": candidate.identity, "opponent": opponent_id,
                "seed": args.seed, "games": args.games, "device": args.device,
                "torch": torch.__version__, "platform": platform.platform(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
                "excluded_sources": exclusion_files,
                "contract": "both Bid30Bidder; pip declarations; first to seven marks",
                "seating": "two copies per partnership; paired team swap; fast_batching=True",
                "inputs": "original own hand + public contract and play prefix; no added features",
                "learning": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    result = run_match(bid_a=Bid30Bidder(), bid_b=Bid30Bidder(), play_a=candidate, play_b=opponent,
                       n_games=args.games, cfg=ArenaConfig(base_seed=args.seed), fast_batching=True,
                       label_a="observer", label_b=args.opponent, verbose=True)
    # Keep complete receipts before auditing: a failure must remain diagnosable.
    with (out / "games.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as stream:
            for game in result.games:
                stream.write((json.dumps(asdict(game), sort_keys=True) + "\n").encode())
    report = {**paired_summary(result.games), "audit": verify_replays(result.games, excluded),
              "elapsed_s": result.elapsed_s}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--opponent", required=True, help="checkpoint, random, or walt:<spec>")
    parser.add_argument("--out", required=True)
    parser.add_argument("--games", type=int, default=2,
                        help="total games; default 2 is one duplicate pair, a plumbing smoke only")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--exclude-deals", action="append", default=[])
    run(parser.parse_args())


if __name__ == "__main__":
    main()
