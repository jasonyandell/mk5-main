"""CLI: export games to the otis interchange JSONL + a fate-row table.

Source-agnostic: reads games from eq-corpus ``.pt`` chunks (``--chunk``) and/or
arena snapshot ``.json`` files (``--snapshots``), then writes two artifacts:

  1. The **interchange JSONL** (one game per line) that feeds the TypeScript referee:

     ``{"game_id": str, "hands": [["6-4",...x7] x4 seats 0-3], "decl_id": int,
        "decl_name": str, "bidder": int, "bid_value": int,
        "plays": [{"seat": int, "domino": "a-b"} x28]}``

     Dominoes are ``"hi-lo"`` pip strings; seats are absolute 0-3; teams are {0,2}
     vs {1,3}; the bid winner leads trick 1 and each trick's winner leads next.

  2. The **fate rows** (one row per count tile per game) as CSV (default) or parquet.

Usage:

    .venv/bin/python -u -m otis.export_games \
        --chunk gus/data/corpus_train_chunk_0-99.pt \
        --snapshots scratch/otis-night/corpus/chunk_selfplay_0.snapshots.json \
        --limit 20 \
        --out-jsonl scratch/otis-night/games.jsonl \
        --out-fates scratch/otis-night/fates.csv

Snapshot rows carry join/label metadata (seed, game_idx, hand_idx, a_team,
bidder_team_pts, opp_team_pts, made); those columns populate for snapshot
sources and are blank for corpus sources. Every snapshot hand's parser-computed
per-team points are cross-checked against the arena's recorded points (a free
second referee — raises on mismatch).

All CPU. otis does not own the GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from otis.corpus import load_corpus_games
from otis.fates import (
    GameFates,
    NeutralGame,
    domino_id_to_pips,
    parse_game_fates,
)
from otis.snapshots import SnapshotMeta, cross_check_points, load_snapshot_hands

# Column order for the fate-row table. The trailing block is snapshot-only
# metadata (blank for corpus sources).
FATE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "decl_id",
    "decl_name",
    "bidder",
    "bid_value",
    "tile",
    "tile_id",
    "holder_seat",
    "holder_team",
    "trick_idx",
    "played_mode",
    "winner_seat",
    "winner_team",
    "capture_side",
    "won_by_trump",
    "team0_count",
    "team0_tricks",
    "team1_count",
    "team1_tricks",
    "team0_points",
    "team1_points",
    # snapshot-only join/label metadata
    "source",
    "seed",
    "game_idx",
    "hand_idx",
    "a_team",
    "bidder_team_pts",
    "opp_team_pts",
    "made",
)


def game_to_interchange(game: NeutralGame, decl_name: str) -> dict:
    """Build the interchange dict for one neutral game."""
    return {
        "game_id": game.game_id,
        "hands": [[domino_id_to_pips(d) for d in hand] for hand in game.hands],
        "decl_id": game.decl_id,
        "decl_name": decl_name,
        "bidder": game.bidder,
        "bid_value": game.bid_value,
        "plays": [{"seat": seat, "domino": domino_id_to_pips(did)} for seat, did in game.plays],
    }


def fate_rows(
    fates: GameFates, source: str, meta: SnapshotMeta | None = None
) -> list[dict]:
    """Flatten a GameFates into one row per count tile.

    ``source`` labels the origin (``"corpus"`` or ``"snapshot"``). When ``meta``
    is given (snapshot source), its join/label columns are populated; otherwise
    they are left blank.
    """
    meta_cols: dict = {
        "source": source,
        "seed": "",
        "game_idx": "",
        "hand_idx": "",
        "a_team": "",
        "bidder_team_pts": "",
        "opp_team_pts": "",
        "made": "",
    }
    if meta is not None:
        meta_cols.update(
            {
                "seed": meta.seed,
                "game_idx": meta.game_idx,
                "hand_idx": meta.hand_idx,
                "a_team": meta.a_team,
                "bidder_team_pts": meta.bidder_team_pts,
                "opp_team_pts": meta.opp_team_pts,
                "made": meta.made,
            }
        )

    rows: list[dict] = []
    for t in fates.tiles:
        rows.append(
            {
                "game_id": fates.game_id,
                "decl_id": fates.decl_id,
                "decl_name": fates.decl_name,
                "bidder": fates.bidder,
                "bid_value": fates.bid_value,
                "tile": t.tile,
                "tile_id": t.tile_id,
                "holder_seat": t.holder_seat,
                "holder_team": t.holder_seat % 2,
                "trick_idx": t.trick_idx,
                "played_mode": t.played_mode,
                "winner_seat": t.winner_seat,
                "winner_team": t.winner_seat % 2,
                "capture_side": t.capture_side,
                "won_by_trump": t.won_by_trump,
                "team0_count": fates.team0_count,
                "team0_tricks": fates.team0_tricks,
                "team1_count": fates.team1_count,
                "team1_tricks": fates.team1_tricks,
                "team0_points": fates.team0_points,
                "team1_points": fates.team1_points,
                **meta_cols,
            }
        )
    return rows


def _write_fates(rows: list[dict], out_path: Path) -> str:
    """Write fate rows to CSV or parquet (by extension). Returns the format used."""
    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise SystemExit(
                f"parquet output requires pandas/pyarrow: {exc}. "
                f"Use a .csv path instead."
            ) from exc
        pd.DataFrame(rows, columns=list(FATE_COLUMNS)).to_parquet(out_path, index=False)
        return "parquet"
    # default: CSV
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FATE_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)
    return "csv"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export games (eq-corpus .pt and/or arena snapshot .json) to "
        "otis interchange + fate rows."
    )
    parser.add_argument(
        "--chunk",
        action="append",
        default=[],
        help="Path to an eq-corpus chunk .pt file (repeatable).",
    )
    parser.add_argument(
        "--snapshots",
        action="append",
        default=[],
        help="Path to an arena snapshot .json file (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Total games/hands to export across all inputs (default: all).",
    )
    parser.add_argument("--out-jsonl", required=True, help="Path for the interchange JSONL.")
    parser.add_argument("--out-fates", required=True, help="Path for fate rows (.csv or .parquet).")
    args = parser.parse_args(argv)

    if not args.chunk and not args.snapshots:
        parser.error("provide at least one --chunk or --snapshots input")

    out_jsonl = Path(args.out_jsonl)
    out_fates = Path(args.out_fates)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_fates.parent.mkdir(parents=True, exist_ok=True)

    all_fate_rows: list[dict] = []
    n_games = 0
    remaining = args.limit

    def _remaining_exhausted() -> bool:
        return remaining is not None and remaining <= 0

    with out_jsonl.open("w") as jf:
        # Corpus (.pt) sources: NeutralGame only, no snapshot metadata.
        for chunk in args.chunk:
            if _remaining_exhausted():
                break
            for game in load_corpus_games(chunk, limit=remaining):
                fates = parse_game_fates(game)  # asserts the P1 identity internally
                jf.write(json.dumps(game_to_interchange(game, fates.decl_name)) + "\n")
                all_fate_rows.extend(fate_rows(fates, source="corpus"))
                n_games += 1
                if remaining is not None:
                    remaining -= 1
                if n_games % 50 == 0:
                    print(f"  ...parsed {n_games} games", flush=True)

        # Snapshot (.json) sources: NeutralGame + metadata + free second referee.
        for snap in args.snapshots:
            if _remaining_exhausted():
                break
            for hand in load_snapshot_hands(snap, limit=remaining):
                game = hand.game
                fates = parse_game_fates(game)  # asserts the P1 identity internally
                # Free second referee: parser points vs arena's recorded points.
                cross_check_points(fates, hand.meta)
                jf.write(json.dumps(game_to_interchange(game, fates.decl_name)) + "\n")
                all_fate_rows.extend(
                    fate_rows(fates, source="snapshot", meta=hand.meta)
                )
                n_games += 1
                if remaining is not None:
                    remaining -= 1
                if n_games % 50 == 0:
                    print(f"  ...parsed {n_games} games", flush=True)

    fmt = _write_fates(all_fate_rows, out_fates)
    print(
        f"Exported {n_games} games -> {out_jsonl} (JSONL) and "
        f"{len(all_fate_rows)} fate rows -> {out_fates} ({fmt}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
