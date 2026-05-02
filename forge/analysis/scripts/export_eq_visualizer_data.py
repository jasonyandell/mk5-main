#!/usr/bin/env python3
"""Export local JSONL inputs for the E[Q] browser visualizers.

The web visualizers under ``forge/analysis/results/web/`` fetch JSONL files from
``forge/analysis/results/data/``. This script rebuilds those data files from the
portable artifacts currently checked into the repo:

- ``27a_eq_surface.jsonl`` from the 27a CSV summary tables.
- ``27b_eq_per_game.jsonl`` and ``eq_pdf_v3_sample.jsonl`` from the small E[Q]
  PDF sample tensor, if present.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forge.oracle import schema, tables
from forge.oracle.tables import led_suit_for_lead_domino, trick_rank


DEFAULT_OUTPUT_DIR = ROOT / "forge" / "analysis" / "results" / "data"
DEFAULT_TABLE_DIR = ROOT / "forge" / "analysis" / "results" / "tables"
DEFAULT_PDF_SAMPLE = ROOT / "forge" / "data" / "eq_pdf_s9200-9201_d10_10s.pt"

DECL_NAMES = {
    0: "blanks",
    1: "ones",
    2: "twos",
    3: "threes",
    4: "fours",
    5: "fives",
    6: "sixes",
    7: "doubles",
    8: "doubles-suit",
    9: "no-trump",
}


def pips_to_domino_id(pips: str) -> int:
    high_s, low_s = pips.split("-")
    high = int(high_s)
    low = int(low_s)
    hi = max(high, low)
    lo = min(high, low)
    return hi * (hi + 1) // 2 + lo


def domino_label(domino_id: int) -> str:
    high, low = schema.domino_pips(domino_id)
    return f"{high}-{low}"


def finite_or_none(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


def export_surface(table_dir: Path, output_dir: Path) -> Path:
    order_path = table_dir / "27a_domino_order.csv"
    matrix_path = table_dir / "27a_eq_matrix.csv"

    dominoes: list[dict[str, Any]] = []
    with order_path.open(newline="") as f:
        for row in csv.DictReader(f):
            pips = row["pips"]
            high_s, low_s = pips.split("-")
            dominoes.append(
                {
                    "id": int(row["domino_id"]),
                    "pips": pips,
                    "high": int(high_s),
                    "low": int(low_s),
                    "position": int(row["position"]),
                    "mean_eq": float(row["mean_eq"]),
                }
            )

    eq_by_pips: dict[str, list[float | None]] = {}
    with matrix_path.open(newline="") as f:
        for row in csv.DictReader(f):
            eq_by_pips[row["Domino"]] = [
                finite_or_none(float(row[f"Move_{move}"])) for move in range(1, 29)
            ]

    eq_matrix = [eq_by_pips[domino["pips"]] for domino in dominoes]
    record = {
        "game_id": "27a aggregate E[Q] surface",
        "source": "forge/analysis/results/tables/27a_eq_matrix.csv",
        "dominoes": dominoes,
        "eq_matrix": eq_matrix,
        "moves": list(range(1, 29)),
    }

    output_path = output_dir / "27a_eq_surface.jsonl"
    output_path.write_text(json.dumps(record) + "\n")
    return output_path


def determine_trick_winner(trick_plays: list[tuple[int, int]], decl_id: int) -> int:
    if len(trick_plays) != 4:
        raise ValueError("trick must have four plays")

    _, lead_domino = trick_plays[0]
    led_suit = led_suit_for_lead_domino(lead_domino, decl_id)
    best_idx = 0
    best_rank = trick_rank(lead_domino, led_suit, decl_id)

    for idx, (_, domino_id) in enumerate(trick_plays[1:], start=1):
        rank = trick_rank(domino_id, led_suit, decl_id)
        if rank > best_rank:
            best_idx = idx
            best_rank = rank

    return trick_plays[best_idx][0]


def step_hold(values: list[float | None]) -> list[float | None]:
    held: list[float | None] = []
    last: float | None = None
    for value in values:
        if value is not None:
            last = value
        held.append(last)
    return held


def game_players(hands: list[list[int]]) -> tuple[list[dict[str, Any]], list[int]]:
    players: list[dict[str, Any]] = []
    domino_order: list[int] = []
    for player, hand in enumerate(hands):
        dominoes = []
        for slot, domino_id in enumerate(hand):
            dominoes.append({"id": domino_id, "pips": domino_label(domino_id), "slot": slot})
            domino_order.append(domino_id)
        players.append({"id": player, "dominoes": dominoes})
    return players, domino_order


def process_game_record(game: Any, game_idx: int) -> dict[str, Any]:
    n_dominoes = 28
    n_moves = 28
    players, domino_order = game_players(game.hands)
    decl_id = int(game.decl_id)

    eq_matrix: list[list[float | None]] = [[None for _ in range(n_moves)] for _ in range(n_dominoes)]
    active_player: list[int] = []
    domino_played = [[False for _ in range(n_moves)] for _ in range(n_dominoes)]
    pdf_data: list[dict[str, Any]] = []
    played_at: dict[int, int] = {}

    team_scores = [0, 0]
    score_history: list[list[int]] = []
    trick_plays: list[tuple[int, int]] = []
    trick_winners: list[int] = []

    for move_idx, decision in enumerate(game.decisions[:n_moves]):
        player = int(decision.player)
        active_player.append(player)
        hand = game.hands[player]

        for slot, domino_id in enumerate(hand):
            global_idx = player * 7 + slot
            legal = bool(decision.legal_mask[slot].item()) if slot < len(decision.legal_mask) else False
            if legal:
                value = float(decision.e_q[slot].item())
                eq_matrix[global_idx][move_idx] = value

                if decision.e_q_pdf is not None:
                    pdf = decision.e_q_pdf[slot].detach().cpu().numpy()
                    var = (
                        float(decision.e_q_var[slot].item())
                        if decision.e_q_var is not None
                        else 0.0
                    )
                    win_bin_start = 60 if player in (0, 2) else 25
                    pdf_data.append(
                        {
                            "d": global_idx,
                            "m": move_idx,
                            "pdf": pdf.tolist(),
                            "mean": value,
                            "std": float(np.sqrt(max(var, 0.0))),
                            "win": float(pdf[win_bin_start:].sum()),
                            "samples": int(decision.n_samples or 0),
                            "converged": bool(decision.converged)
                            if decision.converged is not None
                            else None,
                        }
                    )

        action = int(decision.action_taken)
        if 0 <= action < len(hand):
            global_idx = player * 7 + action
            domino_id = hand[action]
            domino_played[global_idx][move_idx] = True
            played_at[global_idx] = move_idx
            trick_plays.append((player, domino_id))

            if len(trick_plays) == 4:
                winner = determine_trick_winner(trick_plays, decl_id)
                trick_winners.append(winner)
                trick_points = 1 + sum(tables.DOMINO_COUNT_POINTS[d] for _, d in trick_plays)
                team_scores[0 if winner in (0, 2) else 1] += int(trick_points)
                trick_plays = []

        score_history.append([team_scores[0], team_scores[1]])

    for global_idx, row in enumerate(eq_matrix):
        held = step_hold(row)
        if global_idx in played_at:
            for move_idx in range(played_at[global_idx] + 1, n_moves):
                held[move_idx] = None
        eq_matrix[global_idx] = held

    return {
        "game_id": f"sample_{game_idx:03d}_decl_{decl_id}",
        "game_idx": game_idx,
        "trump_id": decl_id,
        "trump_name": DECL_NAMES.get(decl_id, f"unknown-{decl_id}"),
        "players": players,
        "domino_order": domino_order,
        "eq_matrix": eq_matrix,
        "active_player": active_player,
        "domino_played": domino_played,
        "pdf_data": pdf_data,
        "score_history": score_history,
        "trick_winners": trick_winners,
    }


def export_game_and_pdf(sample_path: Path, output_dir: Path, limit: int | None) -> tuple[Path, Path] | None:
    if not sample_path.exists():
        return None

    payload = torch.load(sample_path, weights_only=False)
    games = payload["results"]
    if limit is not None:
        games = games[:limit]

    records = [process_game_record(game, idx) for idx, game in enumerate(games)]

    per_game_path = output_dir / "27b_eq_per_game.jsonl"
    pdf_path = output_dir / "eq_pdf_v3_sample.jsonl"
    per_game_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    pdf_path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return per_game_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--pdf-sample", type=Path, default=DEFAULT_PDF_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=5, help="Number of sample games for PDF/journey.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = [export_surface(args.table_dir, args.output_dir)]
    sample_outputs = export_game_and_pdf(args.pdf_sample, args.output_dir, args.limit)
    if sample_outputs:
        outputs.extend(sample_outputs)

    for output in outputs:
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
