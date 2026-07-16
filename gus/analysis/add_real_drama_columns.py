"""Add corrected drama columns to drama_atlas.parquet.

New columns (no model inference needed — pure oracle tensor arithmetic):
  marginal_eq_gap    : max - second_max of marginal E[Q] over legal actions
  marginal_pmake_gap : max - second_max of p_make(a)>=0 over legal actions
  dominoes_in_hand   : how many dominoes the acting player still holds
  count_unplayed     : sum of count-pip values still in play (max 35)

These enable the corrected drama definition:
  real_drama_midgame : (marginal_eq_gap <= 1.0) AND (count_unplayed >= 15)
                       AND (10 <= d_idx <= 18) AND (4 <= dominoes_in_hand <= 6)
  real_drama_endgame : (marginal_eq_gap <= 1.0) AND (count_unplayed >= 10)
                       AND (dominoes_in_hand <= 3)

Usage:
  python -u gus/analysis/add_real_drama_columns.py
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd

from gus.hf_data import resolve

# Count domino IDs and their pip values (from forge.oracle.tables)
# domino 8: 3-2 = 5pts, 11: 4-1 = 5pts, 15: 5-0 = 5pts, 20: 5-5 = 10pts, 25: 6-4 = 10pts
COUNT_POINTS = {8: 5, 11: 5, 15: 5, 20: 10, 25: 10}
COUNT_TOTAL = 35  # sum of all count pips


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
SCRATCH_DIR = Path(PROJECT_ROOT) / "scratch/drama_atlas"
EVENTS_LOG = SCRATCH_DIR / "events.jsonl"
TAIL_LOG = SCRATCH_DIR / "tail.log"
LIVE_LOG = SCRATCH_DIR / "live.log"
_start_time = time.time()


def _tail(msg: str):
    with open(TAIL_LOG, "a") as f:
        f.write(f"[{time.time() - _start_time:6.0f}s] {msg}\n")


def _live(stage: str, msg: str):
    with open(LIVE_LOG, "w") as f:
        f.write(f"Stage: {stage}\nElapsed: {time.time() - _start_time:.0f}s\n{msg}\n")


def _emit(label: str, payload: dict | None = None):
    rec = {"ts": time.time(), "elapsed_s": round(time.time() - _start_time, 1),
           "label": label}
    if payload:
        rec.update(payload)
    with open(EVENTS_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------------------
# New column computation — pure tensor ops, no model needed
# ---------------------------------------------------------------------------

def marginal_eq_gap_from_eq(e_q: torch.Tensor, legal_mask: torch.Tensor) -> float:
    """top1 - top2 of marginal E[Q] over legal actions.
    If only 1 legal action, gap = 0 (forced play).
    """
    e_q_legal = e_q.float().clone()
    e_q_legal[~legal_mask.bool()] = float("-inf")
    sorted_vals, _ = e_q_legal.sort(descending=True)
    top1 = float(sorted_vals[0].item())
    if legal_mask.sum() < 2:
        return 0.0
    top2 = float(sorted_vals[1].item())
    if top2 == float("-inf"):
        return 0.0
    return top1 - top2


def marginal_pmake_gap_from_worlds(
    q_per_world: torch.Tensor, legal_mask: torch.Tensor, threshold: float = 0.0
) -> float:
    """top1 - top2 of p_make(a)>=threshold over legal actions.
    p_make(a) = fraction of worlds where Q(world, a) >= threshold.
    """
    q = q_per_world.float()  # [M, 7]
    p_make = (q >= threshold).float().mean(dim=0)  # [7]
    p_make_legal = p_make.clone()
    p_make_legal[~legal_mask.bool()] = float("-inf")
    sorted_vals, _ = p_make_legal.sort(descending=True)
    top1 = float(sorted_vals[0].item())
    if legal_mask.sum() < 2:
        return 0.0
    top2 = float(sorted_vals[1].item())
    if top2 == float("-inf"):
        return 0.0
    return top1 - top2


def dominoes_in_hand(
    decisions: list,
    d_idx: int,
    current_player: int,
) -> int:
    """Count dominoes still in acting player's hand at decision d_idx.
    = 7 - number of prior decisions where player == current_player.
    """
    played_by_me = sum(1 for j in range(d_idx) if int(decisions[j].player) == current_player)
    return 7 - played_by_me


def count_unplayed_sum(
    game_hands: list[list[int]],
    decisions: list,
    d_idx: int,
) -> int:
    """Sum of count-pip values for count dominoes not yet played.
    Count dominos: {8: 5, 11: 5, 15: 5, 20: 10, 25: 10} → max 35.
    """
    # Build played set from prior decisions
    played_set: set[int] = set()
    for j in range(d_idx):
        dec = decisions[j]
        player = int(dec.player)
        slot = int(dec.action_taken)
        hand = game_hands[player]
        if 0 <= slot < len(hand):
            d = int(hand[slot])
            if d >= 0:
                played_set.add(d)

    # Sum count pips not yet played
    total = sum(pts for d_id, pts in COUNT_POINTS.items() if d_id not in played_set)
    return total


# ---------------------------------------------------------------------------
# Process one game
# ---------------------------------------------------------------------------

def process_game_new_cols(
    game,
    game_idx: int,
    seed: int,
    split: str,
) -> list[dict]:
    rows = []
    for d_idx in range(len(game.decisions)):
        decision = game.decisions[d_idx]
        if decision.q_per_world is None or decision.e_q is None:
            continue

        legal_mask = decision.legal_mask.bool()
        q_per_world = decision.q_per_world.float()  # [M, 7]
        e_q = decision.e_q.float()  # [7]
        current_player = int(decision.player)

        eq_gap = marginal_eq_gap_from_eq(e_q, legal_mask)
        pmake_gap = marginal_pmake_gap_from_worlds(q_per_world, legal_mask, threshold=0.0)
        dom_in_hand = dominoes_in_hand(game.decisions, d_idx, current_player)
        count_up = count_unplayed_sum(game.hands, game.decisions, d_idx)

        rows.append({
            "split": split,
            "seed": seed,
            "game_idx": game_idx,
            "d_idx": d_idx,
            "marginal_eq_gap": eq_gap,
            "marginal_pmake_gap": pmake_gap,
            "dominoes_in_hand": dom_in_hand,
            "count_unplayed": count_up,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _emit("v2_started")
    _tail("add_real_drama_columns: started")

    # Load existing atlas
    atlas_path = Path(PROJECT_ROOT) / "gus/analysis/drama_atlas.parquet"
    _live("LOADING ATLAS", str(atlas_path))
    df_old = pd.read_parquet(atlas_path)
    _tail(f"Loaded existing atlas: {len(df_old)} rows")

    all_new: list[dict] = []

    # --- Eval pass ---
    _live("EVAL PASS", "corpus_eval_20.pt")
    eval_path = Path(PROJECT_ROOT) / "gus/data/corpus_eval_20.pt"
    blob = torch.load(resolve(eval_path), weights_only=False)
    eval_games = blob["results"]
    eval_seeds = blob.get("seeds", list(range(len(eval_games))))

    for g_idx, game in enumerate(eval_games):
        seed = eval_seeds[g_idx] if g_idx < len(eval_seeds) else g_idx
        rows = process_game_new_cols(game, g_idx, seed, "eval")
        all_new.extend(rows)

    del blob, eval_games
    gc.collect()
    _tail(f"Eval pass: {len([r for r in all_new if r['split']=='eval'])} decisions")

    # --- Train pass ---
    DATA_DIR = Path(PROJECT_ROOT) / "gus/data"
    chunk_paths = sorted(DATA_DIR.glob("corpus_train_chunk_*-*.pt"))
    n_chunks = len(chunk_paths)
    _emit("v2_train_start", {"n_chunks": n_chunks})

    train_rows_so_far = 0
    games_total = 0

    for chunk_i, chunk_path in enumerate(chunk_paths):
        blob = torch.load(resolve(chunk_path), weights_only=False)
        chunk_games = blob["results"]
        chunk_seeds = blob.get("seeds", list(range(len(chunk_games))))

        for g_idx, game in enumerate(chunk_games):
            seed = chunk_seeds[g_idx] if g_idx < len(chunk_seeds) else g_idx
            rows = process_game_new_cols(game, g_idx + games_total, seed, "train")
            all_new.extend(rows)
            train_rows_so_far += len(rows)

        games_total += len(chunk_games)
        del blob, chunk_games
        gc.collect()

        if (chunk_i + 1) % 10 == 0 or chunk_i == n_chunks - 1:
            elapsed = time.time() - _start_time
            rate = (chunk_i + 1) / elapsed
            eta = (n_chunks - chunk_i - 1) / rate if rate > 0 else 0
            msg = f"Chunk {chunk_i+1}/{n_chunks} | {train_rows_so_far} train rows | {elapsed:.0f}s | ETA {eta:.0f}s"
            _tail(msg)
            _live("TRAIN PASS", msg)

    _emit("v2_train_complete", {"n_rows": len(all_new)})

    # --- Merge new columns into existing atlas ---
    _live("MERGING", f"Merging {len(all_new)} new rows into existing atlas")
    df_new = pd.DataFrame(all_new)

    # Join on (split, seed, game_idx, d_idx)
    df_merged = df_old.merge(
        df_new,
        on=["split", "seed", "game_idx", "d_idx"],
        how="left",
    )

    # Sanity check
    n_unmatched = df_merged["marginal_eq_gap"].isna().sum()
    if n_unmatched > 0:
        _tail(f"WARNING: {n_unmatched} rows have NaN marginal_eq_gap")
    else:
        _tail("All rows matched successfully")

    # Add real drama boolean columns
    df_merged["real_drama_midgame"] = (
        (df_merged["marginal_eq_gap"] <= 1.0) &
        (df_merged["count_unplayed"] >= 15) &
        (df_merged["d_idx"] >= 10) & (df_merged["d_idx"] <= 18) &
        (df_merged["dominoes_in_hand"] >= 4) & (df_merged["dominoes_in_hand"] <= 6)
    )
    df_merged["real_drama_endgame"] = (
        (df_merged["marginal_eq_gap"] <= 1.0) &
        (df_merged["count_unplayed"] >= 10) &
        (df_merged["dominoes_in_hand"] <= 3)
    )

    # Save v2
    output_path = Path(PROJECT_ROOT) / "gus/analysis/drama_atlas_v2.parquet"
    df_merged.to_parquet(str(output_path), index=False)

    _emit("v2_parquet_saved", {
        "path": str(output_path),
        "n_rows": len(df_merged),
        "columns": list(df_merged.columns),
        "real_drama_midgame": int(df_merged["real_drama_midgame"].sum()),
        "real_drama_endgame": int(df_merged["real_drama_endgame"].sum()),
    })
    _tail(f"drama_atlas_v2.parquet saved: {len(df_merged)} rows, {len(df_merged.columns)} cols")

    print(f"\nv2 atlas complete: {len(df_merged)} rows")
    print(f"Output: {output_path}")
    print(f"real_drama_midgame: {df_merged['real_drama_midgame'].sum():,} ({df_merged['real_drama_midgame'].mean():.2%})")
    print(f"real_drama_endgame: {df_merged['real_drama_endgame'].sum():,} ({df_merged['real_drama_endgame'].mean():.2%})")
    print(f"\nSample of new columns:")
    print(df_merged[["split", "d_idx", "marginal_eq_gap", "marginal_pmake_gap",
                      "dominoes_in_hand", "count_unplayed",
                      "real_drama_midgame", "real_drama_endgame"]].head(10))


if __name__ == "__main__":
    main()
