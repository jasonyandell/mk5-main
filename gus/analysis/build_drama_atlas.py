"""Build the §22 Drama Atlas: per-decision analytics over the Gus training corpus.

Produces gus/analysis/drama_atlas.parquet with one row per corpus decision.
Columns:
  Identifiers:      split, seed, game_idx, d_idx, current_player, decl_id
  Drama quantities: outcome_variance, action_fragility, belief_sharpness
  Context:          p_make_taken, mode_action, gus_pi_action, gus_matches_mode
  Distribution:     per_world_top_action_distribution (JSON-encoded dict)
  Legal info:       n_legal_actions, action_taken, e_q_taken, e_q_max

Usage:
  python -u gus/analysis/build_drama_atlas.py
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
import pandas as pd

from gus.model.load import load_student
from gus.model.tokenize import tokenize_decision
from gus.model.features import extract_belief_target, reconstruct_prior_plays
from gus.model.voids import voids_feature_vector

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

SCRATCH_DIR = Path(PROJECT_ROOT) / "scratch/drama_atlas"
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
(SCRATCH_DIR / "thoughts").mkdir(exist_ok=True)

EVENTS_LOG = SCRATCH_DIR / "events.jsonl"
TAIL_LOG = SCRATCH_DIR / "tail.log"
LIVE_LOG = SCRATCH_DIR / "live.log"

_start_time = time.time()


def _emit_event(label: str, payload: dict | None = None):
    rec = {"ts": time.time(), "elapsed_s": round(time.time() - _start_time, 1),
           "label": label}
    if payload:
        rec.update(payload)
    with open(EVENTS_LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _update_live(stage: str, msg: str):
    with open(LIVE_LOG, "w") as f:
        f.write(f"Stage: {stage}\nElapsed: {time.time() - _start_time:.0f}s\n{msg}\n")


def _tail(msg: str):
    with open(TAIL_LOG, "a") as f:
        f.write(f"[{time.time() - _start_time:6.0f}s] {msg}\n")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

ADAPTER_PATH = Path(PROJECT_ROOT) / "gus/adapters/v3_consistency_10000g.pt"


# ---------------------------------------------------------------------------
# Core analytics — three quantities
# ---------------------------------------------------------------------------

def compute_outcome_variance(q_per_world: torch.Tensor, action_taken: int) -> float:
    """How much does the Q of the taken action swing across worlds?
    q_per_world: [M, 7]  (float)
    Returns std of q_per_world[:, action_taken] over M worlds.
    """
    q_taken = q_per_world[:, action_taken].float()
    return float(q_taken.std().item())


def compute_action_fragility(q_per_world: torch.Tensor, legal_mask: torch.Tensor) -> int:
    """Count distinct oracle-best actions across M worlds.
    q_per_world: [M, 7]
    legal_mask:  [7] bool
    Returns integer 1-7 (1 = oracle agrees across all worlds, 7 = max disagreement).
    """
    # Mask illegal actions with -inf, then argmax
    q = q_per_world.float()
    illegal = ~legal_mask.bool()  # [7]
    q[:, illegal] = float("-inf")
    per_world_best = q.argmax(dim=-1)  # [M]
    return int(per_world_best.unique().numel())


def compute_belief_sharpness(
    belief_logits: torch.Tensor,
    belief_mask: torch.Tensor,
) -> float:
    """1 - H / H_max averaged over unseen dominoes.
    belief_logits: [28, 3]  (logits from belief head)
    belief_mask:   [28] bool — True = this domino is unseen (compute belief for it)
    Returns float in [0, 1]: 0 = pure prior, 1 = perfect certainty.
    """
    # Softmax over 3 seats per domino
    probs = torch.softmax(belief_logits.float(), dim=-1)  # [28, 3]
    log_probs = torch.log(probs + 1e-12)
    h = -(probs * log_probs).sum(dim=-1)  # [28]  Shannon entropy in nats
    h_max = math.log(3)  # max entropy = log(3) for 3 seats uniform

    mask = belief_mask.bool()  # [28]
    if mask.sum() == 0:
        return 1.0  # all dominoes seen — maximum certainty

    h_masked = h[mask]
    sharpness = 1.0 - float((h_masked / h_max).mean().item())
    return max(0.0, min(1.0, sharpness))  # clamp for safety


def compute_p_make_taken(q_per_world: torch.Tensor, action_taken: int, threshold: float = 0.0) -> float:
    """P(Q >= threshold | action_taken) from the per-world histogram.
    Threshold = 0: 'what fraction of worlds does this action yield non-negative Q?'
    """
    q_taken = q_per_world[:, action_taken].float()
    return float((q_taken >= threshold).float().mean().item())


def compute_per_world_distribution(q_per_world: torch.Tensor, legal_mask: torch.Tensor) -> str:
    """JSON-encoded dict: {action_idx: n_worlds_where_this_is_oracle_best}"""
    q = q_per_world.float()
    illegal = ~legal_mask.bool()
    q[:, illegal] = float("-inf")
    per_world_best = q.argmax(dim=-1)  # [M]
    unique, counts = per_world_best.unique(return_counts=True)
    dist = {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}
    return json.dumps(dist)


def compute_mode_action(q_per_world: torch.Tensor, legal_mask: torch.Tensor) -> int:
    """Mode of the marginal E[Q]: argmax of mean Q over worlds, among legal actions."""
    e_q_marginal = q_per_world.float().mean(dim=0)  # [7]
    illegal = ~legal_mask.bool()
    e_q_marginal[illegal] = float("-inf")
    return int(e_q_marginal.argmax().item())


# ---------------------------------------------------------------------------
# Student inference for one decision
# ---------------------------------------------------------------------------

def infer_gus(
    model,
    is_voids: bool,
    game,
    d_idx: int,
    device: str,
) -> tuple[int, torch.Tensor]:
    """Run student inference, return (gus_pi_action, belief_logits [28,3]).

    We pass zero world_assignment because we only want belief and pi_me outputs.
    (belief_head depends only on state_emb; world_emb only feeds q_head.)
    """
    decision = game.decisions[d_idx]
    current_player = int(decision.player)

    tokens, attn_mask = tokenize_decision(
        game.hands,
        int(game.decl_id),
        game.decisions,
        d_idx,
    )
    prior_plays = reconstruct_prior_plays(game.hands, game.decisions, d_idx)

    # Tensors to device
    tokens = tokens.unsqueeze(0).to(device)       # [1, L, 5]
    attn_mask = attn_mask.unsqueeze(0).to(device) # [1, L]
    world_assignment = torch.zeros(1, 28, 3, device=device)  # dummy

    with torch.no_grad():
        if is_voids:
            voids = voids_feature_vector(prior_plays, int(game.decl_id), current_player)
            voids = voids.unsqueeze(0).to(device)  # [1, 24]
            out = model(tokens, attn_mask, world_assignment, voids)
        else:
            out = model(tokens, attn_mask, world_assignment)

    # π_me argmax (legal-masked)
    pi_logits = out["pi_me_logits"][0]  # [7]
    legal = decision.legal_mask.bool().to(device)
    pi_logits_masked = pi_logits.masked_fill(~legal, float("-inf"))
    gus_action = int(pi_logits_masked.argmax().item())

    belief_logits = out["belief_logits"][0].cpu()  # [28, 3]
    return gus_action, belief_logits


# ---------------------------------------------------------------------------
# Process one game — returns list of row dicts
# ---------------------------------------------------------------------------

def process_game(
    game,
    game_idx: int,
    seed: int,
    split: str,
    model,
    is_voids: bool,
    device: str,
) -> list[dict[str, Any]]:
    rows = []
    n_decisions = len(game.decisions)

    for d_idx in range(n_decisions):
        decision = game.decisions[d_idx]
        if decision.q_per_world is None or decision.world_hands is None:
            continue

        action_taken = int(decision.action_taken)
        legal_mask = decision.legal_mask.bool()
        q_per_world = decision.q_per_world.float()  # [M, 7]
        e_q = decision.e_q.float() if decision.e_q is not None else None

        # --- Three drama quantities ---
        outcome_var = compute_outcome_variance(q_per_world, action_taken)
        fragility = compute_action_fragility(q_per_world.clone(), legal_mask)

        # Belief sharpness via student's belief head
        gus_action, belief_logits = infer_gus(model, is_voids, game, d_idx, device)

        # Belief mask: which dominoes are unseen by current player?
        prior_plays = reconstruct_prior_plays(game.hands, game.decisions, d_idx)
        current_player = int(decision.player)
        _, belief_mask = extract_belief_target(game.hands, prior_plays, current_player)

        sharpness = compute_belief_sharpness(belief_logits, belief_mask)

        # --- Context columns ---
        p_make = compute_p_make_taken(q_per_world, action_taken, threshold=0.0)
        mode_action = compute_mode_action(q_per_world.clone(), legal_mask)
        dist_json = compute_per_world_distribution(q_per_world.clone(), legal_mask)

        n_legal = int(legal_mask.sum().item())
        e_q_taken = None
        e_q_max = None
        if e_q is not None:
            e_q_legal = e_q.clone()
            e_q_legal[~legal_mask] = float("-inf")
            e_q_taken = float(e_q[action_taken].item())
            e_q_max = float(e_q_legal.max().item())

        rows.append({
            "split": split,
            "seed": seed,
            "game_idx": game_idx,
            "d_idx": d_idx,
            "current_player": current_player,
            "decl_id": int(game.decl_id),
            # Three drama quantities
            "outcome_variance": outcome_var,
            "action_fragility": fragility,
            "belief_sharpness": sharpness,
            # Context
            "p_make_taken": p_make,
            "mode_action": mode_action,
            "gus_pi_action": gus_action,
            "gus_matches_mode": int(gus_action == mode_action),
            "action_taken": action_taken,
            "n_legal_actions": n_legal,
            "e_q_taken": e_q_taken,
            "e_q_max": e_q_max,
            # Distribution (JSON)
            "per_world_top_action_distribution": dist_json,
        })

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _emit_event("started", {"adapter": str(ADAPTER_PATH)})
    _tail("Drama Atlas build started")

    device = "cpu"  # Per spec: don't use MPS (arena-runner is on it)
    _update_live("LOADING MODEL", f"adapter: {ADAPTER_PATH.name}")
    model, is_voids = load_student(str(ADAPTER_PATH), device)
    _emit_event("model_loaded", {"is_voids": is_voids})
    _tail(f"Model loaded (is_voids={is_voids}), device={device}")

    all_rows: list[dict] = []

    # -----------------------------------------------------------------------
    # Pass 1: eval corpus (small — do all at once)
    # -----------------------------------------------------------------------
    _update_live("EVAL PASS", "Loading corpus_eval_20.pt")
    eval_path = Path(PROJECT_ROOT) / "gus/data/corpus_eval_20.pt"
    blob = torch.load(str(eval_path), weights_only=False)
    eval_games = blob["results"]
    eval_seeds = blob.get("seeds", list(range(len(eval_games))))

    eval_rows = []
    for g_idx, game in enumerate(eval_games):
        seed = eval_seeds[g_idx] if g_idx < len(eval_seeds) else g_idx
        rows = process_game(game, g_idx, seed, "eval", model, is_voids, device)
        eval_rows.extend(rows)

    all_rows.extend(eval_rows)
    _emit_event("eval_pass_complete", {"n_decisions": len(eval_rows), "n_games": len(eval_games)})
    _tail(f"Eval pass complete: {len(eval_rows)} decisions from {len(eval_games)} games")

    del blob, eval_games
    gc.collect()

    # -----------------------------------------------------------------------
    # Pass 2: train corpus (100 chunks × ~140 decisions each)
    # -----------------------------------------------------------------------
    DATA_DIR = Path(PROJECT_ROOT) / "gus/data"
    chunk_paths = sorted(DATA_DIR.glob("corpus_train_chunk_*-*.pt"))
    n_chunks = len(chunk_paths)
    _emit_event("train_pass_start", {"n_chunks": n_chunks})
    _tail(f"Train pass: {n_chunks} chunks to process")

    train_rows_total = 0
    games_total = 0
    chunk_t0 = time.time()

    for chunk_i, chunk_path in enumerate(chunk_paths):
        blob = torch.load(str(chunk_path), weights_only=False)
        chunk_games = blob["results"]
        chunk_seeds = blob.get("seeds", list(range(len(chunk_games))))

        chunk_rows = []
        for g_idx, game in enumerate(chunk_games):
            seed = chunk_seeds[g_idx] if g_idx < len(chunk_seeds) else g_idx
            rows = process_game(game, g_idx + games_total, seed, "train",
                                model, is_voids, device)
            chunk_rows.extend(rows)

        all_rows.extend(chunk_rows)
        train_rows_total += len(chunk_rows)
        games_total += len(chunk_games)

        del blob, chunk_games
        gc.collect()

        # Progress logging every 5 chunks
        if (chunk_i + 1) % 5 == 0 or chunk_i == n_chunks - 1:
            elapsed = time.time() - _start_time
            rate = (chunk_i + 1) / elapsed
            eta = (n_chunks - chunk_i - 1) / rate if rate > 0 else 0
            msg = (f"Train chunk {chunk_i+1}/{n_chunks} | "
                   f"{train_rows_total} decisions | {elapsed:.0f}s elapsed | "
                   f"ETA {eta:.0f}s")
            _tail(msg)
            _update_live("TRAIN PASS", msg)

    _emit_event("train_pass_complete", {
        "n_chunks": n_chunks,
        "n_decisions": train_rows_total,
        "n_games": games_total,
    })

    # -----------------------------------------------------------------------
    # Build parquet
    # -----------------------------------------------------------------------
    _update_live("BUILDING PARQUET", f"Total rows: {len(all_rows)}")
    output_path = Path(PROJECT_ROOT) / "gus/analysis/drama_atlas.parquet"
    df = pd.DataFrame(all_rows)
    df.to_parquet(str(output_path), index=False)

    _emit_event("parquet_saved", {
        "path": str(output_path),
        "n_rows": len(df),
        "columns": list(df.columns),
        "split_counts": df["split"].value_counts().to_dict(),
    })
    _tail(f"Parquet saved: {output_path} ({len(df)} rows)")
    _update_live("DONE", f"Parquet saved with {len(df)} rows. Total elapsed: {time.time()-_start_time:.0f}s")

    print(f"\nDrama Atlas complete: {len(df)} rows")
    print(f"Output: {output_path}")
    print(df.groupby("split").size())
    print(df.describe())


if __name__ == "__main__":
    main()
