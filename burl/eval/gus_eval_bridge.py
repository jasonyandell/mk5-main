"""Resolve ``global_idx`` from gus's ``corpus_eval_20.pt`` to ``BurlDecision``.

The gus eval corpus is indexed as 20 games x 28 decisions flattened, so a
row's ``global_idx`` maps to ``(game_idx = i // 28, dec_in_game = i % 28)``.
Each game's ``GameRecordGPU.decisions`` mirrors that ordering (verified at
ingest in the diagnostic harness; see ``scratch/belief_trajectory_rollout/
diagnostic/gus_eval_bridge.py`` for the original).

This module reconstructs a ``BurlDecision`` for any ``global_idx`` by replaying
the game's action history up to (but not including) the target decision.  It
is read-only: no GPU, no model load, no oracle resolution beyond what
``BurlDecision`` already exposes.

Promoted from ``scratch/belief_trajectory_rollout/diagnostic/`` so tracked
benches and evals can rely on it without sourcing from scratch.
"""
from __future__ import annotations

from pathlib import Path

import torch

from burl.eval.decision_dataset import BurlDecision, _replay_state


DEFAULT_CORPUS = Path("/Users/jason/code/mk5-main/gus/data/corpus_eval_20.pt")
DECISIONS_PER_GAME = 28


def load_corpus(path: Path = DEFAULT_CORPUS) -> dict:
    return torch.load(path, weights_only=False)


def build_burl_decision(corpus: dict, global_idx: int) -> BurlDecision:
    game_idx = global_idx // DECISIONS_PER_GAME
    dec_idx = global_idx % DECISIONS_PER_GAME

    rec = corpus["results"][game_idx]
    seed = int(corpus["seeds"][game_idx])
    decl_id = int(rec.decl_id)
    narrator = int(rec.decisions[dec_idx].player)
    target_dec = rec.decisions[dec_idx]

    hands_remaining = [list(h) for h in rec.hands]
    history: list[tuple[int, int]] = []
    for j in range(dec_idx):
        prev = rec.decisions[j]
        slot = int(prev.action_taken)
        player = int(prev.player)
        dom = hands_remaining[player][slot]
        history.append((player, int(dom)))
        hands_remaining[player][slot] = -1

    state = _replay_state(seed, decl_id, history)

    initial_hand = list(rec.hands[narrator])
    legal_mask = target_dec.legal_mask
    e_q = target_dec.e_q
    legal_plays = [int(initial_hand[s]) for s in range(7) if bool(legal_mask[s])]
    per_play_eq = {
        int(initial_hand[s]): float(e_q[s].item())
        for s in range(7)
        if bool(legal_mask[s])
    }
    bot_slot = int(target_dec.action_taken)
    bot_play = int(initial_hand[bot_slot])
    bot_eq = float(e_q[bot_slot].item())

    eq_sorted = sorted(per_play_eq.values(), reverse=True)
    eq_gap = float(eq_sorted[0] - eq_sorted[1]) if len(eq_sorted) >= 2 else 0.0

    return BurlDecision(
        seed=seed,
        declaration=decl_id,
        narrator_seat=narrator,
        trick_idx=dec_idx,
        game_state=state,
        legal_plays=legal_plays,
        per_play_eq=per_play_eq,
        bot_play=bot_play,
        bot_eq=bot_eq,
        eq_gap=eq_gap,
    )


def build_decisions(
    corpus: dict, global_indices: list[int]
) -> list[BurlDecision]:
    return [build_burl_decision(corpus, gi) for gi in global_indices]
