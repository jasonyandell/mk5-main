"""Bayes-optimal belief ceiling on a corpus.

For each decision, use the oracle's own sampled worlds (decision.world_hands,
shape [M, 3, 7]) to compute the empirical posterior P(seat | domino) for each
unseen domino. Argmax gives the Bayes-optimal top-1 prediction. Compare to
truth (the real deal's hand).

This is the ceiling no neural belief head can exceed on this corpus. It
answers the question "is belief-top-1 hard because of architecture/capacity,
or hard because the signal isn't there?"

Usage:
    python -u -m gus.eval.belief_ceiling --corpus gus/data/corpus_eval_20.pt

Result on eval_20 (2026-04-22): 39.184% top-1. Gus's belief head matches
this to within noise (§6 receipts), so top-1 accuracy is at ceiling on this
corpus. See PRACTICALITIES §21 for the full receipt and reorientation of
the belief roadmap toward posterior-shape (calibration) work.
"""
from __future__ import annotations
import argparse
import sys
from collections import defaultdict

import torch

N_DOMINOES = 28


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    args = ap.parse_args()

    blob = torch.load(args.corpus, weights_only=False)
    games = blob["results"]

    n_games = len(games)
    print(f"Loaded {n_games} games from {args.corpus}")

    # Accumulators.
    total_correct = 0
    total_slots = 0
    # Per-decision-idx breakdown.
    by_d: dict[int, list[int]] = defaultdict(list)
    # Per-world-count sanity: how many worlds per decision on avg?
    worlds_counts: list[int] = []

    for g_idx, game in enumerate(games):
        hands = game.hands  # [4][7] absolute seat rows
        # Build absolute truth: for each domino, which seat holds it?
        dom_to_seat = [-1] * N_DOMINOES
        for seat in range(4):
            for d in hands[seat]:
                d = int(d)
                if 0 <= d < N_DOMINOES:
                    dom_to_seat[d] = seat

        for d_idx, decision in enumerate(game.decisions):
            P = int(decision.player)

            # Reconstruct played dominoes up to this decision.
            played = set()
            for prior in game.decisions[:d_idx]:
                a = int(prior.action_taken)
                # action_taken is a slot; the actual domino is hands[prior.player][a]
                d_played = int(hands[int(prior.player)][a])
                if d_played >= 0:
                    played.add(d_played)

            my_hand = {int(x) for x in hands[P] if int(x) >= 0}
            # Unseen-to-P dominoes = all 28 minus my hand minus already played.
            unseen = [d for d in range(N_DOMINOES) if d not in my_hand and d not in played]

            # world_hands: [M, 3, 7]; row 0 = (P+1)%4 left_opp, row 1 = (P+2)%4 partner, row 2 = (P+3)%4 right_opp
            wh = decision.world_hands
            if wh is None:
                continue
            if hasattr(wh, "shape"):
                M = wh.shape[0]
            else:
                M = len(wh)
            worlds_counts.append(M)

            # For each unseen domino, count occurrences per relative seat across M worlds.
            counts = torch.zeros(N_DOMINOES, 3, dtype=torch.float32)
            for m in range(M):
                world_m = wh[m]  # [3, 7]
                if hasattr(world_m, "tolist"):
                    rows = world_m.tolist()
                else:
                    rows = world_m
                for rel in range(3):
                    for d in rows[rel]:
                        d = int(d)
                        if 0 <= d < N_DOMINOES:
                            counts[d, rel] += 1.0

            # Argmax over rel seats → best-guess rel seat.
            best_rel = counts.argmax(dim=-1)  # [28]
            # Map rel → absolute seat: abs = (P + rel + 1) % 4.
            for d in unseen:
                truth_seat = dom_to_seat[d]
                truth_rel = (truth_seat - P - 1) % 4
                if truth_rel == 3:
                    # truth_rel must be in {0,1,2} (not me). If it's 3 (= me), skip — shouldn't happen since d is unseen.
                    continue
                pred_rel = int(best_rel[d].item())
                ok = 1 if pred_rel == truth_rel else 0
                total_correct += ok
                total_slots += 1
                by_d[d_idx].append(ok)

        if (g_idx + 1) % 5 == 0 or g_idx == n_games - 1:
            acc = total_correct / max(total_slots, 1)
            print(f"  game {g_idx+1}/{n_games}  running top-1={acc:.3%}  slots={total_slots}")

    overall = total_correct / max(total_slots, 1)
    avg_worlds = sum(worlds_counts) / max(len(worlds_counts), 1)
    print()
    print(f"=== Bayes-optimal belief top-1 ceiling ===")
    print(f"  Overall top-1:        {overall:.3%}")
    print(f"  Total slot preds:     {total_slots:,}")
    print(f"  Avg worlds/decision:  {avg_worlds:.0f}")
    print(f"  Chance (3 seats):     33.33%")
    print()
    print("By decision_idx (belief should sharpen over the game):")
    print(f"  {'d_idx':>5s}  {'acc':>7s}  {'n':>5s}")
    for d in sorted(by_d.keys()):
        bits = by_d[d]
        acc = sum(bits) / max(len(bits), 1)
        print(f"  {d:>5d}  {acc:>6.2%}  {len(bits):>5d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
