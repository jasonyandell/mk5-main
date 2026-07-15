"""Independent read-side referee for joint-world corpus files.

Grades regenerated corpus chunks for otis Phase R (issues #51/#52/#55):

- R1/R3: every stored world must be a real 28-domino deal — the in-range
  entries of world_hands[m] form EXACTLY the decision's unseen set and each
  relative-seat row carries that opponent's true remaining count.
- R4: where world_weights are present they must be a distribution (sum 1,
  nonnegative) with the same M as world_hands.
- R5: no game may carry decl_id 8 (doubles-suit, purged from enumeration).

This module intentionally does NOT import forge.eq.validate_worlds (the
write-time assertion): the referee re-derives validity from the saved records
alone, per the independence discipline in issue #55. The implementation was
validated against the otis-v0 contamination numbers (54.2% valid at d0 on the
April corpus_eval_20.pt; issue #52's 66.9%/57.5% invalid on the v2/v1 probes).

Usage:
    python -u scripts/referee_worlds.py FILE.pt [FILE2.pt ...] [--json OUT]

Exit code 0 iff every file is fully clean. Prints a per-file verdict line
`REFEREE <file> VERDICT=<CLEAN|DIRTY> ...` for machine parsing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

N = 28


def referee_file(path: str) -> dict:
    blob = torch.load(path, weights_only=False)
    games = blob["results"]

    total_worlds = 0
    invalid_worlds = 0
    total_decisions = 0
    decisions_with_worlds = 0
    weights_decisions = 0
    weights_bad = 0
    decl8_games = 0
    by_d_invalid: dict[int, list[int]] = defaultdict(lambda: [0, 0])  # d -> [inv, tot]

    for game in games:
        hands = game.hands
        if int(game.decl_id) == 8:
            decl8_games += 1
        for d_idx, dec in enumerate(game.decisions):
            total_decisions += 1
            wh = dec.world_hands
            if wh is None:
                continue
            decisions_with_worlds += 1

            # Reconstruct the unseen set + per-seat remaining counts from the
            # record alone (slot convention: action indexes the initial row).
            P = int(dec.player)
            played: set[int] = set()
            plays_by_seat = [0, 0, 0, 0]
            for prior in game.decisions[:d_idx]:
                sp = int(prior.player)
                tile = int(hands[sp][int(prior.action_taken)])
                if tile >= 0:
                    played.add(tile)
                plays_by_seat[sp] += 1
            my_initial = {int(x) for x in hands[P] if int(x) >= 0}
            my_remaining = my_initial - played
            unseen = sorted(set(range(N)) - my_remaining - played)
            unseen_mask = torch.zeros(N, dtype=torch.bool)
            unseen_mask[unseen] = True

            wh = wh.long()
            M = wh.shape[0]
            in_range = (wh >= 0) & (wh < N)
            occ = torch.zeros(M, N, dtype=torch.int32)
            occ.scatter_add_(1, wh.clamp(0, N - 1).reshape(M, -1), in_range.reshape(M, -1).int())
            cover_ok = (occ[:, unseen_mask] == 1).all(dim=1) & (occ[:, ~unseen_mask] == 0).all(dim=1)
            expected = torch.tensor(
                [7 - plays_by_seat[(P + r + 1) % 4] for r in range(3)], dtype=torch.long
            )
            rows_ok = (in_range.sum(dim=2) == expected.unsqueeze(0)).all(dim=1)
            valid = cover_ok & rows_ok

            n_inv = int((~valid).sum())
            total_worlds += M
            invalid_worlds += n_inv
            cell = by_d_invalid[d_idx]
            cell[0] += n_inv
            cell[1] += M

            ww = getattr(dec, "world_weights", None)
            if ww is not None:
                weights_decisions += 1
                ok = (
                    ww.shape[0] == M
                    and bool((ww >= 0).all())
                    and abs(float(ww.sum()) - 1.0) <= 1e-4
                )
                if not ok:
                    weights_bad += 1

    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    clean = invalid_worlds == 0 and weights_bad == 0 and decl8_games == 0
    return {
        "file": path,
        "sha256": sha,
        "bytes": Path(path).stat().st_size,
        "games": len(games),
        "decisions": total_decisions,
        "decisions_with_worlds": decisions_with_worlds,
        "worlds": total_worlds,
        "invalid_worlds": invalid_worlds,
        "invalid_fraction": (invalid_worlds / total_worlds) if total_worlds else 0.0,
        "weights_decisions": weights_decisions,
        "weights_bad": weights_bad,
        "decl8_games": decl8_games,
        "by_d_invalid": {
            d: {"invalid": c[0], "total": c[1]} for d, c in sorted(by_d_invalid.items())
        },
        "clean": clean,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", default=None, help="write per-file reports to this JSON path")
    args = ap.parse_args()

    reports = []
    all_clean = True
    for path in args.files:
        r = referee_file(path)
        reports.append(r)
        verdict = "CLEAN" if r["clean"] else "DIRTY"
        all_clean &= r["clean"]
        print(
            f"REFEREE {path} VERDICT={verdict} games={r['games']} "
            f"worlds={r['worlds']} invalid={r['invalid_worlds']} "
            f"({r['invalid_fraction']:.4%}) weights_dec={r['weights_decisions']} "
            f"weights_bad={r['weights_bad']} decl8={r['decl8_games']} "
            f"sha256={r['sha256'][:16]}",
            flush=True,
        )

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(reports, f, indent=2)

    return 0 if all_clean else 1


if __name__ == "__main__":
    sys.exit(main())
