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

# Corpus pickles reference forge classes; running as a script puts scripts/
# on sys.path instead of the repo root, so add the root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

N = 28


def _scan_game(game) -> dict:
    """Per-game tallies (top-level so multiprocessing can fork it)."""
    hands = game.hands
    out = {
        "worlds": 0, "invalid": 0, "decisions": 0, "with_worlds": 0,
        "weights_dec": 0, "weights_bad": 0,
        "decl8": 1 if int(game.decl_id) == 8 else 0,
        "by_d": {},
    }
    for d_idx, dec in enumerate(game.decisions):
        out["decisions"] += 1
        wh = dec.world_hands
        if wh is None:
            continue
        out["with_worlds"] += 1

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
        out["worlds"] += M
        out["invalid"] += n_inv
        cell = out["by_d"].setdefault(d_idx, [0, 0])
        cell[0] += n_inv
        cell[1] += M

        ww = getattr(dec, "world_weights", None)
        if ww is not None:
            out["weights_dec"] += 1
            ok = (
                ww.shape[0] == M
                and bool((ww >= 0).all())
                and abs(float(ww.sum()) - 1.0) <= 1e-4
            )
            if not ok:
                out["weights_bad"] += 1
    return out


def referee_file(path: str, workers: int = 0) -> dict:
    import multiprocessing as mp

    # Torch's default fd-based tensor sharing exhausts the fd limit when 100
    # games of world tensors cross the pool boundary; use /dev/shm files and
    # raise the soft limit as belt-and-suspenders.
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
    except Exception:
        pass

    blob = torch.load(path, weights_only=False)
    games = blob["results"]

    torch.set_num_threads(1)  # per-decision tensors are small; parallelism is per game
    if workers <= 0:
        workers = min(32, mp.cpu_count() or 1)
    if workers > 1 and len(games) > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            partials = pool.map(_scan_game, games, chunksize=1)
    else:
        partials = [_scan_game(g) for g in games]

    total_worlds = sum(p["worlds"] for p in partials)
    invalid_worlds = sum(p["invalid"] for p in partials)
    total_decisions = sum(p["decisions"] for p in partials)
    decisions_with_worlds = sum(p["with_worlds"] for p in partials)
    weights_decisions = sum(p["weights_dec"] for p in partials)
    weights_bad = sum(p["weights_bad"] for p in partials)
    decl8_games = sum(p["decl8"] for p in partials)
    by_d_invalid: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for p in partials:
        for d, (inv, tot) in p["by_d"].items():
            by_d_invalid[d][0] += inv
            by_d_invalid[d][1] += tot

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
