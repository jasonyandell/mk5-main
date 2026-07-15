"""otis/play_data.py — per-ply info-state rows for OtisPlayNet (issue #53, V2).

Joins the on-policy selfplay snapshots (raw deals + full play histories) to the W2
master parquet labels on ``(source, seed, game_idx, hand_idx, a_team)``, replays each
hand through ``forge.zeb.game`` to reconstruct every mid-hand information state, and
emits featurized rows from the ACTING seat's perspective.

Sampling: one ply per trick (7 tricks → 7 rows per hand), the within-trick position
chosen by a per-hand seeded RNG so trick-position coverage is uniform-ish.

Labels are RELABELED to the actor's team. The parquet's fate classes use
capture axis 0 = bidding team; if the acting seat is NOT on the bidder's team we flip
the capture bit so class 0..3 always means "the actor's team captures this tile".
The trick count is likewise mirrored (``7 - bidder_tricks``) for the opposing team.

The parquet's deal-hash split rides along per hand — every ply of a hand inherits its
hand's split (asserted in the tests). Pure CPU; otis does not own the GPU.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import time
from pathlib import Path

import torch

from forge.zeb.game import apply_action, current_player, is_terminal
from forge.zeb.types import BidState, GamePhase, ZebGameState
from otis.data import _build_label_table
from otis.model import N_FATE_CLASSES, N_TILES, N_TRICKS
from otis.play_model import FEATURE_DIM, FEATURE_SCHEMA_VERSION, featurize_play_state

# Read-only source data lives in the sibling otis-v0 worktree.
SIBLING = Path("/Users/jason/code/mk5-main/.claude/worktrees/otis-v0/scratch/otis-night")
DEFAULT_PARQUET_DIR = SIBLING
DEFAULT_CORPUS_GLOB = str(SIBLING / "corpus" / "chunk_selfplay_*.snapshots.json")

N_TRICKS_PER_HAND = 7
PLAYS_PER_TRICK = 4


def _relabel_fate(cls: int, flip: bool) -> int:
    """Fate class in the actor's frame; flip the capture bit when actor != bidder team."""
    if flip:
        return (1 - cls // 4) * 4 + cls % 4
    return cls


def _build_playing_state(snap: dict) -> ZebGameState:
    """The initial PLAYING ZebGameState for a hand, exactly as arena/engine builds it."""
    hands = tuple(tuple(int(x) for x in h) for h in snap["hands"])
    bidder = int(snap["bidder"])
    return ZebGameState(
        hands=hands,
        dealer=int(snap.get("dealer", 0)),
        phase=GamePhase.PLAYING,
        bid_state=BidState(
            bids=tuple(int(b) for b in snap["bids"]),
            high_bidder=bidder,
            high_bid=int(snap["bid_value"]),
        ),
        decl_id=int(snap["decl_id"]),
        bidder=bidder,
        played=frozenset(),
        play_history=(),
        current_trick=(),
        trick_leader=bidder,
        team_points=(0, 0),
    )


def _sampled_plies(seed: int, game_idx: int, hand_idx: int, a_team: int) -> set[int]:
    """One global ply index per trick (7 total), chosen by a per-hand seeded RNG."""
    rng = random.Random(hash((int(seed), int(game_idx), int(hand_idx), int(a_team))))
    return {t * PLAYS_PER_TRICK + rng.randint(0, PLAYS_PER_TRICK - 1)
            for t in range(N_TRICKS_PER_HAND)}


def _hand_rows(snap: dict, label: tuple) -> list[tuple]:
    """Replay one hand, emitting (feature, fate5, trick, trick_idx) per sampled ply.

    ``label`` = (y_fate5, y_trick) in bidding-team frame (from the parquet).
    ``trick_idx`` is the completed-trick count 0..6 at the decision (for the by-trick gate).
    """
    y_fate5, y_trick = label
    bidder_team = int(snap["bidder"]) % 2
    sampled = _sampled_plies(
        snap["seed"], snap["game_idx"], snap["hand_idx"], snap["a_team"]
    )

    state = _build_playing_state(snap)
    out: list[tuple] = []
    for ply, (seat, dom) in enumerate(snap["plays"]):
        seat = int(seat)
        dom = int(dom)
        cp = current_player(state)
        if cp != seat:
            raise ValueError(f"seat mismatch at ply {ply}: current_player {cp} != {seat}")
        if ply in sampled:
            flip = (seat % 2) != bidder_team
            fate = [_relabel_fate(int(y_fate5[i]), flip) for i in range(N_TILES)]
            trick = int(y_trick) if not flip else (N_TRICKS - 1) - int(y_trick)
            out.append((featurize_play_state(state, seat), fate, trick, ply // PLAYS_PER_TRICK))
        slot = list(state.hands[cp]).index(dom)
        state = apply_action(state, slot)
    if not is_terminal(state):
        raise ValueError("hand did not terminate after replaying all plays")
    return out


def build_rows(
    *,
    parquet_dir: Path = DEFAULT_PARQUET_DIR,
    corpus_glob: str = DEFAULT_CORPUS_GLOB,
    n_hands: int = 40000,
    log_every_s: float = 30.0,
) -> dict:
    """Build per-ply rows for selfplay hands, grouped by the parquet split.

    Returns a dict keyed by split ∈ {"train","val","test"} → dict of stacked tensors,
    plus a ``"meta"`` block (feature dim, counts, per-tile/trick train marginals).
    """
    tab = _build_label_table(Path(parquet_dir))
    tab = tab[tab["source"] == "selfplay"]
    label: dict[tuple, tuple] = {}
    split_of: dict[tuple, str] = {}
    for r in tab.itertuples(index=False):
        key = (int(r.seed), int(r.game_idx), int(r.hand_idx), int(r.a_team))
        label[key] = (
            (int(r.y_fate_0), int(r.y_fate_1), int(r.y_fate_2), int(r.y_fate_3), int(r.y_fate_4)),
            int(r.y_trick),
        )
        split_of[key] = str(r.split)

    buckets: dict[str, dict[str, list]] = {
        s: {"X": [], "fate": [], "trick": [], "trick_idx": []} for s in ("train", "val", "test")
    }
    chunks = sorted(glob.glob(corpus_glob))
    seen_hands = 0
    seen_rows = 0
    t0 = last = time.time()
    stop = False
    for path in chunks:
        if stop:
            break
        snaps = json.loads(Path(path).read_text())["snapshots"]
        for snap in snaps:
            key = (int(snap["seed"]), int(snap["game_idx"]),
                   int(snap["hand_idx"]), int(snap["a_team"]))
            if key not in label:
                continue
            split = split_of[key]
            for feat, fate, trick, trick_idx in _hand_rows(snap, label[key]):
                buckets[split]["X"].append(feat)
                buckets[split]["fate"].append(fate)
                buckets[split]["trick"].append(trick)
                buckets[split]["trick_idx"].append(trick_idx)
                seen_rows += 1
            seen_hands += 1
            if time.time() - last >= log_every_s:
                last = time.time()
                print(
                    f"[play_data] {seen_hands} hands, {seen_rows} rows, "
                    f"{last - t0:5.0f}s  (train {len(buckets['train']['X'])} "
                    f"val {len(buckets['val']['X'])})",
                    flush=True,
                )
            if n_hands and seen_hands >= n_hands:
                stop = True
                break

    result: dict = {}
    for split, cols in buckets.items():
        if not cols["X"]:
            continue
        result[split] = {
            "X": torch.stack(cols["X"]).to(torch.float32),
            "y_fate": torch.tensor(cols["fate"], dtype=torch.long),          # [N,5]
            "y_trick": torch.tensor(cols["trick"], dtype=torch.long),        # [N]
            "trick_idx": torch.tensor(cols["trick_idx"], dtype=torch.long),  # [N] 0..6
        }

    # Train marginals (base-rate reference for the V2-c gate).
    tr = result.get("train")
    fate_marg = None
    trick_marg = None
    if tr is not None:
        fate_marg = []
        for i in range(N_TILES):
            counts = torch.bincount(tr["y_fate"][:, i], minlength=N_FATE_CLASSES).to(torch.float64)
            fate_marg.append((counts / counts.sum()).tolist())
        tc = torch.bincount(tr["y_trick"], minlength=N_TRICKS).to(torch.float64)
        trick_marg = (tc / tc.sum()).tolist()

    result["meta"] = {
        "feature_dim": FEATURE_DIM,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "n_hands": seen_hands,
        "n_rows": seen_rows,
        "counts": {s: int(result[s]["X"].shape[0]) for s in result if s != "meta"},
        "wall_s": round(time.time() - t0, 1),
        "fate_train_marginal": fate_marg,   # [5][8]
        "trick_train_marginal": trick_marg,  # [8]
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratch/otis-night2/play_rows")
    ap.add_argument("--n-hands", type=int, default=40000, help="0 = all selfplay hands")
    ap.add_argument("--parquet-dir", default=str(DEFAULT_PARQUET_DIR))
    ap.add_argument("--corpus-glob", default=DEFAULT_CORPUS_GLOB)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = build_rows(
        parquet_dir=Path(args.parquet_dir),
        corpus_glob=args.corpus_glob,
        n_hands=args.n_hands,
    )
    for split in ("train", "val", "test"):
        if split in data:
            torch.save(data[split], out / f"{split}.pt")
            print(f"[play_data] wrote {out / f'{split}.pt'} "
                  f"(N={data[split]['X'].shape[0]})", flush=True)
    (out / "meta.json").write_text(json.dumps(data["meta"], indent=2))
    print(f"[play_data] meta → {out / 'meta.json'}", flush=True)
    print(f"[play_data] DONE {data['meta']['counts']} in {data['meta']['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
