#!/usr/bin/env python3
"""#26 bridge: arena self-play deals + REAL auctions -> belief-training corpus.

The arena (``arena.cli --emit-snapshots``) plays full games with real
auctions and dumps, for every hand, the seat-ordered deal plus the auction
that actually happened (per-seat bids, winning seat, contract, declaration).

This bridge runs the SAME oracle E[Q] generation used for the belief corpus
(``forge.eq.generate`` with ``save_joint_worlds=True``) on those real deals,
then stamps each produced ``GameRecordGPU`` with the auction provenance
(``bids`` / ``bidder`` / ``bid_value``). The output is a
``{"results": [...], "seeds": [...]}`` corpus loadable by
``gus.model.dataset_seq_world.JointWorldFullDataset`` — the data path that
unlocks auction-conditioned belief (#24).

Usage:
    python -u -m forge.cli.generate_eq_from_snapshots \\
        --snapshots arena/results/snapshots.json \\
        --checkpoint forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt \\
        --out gus/data/corpus_from_arena.pt \\
        --device mps --n-samples 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from forge.eq.generate.pipeline import generate_eq_games_gpu
from forge.eq.generate.types import GameRecordGPU


def load_snapshots(path: Path) -> list[dict]:
    """Read the arena ``--emit-snapshots`` JSON payload.

    Returns the list of per-hand snapshot dicts; passed-out hands carry no
    contract and are simply absent from the arena output.
    """
    payload = json.loads(path.read_text())
    snaps = payload["snapshots"]
    if not snaps:
        raise ValueError(f"{path} contains no snapshots")
    return snaps


def attach_auction(record: GameRecordGPU, snap: dict) -> GameRecordGPU:
    """Stamp the real-auction provenance from a snapshot onto a game record."""
    record.bids = tuple(int(b) for b in snap["bids"])
    record.bidder = int(snap["bidder"])
    record.bid_value = int(snap["bid_value"])
    return record


def forced_actions_from_snapshot(snap: dict) -> list[int]:
    """The snapshot's recorded play line as slot indices (0-6) for teacher-forcing.

    Each recorded play is ``(seat, domino_id)``. ``GameStateTensor`` lays a
    seat's hand out in ``snap["hands"][seat]`` order and never permutes it (a
    played slot is set to -1 in place), so the acting slot of a domino is simply
    its index in that seat's original 7-tile hand — the exact slot the oracle's
    ``e_q[7]`` / ``legal_mask[7]`` are indexed by. This is the inverse of
    ``apply_actions``: feeding these advances the state along the recorded line.
    """
    slot_of = [{int(d): i for i, d in enumerate(seat_hand)} for seat_hand in snap["hands"]]
    return [slot_of[int(seat)][int(domino)] for seat, domino in snap["plays"]]


def verify_teacher_forced_alignment(record: GameRecordGPU, snap: dict) -> None:
    """Fail-fast that the produced record is byte-aligned to the recorded line.

    ``record.decisions[k].player`` is the engine's current player at ply k; if
    teacher-forcing kept the simulation in lock-step with the snapshot it must
    equal ``plays[k][0]`` for every k (and there must be exactly one decision per
    recorded play). This is the property the (deal, decision-index, acting-seat)
    join in ``champion.jud_net.load_aux_table`` needs to land decision k's E[Q]
    on the corpus's play step k with 100% decision-row coverage.
    """
    plays = snap["plays"]
    if len(record.decisions) != len(plays):
        raise ValueError(
            f"teacher-forced record has {len(record.decisions)} decisions but the "
            f"snapshot has {len(plays)} plays — alignment broken."
        )
    for k, dec in enumerate(record.decisions):
        if int(dec.player) != int(plays[k][0]):
            raise ValueError(
                f"teacher-forced decision {k} mover {int(dec.player)} != recorded "
                f"mover {int(plays[k][0])} — alignment broken."
            )


def generate_corpus(
    *,
    model,
    snapshots: list[dict],
    n_samples: int,
    device: str,
    batch_size: int,
    teacher_forced: bool = False,
    log_every_s: float = 30.0,
) -> list[GameRecordGPU]:
    """Run the joint-world oracle on each snapshot's deal and attach the auction.

    Batches snapshots through ``generate_eq_games_gpu`` (the canonical corpus
    path, ``save_joint_worlds=True`` so each decision carries
    ``world_hands``/``q_per_world``), then stamps ``bids``/``bidder``/
    ``bid_value`` from the originating snapshot onto each record.

    With ``teacher_forced=True`` the oracle still evaluates E[Q] for every legal
    action at every state, but the state is advanced along the snapshot's
    RECORDED play line rather than the oracle's own p_make-greedy line. Decision
    k then corresponds exactly to arena play step k, so the aux-label join
    (``champion.jud_net.load_aux_table``) lands on every decision row — the
    guaranteed-aligned path that fixes the greedy-replay prefix mismatch.
    """
    results: list[GameRecordGPU] = []
    n = len(snapshots)
    t0 = last_log = time.time()
    for start in range(0, n, batch_size):
        batch = snapshots[start : start + batch_size]
        hands = [[list(h) for h in snap["hands"]] for snap in batch]
        decl_ids = [int(snap["decl_id"]) for snap in batch]
        bid_values = [int(snap["bid_value"]) for snap in batch]
        bidders = [int(snap["bidder"]) for snap in batch]
        forced = (
            [forced_actions_from_snapshot(snap) for snap in batch]
            if teacher_forced else None
        )

        records = generate_eq_games_gpu(
            model=model,
            hands=hands,
            decl_ids=decl_ids,
            n_samples=n_samples,
            device=device,
            save_joint_worlds=True,
            bid_values=bid_values,
            # The real bid winner leads the first trick, so the generated play
            # matches the auction the belief head conditions on (avoids a
            # train/inference mismatch where seat 0 always led).
            bidders=bidders,
            forced_actions=forced,
        )
        for record, snap in zip(records, batch):
            if teacher_forced:
                verify_teacher_forced_alignment(record, snap)
            results.append(attach_auction(record, snap))

        now = time.time()
        if now - last_log >= log_every_s or start + batch_size >= n:
            last_log = now
            done = len(results)
            rate = done / (now - t0) if now > t0 else 0.0
            print(
                f"  {done}/{n} games  {rate:.2f} g/s  t={now - t0:5.0f}s",
                flush=True,
            )
        if device == "cuda":
            torch.cuda.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bridge arena real-auction snapshots -> belief-training corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--snapshots", required=True,
                        help="Arena --emit-snapshots JSON payload")
    parser.add_argument("--checkpoint", required=True,
                        help="Stage 1 oracle checkpoint")
    parser.add_argument("--out", required=True,
                        help="Output corpus .pt (loadable by JointWorldFullDataset)")
    parser.add_argument("--device", default="mps",
                        help="Device: mps (default) / cuda / cpu")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Worlds sampled per decision (default: 50)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Snapshots (games) per GPU batch (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many snapshots (smoke runs)")
    parser.add_argument("--teacher-forced", action="store_true",
                        help="Advance each hand along the snapshot's RECORDED play "
                             "line instead of the oracle's greedy line, so decision "
                             "k == arena play step k. Required for aligned Lane B "
                             "E[Q] labels (100%% aux-join coverage); without it the "
                             "greedy-replay prefix drifts and labels silently "
                             "mis-key. Needs snapshots that carry full 'plays'.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed torch RNG so the MRV world sampling is reproducible. "
                             "For the #26 self-play loop this makes round-over-round "
                             "corpora differ by the real auction shift, not by sampler "
                             "jitter — essential for a clean belief-KL convergence read.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    snap_path = Path(args.snapshots)
    if not snap_path.exists():
        print(f"Error: snapshots not found: {snap_path}", flush=True)
        return 1

    snapshots = load_snapshots(snap_path)
    if args.limit is not None:
        snapshots = snapshots[: args.limit]
    print(f"Loaded {len(snapshots)} snapshots from {snap_path}", flush=True)

    if args.teacher_forced:
        missing = [i for i, s in enumerate(snapshots) if "plays" not in s]
        if missing:
            print(
                f"Error: --teacher-forced needs full 'plays' on every snapshot; "
                f"{len(missing)} row(s) lack it (first at index {missing[0]}). "
                "Regenerate the corpus with a post-jud-v1 arena (--emit-snapshots).",
                flush=True,
            )
            return 1
        print("Teacher-forced: replaying recorded play lines (aligned E[Q] labels).",
              flush=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("Error: CUDA requested but unavailable.", flush=True)
        return 1
    if device == "mps" and not torch.backends.mps.is_available():
        print("Error: MPS requested but unavailable.", flush=True)
        return 1

    from forge.eq.oracle import Stage1Oracle

    print(f"Loading oracle: {args.checkpoint} on {device}", flush=True)
    oracle = Stage1Oracle(args.checkpoint, device=device, compile=False)
    print(f"  samples/decision={args.n_samples}  batch={args.batch_size}", flush=True)

    t0 = time.perf_counter()
    results = generate_corpus(
        model=oracle.model,
        snapshots=snapshots,
        n_samples=args.n_samples,
        device=device,
        batch_size=args.batch_size,
        teacher_forced=args.teacher_forced,
    )
    elapsed = time.perf_counter() - t0
    rate = len(results) / elapsed if elapsed > 0 else 0.0
    print(f"Generated {len(results)} games in {elapsed:.1f}s ({rate:.2f} g/s)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "results": results,
        # No fresh-deal seeds: provenance is the auction, not a deal seed.
        # `seeds` stays present-and-empty so JointWorldFullDataset's
        # `blob.get("seeds", [])` extension is a no-op.
        "seeds": [],
        "snapshot_file": str(snap_path),
        "checkpoint": args.checkpoint,
        "n_samples": args.n_samples,
        "source": "arena_real_auction_teacher_forced" if args.teacher_forced
                  else "arena_real_auction",
        "teacher_forced": bool(args.teacher_forced),
    }
    torch.save(save_dict, str(out_path))
    print(f"Saved corpus -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
