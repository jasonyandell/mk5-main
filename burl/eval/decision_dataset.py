"""Move 3 held-out decision dataset for grading Burl.

Each ``BurlDecision`` is a trick-6 decision with a real choice (|legal|>=2) and
a real consequence (E[Q] gap between the best and second-best legal play is
>=1.0 points). Every decision is pre-paired with the E[Q]-greedy bot's baseline
play so the grader can compare Burl's choice against the bot without re-running
the oracle.

Held-out seeds are 900000..909999. We assert this; training must not leak.

Pipeline per seed (mirrors ``lem/narrate/batch.py``):
    seed -> deal_from_seed -> 10 games (one per declaration)
         -> generate_eq_games_gpu (N=10 worlds, greedy)
         -> for each game, for each of 4 narrators, inspect narrator_turns[5]
            (the narrator's 6th play = the 6th trick = last non-trivial trick).
         -> keep if n_legal >= 2 and eq_gap >= 1.0.

Game-state serialization: instead of dumping the ``ZebGameState`` dataclass
directly, we record ``(seed, decl_id, play_history)`` — the minimum needed to
replay the exact position. ``load_dataset`` rebuilds the state by running
``apply_action`` on the slot corresponding to each (player, domino_id) tuple.
This keeps the jsonl human-readable and the round-trip loss-free.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

EVAL_SEED_START = 900000
EVAL_SEED_END = 910000  # half-open

N_DECLS = 10
N_TRICKS = 7
NARRATOR_TURN_IDX = 5  # 0-indexed: narrator's 6th play = trick 6
DEFAULT_TRICK_IDX = 6  # 1-indexed label stored on the record
DEFAULT_OUT = Path("burl/eval/data/move3_decisions.jsonl")


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


@dataclass
class BurlDecision:
    seed: int
    declaration: int
    narrator_seat: int
    trick_idx: int
    game_state: Any
    legal_plays: list[int]
    per_play_eq: dict[int, float]
    bot_play: int
    bot_eq: float
    eq_gap: float


def _zeb_state_from_deal(seed: int, decl_id: int, bidder: int = 0):
    """Build an initial ``ZebGameState`` matching the pipeline's conventions.

    Matches ``GameStateTensor.from_deals``: trick_leader=0, bidder=0 unless
    overridden. We skip bidding and place the state directly into the PLAYING
    phase.
    """
    from forge.oracle.rng import deal_from_seed
    from forge.zeb.types import BidState, GamePhase, ZebGameState

    hands = tuple(tuple(h) for h in deal_from_seed(seed))
    bid_state = BidState(bids=(30, 0, 0, 0), high_bidder=bidder, high_bid=30)
    return ZebGameState(
        hands=hands,
        dealer=0,
        phase=GamePhase.PLAYING,
        bid_state=bid_state,
        decl_id=decl_id,
        bidder=bidder,
        played=frozenset(),
        play_history=(),
        current_trick=(),
        trick_leader=0,
        team_points=(0, 0),
    )


def _slot_for(hand: tuple[int, ...] | list[int], domino_id: int) -> int:
    for i, d in enumerate(hand):
        if d == domino_id:
            return i
    raise ValueError(f"domino {domino_id} not in hand {tuple(hand)}")


def _replay_state(seed: int, decl_id: int, play_history: list[tuple[int, int]], bidder: int = 0):
    """Rebuild a ``ZebGameState`` at the point just before the next play."""
    from forge.zeb.game import apply_action

    state = _zeb_state_from_deal(seed, decl_id, bidder=bidder)
    for player, domino_id in play_history:
        slot = _slot_for(state.hands[player], domino_id)
        state = apply_action(state, slot)
    return state


def _compute_eq_gap(e_q, legal_mask) -> float:
    eq = e_q.clone()
    eq[~legal_mask] = float("-inf")
    sorted_eq = eq[legal_mask].sort(descending=True).values
    if sorted_eq.numel() < 2:
        return 0.0
    return float((sorted_eq[0] - sorted_eq[1]).item())


def _collect_from_record(
    record,
    seed: int,
    min_legal: int,
    min_gap: float,
    decl_cap: int | None,
    decl_counts: dict[int, int],
) -> list[BurlDecision]:
    out: list[BurlDecision] = []
    decl_id = record.decl_id

    for narrator in range(4):
        if decl_cap is not None and decl_counts[decl_id] >= decl_cap:
            break

        narrator_turns = [(i, d) for i, d in enumerate(record.decisions) if d.player == narrator]
        if len(narrator_turns) <= NARRATOR_TURN_IDX:
            continue

        dec_idx, decision = narrator_turns[NARRATOR_TURN_IDX]

        n_legal = int(decision.legal_mask.sum().item())
        if n_legal < min_legal:
            continue

        gap = _compute_eq_gap(decision.e_q, decision.legal_mask)
        if gap < min_gap:
            continue

        # Replay state up to (but not including) this decision.
        history: list[tuple[int, int]] = []
        initial_hands = [list(h) for h in record.hands]
        hands_remaining = [list(h) for h in record.hands]
        for j in range(dec_idx):
            prev = record.decisions[j]
            dom = hands_remaining[prev.player][prev.action_taken]
            history.append((prev.player, int(dom)))
            hands_remaining[prev.player][prev.action_taken] = -1

        try:
            state = _replay_state(seed, decl_id, history)
        except Exception as e:
            log(f"[warn] replay failed seed={seed} decl={decl_id} narrator={narrator}: {e}")
            continue

        hand_for_narrator = initial_hands[narrator]
        legal_plays = [int(hand_for_narrator[s]) for s in range(7) if decision.legal_mask[s]]
        per_play_eq = {
            int(hand_for_narrator[s]): float(decision.e_q[s].item())
            for s in range(7)
            if decision.legal_mask[s]
        }
        bot_slot = decision.action_taken
        bot_play = int(hand_for_narrator[bot_slot])
        bot_eq = float(decision.e_q[bot_slot].item())

        out.append(
            BurlDecision(
                seed=seed,
                declaration=decl_id,
                narrator_seat=narrator,
                trick_idx=DEFAULT_TRICK_IDX,
                game_state=state,
                legal_plays=legal_plays,
                per_play_eq=per_play_eq,
                bot_play=bot_play,
                bot_eq=bot_eq,
                eq_gap=gap,
            )
        )
        decl_counts[decl_id] += 1

    return out


def generate_dataset(
    n_decisions: int = 50,
    seeds: Iterable[int] | None = None,
    min_legal: int = 2,
    min_gap: float = 1.0,
    trick_idx: int = DEFAULT_TRICK_IDX,
    n_worlds: int = 10,
    checkpoint: str | Path | None = None,
    device: str | None = None,
    balance_declarations: bool = True,
    progress_every: int = 10,
) -> list[BurlDecision]:
    """Generate ``n_decisions`` held-out eval decisions.

    Iterates ``seeds`` (default: 900000..) and, per seed, plays all 10
    declarations via E[Q]-greedy (N=``n_worlds``). Keeps decisions passing the
    trick-6 / |legal|>=2 / eq_gap>=``min_gap`` filter until we've collected
    ``n_decisions``. If ``balance_declarations``, caps per-declaration count at
    ``ceil(n_decisions / 10)`` to encourage coverage across all 10 decls.
    """
    if trick_idx != DEFAULT_TRICK_IDX:
        # We only compute E[Q] at the narrator's 6th play; other trick indices
        # would require a different turn index (and the "last non-trivial"
        # framing is specific to trick 6).
        raise NotImplementedError(f"trick_idx={trick_idx} not supported; use 6")

    # Heavy imports deferred so --help stays snappy.
    import torch

    from burl.tools.eq_distribution import load_eq_oracle
    from forge.eq.generate.pipeline import generate_eq_games_gpu
    from forge.oracle.rng import deal_from_seed

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    oracle = load_eq_oracle(checkpoint_path=checkpoint, device=device)

    if seeds is None:
        seed_iter = iter(range(EVAL_SEED_START, EVAL_SEED_END))
    else:
        seed_iter = iter(seeds)

    cap = None
    if balance_declarations:
        # ceil-div so the dataset can over-shoot by at most (N_DECLS-1) during
        # the last seed pass.
        cap = (n_decisions + N_DECLS - 1) // N_DECLS

    out: list[BurlDecision] = []
    decl_counts: dict[int, int] = defaultdict(int)
    seeds_processed = 0
    scanned = 0
    kept = 0
    t_start = time.time()

    for seed in seed_iter:
        if len(out) >= n_decisions:
            break
        if not (EVAL_SEED_START <= seed < EVAL_SEED_END):
            raise AssertionError(
                f"seed {seed} outside held-out range [{EVAL_SEED_START}, {EVAL_SEED_END})"
            )

        hands = deal_from_seed(seed)
        batch_hands = [hands] * N_DECLS
        batch_decls = list(range(N_DECLS))

        try:
            records = generate_eq_games_gpu(
                model=oracle.model,
                hands=batch_hands,
                decl_ids=batch_decls,
                n_samples=n_worlds,
                device=device,
                greedy=True,
                seeds=[seed * N_DECLS + d for d in range(N_DECLS)],
            )
        except Exception as e:
            log(f"[error] seed {seed} E[Q] batch failed: {e}")
            continue

        for record in records:
            scanned += sum(1 for d in record.decisions if d.player == record.decisions[0].player) or 1
            collected = _collect_from_record(
                record, seed, min_legal, min_gap, cap, decl_counts,
            )
            for dec in collected:
                out.append(dec)
                kept += 1
                if len(out) >= n_decisions:
                    break
            if len(out) >= n_decisions:
                break

        seeds_processed += 1
        if seeds_processed % progress_every == 0:
            elapsed = time.time() - t_start
            rate = seeds_processed / elapsed if elapsed > 0 else 0.0
            log(
                f"[progress] seed={seed} kept={kept}/{n_decisions} "
                f"seeds={seeds_processed} rate={rate:.2f} seeds/s"
            )

    elapsed = time.time() - t_start
    log(f"[done] kept={kept} scanned_seeds={seeds_processed} time={elapsed:.1f}s")
    return out[:n_decisions]


def save_dataset(path: Path, decisions: list[BurlDecision]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in decisions:
            state = d.game_state
            play_history = [[int(p), int(dom)] for p, dom in state.play_history]
            entry = {
                "seed": d.seed,
                "declaration": d.declaration,
                "narrator_seat": d.narrator_seat,
                "trick_idx": d.trick_idx,
                "bidder": int(state.bidder),
                "play_history": play_history,
                "legal_plays": list(d.legal_plays),
                "per_play_eq": {str(k): float(v) for k, v in d.per_play_eq.items()},
                "bot_play": int(d.bot_play),
                "bot_eq": float(d.bot_eq),
                "eq_gap": float(d.eq_gap),
            }
            f.write(json.dumps(entry) + "\n")


def load_dataset(path: Path) -> list[BurlDecision]:
    path = Path(path)
    out: list[BurlDecision] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            history = [(int(p), int(dom)) for p, dom in e["play_history"]]
            state = _replay_state(
                e["seed"], e["declaration"], history, bidder=int(e.get("bidder", 0)),
            )
            out.append(
                BurlDecision(
                    seed=int(e["seed"]),
                    declaration=int(e["declaration"]),
                    narrator_seat=int(e["narrator_seat"]),
                    trick_idx=int(e["trick_idx"]),
                    game_state=state,
                    legal_plays=[int(x) for x in e["legal_plays"]],
                    per_play_eq={int(k): float(v) for k, v in e["per_play_eq"].items()},
                    bot_play=int(e["bot_play"]),
                    bot_eq=float(e["bot_eq"]),
                    eq_gap=float(e["eq_gap"]),
                )
            )
    return out


def _summarize(decisions: list[BurlDecision]) -> str:
    from forge.oracle.declarations import DECL_ID_TO_NAME

    counts: dict[int, int] = defaultdict(int)
    for d in decisions:
        counts[d.declaration] += 1
    gaps = sorted(d.eq_gap for d in decisions)
    n = len(gaps)
    mean_gap = sum(gaps) / n if n else 0.0
    median_gap = gaps[n // 2] if n else 0.0

    lines = [
        f"n={n}",
        f"mean_eq_gap={mean_gap:.3f}  median_eq_gap={median_gap:.3f}",
        "declaration mix:",
    ]
    for decl_id in range(N_DECLS):
        c = counts.get(decl_id, 0)
        lines.append(f"  {decl_id} {DECL_ID_TO_NAME[decl_id]:<15s} {c}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50, dest="n_decisions")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-legal", type=int, default=2)
    parser.add_argument("--min-gap", type=float, default=1.0)
    parser.add_argument("--n-worlds", type=int, default=10)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--no-balance", action="store_true",
                        help="Disable per-declaration cap (allow uneven mix)")
    parser.add_argument("--seed-start", type=int, default=EVAL_SEED_START)
    parser.add_argument("--seed-stop", type=int, default=EVAL_SEED_END,
                        help="Exclusive upper bound on seeds (default: 910000).")
    args = parser.parse_args()

    if not (EVAL_SEED_START <= args.seed_start < EVAL_SEED_END and
            args.seed_stop <= EVAL_SEED_END and args.seed_stop > args.seed_start):
        raise SystemExit(
            f"seed range [{args.seed_start}, {args.seed_stop}) must lie in held-out "
            f"[{EVAL_SEED_START}, {EVAL_SEED_END})"
        )

    decisions = generate_dataset(
        n_decisions=args.n_decisions,
        seeds=range(args.seed_start, args.seed_stop),
        min_legal=args.min_legal,
        min_gap=args.min_gap,
        n_worlds=args.n_worlds,
        checkpoint=args.checkpoint,
        balance_declarations=not args.no_balance,
    )

    save_dataset(args.out, decisions)

    # Round-trip sanity check.
    reloaded = load_dataset(args.out)
    assert len(reloaded) == len(decisions), f"round-trip count mismatch: {len(reloaded)} vs {len(decisions)}"
    for orig, rel in zip(decisions, reloaded):
        assert orig.seed == rel.seed
        assert orig.declaration == rel.declaration
        assert orig.bot_play == rel.bot_play
        assert abs(orig.eq_gap - rel.eq_gap) < 1e-5
        assert set(orig.legal_plays) == set(rel.legal_plays)
        assert tuple(orig.game_state.play_history) == tuple(rel.game_state.play_history)
        assert int(orig.game_state.decl_id) == int(rel.game_state.decl_id)

    log("\n" + _summarize(decisions))
    log(f"\nsaved {len(decisions)} decisions -> {args.out}")
    log("round-trip: OK")


if __name__ == "__main__":
    main()
