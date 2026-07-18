"""walt/grade.py — WaltPlay arena policy + paired grading driver (builder B4).

WaltPlay is an arena play policy (``choose(states, bid_values, marks,
marks_to_win) -> list[slot]``). It delegates to ``arena.jud_play.JudPlay``
while the mover holds > HORIZON tiles and, at <= HORIZON tiles, builds an
``EndgameRoot`` from the ZebGameState, enumerates the physics-exact worlds,
optionally filters them with the sigma (W1) belief filter, solves the exact
information-set best response against the deterministic jud field, and
converts the chosen domino id back to a fixed-hand slot index at the arena
boundary.

Per DESIGN.md: solver + field-oracle imports are lazy (grade.py imports
cleanly before walt.solver / walt.worlds / walt.field exist); per-state
solves fan out over a multiprocessing pool whose workers each construct their
own CPU FieldOracle.

main() drives a paired-halves match over identical deal seeds:
    team A = margin:wp(r8) bidder + WaltPlay
    team B = margin:wp(r8) bidder + JudPlay
and writes summary.json + per_hand.csv (via arena.match) plus walt-specific
decisions.jsonl, events.jsonl, and tail.log heartbeats.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from walt.contracts import EndgameRoot, HORIZON

# The jud field net (deterministic field for both WaltPlay's solver and the
# team-B JudPlay baseline) and the incumbent bidder.
DEFAULT_FIELD_NET = "champion/jud_net.pt"
INCUMBENT_BIDDER = "margin:wp,model=champion/margin_net_r8.pt"


# --------------------------------------------------------------------------
# Worker-side solve (module-level so it survives spawn pickling)
# --------------------------------------------------------------------------
_WORKER_ORACLE = None


def _worker_init(net_path: str) -> None:
    """Pool initializer: pin threads, build one CPU FieldOracle per worker."""
    global _WORKER_ORACLE
    import torch

    torch.set_num_threads(1)
    from walt.field import FieldOracle

    _WORKER_ORACLE = FieldOracle(net_path=net_path, device="cpu")


def _make_moves_filter(root: EndgameRoot, horizon: int = HORIZON):
    """sigma moves_filter: opponents always; partner only for moves it made
    while still holding > horizon tiles (pre-horizon partner reading)."""
    me = root.me
    hist = root.play_history

    def moves_filter(seat: int, k: int) -> bool:
        if seat % 2 != me % 2:
            return True  # opponent — always sigma-consistent
        # partner (seat != me): tiles held at move k = 7 - prior partner plays
        prior = 0
        for j in range(k):
            if hist[j][0] == seat:
                prior += 1
        return (7 - prior) > horizon

    return moves_filter


def _world_cap_rng(root: EndgameRoot, world_cap: int) -> np.random.Generator:
    """Deterministic rng for the world-cap subsample: a function of the root's
    information set and K only, so re-solving the same root reproduces the
    same subsample (and the same move) across runs and processes."""
    key = (root.me, root.decl_id, root.bidder, tuple(root.my_hand),
           tuple(root.play_history), world_cap)
    return np.random.default_rng(abs(hash(key)) % (2**63))


def _run_solve(root: EndgameRoot, beliefs: str, payoff: str, oracle,
               horizon: int = HORIZON, world_cap: int = 0):
    """Enumerate worlds, optionally sigma-filter, optionally cap, solve.

    world_cap=0 keeps the solve exact-given-worlds. world_cap=K>0 uniformly
    subsamples (without replacement) whenever more than K worlds survive the
    filter — MC error measured 2026-07-18 (scratch/worldcap): p90 |dValue|
    0.45 pts at K=512, 0.58 at K=256; material argmax flips (|dV|>0.5) 0% at
    K>=512, 1.3% at K=256. Flips below that are near-ties.
    """
    from walt.worlds import enumerate_worlds
    from walt.solver import solve

    worlds = enumerate_worlds(root)
    if beliefs == "sigma" and len(worlds) > 0:
        from walt.field import sigma_consistent

        keep = sigma_consistent(root, worlds, oracle,
                                _make_moves_filter(root, horizon))
        filtered = worlds[keep]
        # A deterministic field should never filter out the realized world; if
        # a tie-break edge empties the set, fall back to unfiltered u so play
        # never crashes (logged via walker_flags downstream).
        if len(filtered) > 0:
            worlds = filtered
    if world_cap and len(worlds) > world_cap:
        idx = _world_cap_rng(root, world_cap).choice(
            len(worlds), size=world_cap, replace=False
        )
        worlds = worlds[np.sort(idx)]
    n = max(len(worlds), 1)
    weights = np.full(len(worlds), 1.0 / n, dtype=np.float64)
    return solve(root, worlds, weights, oracle, payoff=payoff)


def _solve_task(task):
    """Pool task: (root, beliefs, payoff) -> compact result dict."""
    root, beliefs, payoff, horizon, world_cap = task
    t0 = time.perf_counter()
    res = _run_solve(root, beliefs, payoff, _WORKER_ORACLE, horizon, world_cap)
    solve_ms = (time.perf_counter() - t0) * 1e3
    _WORKER_ORACLE.evict_if_huge()   # solve boundary: safe to drop the memo
    return {
        "best_move": int(res.best_move),
        "value": float(res.value),
        "n_worlds": int(res.n_worlds),
        "n_nodes": int(res.n_nodes),
        "n_field_queries": int(res.n_field_queries),
        "walker_flags": dict(res.walker_flags),
        "root_values": {int(k): float(v) for k, v in res.root_values.items()},
        "solve_ms": solve_ms,
    }


# --------------------------------------------------------------------------
# WaltPlay
# --------------------------------------------------------------------------
class WaltPlay:
    """Arena play policy: JudPlay above the horizon, exact solver at/below it.

    Parameters
    ----------
    field_net : path to the jud field net (deterministic field).
    beliefs   : 'u' (W0 uniform) or 'sigma' (W1 exact B(sigma) filter).
    payoff    : 'points' or 'make' (passed through to the solver).
    pool_size : multiprocessing workers for per-state solves. 0 => run
                in-process sequentially with one shared FieldOracle (used by
                the smoke test to avoid spawning workers).
    decisions_path : optional JSONL sink; one record per walt solve.
    """

    def __init__(
        self,
        field_net: str = DEFAULT_FIELD_NET,
        *,
        beliefs: str = "sigma",
        payoff: str = "points",
        horizon: int = HORIZON,
        world_cap: int = 0,
        pool_size: int = 12,
        decisions_path: Optional[str] = None,
    ):
        if beliefs not in ("u", "sigma"):
            raise ValueError(f"beliefs must be 'u' or 'sigma', got {beliefs!r}")
        if payoff not in ("points", "make"):
            raise ValueError(f"payoff must be 'points' or 'make', got {payoff!r}")
        self.field_net = field_net
        self.beliefs = beliefs
        self.payoff = payoff
        self.horizon = int(horizon)
        self.world_cap = int(world_cap)
        self.pool_size = pool_size
        self.decisions_path = decisions_path

        # Team-B-style jud head, in-process, for the > HORIZON delegation.
        from arena.jud_play import JudPlay
        from champion.jud_net import load_jud_net

        self._jud = JudPlay(load_jud_net(field_net, device="cpu"))

        # ctx maps id(state) -> (game_idx, hand_idx, a_team); the driver sets
        # it each tick so decision records can be joined to games.
        self.ctx: dict = {}

        self._pool = None            # lazy multiprocessing.Pool
        self._local_oracle = None    # lazy in-process FieldOracle (pool_size 0)
        self._dec_fh = None
        if decisions_path:
            Path(decisions_path).parent.mkdir(parents=True, exist_ok=True)
            self._dec_fh = open(decisions_path, "a", buffering=1)

    # -- infra -------------------------------------------------------------
    def _ensure_pool(self):
        if self._pool is None:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.pool_size,
                initializer=_worker_init,
                initargs=(self.field_net,),
            )
        return self._pool

    def _ensure_local_oracle(self):
        if self._local_oracle is None:
            import torch

            torch.set_num_threads(1)
            from walt.field import FieldOracle

            self._local_oracle = FieldOracle(net_path=self.field_net, device="cpu")
        return self._local_oracle

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        if self._dec_fh is not None:
            self._dec_fh.close()
            self._dec_fh = None

    # -- root construction -------------------------------------------------
    @staticmethod
    def _build_root(state, mover: int, remaining: list[int], bid_value: int) -> EndgameRoot:
        return EndgameRoot(
            decl_id=state.decl_id,
            bidder=state.bidder,
            bid_value=int(bid_value),
            bids=tuple(state.bid_state.bids),
            dealer=state.dealer,
            me=mover,
            my_hand=tuple(sorted(remaining)),
            play_history=tuple(state.play_history),
            trick_leader=state.trick_leader,
            current_trick=tuple(state.current_trick),
            team_points=tuple(state.team_points),
        )

    # -- policy interface --------------------------------------------------
    def choose(
        self,
        states: Sequence,
        bid_values: Sequence[int],
        marks=None,
        marks_to_win: int = 7,
    ) -> list[int]:
        from forge.zeb.game import current_player

        out: list[Optional[int]] = [None] * len(states)

        jud_idx: list[int] = []
        jud_states: list = []
        jud_bids: list[int] = []

        solve_meta: list[tuple] = []  # (i, state, mover, root)
        tasks: list[tuple] = []

        for i, s in enumerate(states):
            mover = current_player(s)
            remaining = [d for d in s.hands[mover] if d not in s.played]
            if len(remaining) > self.horizon:
                jud_idx.append(i)
                jud_states.append(s)
                jud_bids.append(bid_values[i])
            else:
                root = self._build_root(s, mover, remaining, bid_values[i])
                solve_meta.append((i, s, mover, root))
                tasks.append((root, self.beliefs, self.payoff, self.horizon,
                              self.world_cap))

        # Above-horizon: one batched JudPlay forward pass.
        if jud_states:
            acts = self._jud.choose(jud_states, jud_bids, marks, marks_to_win)
            for j, i in enumerate(jud_idx):
                out[i] = acts[j]

        # At/below horizon: exact solves (pool or in-process).
        if tasks:
            if self.pool_size and self.pool_size > 0:
                results = self._ensure_pool().map(_solve_task, tasks)
            else:
                oracle = self._ensure_local_oracle()
                results = []
                for root, beliefs, payoff, horizon, world_cap in tasks:
                    t0 = time.perf_counter()
                    res = _run_solve(root, beliefs, payoff, oracle, horizon,
                                     world_cap)
                    oracle.evict_if_huge()
                    results.append({
                        "best_move": int(res.best_move),
                        "value": float(res.value),
                        "n_worlds": int(res.n_worlds),
                        "n_nodes": int(res.n_nodes),
                        "n_field_queries": int(res.n_field_queries),
                        "walker_flags": dict(res.walker_flags),
                        "root_values": {int(k): float(v)
                                        for k, v in res.root_values.items()},
                        "solve_ms": (time.perf_counter() - t0) * 1e3,
                    })
            for (i, s, mover, root), r in zip(solve_meta, results):
                chosen_id = r["best_move"]
                out[i] = s.hands[mover].index(chosen_id)
                self._record(s, mover, root, r)

        return [o for o in out]

    def _record(self, state, mover: int, root: EndgameRoot, r: dict) -> None:
        if self._dec_fh is None:
            return
        game_idx, hand_idx, a_team = self.ctx.get(id(state), (-1, -1, -1))
        # jud-would-have (as a domino id) for offense/defense divergence audit.
        jud_slot = self._jud.choose([state], [root.bid_value])[0]
        jud_id = state.hands[mover][jud_slot]
        rec = {
            "game": game_idx,
            "hand": hand_idx,
            "a_team": a_team,
            "seat": mover,
            "decl_id": root.decl_id,
            "bidder": root.bidder,
            "bid_value": root.bid_value,
            "tiles_left": len(root.my_hand),
            "beliefs": self.beliefs,
            "payoff": self.payoff,
            "world_cap": self.world_cap,
            "n_worlds": r["n_worlds"],
            "n_nodes": r["n_nodes"],
            "n_field_queries": r["n_field_queries"],
            "solve_ms": round(r["solve_ms"], 3),
            "value": round(r["value"], 4),
            "root": {
                "decl_id": root.decl_id, "bidder": root.bidder,
                "bid_value": root.bid_value, "bids": list(root.bids),
                "dealer": root.dealer, "me": root.me,
                "my_hand": list(root.my_hand),
                "play_history": [list(x) for x in root.play_history],
                "trick_leader": root.trick_leader,
                "current_trick": list(root.current_trick),
                "team_points": list(root.team_points),
            },
            "root_values": r.get("root_values", {}),
            "chosen_id": r["best_move"],
            "jud_would_id": int(jud_id),
            "diverged": int(r["best_move"] != int(jud_id)),
            "walker_flags": r["walker_flags"],
        }
        self._dec_fh.write(json.dumps(rec) + "\n")

    def __repr__(self) -> str:
        return (
            f"WaltPlay(beliefs={self.beliefs}, payoff={self.payoff}, "
            f"pool={self.pool_size}, field={self.field_net})"
        )


# --------------------------------------------------------------------------
# Driver: paired-halves lockstep with per-game ctx + heartbeats
# --------------------------------------------------------------------------
class _Heartbeat:
    """events.jsonl + tail.log, a line at least every `every_s` seconds."""

    def __init__(self, out_dir: Path, every_s: float = 30.0):
        self.events = open(out_dir / "events.jsonl", "a", buffering=1)
        self.tail = open(out_dir / "tail.log", "a", buffering=1)
        self.every_s = every_s
        self.t0 = time.time()
        self._last = 0.0

    def maybe(self, **fields) -> None:
        now = time.time()
        if now - self._last < self.every_s:
            return
        self._last = now
        self.log(**fields)

    def log(self, **fields) -> None:
        rec = {"t": round(time.time() - self.t0, 1), **fields}
        line = json.dumps(rec)
        self.events.write(line + "\n")
        self.tail.write(line + "\n")

    def close(self) -> None:
        self.events.close()
        self.tail.close()


def _run_paired_walt(
    *, n_games: int, cfg, bid, walt_play: WaltPlay, jud_play, heartbeat: _Heartbeat,
    hand_stream_path=None,
):
    """Both halves in one lockstep pool over identical seeds. Team A = walt,
    team B = jud; the same margin bidder fills every seat.

    Mirrors arena.engine._run_lockstep but tags each game so WaltPlay can join
    decisions and emits heartbeats. Policies are deterministic, so pooling the
    halves is byte-identical to the sequential halves.
    """
    from arena.engine import _LiveGame, _seat_policies
    from forge.zeb.game import current_player

    half = n_games // 2
    if half == 0:
        raise ValueError("n_games must be at least 2")

    games = [
        _LiveGame(i, a_team, cfg, _seat_policies(a_team, bid, bid))
        for a_team in (0, 1)
        for i in range(half)
    ]

    # logging-only: stream each completed hand as a JSONL row (prelim signal)
    stream_fh = open(hand_stream_path, "a") if hand_stream_path else None
    streamed = [0] * len(games)

    while True:
        live = [g for g in games if not g.done]
        if not live:
            break
        hands_done = sum(len(g.hands) for g in games)
        heartbeat.maybe(live=len(live), total=len(games), hands=hands_done)

        # ctx for decision joins this tick.
        walt_play.ctx = {
            id(g.state): (g.game_idx, g.hand_idx, g.a_team) for g in live
        }

        a_games, b_games = [], []
        for g in live:
            side = a_games if current_player(g.state) % 2 == g.a_team else b_games
            side.append(g)
        for side_games, policy in ((a_games, walt_play), (b_games, jud_play)):
            if not side_games:
                continue
            actions = policy.choose(
                [g.state for g in side_games],
                [g.bid_value for g in side_games],
                [(g.marks[0], g.marks[1]) for g in side_games],
                cfg.marks_to_win,
            )
            for g, action in zip(side_games, actions):
                g.apply(action)

        if stream_fh is not None:
            wrote = False
            for gi, g in enumerate(games):
                while streamed[gi] < len(g.hands):
                    h = g.hands[streamed[gi]]
                    streamed[gi] += 1
                    a_pts = h.team_points[h.a_team]
                    b_pts = h.team_points[1 - h.a_team]
                    stream_fh.write(json.dumps({
                        "game": h.game_idx, "hand": h.hand_idx, "seed": h.seed,
                        "a_team": h.a_team, "bidder": h.bidder,
                        "bid_value": h.bid_value, "decl_id": h.decl_id,
                        "made": h.made, "a_pts": a_pts, "b_pts": b_pts,
                        "marks_after": list(h.marks_after),
                    }) + "\n")
                    wrote = True
            if wrote:
                stream_fh.flush()

    if stream_fh is not None:
        stream_fh.close()
    return [g.record() for g in games]


def main() -> int:
    parser = argparse.ArgumentParser(description="walt paired grading driver")
    parser.add_argument("--n-games", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=HORIZON,
                        help="informational; solver consults at <= this many tiles")
    parser.add_argument("--world-cap", type=int, default=0,
                        help="0 = exact; K>0 uniformly subsamples worlds to K "
                             "after the belief filter (measured: K=512 -> p90 "
                             "|dV| 0.45 pts, zero material argmax flips)")
    parser.add_argument("--beliefs", choices=("u", "sigma"), default="sigma")
    parser.add_argument("--payoff", choices=("points", "make"), default="points")
    parser.add_argument("--pool-size", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--marks-to-win", type=int, default=7)
    parser.add_argument("--max-redeals", type=int, default=3)
    parser.add_argument("--field-net", type=str, default=DEFAULT_FIELD_NET)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    from arena.cli import parse_bidder
    from arena.engine import ArenaConfig
    from arena.jud_play import JudPlay
    from arena.match import MatchResult, hand_rows, summarize
    from champion.jud_net import load_jud_net

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = _Heartbeat(out_dir, every_s=30.0)

    bid = parse_bidder(INCUMBENT_BIDDER, device="cpu", gus_adapter=None, model=None)
    jud_play = JudPlay(load_jud_net(args.field_net, device="cpu"))
    walt_play = WaltPlay(
        field_net=args.field_net,
        beliefs=args.beliefs,
        payoff=args.payoff,
        horizon=args.horizon,
        world_cap=args.world_cap,
        pool_size=args.pool_size,
        decisions_path=str(out_dir / "decisions.jsonl"),
    )

    cfg = ArenaConfig(
        marks_to_win=args.marks_to_win,
        max_redeals=args.max_redeals,
        base_seed=args.base_seed,
    )
    cap_tag = f",cap{args.world_cap}" if args.world_cap else ""
    label_a = (f"{INCUMBENT_BIDDER}+walt:{args.beliefs},{args.payoff},"
               f"h{args.horizon}{cap_tag}")
    label_b = f"{INCUMBENT_BIDDER}+judplay"

    print(f"walt grading: {label_a} vs {label_b}  n={args.n_games}", flush=True)
    heartbeat.log(event="start", label_a=label_a, label_b=label_b,
                  n_games=args.n_games)

    t0 = time.time()
    try:
        games = _run_paired_walt(
            n_games=args.n_games, cfg=cfg, bid=bid,
            walt_play=walt_play, jud_play=jud_play, heartbeat=heartbeat,
            hand_stream_path=out_dir / "hand_stream.jsonl",
        )
    finally:
        walt_play.close()
    elapsed = time.time() - t0

    result = MatchResult(
        label_a=label_a, label_b=label_b, cfg=cfg, elapsed_s=elapsed, games=games,
    )
    s = summarize(result)
    (out_dir / "summary.json").write_text(json.dumps({
        **s,
        "cfg": {
            "marks_to_win": cfg.marks_to_win,
            "max_redeals": cfg.max_redeals,
            "base_seed": cfg.base_seed,
            "beliefs": args.beliefs,
            "payoff": args.payoff,
            "horizon": args.horizon,
            "world_cap": args.world_cap,
            "field_net": args.field_net,
        },
    }, indent=2) + "\n")

    # per_hand.csv (reuse arena.match.hand_rows).
    import csv

    rows = hand_rows(result)
    if rows:
        with (out_dir / "per_hand.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    heartbeat.log(event="done", elapsed_s=round(elapsed, 1),
                  a_wins=s["a_wins"], n_games=s["n_games"],
                  mark_margin=round(s["mean_mark_margin"], 3))
    heartbeat.close()

    print(
        f"A wins {s['a_wins']}/{s['n_games']} "
        f"({s['a_game_win_rate']:.1%})  mark margin {s['mean_mark_margin']:+.2f}/game "
        f"[{s['mark_margin_ci_lo_95']:+.2f}, {s['mark_margin_ci_hi_95']:+.2f}]  "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    print(f"Wrote {out_dir}/summary.json, per_hand.csv, decisions.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
