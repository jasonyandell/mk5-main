#!/usr/bin/env python3
"""
host — the local game authority for table42.

Owns the whole game: deals, auction, play (via forge's zeb engine — the
project's real Python rules), scoring, marks race, set-detection
fast-forward, and a complete local log (seed, hands, every move with the
mover's reasoning, chat). Nothing hidden ever leaves this process except
per-seat filtered view files.

Seats are driven by pluggable movers:
  file:<Name>    local inbox — a session or persistent agent teammate reads
                 its view via wait.py and writes moves via act.py
  random:<Name>  instant uniform-legal baseline (smoke)
  jud:<Name>     the graded champion in the seat (serving rule, #69):
                 margin:wp(r8) ValueBidder at auction, lens:ev (forge E[Q],
                 10 worlds) at play; its numbers — P(make)+utility per
                 candidate bid, E[pts] per candidate play — go in reasoning

Usage:
  python -u host.py new --seats "file:Jason,file:Claude,jud:Jud,jud:Jed" \
      [--seed N] [--marks 7] [--tick 1.0]

Runs are written under scratch/table42/run/<gid>/ (gitignored; durable
copies go to the HF evidence dataset per the run-artifacts policy).

Honor system: everyone playing agrees not to read run/<gid>/log.jsonl or
other seats' view files until the review.
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from arena.auction import (  # noqa: E402
    ONE_MARK,
    bidding_order,
    contract_points,
    legal_bids,
    marks_for_bid,
    score_hand,
)
from arena.engine import hand_seed  # noqa: E402
from forge.oracle.declarations import DECL_ID_TO_NAME, GAME_DECL_IDS  # noqa: E402
from forge.oracle.rng import deal_from_seed  # noqa: E402
from forge.oracle.tables import DOMINOES  # noqa: E402
from forge.zeb.game import apply_action, current_player, is_terminal, legal_actions  # noqa: E402
from forge.zeb.types import BidState, GamePhase, ZebGameState  # noqa: E402

TICK_S = 1.0  # overridden by --tick
IDLE_LIMIT_S = 30 * 60
MAX_REDEALS = 2

DOM_NAME = [f"{h}-{l}" for h, l in DOMINOES]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"


def bid_label(b: int) -> str:
    if b == 0:
        return "pass"
    return f"bid {b}" if b <= ONE_MARK else f"bid {b} ({marks_for_bid(b)} marks)"


@dataclass
class Seat:
    idx: int
    driver: str  # file | random | jud
    name: str


class Host:
    def __init__(self, seats: list[Seat], seed: int, marks_to_win: int, rundir: Path):
        self.seats = seats
        self.base_seed = seed
        self.marks_to_win = marks_to_win
        self.rundir = rundir
        self.gid = rundir.name
        self.rng = __import__("random").Random(seed ^ 0x42)

        self.marks = [0, 0]
        self.hand_idx = 0
        self.dealer = 0
        self.turn_id = 0
        self.chat: list[dict] = []
        self.last_activity = time.time()
        self._view_versions: dict[int | str, str] = {}

        (rundir / "moves").mkdir(parents=True, exist_ok=True)
        (rundir / "chat_in").mkdir(exist_ok=True)
        self.logf = open(rundir / "log.jsonl", "a")
        self._deal()

    # ---------- logging ----------

    def log(self, rec: dict) -> None:
        rec = {"ts": now(), **rec}
        self.logf.write(json.dumps(rec) + "\n")
        self.logf.flush()

    def say(self, seat: int | None, name: str, text: str) -> None:
        msg = {"id": len(self.chat) + 1, "seat": seat, "name": name,
               "text": text, "ts": now()}
        self.chat.append(msg)
        self.log({"type": "chat", **msg})

    # ---------- hand lifecycle ----------

    def _deal(self) -> None:
        seed = hand_seed(self.base_seed, 0, self.hand_idx, 0)
        hands = tuple(tuple(h) for h in deal_from_seed(seed))
        self.hands = hands
        self.hand_seed_val = seed
        self.redeals = 0
        self.phase = "bidding"
        self.bids: dict[int, int] = {}
        self.bid_order = list(bidding_order(self.dealer))
        self.bid_ptr = 0
        self.high_bid = 0
        self.high_bidder = -1
        self.state: ZebGameState | None = None
        self.tricks: list[dict] = []
        self.log({"type": "deal", "hand_idx": self.hand_idx, "seed": seed,
                  "dealer": self.dealer, "redeals": 0,
                  "hands": {self.seats[i].name: [DOM_NAME[d] for d in hands[i]]
                            for i in range(4)}})

    def _redeal(self) -> None:
        self.say(None, "table", "all passed — reshake")
        self.log({"type": "redeal", "hand_idx": self.hand_idx})
        self.dealer = (self.dealer + 1) % 4
        redeals = self.redeals + 1
        seed = hand_seed(self.base_seed, 0, self.hand_idx, redeals)
        hands = tuple(tuple(h) for h in deal_from_seed(seed))
        self.hands = hands
        self.hand_seed_val = seed
        self.redeals = redeals
        self.bids = {}
        self.bid_order = list(bidding_order(self.dealer))
        self.bid_ptr = 0
        self.high_bid = 0
        self.high_bidder = -1
        self.log({"type": "deal", "hand_idx": self.hand_idx, "seed": seed,
                  "dealer": self.dealer, "redeals": redeals,
                  "hands": {self.seats[i].name: [DOM_NAME[d] for d in hands[i]]
                            for i in range(4)}})

    # ---------- turn plumbing ----------

    def actor(self) -> int:
        if self.phase == "bidding":
            return self.bid_order[self.bid_ptr]
        if self.phase == "declare":
            return self.high_bidder
        if self.phase == "playing":
            return current_player(self.state)
        return -1

    def menu(self) -> list[str]:
        if self.phase == "bidding":
            opts = [bid_label(b) for b in legal_bids(self.high_bid)]
            forced = (self.redeals >= MAX_REDEALS and self.high_bid == 0
                      and self.bid_ptr == 3)
            return opts if forced else (["pass"] + opts)
        if self.phase == "declare":
            return [f"trump {DECL_ID_TO_NAME[d]}" for d in GAME_DECL_IDS]
        if self.phase == "playing":
            cur = current_player(self.state)
            return [f"play {DOM_NAME[self.state.hands[cur][s]]}"
                    for s in legal_actions(self.state)]
        return []

    def apply_move(self, seat: int, move: str, reasoning: str, source: str) -> bool:
        move = move.strip().lower()
        menu = [m.lower() for m in self.menu()]
        # allow bare shorthand: "30" for "bid 30", "6-4" for "play 6-4", suit for trump
        matches = [i for i, m in enumerate(menu)
                   if m == move or m.split(" ", 1)[-1].split(" (")[0] == move]
        if seat != self.actor() or not matches:
            return False
        chosen = self.menu()[matches[0]]
        self.log({"type": "action", "hand_idx": self.hand_idx, "turn_id": self.turn_id,
                  "seat": seat, "name": self.seats[seat].name, "phase": self.phase,
                  "move": chosen, "reasoning": reasoning, "source": source})
        self._execute(seat, chosen)
        self.turn_id += 1
        self.last_activity = time.time()
        return True

    def _execute(self, seat: int, move: str) -> None:
        if self.phase == "bidding":
            val = 0 if move == "pass" else int(move.split()[1])
            self.bids[seat] = val
            if val > self.high_bid:
                self.high_bid, self.high_bidder = val, seat
            self.bid_ptr += 1
            if self.bid_ptr == 4:
                if self.high_bid == 0:
                    self._redeal()
                else:
                    self.phase = "declare"
                    self.say(None, "table",
                             f"{self.seats[self.high_bidder].name} takes it at "
                             f"{bid_label(self.high_bid)[4:] or self.high_bid}")
            return
        if self.phase == "declare":
            name = move.split(" ", 1)[1]
            decl = next(d for d in GAME_DECL_IDS if DECL_ID_TO_NAME[d] == name)
            self.state = ZebGameState(
                hands=self.hands, dealer=self.dealer, phase=GamePhase.PLAYING,
                bid_state=BidState(
                    bids=tuple(self.bids.get(i, 0) for i in range(4)),
                    high_bidder=self.high_bidder, high_bid=self.high_bid),
                decl_id=decl, bidder=self.high_bidder, played=frozenset(),
                play_history=(), current_trick=(), trick_leader=self.high_bidder,
                team_points=(0, 0))
            self.phase = "playing"
            # Announce the declaration on the table channel. Game night 1's
            # worst bug was a trump misread propagating through chat with no
            # authoritative statement anywhere a seat would definitely see.
            self.say(None, "table",
                     f"{self.seats[seat].name} declares trump {name}")
            return
        # playing: move is "play H-L"
        dom = move.split()[1]
        cur = current_player(self.state)
        slot = next(s for s in legal_actions(self.state)
                    if DOM_NAME[self.state.hands[cur][s]] == dom)
        before = self.state
        self.state = apply_action(self.state, slot)
        if len(before.current_trick) == 3:  # this play completed a trick
            plays = self.state.play_history[-4:]
            pts = (self.state.team_points[0] - before.team_points[0]) + \
                  (self.state.team_points[1] - before.team_points[1])
            # apply_action sets trick_leader to the trick's winner in both the
            # terminal and non-terminal branches of the zeb engine.
            winner = self.state.trick_leader
            self.tricks.append({
                "plays": [[self.seats[p].name, DOM_NAME[d]] for p, d in plays],
                "winner": self.seats[winner].name, "points": pts})
            self.log({"type": "trick", "hand_idx": self.hand_idx, **self.tricks[-1]})
        if is_terminal(self.state):
            self._score_hand()
        else:
            self._maybe_fast_forward()

    def _decided(self) -> str | None:
        target = contract_points(self.high_bid)
        bt = self.high_bidder % 2
        tp = self.state.team_points
        if tp[1 - bt] >= 43 - target:
            return "set"
        if tp[bt] >= target:
            return "made"
        return None

    def _maybe_fast_forward(self) -> None:
        verdict = self._decided()
        if not verdict:
            return
        n = 0
        while not is_terminal(self.state):
            cur = current_player(self.state)
            slot = legal_actions(self.state)[0]
            dom = DOM_NAME[self.state.hands[cur][slot]]
            self.log({"type": "action", "hand_idx": self.hand_idx,
                      "turn_id": self.turn_id, "seat": cur,
                      "name": self.seats[cur].name, "phase": "playing",
                      "move": f"play {dom}",
                      "reasoning": f"fast-forward (hand {verdict})",
                      "source": "host"})
            before = self.state
            self.state = apply_action(self.state, slot)
            self.turn_id += 1
            n += 1
            if len(before.current_trick) == 3:
                plays = self.state.play_history[-4:]
                pts = sum(self.state.team_points) - sum(before.team_points)
                self.tricks.append({
                    "plays": [[self.seats[p].name, DOM_NAME[d]] for p, d in plays],
                    "winner": "-", "points": pts})
        self.say(None, "table", f"hand {verdict} — fast-forwarded {n} dead plays")
        self._score_hand()

    def _score_hand(self) -> None:
        hs = score_hand(self.high_bid, self.high_bidder % 2, self.state.team_points)
        self.marks[0] += hs.marks[0]
        self.marks[1] += hs.marks[1]
        self.log({"type": "hand_score", "hand_idx": self.hand_idx,
                  "bid": self.high_bid, "bidder": self.seats[self.high_bidder].name,
                  "decl": DECL_ID_TO_NAME[self.state.decl_id],
                  "team_points": list(self.state.team_points),
                  "made": hs.made, "marks_awarded": list(hs.marks),
                  "marks": list(self.marks)})
        self.say(None, "table",
                 f"hand {self.hand_idx + 1}: {self.seats[self.high_bidder].name} "
                 f"{'made' if hs.made else 'SET'} at {self.high_bid} "
                 f"({self.state.team_points[0]}-{self.state.team_points[1]}) — "
                 f"marks {self.marks[0]}-{self.marks[1]}")
        if max(self.marks) >= self.marks_to_win:
            self.phase = "game_end"
            self.log({"type": "game_end", "marks": list(self.marks),
                      "winner_team": 0 if self.marks[0] > self.marks[1] else 1})
        else:
            self.hand_idx += 1
            self.dealer = (self.dealer + 1) % 4
            self._deal()

    # ---------- views ----------

    def view_for(self, seat: int | None) -> dict:
        actor = self.actor()
        v = {
            "gid": self.gid, "phase": self.phase, "turn_id": self.turn_id,
            "seat": seat, "name": self.seats[seat].name if seat is not None else "spectator",
            "seatNames": [s.name for s in self.seats],
            "dealer": self.dealer, "hand_idx": self.hand_idx,
            "marks": self.marks, "marks_to_win": self.marks_to_win,
            "bids": {self.seats[i].name: bid_label(b) for i, b in self.bids.items()},
            "high_bid": self.high_bid,
            "bidder": self.seats[self.high_bidder].name if self.high_bidder >= 0 else None,
            "decl": DECL_ID_TO_NAME.get(self.state.decl_id) if self.state else None,
            "team_points": list(self.state.team_points) if self.state else [0, 0],
            "current_trick": [[self.seats[p].name, DOM_NAME[d]]
                              for p, d in zip(self._trick_players(), self.state.current_trick)]
                             if self.state else [],
            "tricks": self.tricks,
            "actor": self.seats[actor].name if actor >= 0 else None,
            "turn": seat is not None and actor == seat and self.phase != "game_end",
            "menu": self.menu() if (seat is not None and actor == seat) else [],
            "hand": [DOM_NAME[d] for d in self._remaining(seat)] if seat is not None else [],
            "chat": self.chat[-20:], "ts": now(),
        }
        return v

    def _trick_players(self) -> list[int]:
        lead = self.state.trick_leader
        return [(lead + i) % 4 for i in range(len(self.state.current_trick))]

    def _remaining(self, seat: int) -> list[int]:
        if self.state is None:
            return list(self.hands[seat])
        return [d for d in self.state.hands[seat] if d not in self.state.played]

    def write_views(self) -> None:
        for s in self.seats:
            v = self.view_for(s.idx)
            key = json.dumps(v, sort_keys=True)
            if self._view_versions.get(s.idx) != key:
                self._view_versions[s.idx] = key
                (self.rundir / f"view-seat-{s.idx}.json").write_text(json.dumps(v, indent=1))
        sv = self.view_for(None)
        key = json.dumps(sv, sort_keys=True)
        if self._view_versions.get("spec") != key:
            self._view_versions["spec"] = key
            (self.rundir / "view-spectator.json").write_text(json.dumps(sv, indent=1))

    # ---------- inputs ----------

    def poll_local(self) -> None:
        for s in self.seats:
            if s.driver != "file":
                continue
            f = self.rundir / "moves" / f"seat-{s.idx}.json"
            if not f.exists():
                continue
            try:
                m = json.loads(f.read_text())
            except Exception:
                continue
            if m.get("turn_id") == self.turn_id:
                if self.apply_move(s.idx, str(m.get("move", "")),
                                   str(m.get("reasoning", "")), "file"):
                    f.unlink(missing_ok=True)
            elif isinstance(m.get("turn_id"), int) and m["turn_id"] < self.turn_id:
                f.unlink(missing_ok=True)  # stale
        for f in sorted((self.rundir / "chat_in").glob("*.json")):
            try:
                m = json.loads(f.read_text())
                self.say(m.get("seat"), m.get("name", "?"), str(m.get("text", ""))[:2000])
            except Exception:
                pass
            f.unlink(missing_ok=True)

    # ---------- jud seat (the graded champion: margin:wp r8 + lens:ev) ----------

    def _jud_brain(self):
        if not hasattr(self, "_jud"):
            import torch
            from arena.lens_play import LensPlay
            from champion.utility import MarksToSeven
            from champion.value_bidder import ValueBidder, load_margin_net
            from forge.zeb.eval.loading import DEFAULT_ORACLE, load_oracle

            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "jud seat serves lens:ev (forge E[Q]) — GPU required, "
                    "no CPU fallback")

            class TableLens(LensPlay):
                """LensPlay that keeps the per-slot E[Q] of its last call,
                so the host can log the numbers as the seat's reasoning."""

                def _select(self, e_q, e_q_pdf, gst, bid_values, marks,
                            marks_to_win):
                    self.last_e_q = e_q.detach().cpu()
                    return super()._select(e_q, e_q_pdf, gst, bid_values,
                                           marks, marks_to_win)

            net = load_margin_net(REPO / "champion" / "margin_net_r8.pt",
                                  device="cpu")
            oracle = load_oracle(str(REPO / DEFAULT_ORACLE), "mps")
            self._jud = {
                "bidder": ValueBidder(net, MarksToSeven()),
                "play": TableLens(oracle, utility="ev", n_samples=10,
                                  device="mps"),
            }
            print("jud brain loaded: margin:wp(r8) + lens:ev — the graded "
                  "champion (serving rule, #69)", flush=True)
        return self._jud

    def poll_jud(self) -> None:
        a = self.actor()
        if a < 0 or self.seats[a].driver != "jud":
            return
        brain = self._jud_brain()
        rng = self.rng

        if self.phase == "bidding":
            from arena.auction import PASS, BidContext, contract_points
            ctx = BidContext(
                hand=self.hands[a], seat=a, dealer=self.dealer,
                bids=tuple(self.bids.get(i, -1) for i in range(4)),
                high_bid=self.high_bid, high_seat=self.high_bidder,
                legal=tuple(legal_bids(self.high_bid)),
                marks=(self.marks[0], self.marks[1]),
                marks_to_win=self.marks_to_win,
            )
            bidder = brain["bidder"]
            table = bidder._table(ctx)
            lines = []
            for v in ctx.legal[:6]:
                thr = contract_points(v)
                p = max(row[thr] for row in table.values())
                u = bidder._utility_of(v, table, ctx)
                lines.append(f"{bid_label(v)}: P(make)={p:.2f} u={u:+.3f}")
            choice = bidder.bid(ctx, rng)
            menu = self.menu()
            if choice == PASS:
                move = "pass" if "pass" in [m.lower() for m in menu] else menu[0]
                why = "no bid clears the marks-utility margin"
                if move != "pass":
                    why = "would pass, but forced to open"
            else:
                move = bid_label(choice)
                why = "cheapest bid with positive marks utility"
            self.apply_move(a, move,
                            f"margin:wp(r8) prices [{'; '.join(lines)}] — {why}",
                            "jud")
            return

        if self.phase == "declare":
            decl = brain["bidder"].declare(self.hands[a], self.high_bid, rng)
            self.apply_move(
                a, f"trump {DECL_ID_TO_NAME[decl]}",
                "margin:wp(r8): the declaration the winning bid was priced "
                "under (argmax-exceedance at the contract threshold)", "jud")
            return

        if self.phase == "playing":
            st = self.state
            mover = current_player(st)
            play = brain["play"]
            (slot,) = play.choose([st], [self.high_bid],
                                  [(self.marks[0], self.marks[1])],
                                  self.marks_to_win)
            legal = legal_actions(st)
            eq = play.last_e_q[0]
            evs = "; ".join(
                f"{DOM_NAME[st.hands[mover][s]]}→{float(eq[s]):+.1f}"
                for s in legal)
            role = ("declaring side" if mover % 2 == st.bidder % 2
                    else "defending")
            self.apply_move(
                a, f"play {DOM_NAME[st.hands[mover][slot]]}",
                f"lens:ev E[Q] (own-team point margin) per option over "
                f"{play.n_samples} worlds [{evs}] — {role}, argmax", "jud")

    def poll_random(self) -> None:
        a = self.actor()
        if a >= 0 and self.seats[a].driver == "random":
            menu = self.menu()
            pick = "pass" if "pass" in menu and self.rng.random() < 0.7 else self.rng.choice(menu)
            self.apply_move(a, pick, "random baseline", "random")

    # ---------- main ----------

    def run(self) -> None:
        print(f"host up: game {self.gid}, seed {self.base_seed}, "
              f"seats {[f'{s.idx}:{s.driver}:{s.name}' for s in self.seats]}", flush=True)
        last_beat = 0.0
        while True:
            self.poll_local()
            self.poll_random()
            self.poll_jud()
            self.write_views()
            if self.phase == "game_end":
                print(f"GAME END marks {self.marks}", flush=True)
                return
            if time.time() - self.last_activity > IDLE_LIMIT_S:
                self.log({"type": "timeout"})
                print("30 min inactivity — host exiting; restart manually", flush=True)
                return
            if time.time() - last_beat > 60:
                last_beat = time.time()
                print(f"{now()} hand {self.hand_idx + 1} {self.phase} "
                      f"actor={self.actor()} marks={self.marks}", flush=True)
            time.sleep(getattr(self, 'tick_s', TICK_S))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    new = sub.add_parser("new")
    new.add_argument("--seats", required=True,
                     help="4 comma-separated driver:Name (driver: file|random|jud)")
    new.add_argument("--seed", type=int, default=secrets.randbelow(2**31))
    new.add_argument("--marks", type=int, default=7)
    new.add_argument("--tick", type=float, default=1.0)
    new.add_argument("--run-root", default=str(REPO / "scratch" / "table42" / "run"),
                     help="parent dir for run/<gid>/ (default: scratch/table42/run)")
    args = ap.parse_args()

    specs = args.seats.split(",")
    assert len(specs) == 4, "need 4 seats"
    seats = []
    for i, spec in enumerate(specs):
        driver, name = spec.split(":", 1)
        assert driver in ("file", "random", "jud"), driver
        seats.append(Seat(idx=i, driver=driver, name=name))

    gid = time.strftime("%m%d-%H%M%S")
    rundir = Path(args.run_root) / gid
    rundir.mkdir(parents=True)
    host = Host(seats, args.seed, args.marks, rundir)
    host.tick_s = args.tick
    (rundir / "config.json").write_text(json.dumps({
        "gid": gid, "seed": args.seed, "marks_to_win": args.marks,
        "seats": [{"idx": s.idx, "driver": s.driver, "name": s.name}
                  for s in seats]}, indent=2))
    host.log({"type": "start", "gid": gid, "seed": args.seed,
              "seats": [f"{s.driver}:{s.name}" for s in seats],
              "honor": "players agree not to read log.jsonl or other seats' views until review"})
    host.run()


if __name__ == "__main__":
    main()
