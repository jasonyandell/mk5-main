#!/usr/bin/env python
"""walt/tests/test_field.py — parity + sanity gates for the jud field oracle (B2).

Plain executable script (no pytest). Prints one PASS/FAIL line per gate; exits
nonzero on any FAIL.

Gates:
  (1)  featurizer bit-equality (torch.equal) vs champion.jud_net.featurize_state
       on >= 2000 random (state, seat) pairs, incl. mid-trick child states.
  (1b) incremental child block bit-equality vs the full featurizer (the path
       FieldOracle.decisions actually uses).
  (2)  decision equality vs a real arena.jud_play.JudPlay on >= 500 random
       mid-hand states — require 100%.
  (3)  sigma_consistent sanity: true world survives; all survivors reproduce
       the observed moves.
  (4)  timing: featurization + forward wall for a 1000-world x 1-step batch.
"""
import sys
import time


import numpy as np
import torch

from champion.jud_net import featurize_state, load_jud_net
from arena.jud_play import JudPlay
from forge.zeb.game import (
    apply_action,
    current_player,
    is_terminal,
    legal_actions,
    new_game,
)

import walt.field as field
from walt.field import FieldOracle, PubState, featurize, sigma_consistent

from pathlib import Path as _P
NET = str(_P(__file__).resolve().parents[2] / "champion" / "jud_net.pt")

_fail = 0


def report(name, ok, extra=""):
    global _fail
    tag = "PASS" if ok else "FAIL"
    if not ok:
        _fail += 1
    print(f"{tag} {name}{(' — ' + extra) if extra else ''}", flush=True)


# --------------------------------------------------------------------- #
#  Playout helpers                                                       #
# --------------------------------------------------------------------- #

def random_states(seed, rng):
    """Yield every intermediate PLAYING state of one random playout."""
    s = new_game(seed)
    states = []
    while not is_terminal(s):
        states.append(s)
        legal = legal_actions(s)
        s = apply_action(s, int(rng.choice(legal)))
    return states


def feat_from_state(state, seat):
    """Full walt featurizer applied to a live state (for a given POV seat)."""
    return featurize(
        seat,
        state.hands[seat],
        state.bid_state.bids,
        state.bidder,
        state.dealer,
        state.decl_id,
        state.play_history,
    )


# --------------------------------------------------------------------- #
#  Gate 1 — featurizer bit-equality                                     #
# --------------------------------------------------------------------- #

def gate1(rng):
    n = 0
    mism = 0
    first_bad = None
    seed = 0
    while n < 2200:
        states = random_states(seed, rng)
        seed += 1
        for st in states:
            for seat in range(4):
                ref = featurize_state(st, seat=seat)          # torch [350]
                mine = torch.from_numpy(feat_from_state(st, seat))
                if not torch.equal(ref, mine):
                    mism += 1
                    if first_bad is None:
                        diff = (ref - mine).abs()
                        idx = int(torch.argmax(diff))
                        first_bad = (seat, idx, float(ref[idx]), float(mine[idx]))
                n += 1
    report("gate1_featurizer_bit_equality", mism == 0,
           f"{n} (state,seat) pairs, {mism} mismatch"
           + (f", first={first_bad}" if first_bad else ""))


# --------------------------------------------------------------------- #
#  Gate 1b — incremental child block == full featurizer                 #
# --------------------------------------------------------------------- #

def gate1b(rng):
    checked = 0
    mism = 0
    seed = 5000
    while checked < 1500:
        s = new_game(seed)
        seed += 1
        while not is_terminal(s):
            mover = current_player(s)
            legal = legal_actions(s)              # slot indices
            pub = PubState.from_state(s)
            hand_mask = 0
            for d in s.hands[mover]:
                if d not in s.played:
                    hand_mask |= 1 << d
            orig_mask = hand_mask | field._played_by(pub.play_history, mover)
            orig_hand = [d for d in range(28) if (orig_mask >> d) & 1]
            h91 = field._hand_auction_91(
                mover, orig_hand, pub.bids, pub.bidder, pub.dealer, pub.decl_id
            )
            pblock, ppts = field._featurize_play_np(
                pub.play_history, mover, pub.bidder, pub.decl_id
            )
            for slot in legal:
                dom = s.hands[mover][slot]
                child = apply_action(s, slot)
                cblock = field._child_play_block(
                    pblock, ppts, pub.play_history, mover, dom, pub.bidder, pub.decl_id
                )
                incr = np.concatenate([h91, cblock])
                ref = featurize_state(child, seat=mover).numpy()
                if not np.array_equal(incr, ref):
                    mism += 1
                checked += 1
            s = apply_action(s, int(rng.choice(legal)))
    report("gate1b_incremental_child_block", mism == 0,
           f"{checked} children, {mism} mismatch")


# --------------------------------------------------------------------- #
#  Gate 2 — decision equality vs JudPlay                                 #
# --------------------------------------------------------------------- #

def gate2(rng, oracle):
    model = load_jud_net(NET, device="cpu")
    judplay = JudPlay(model, device="cpu")

    # Collect >=500 mid-hand states (at least one tile already played so worlds
    # / follow-suit legality actually exercise; mix of leading and following).
    states = []
    seed = 20000
    while len(states) < 600:
        s = new_game(seed)
        seed += 1
        depth = 0
        while not is_terminal(s):
            if depth >= 1:
                states.append(s)
            legal = legal_actions(s)
            s = apply_action(s, int(rng.choice(legal)))
            depth += 1

    # JudPlay returns a SLOT index; convert to a domino id to compare.
    jp_slots = judplay.choose(states, [s.bid_state.high_bid for s in states])
    jp_moves = [st.hands[current_player(st)][slot] for st, slot in zip(states, jp_slots)]

    queries = []
    for st in states:
        mover = current_player(st)
        hand_mask = 0
        for d in st.hands[mover]:
            if d not in st.played:
                hand_mask |= 1 << d
        queries.append((mover, hand_mask, PubState.from_state(st)))
    my_moves = oracle.decisions(queries)

    mism = [(i, jp_moves[i], my_moves[i]) for i in range(len(states))
            if jp_moves[i] != my_moves[i]]
    ok = len(mism) == 0
    extra = f"{len(states)} states, {len(mism)} mismatch"
    if not ok:
        # Diagnose: float-order vs tie-break. Re-evaluate the first miss.
        i, jm, mm = mism[0]
        st = states[i]
        mover = current_player(st)
        legal_slots = legal_actions(st)
        feats = torch.stack([
            featurize_state(apply_action(st, a), seat=mover) for a in legal_slots
        ])
        from champion.jud_net import mean_points
        with torch.no_grad():
            ev = mean_points(model(feats))
        sign = 1.0 if mover % 2 == st.bidder % 2 else -1.0
        extra += f"; first miss slot-evs={[round(float(x),6) for x in sign*ev]}"
        extra += f" jp_dom={jm} my_dom={mm}"
    report("gate2_decision_equality_vs_judplay", ok, extra)


# --------------------------------------------------------------------- #
#  Gate 3 — sigma_consistent sanity                                     #
# --------------------------------------------------------------------- #

class _Root:
    def __init__(self, state, me):
        self.me = me
        self.decl_id = state.decl_id
        self.bidder = state.bidder
        self.bid_value = state.bid_state.high_bid
        self.bids = tuple(state.bid_state.bids)
        self.dealer = state.dealer
        self.play_history = tuple((int(p), int(d)) for p, d in state.play_history)


def gate3(oracle):
    # Full jud-vs-jud playout, then take an endgame root for some seat.
    model = load_jud_net(NET, device="cpu")
    jp = JudPlay(model, device="cpu")
    seed = 90001
    s = new_game(seed)
    while not is_terminal(s):
        slot = jp.choose([s], [s.bid_state.high_bid])[0]
        s = apply_action(s, slot)
    final = s
    true_hands = final.hands  # original 7-tuples per seat

    # Pick an endgame root: replay to a point where `me` has HORIZON tiles left.
    me = 0
    root_state = new_game(seed)
    hist = final.play_history
    # rebuild states step by step to find where `me`'s remaining count hits 3.
    st = new_game(seed)
    root_state = None
    for (player, dom) in hist:
        mover = current_player(st)
        remaining_me = sum(1 for d in st.hands[me] if d not in st.played)
        if remaining_me == 3 and root_state is None:
            root_state = st
        # advance by the recorded move (find its slot)
        slot = st.hands[mover].index(dom)
        st = apply_action(st, slot)
    if root_state is None:
        root_state = st
    root = _Root(root_state, me)

    # Hidden seats ascending; true world = their CURRENT remaining hands.
    hidden = [x for x in range(4) if x != me]
    def remaining_mask(seat):
        return sum(
            1 << d for d in root_state.hands[seat]
            if d not in root_state.played
        )
    true_world = np.array([[remaining_mask(s) for s in hidden]], dtype=np.uint32)

    # Build a small world sample INCLUDING the true world: permute the union of
    # the three hidden seats' current tiles into the same per-seat counts.
    counts = [bin(int(true_world[0, i])).count("1") for i in range(3)]
    pool = []
    for i in range(3):
        for d in range(28):
            if (int(true_world[0, i]) >> d) & 1:
                pool.append(d)
    rng = np.random.default_rng(7)
    sampled = [true_world[0].copy()]
    for _ in range(60):
        perm = list(pool)
        rng.shuffle(perm)
        row = []
        off = 0
        for i in range(3):
            row.append(sum(1 << perm[off + j] for j in range(counts[i])))
            off += counts[i]
        sampled.append(np.array(row, dtype=np.uint32))
    worlds = np.array(sampled, dtype=np.uint32)

    alive = sigma_consistent(root, worlds, oracle, lambda seat, k: True)

    true_survives = bool(alive[0])

    # Every survivor must reproduce every filtered observed move. Re-verify by
    # replaying each survivor world through the oracle independently.
    all_reproduce = True
    hist_root = root.play_history
    for w in np.nonzero(alive)[0]:
        ok = _world_reproduces(root, worlds[w], hidden, oracle)
        all_reproduce = all_reproduce and ok
    # And a world that does NOT reproduce must be filtered out (soundness):
    dead_ok = True
    for w in np.nonzero(~alive)[0]:
        if _world_reproduces(root, worlds[w], hidden, oracle):
            dead_ok = False

    report("gate3_sigma_true_world_survives", true_survives,
           f"{worlds.shape[0]} worlds, {int(alive.sum())} survive")
    report("gate3_sigma_survivors_reproduce", all_reproduce and dead_ok,
           f"survivors_ok={all_reproduce} dead_partition_ok={dead_ok}")


def _world_reproduces(root, world_row, hidden, oracle):
    """Independent check: does σ under this world's hands reproduce every
    observed non-me move?"""
    col = {s: i for i, s in enumerate(hidden)}
    hist = root.play_history
    all_plays = {s: field._played_by(hist, s) for s in hidden}
    orig = {s: int(world_row[col[s]]) | all_plays[s] for s in hidden}
    played_before = {s: 0 for s in range(4)}
    ok = True
    for k, (seat, tile) in enumerate(hist):
        if seat != root.me:
            hand_k = orig[seat] & ~played_before[seat]
            pub = PubState.from_history(
                root.decl_id, root.bidder, root.bids, root.dealer, hist[:k]
            )
            mv = oracle.decisions([(seat, hand_k, pub)])[0]
            if mv != int(tile):
                ok = False
                break
        played_before[seat] |= 1 << int(tile)
    return ok


# --------------------------------------------------------------------- #
#  Gate 4 — timing (1000 worlds x 1 step)                                #
# --------------------------------------------------------------------- #

def gate4(rng):
    fresh = FieldOracle(NET, device="cpu")  # cold memo for honest timing
    # One decision step, one seat, one public state, 1000 distinct worlds
    # (distinct hands) -> 1000 memo misses -> one forward.
    seed = 40000
    st = new_game(seed)
    # advance a couple plays so there is a play block + follow-suit structure
    for _ in range(2):
        st = apply_action(st, int(rng.choice(legal_actions(st))))
    mover = current_player(st)
    pub = PubState.from_state(st)
    # Fabricate 1000 plausible remaining-hand masks for `mover`: random subsets
    # of the unplayed tiles of the mover's own size (identity of the field is
    # not under test here — throughput is).
    base_hand = [d for d in st.hands[mover] if d not in st.played]
    k = len(base_hand)
    unplayed = [d for d in range(28) if d not in st.played]
    queries = []
    seen = set()
    while len(queries) < 1000:
        pick = tuple(sorted(rng.choice(unplayed, size=k, replace=False).tolist()))
        if pick in seen:
            continue
        seen.add(pick)
        hm = 0
        for d in pick:
            hm |= 1 << int(d)
        queries.append((mover, hm, pub))

    t0 = time.perf_counter()
    fresh.decisions(queries)
    dt = time.perf_counter() - t0
    report("gate4_timing_1000x1", True,
           f"{dt*1000:.1f} ms wall, {fresh.n_forward} forward(s), "
           f"{fresh.n_rows} rows featurized")


# --------------------------------------------------------------------- #
#  main                                                                  #
# --------------------------------------------------------------------- #

def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(12345)
    oracle = FieldOracle(NET, device="cpu")

    t = time.perf_counter()
    gate1(rng)
    print(f"  (gate1 {time.perf_counter()-t:.1f}s)", flush=True)

    t = time.perf_counter()
    gate1b(rng)
    print(f"  (gate1b {time.perf_counter()-t:.1f}s)", flush=True)

    t = time.perf_counter()
    gate2(rng, oracle)
    print(f"  (gate2 {time.perf_counter()-t:.1f}s)", flush=True)

    gate3(oracle)
    gate4(rng)

    print(("ALL PASS" if _fail == 0 else f"{_fail} GATE(S) FAILED"), flush=True)
    sys.exit(1 if _fail else 0)


if __name__ == "__main__":
    main()
