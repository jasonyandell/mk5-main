"""test_tables.py — parity of walt LUTs against the zeb engine & forge tables.

Plain executable script: prints one PASS/FAIL line per gate, exits nonzero on
any FAIL. Run:  /Users/jason/code/mk5-main/.venv/bin/python -u test_tables.py
"""
from __future__ import annotations

import random
import sys

import numpy as np


from forge.oracle.tables import resolve_trick, trick_rank  # noqa: E402
from forge.zeb import game as zeb  # noqa: E402
from forge.zeb.types import GamePhase  # noqa: E402
import dataclasses  # noqa: E402

from walt.tables import (  # noqa: E402
    N_DOMINOES,
    get_luts,
    hand_to_mask,
    legal_moves_mask,
    mask_to_tiles,
    resolve_tricks,
)

REAL_DECLS = (0, 1, 2, 3, 4, 5, 6, 7, 9)

_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  {detail}"
    print(line)
    if not ok:
        _failures.append(name)


def gen_states(n_states: int, seed0: int = 0):
    """Yield PLAYING zeb states across all real decls via random legal play."""
    rng = random.Random(seed0)
    produced = 0
    g = 0
    while produced < n_states:
        state = zeb.new_game(zeb.game_seed(seed0, g))
        g += 1
        decl = rng.choice(REAL_DECLS)
        state = dataclasses.replace(state, decl_id=decl)
        # random-length random-legal playout
        steps = rng.randint(0, 24)
        for _ in range(steps):
            if state.phase != GamePhase.PLAYING:
                break
            legal = zeb.legal_actions(state)
            if not legal:
                break
            # emit this state before advancing
            yield state
            produced += 1
            if produced >= n_states:
                return
            state = zeb.apply_action(state, rng.choice(legal))


def test_legal_parity() -> None:
    n = 0
    mismatches = 0
    for state in gen_states(2200, seed0=11):
        if state.phase != GamePhase.PLAYING:
            continue
        player = zeb.current_player(state)
        remaining = [d for d in state.hands[player] if d not in state.played]
        hand_mask = hand_to_mask(remaining)
        led_tile = state.current_trick[0] if state.current_trick else None
        walt_mask = int(legal_moves_mask(hand_mask, led_tile, state.decl_id))
        # engine legal slots -> domino ids -> mask
        legal_slots = zeb.legal_actions(state)
        engine_ids = [state.hands[player][s] for s in legal_slots]
        engine_mask = int(hand_to_mask(engine_ids))
        if walt_mask != engine_mask:
            mismatches += 1
        n += 1
    check(
        "legal_moves_mask == zeb legal_actions",
        mismatches == 0 and n >= 2000,
        f"n={n} mismatches={mismatches}",
    )


def test_resolve_parity() -> None:
    rng = random.Random(7)
    B = 2500
    leaders = []
    tiles4 = []
    decls = []
    for _ in range(B):
        decl = rng.choice(REAL_DECLS)
        four = rng.sample(range(N_DOMINOES), 4)
        leaders.append(rng.randint(0, 3))
        tiles4.append(four)
        decls.append(decl)
    mismatches = 0
    # group by decl for the vectorized path
    for decl in REAL_DECLS:
        idx = [i for i in range(B) if decls[i] == decl]
        if not idx:
            continue
        L = np.array([leaders[i] for i in idx])
        T = np.array([tiles4[i] for i in idx])
        w, p = resolve_tricks(L, T, decl)
        for j, i in enumerate(idx):
            out = resolve_trick(tiles4[i][0], tuple(tiles4[i]), decl)
            exp_winner = (leaders[i] + out.winner_offset) % 4
            if int(w[j]) != exp_winner or int(p[j]) != out.points:
                mismatches += 1
    check(
        "resolve_tricks == forge resolve_trick",
        mismatches == 0,
        f"B={B} mismatches={mismatches}",
    )


def test_beat_count() -> None:
    bad = 0
    boss_ok = True
    for decl in REAL_DECLS:
        luts = get_luts(decl)
        for t in range(N_DOMINOES):
            led = int(luts.led_suit[t])
            col = np.array([trick_rank(o, led, decl) for o in range(N_DOMINOES)])
            expect = int(np.sum(col <= col[t]) - 1)
            if int(luts.beat_count[t]) != expect:
                bad += 1
        # the trump/led boss beats all 27 others
        if int(luts.beat_count.max()) != 27:
            boss_ok = False
    check("beat_count == recomputed via trick_rank", bad == 0, f"mismatches={bad}")
    check("beat_count boss beats 27 in every decl", boss_ok)


def test_masks_roundtrip() -> None:
    rng = random.Random(3)
    ok = True
    for _ in range(500):
        k = rng.randint(0, 7)
        tiles = sorted(rng.sample(range(N_DOMINOES), k))
        if mask_to_tiles(hand_to_mask(tiles)) != tiles:
            ok = False
            break
    check("hand_to_mask / mask_to_tiles round-trip", ok)


if __name__ == "__main__":
    test_legal_parity()
    test_resolve_parity()
    test_beat_count()
    test_masks_roundtrip()
    print(f"\n{len(_failures)} failure(s)" if _failures else "\nall gates green")
    sys.exit(1 if _failures else 0)
