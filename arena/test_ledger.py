# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""The arena's ledger, written down for Silas (silas/tests/data/).

Two ledgers of a Silas-only field, written by the arena's own auction and
play engine (`arena.ledger.run_ledger`): one in marks from 0-0 with every
chair rotated and the straight auction (hands pass out), one in the book's
points from 200-180 with one chair and the forced shaker. Silas's
`tests/ledger.rs` replays every a and b row by its own `play_out` and must get
the same hand tile for tile (the differential check across the boundary), and
reads each ledger to the arena's own paired mean and flips, which this file
writes beside them (`ledger-arena-means.txt`). The chance arm is the arena's
random bidder, which Silas cannot replay, so it is read, not replayed.

To rewrite the fixtures after a deliberate change to the arena:

    python -u -c "import arena.test_ledger as t; t.write()"
"""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from arena.ledger import FORMAT, overbidding_field, paired_mean, parse_seat, play_name, run_ledger

DATA = Path(__file__).resolve().parent.parent / "silas" / "tests" / "data"
MEANS = DATA / "ledger-arena-means.txt"
# The field's four seats and the two players, as the Rust test builds them.
FIELD = ("silas:bid-only-enough", "silas:book/pounce", "silas:risk-budget", "silas:book/protect-the-count")
A, B = "silas:risk-budget", "silas:never-32-or-33"
LEDGERS = {
    "ledger-marks.json": dict(n=4, seed=20261015, seats="rotate", scoring="marks", target=7, score=(0, 0)),
    "ledger-points.json": dict(n=8, seed=20261016, seats=1, scoring="points", target=250, score=(200, 180),
                               forced=True),
}


def ledger(name: str) -> dict:
    return run_ledger(field=tuple(parse_seat(s) for s in FIELD), a=parse_seat(A), b=parse_seat(B), **LEDGERS[name])


def means(ledgers: dict[str, dict]) -> str:
    lines = [
        "# The arena's paired mean of each ledger beside this file, in its own unit,",
        "# settled by the arena's scoring (arena.ledger.paired_mean), exactly; then",
        "# the flips: deals a did better, b did, tied. Written and checked by",
        "# arena/test_ledger.py; read by silas/tests/ledger.rs.",
    ]
    for name, led in ledgers.items():
        mean, (w, l, t) = paired_mean(led)
        lines.append(f"{name} {led['scoring']['kind']} {mean} {w} {l} {t}")
    return "\n".join(lines) + "\n"


def write() -> None:
    ledgers = {name: ledger(name) for name in LEDGERS}
    for name, led in ledgers.items():
        (DATA / name).write_text(json.dumps(led) + "\n")
    MEANS.write_text(means(ledgers))


def test_the_ledger_fixtures_are_the_arenas():
    ledgers = {name: ledger(name) for name in LEDGERS}
    for name, led in ledgers.items():
        assert json.loads((DATA / name).read_text()) == led, name
    assert MEANS.read_text() == means(ledgers)


def test_a_ledger_carries_outcomes_and_every_arm_of_every_seating():
    led = json.loads((DATA / "ledger-marks.json").read_text())
    assert led["ledger"] == FORMAT
    assert set(led) == {"ledger", "field", "players", "sample", "seats", "auction", "offered", "scoring", "score", "rows"}
    keys = {"deal", "hands", "shaker", "chair", "arm", "calls", "contract", "outcome", "plays"}
    assert all(set(r) == keys for r in led["rows"])
    assert len(led["rows"]) == 4 * 16 * 3
    played = [r for r in led["rows"] if r["contract"] is not None]
    assert played and len(played) < len(led["rows"]), "the straight auction passes some hands out"
    assert all(len(r["plays"]) == 28 and sum(r["outcome"].values()) == 42 for r in played)
    forced = json.loads((DATA / "ledger-points.json").read_text())
    assert all(r["contract"] is not None for r in forced["rows"]), "a forced shaker never passes out"


def test_the_overbidding_field_bids_past_the_heuristic_ceiling():
    plain = run_ledger(field=overbidding_field(0), a=parse_seat("heuristic+lowest"), b=parse_seat("heuristic+lowest"),
                       n=6, seed=3)
    over = run_ledger(field=overbidding_field(10), a=parse_seat("heuristic+lowest"), b=parse_seat("heuristic+lowest"),
                      n=6, seed=3)
    contracts = lambda led: sum(r["contract"] is not None for r in led["rows"])  # noqa: E731
    assert contracts(over) > contracts(plain)
    assert over["field"].startswith("S0 heuristic:3,-10+lowest")
    # The same player in both arms reads zero on every deal.
    assert paired_mean(over) == (Fraction(0), (0, 0, 6))


def test_a_net_seat_is_named_by_its_net_and_seed(tmp_path):
    # Two ledgers name one field alike only when they seated the same nets at
    # the same seed (#183): the file name alone is not the net.
    net = tmp_path / "f.fnet"
    net.write_bytes(b"one net")
    name = play_name(f"walt:k0,n16,F={net},d=posterior", 3)
    assert name == f"walt:k0,n16,F={net}#{hashlib.sha256(b'one net').hexdigest()[:16]},d=posterior seed=3"
    net.write_bytes(b"another net")
    assert play_name(f"walt:k0,n16,F={net},d=posterior", 3) != name
    assert play_name("random", 3) == "random seed=3"


def test_zeb_sits_in_the_chair_and_its_ledger_reads(tmp_path):
    # The runner seats a planner as readily as a Silas policy: deployed Zeb's
    # net and plan (at 16 worlds, to keep the test quick) against the lowest
    # tile, in one chair of the heuristic field.
    from scenario.field import FieldNet
    shipped = Path(__file__).resolve().parent.parent / "zeb" / "net" / "margin_d250k_h128x64.fnet"
    net = tmp_path / "zeb.npz"
    FieldNet.from_bytes(shipped.read_bytes()).save(net)
    zeb = parse_seat(f"heuristic+walt:k0,n16,F={net},f=16,d=posterior,t=margin", seed=0)
    assert f"#{hashlib.sha256(net.read_bytes()).hexdigest()[:16]}" in zeb.name
    led = run_ledger(field=overbidding_field(0), a=zeb, b=parse_seat("heuristic+lowest"), n=2, seed=5, seats=0)
    assert led["players"]["a"] == zeb.name
    played = [r for r in led["rows"] if r["contract"] is not None]
    assert played and all(len(r["plays"]) == 28 and sum(r["outcome"].values()) == 42 for r in played)
    mean, flips = paired_mean(led)
    assert sum(flips) == 2
