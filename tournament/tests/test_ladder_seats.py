# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Configured players are seats: a configured Zeb and the overcalling Demon are
participants the standing worker can advertise a capability for and schedule.

Offline and fast: the build and the games are faked; the records are the real ones.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import textwrap

import pytest

from tournament import artifacts, seats
from tournament import worker as W
from tournament import zeb_native_seat as Z
from tournament.participants import (DEMON_OVERCALL_N100, ROSTER, WALT, ZEB_040_MATCH,
                                     ZEB_041_CONTESTED, Participant, load_all,
                                     supports_straight_games)
from tournament.schedule import next_pair
from tournament.store import GitStore, dumps

# The rematch (wiki/experiments/rematch-zeb-041-vs-demon-2026-10-15.md) recorded games under
# these ids and these bytes. A ladder that redefined them would disagree with its faceoffs.
RECORD_SHA256 = {
    "zeb-0.4.0-match": "1a70abef930a3b76008fa29fb99ff91adea03facdaa3c6bad50c4a3622f98497",
    "zeb-0.4.1-contested": "87bba2ab803171d8efff429d1de4d8a87ff84a82521826753166f676a42aa53b",
    "demon-pimc-n100-bidc102a137-straight-2e233d29":
        "0c05f48ca2c73008f8faba83e5f69cf5d8490cc0a453acb19546c6393314fee2",
}
CONFIGURED = (ZEB_040_MATCH, ZEB_041_CONTESTED)


def test_the_rematchs_participants_are_unchanged():
    for p in (*CONFIGURED, DEMON_OVERCALL_N100):
        assert hashlib.sha256(dumps(p.to_record())).hexdigest() == RECORD_SHA256[p.id]
    assert [p.id for p in CONFIGURED] == ["zeb-0.4.0-match", "zeb-0.4.1-contested"]


def test_configured_zebs_pin_their_tag_and_config_and_round_trip(tmp_path):
    for p in CONFIGURED:
        assert Participant.from_record(json.loads(dumps(p.to_record()))) == p
        assert p.kind == "zeb-native" and p.pin["tag"] == f"zeb-v{p.id.split('-')[1]}"
        assert set(p.pin["files"]) == {"package/index.js", "package/match.js"}
        assert p.pin["commit"] in ("29f519953467f6aacd78bd17700f3c67fda6b4eb",
                                   "0b57a8f5dff9fc7b5249c01be88927b7c8c7ecd0")
        assert p.config["matchAware"] is True and p.config["legalDeclarations"] == [0, 1, 2, 3, 4, 5, 6]
    assert ZEB_041_CONTESTED.config["overcall"] is True and ZEB_040_MATCH.config["overcall"] is False
    # two configs, two seats: distinct ids, one native binary
    assert ZEB_040_MATCH.id != ZEB_041_CONTESTED.id
    assert ZEB_040_MATCH.pin["native_binary_sha256"] == ZEB_041_CONTESTED.pin["native_binary_sha256"]
    # immutable records in the store: seeded once, the same bytes again is no second add
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    store = GitStore(tmp_path)
    store.append(ZEB_041_CONTESTED.path(), dumps(ZEB_041_CONTESTED.to_record()), "add")
    assert not store.append(ZEB_041_CONTESTED.path(), dumps(ZEB_041_CONTESTED.to_record()), "again")
    assert load_all(store)[ZEB_041_CONTESTED.id] == ZEB_041_CONTESTED


def test_the_roster_seats_them_and_the_overcalling_demon_stays_a_fenced_seat():
    ids = [p.id for p in ROSTER]
    assert len(ids) == len(set(ids))
    assert {*(p.id for p in CONFIGURED), DEMON_OVERCALL_N100.id} <= set(ids)
    # STRATEGY-FUSION FENCE: the Demon is a seat (a participant of kind demon), never a component
    assert DEMON_OVERCALL_N100.kind == "demon" and DEMON_OVERCALL_N100.needs == ("cargo",)
    assert supports_straight_games(DEMON_OVERCALL_N100.to_record())


def fake_machine(monkeypatch, have: set[str]):
    monkeypatch.setattr(W.shutil, "which", lambda name: f"/bin/{name}" if name in have else None)


def test_the_worker_advertises_zeb_native_only_when_it_can_build_it(monkeypatch):
    fake_machine(monkeypatch, {"node", "cargo", "git"})
    assert {"node", "cargo", "zeb-native"} <= W.capabilities()
    for missing in ("node", "cargo", "git"):
        fake_machine(monkeypatch, {"node", "cargo", "git"} - {missing})
        assert "zeb-native" not in W.capabilities()


def test_the_scheduler_can_pick_the_configured_seats(monkeypatch):
    records = {p.id: p.to_record() for p in (WALT, *CONFIGURED, DEMON_OVERCALL_N100)}
    fake_machine(monkeypatch, {"node", "cargo", "git"})
    caps = W.capabilities()
    assert W.runnable(records, caps) == sorted(records)
    fake_machine(monkeypatch, {"node"})
    assert W.runnable(records, W.capabilities()) == [WALT.id]  # no capability, no seat
    fit = W.ratings(records, [], WALT.id)
    allowed = W.runnable(records, caps)
    choice = next_pair(fit, 2, allowed, rng=random.Random(0))
    assert {choice.a, choice.b} <= set(records) and choice.a != choice.b
    # unplayed pairs sit at the prior, so a configured Zeb is among the first scheduled
    picks = [next_pair(fit, 2, allowed, rng=random.Random(i)) for i in range(30)]
    picked = {name for c in picks for name in (c.a, c.b)}
    assert {p.id for p in CONFIGURED} <= picked
    assert W.runnable(records, caps, only={ZEB_041_CONTESTED.id, DEMON_OVERCALL_N100.id}) == \
        sorted([ZEB_041_CONTESTED.id, DEMON_OVERCALL_N100.id])


def test_the_build_refuses_a_pin_the_source_does_not_match(monkeypatch):
    trees = dict(ZEB_041_CONTESTED.pin["trees"])
    shipped = b"export const x = 1;\n"
    monkeypatch.setattr(Z, "_ensure_commits", lambda *c: None)
    monkeypatch.setattr(Z, "_git", lambda *a, binary=False: shipped if a[0] == "show" else trees["zeb/src"] + "\n")
    with pytest.raises(RuntimeError, match="is tree"):  # the fake serves one tree for every path
        Z.check_pin(ZEB_041_CONTESTED)
    monkeypatch.setattr(Z, "_git", lambda *a, binary=False: shipped if a[0] == "show" else
                        trees[a[1].split(":", 1)[1]] + "\n")
    with pytest.raises(RuntimeError, match="sha256"):  # right trees, wrong bytes in a shipped file
        Z.check_pin(ZEB_041_CONTESTED)


HOST_STAND_IN = textwrap.dedent("""
    import { createInterface } from 'node:readline';
    const [kind, dir, configText] = process.argv.slice(2);
    for await (const line of createInterface({ input: process.stdin })) {
      const r = JSON.parse(line);
      console.log(JSON.stringify(r.op === 'bid' ? { bid: 0, echo: r, config: JSON.parse(configText) } : { error: 'x' }));
    }
""")


def test_a_native_seat_sends_the_auction_context_and_its_config(tmp_path, monkeypatch):
    art = tmp_path / "art"
    art.mkdir()
    (art / "zeb-native").write_bytes(b"binary")
    (art / "zeb-native.sha256").write_text(hashlib.sha256(b"binary").hexdigest() + "\n")
    stand_in = tmp_path / "host.mjs"
    stand_in.write_text(HOST_STAND_IN)
    monkeypatch.setattr(seats, "materialize", lambda p: art)
    monkeypatch.setattr(seats, "NATIVE_HOST", stand_in)
    s = seats.seat(ZEB_041_CONTESTED, "g")
    try:
        from arena.auction import BidContext

        ctx = BidContext(hand=(1, 2, 3, 4, 5, 6, 7), seat=2, dealer=1, bids=(0, -1, -1, -1), high_bid=0,
                         high_seat=-1, legal=(30, 31), marks=(3, 4), marks_to_win=7)
        assert s.bid(ctx, random.Random(0)) == 0
        reply = s.host.ask({"op": "bid", "hand": [1], "seat": 0, "high_bid": 0, "high_seat": -1,
                            "legal": [30], "game": "g", "hand_no": 1})
        assert reply["config"]["overcall"] is True and reply["config"]["matchAware"] is True
        assert reply["config"]["nativeBinary"] == str(art / "zeb-native")
        sent = s.host.ask({"op": "bid", "hand": [1], "seat": 2, "high_bid": 0, "high_seat": -1,
                           "legal": [30], "game": "g", "hand_no": 1})["echo"]
        assert "marks" not in sent  # the plain request is the host's own; context comes from NativeZebSeat
    finally:
        s.close()
    (art / "zeb-native").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="not the binary"):
        seats.seat(ZEB_041_CONTESTED, "g")


def test_native_zeb_materializes_through_its_builder(tmp_path, monkeypatch):
    built = []
    monkeypatch.setattr(Z, "build", lambda p, out, target: built.append((p.id, target.name)))
    target = artifacts.materialize(ZEB_040_MATCH, tmp_path)
    assert built == [("zeb-0.4.0-match", ".zeb-native-target")] and (target / ".verified").exists()
    artifacts.materialize(ZEB_040_MATCH, tmp_path)  # verified once, then reused
    assert len(built) == 1
