# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""The walt1-binary seat: request shape, seed rule, reply checks, fail-closed pins."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from tournament.walt1_seat import (Walt1Error, check_reply, decision_seed, participant, request,
                                   verify)
from tournament.participants import Participant, supports_ladder


def test_request_matches_endpoint_shape():
    s = SimpleNamespace(decl_id=6, bidder=0, hands=((26, 1, 6, 7, 8, 18, 19), (), (), ()),
                        play_history=())
    assert request(s, 30, 0, 40, 8, 1000) == (
        "walt1 play\ndecl 6\nbid 30\nseat 0\nbidder 0\nhand 1 6 7 8 18 19 26\nplays\n"
        "n 40\nn0 8\nseed 1000\n")
    s.play_history = ((0, 19), (1, 16))
    assert "\nplays 0 19 1 16\n" in request(s, 30, 2, 40, 8, 1)


def test_seed_ignores_names_and_claim_time():
    g1 = "20261109T000000Z-walt1-a-1--walt1-b-2-00000000000000ff/0/1"
    g2 = "20261110T120000Z-walt1-x-9--walt1-ref-3-00000000000000ff/0/1"
    assert decision_seed(g1, 2, 3, 5) == decision_seed(g2, 2, 3, 5)
    assert decision_seed(g1, 2, 3, 5) != decision_seed(g1, 2, 3, 6)
    assert decision_seed(g1, 2, 3, 5) != decision_seed(g1[:-1] + "0", 2, 3, 5)
    with pytest.raises(ValueError):
        decision_seed("not-a-game", 1, 0, 0)


OK = '{"v":1,"kind":"play","choice":19,"forced":false,"raced":false,"opts":[[1,1],[19,2]],"leader":0,"points":[0,0]}'


def test_reply_checks():
    assert check_reply(OK, (1, 19), 0, [0, 0]) == 19
    bad = {
        "not json": "illegal",
        OK.replace('"choice":19', '"choice":5'): "illegal choice",
        OK.replace('"leader":0', '"leader":2'): "leader/points",
        OK.replace('"points":[0,0]', '"points":[1,0]'): "leader/points",
        '{"v":1,"error":"seat not to act"}': "error reply",
        OK.replace('"choice":19', '"choice":true'): "no integer choice",
    }
    for line, why in bad.items():
        with pytest.raises(Walt1Error):
            check_reply(line, (1, 19), 0, [0, 0])
    with pytest.raises(Walt1Error, match="forced"):
        check_reply(OK, (19,), 0, [0, 0])


def test_pins_fail_closed(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.rs").write_text("fn main() {}\n")
    binary = tmp_path / "walt-play"
    binary.write_bytes(b"one")
    p = participant("t", binary, tmp_path)
    assert p.id.startswith("walt1-t-") and p.pin["git_commit"] is None
    assert supports_ladder(p.to_record(), "marks7-bid30") and not supports_ladder(p.to_record(), "marks7")
    assert verify(Participant.from_record(p.to_record())) == binary.resolve()
    binary.write_bytes(b"two")
    with pytest.raises(RuntimeError, match="pinned binary"):
        verify(p)
    binary.write_bytes(b"one")
    (tmp_path / "src" / "main.rs").write_text("fn main() { }\n")
    with pytest.raises(RuntimeError, match="sources changed"):
        verify(p)
