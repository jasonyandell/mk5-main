"""The Node seat protocol, against a stand-in for the zeb package (offline).

The stand-in records what it is asked, so the test checks what the host sends
the player (only its own hand and the public plays) and the table conventions
the host applies around it (partner pass, the need, the one-mark cap).
"""
from __future__ import annotations

import json

import pytest

from tournament.seats import HostError, NodeHost

FAKE = """
export function initSync() {}
export const version = () => 'fake-zeb';
export const netId = () => 'no-net';
export const calls = [];
export function forecast({ hand }) {
  // decl 6 makes 34 with p .9, decl 9 makes 42 with p .95: the rule picks 42 under 9
  const p = (top) => Array.from({ length: 13 }, (_, i) => (30 + i <= top ? 0.95 : 0.1));
  return [{ decl: 6, mean: 10, pmake: p(34).map((x, i) => (i <= 4 ? 0.99 : x)) },
          { decl: 9, mean: 12, pmake: p(42) }];
}
export function chooseBid(f, { need, cutoff = 0.8 }) {
  for (let t = 42; t >= need; t--) {
    let best = null;
    for (const e of f) if (e.pmake[t - 30] >= cutoff && (!best || e.pmake[t - 30] > best.pmake[t - 30])) best = e;
    if (best) return { bid: t, decl: best.decl, pmake: best.pmake[t - 30] };
  }
  return null;
}
export function play(req) {
  if (req.hand.length !== 7) throw new Error('hand must be the dealt seven');
  const played = new Set(req.plays.map((p) => p[1]));
  return { tile: req.hand.filter((t) => !played.has(t)).sort((a, b) => b - a)[0], echo: req };
}
"""


@pytest.fixture
def host(tmp_path):
    pkg = tmp_path / "package"
    (pkg / "pkg").mkdir(parents=True)
    (pkg / "index.js").write_text(FAKE)
    (pkg / "package.json").write_text(json.dumps({"type": "module"}))
    (pkg / "pkg" / "zeb_bg.wasm").write_bytes(b"\0asm")
    h = NodeHost("zeb", tmp_path, {"worlds": 160})
    yield h
    h.close()


HAND = [27, 26, 24, 20, 17, 12, 2]


def test_version(host):
    assert host.ask({"op": "version"}) == {"version": "fake-zeb", "net": "no-net"}


def test_bids_by_the_rule_behind_a_partner_pass(host):
    base = {"op": "bid", "hand": HAND, "seat": 1, "game": "g", "hand_no": 1}
    assert host.ask({**base, "high_bid": 0, "high_seat": -1, "legal": [*range(30, 43), 84]}) == {"bid": 42}
    # partner (seat 3) holds the bid: pass, whatever the hand
    assert host.ask({**base, "high_bid": 31, "high_seat": 3, "legal": [*range(32, 43), 84]}) == {"bid": 0}
    # above one mark only marks bids remain, which Zeb never makes
    assert host.ask({**base, "high_bid": 42, "high_seat": 0, "legal": [84]}) == {"bid": 0}


def test_declares_the_likeliest_at_the_contract(host):
    ask = {"op": "declare", "hand": HAND, "seat": 1, "game": "g", "hand_no": 1}
    assert host.ask({**ask, "bid": 30}) == {"decl": 6}
    assert host.ask({**ask, "bid": 84}) == {"decl": 9}  # a marks contract is played to 42


def test_play_sends_the_dealt_hand_and_public_plays(host):
    r = host.ask({"op": "play", "hand": HAND, "seat": 0, "decl": 6, "bid": 84, "bidder": 0,
                  "plays": [[0, 27], [1, 0], [2, 1], [3, 3]], "leader": 0, "points": [0, 0],
                  "game": "g", "hand_no": 1})
    assert r == {"tile": 26}


def test_errors_come_back_as_errors(host):
    with pytest.raises(HostError, match="unknown op"):
        host.ask({"op": "nope"})
    assert host.ask({"op": "version"})["version"] == "fake-zeb"  # and the seat lives on
