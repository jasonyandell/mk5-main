"""#26 bridge: arena --emit-snapshots payload shape and deal validity.

CPU-only, tiny match (heuristic auctions, random play, fixed seed). Asserts
the snapshot rows carry a valid 4x7 deal plus the real auction, ready for the
forge.cli.generate_eq_from_snapshots bridge.
"""
from arena.bidders import HeuristicBidder
from arena.engine import ArenaConfig
from arena.match import hand_rows, run_match, snapshot_rows
from arena.play import RandomPlay

_SNAPSHOT_KEYS = {
    "a_team", "game_idx", "hand_idx", "seed", "dealer",
    "hands", "decl_id", "bids", "bidder", "bid_value", "plays",
    "bidder_team_pts", "opp_team_pts", "made",
}


def _tiny_result(n_games=2):
    return run_match(
        bid_a=HeuristicBidder(), bid_b=HeuristicBidder(),
        play_a=RandomPlay(seed=1), play_b=RandomPlay(seed=2),
        n_games=n_games, cfg=ArenaConfig(marks_to_win=2, base_seed=7),
    )


def test_snapshot_rows_shape_and_valid_deal():
    rows = snapshot_rows(_tiny_result())
    assert rows, "tiny match should produce at least one contracted hand"
    for r in rows:
        # 4 seats x 7 dominoes, every id 0..27, no repeats across the deal.
        assert set(r) == _SNAPSHOT_KEYS
        hands = r["hands"]
        assert len(hands) == 4
        assert all(len(h) == 7 for h in hands)
        flat = [d for h in hands for d in h]
        assert sorted(flat) == list(range(28))
        # Auction provenance.
        assert len(r["bids"]) == 4
        assert 0 <= r["bidder"] <= 3
        assert r["bids"][r["bidder"]] == r["bid_value"]
        assert r["bid_value"] == max(r["bids"])
        assert 0 <= r["decl_id"] <= 6  # arena bidders declare pip trumps
        assert 0 <= r["dealer"] <= 3  # auction runs dealer+1 .. dealer
        # Realized outcome: declaring + opposing points partition the 42, the
        # declaring team captured at least 0, and made is a clean 0/1 flag.
        assert 0 <= r["bidder_team_pts"] <= 42
        assert 0 <= r["opp_team_pts"] <= 42
        assert r["made"] in (0, 1)
        # Play history (jud v1): 28 (seat, domino) pairs covering every domino
        # exactly once, bidder leads trick 1, and within each trick the seats
        # rotate clockwise from the trick's leader.
        plays = r["plays"]
        assert len(plays) == 28
        assert sorted(d for _, d in plays) == list(range(28))
        assert plays[0][0] == r["bidder"]
        for t in range(7):
            leader = plays[4 * t][0]
            assert [p for p, _ in plays[4 * t:4 * t + 4]] == [
                (leader + i) % 4 for i in range(4)
            ]
        # Every play came from the player's own dealt hand.
        for seat, domino in plays:
            assert domino in r["hands"][seat]


def test_snapshot_outcomes_match_per_hand_and_sum_to_42():
    """Realized-outcome fields must agree exactly with per_hand.csv (the same
    MatchResult), and every completed hand partitions the 42 points."""
    result = _tiny_result(n_games=4)
    snaps = snapshot_rows(result)
    rows = hand_rows(result)
    assert len(snaps) == len(rows), "one snapshot per completed hand"

    # The two paired halves replay identical deal seeds, so game_idx/hand_idx/
    # seed all repeat across halves; a_team is what makes the key unique.
    by_key = {(r["a_team"], r["game_idx"], r["hand_idx"]): r for r in rows}
    assert len(by_key) == len(rows), "(a_team, game_idx, hand_idx) uniquely key a hand"

    for s in snaps:
        row = by_key[(s["a_team"], s["game_idx"], s["hand_idx"])]
        assert s["seed"] == row["seed"]
        assert s["dealer"] == row["dealer"]
        # per_hand.csv reports points in A/B terms; the snapshot reports them
        # in declaring/opposing terms. bidder_is_a picks which is which.
        if row["bidder_is_a"]:
            assert s["bidder_team_pts"] == row["team_a_pts"]
            assert s["opp_team_pts"] == row["team_b_pts"]
        else:
            assert s["bidder_team_pts"] == row["team_b_pts"]
            assert s["opp_team_pts"] == row["team_a_pts"]
        assert s["made"] == row["made"]
        # A played hand always distributes all 42 points (redeals are auction
        # pass-outs, not partial play), so this holds for every completed hand.
        if row["redeals"] == 0:
            assert s["bidder_team_pts"] + s["opp_team_pts"] == 42


def test_snapshot_plays_replay_to_recorded_points():
    """The play history must replay — via resolve_trick — to exactly the
    stamped outcome: each trick's winner leads the next, and the accumulated
    team points equal bidder_team_pts / opp_team_pts. This is the contract the
    jud featurizer's points-so-far reconstruction stands on."""
    from forge.oracle.tables import resolve_trick

    for s in snapshot_rows(_tiny_result(n_games=4)):
        pts = [0, 0]
        for t in range(7):
            trick = s["plays"][4 * t:4 * t + 4]
            out = resolve_trick(
                trick[0][1], tuple(d for _, d in trick), s["decl_id"],
            )
            winner = (trick[0][0] + out.winner_offset) % 4
            if t < 6:
                assert s["plays"][4 * (t + 1)][0] == winner
            pts[winner % 2] += out.points
        assert pts[s["bidder"] % 2] == s["bidder_team_pts"]
        assert pts[1 - s["bidder"] % 2] == s["opp_team_pts"]
        assert sum(pts) == 42


def test_cli_emit_snapshots_writes_file(tmp_path):
    """End-to-end CLI: --emit-snapshots writes a well-formed JSON payload."""
    import json
    import sys

    from arena import cli

    out = tmp_path / "snaps.json"
    argv = [
        "arena.cli",
        "--team-a", "heuristic+random",
        "--team-b", "heuristic+random",
        "--n-games", "2",
        "--marks-to-win", "2",
        "--base-seed", "11",
        "--device", "cpu",
        "--out-dir", str(tmp_path / "results"),
        "--emit-snapshots", str(out),
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        rc = cli.main()
    finally:
        sys.argv = old_argv
    assert rc == 0
    assert out.exists()

    payload = json.loads(out.read_text())
    assert "snapshots" in payload and "metadata" in payload
    assert payload["metadata"]["n_snapshots"] == len(payload["snapshots"])
    assert payload["metadata"]["n_games"] == 2
    for s in payload["snapshots"]:
        assert len(s["hands"]) == 4 and all(len(h) == 7 for h in s["hands"])
        assert len(s["bids"]) == 4
        # Realized-outcome fields survive the JSON round-trip.
        assert set(s) == _SNAPSHOT_KEYS
        assert s["made"] in (0, 1)
        assert s["bidder_team_pts"] + s["opp_team_pts"] == 42
