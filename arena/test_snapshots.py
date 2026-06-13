"""#26 bridge: arena --emit-snapshots payload shape and deal validity.

CPU-only, tiny match (heuristic auctions, random play, fixed seed). Asserts
the snapshot rows carry a valid 4x7 deal plus the real auction, ready for the
forge.cli.generate_eq_from_snapshots bridge.
"""
from arena.bidders import HeuristicBidder
from arena.engine import ArenaConfig
from arena.match import run_match, snapshot_rows
from arena.play import RandomPlay


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
        assert set(r) == {"hands", "decl_id", "bids", "bidder", "bid_value"}
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
