"""CPU-only tests for the canonical Arena decision record."""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from dataclasses import replace

import pytest

from arena.bidders import Bid30Bidder, HeuristicBidder
from arena.decision_records import (
    ACTOR_INFORMATION_STATE_ID_VERSION,
    DECISION_RECORD_SCHEMA_VERSION,
    MECHANISM_SECTIONS,
    REPO_ROOT,
    RULES_STATE_ID_VERSION,
    artifact_fingerprint,
    build_decision_records,
    canonical_state_ids,
    git_provenance,
    policy_fingerprint_from_spec,
    write_decision_records,
)
from arena.engine import ArenaConfig
from arena.match import run_match
from arena.play import RandomPlay
from forge.zeb.game import new_game


TEST_CODE = {"status": "test", "head": "test-head", "dirty": False, "error": None}


def _policy(spec: str) -> dict:
    return policy_fingerprint_from_spec(
        spec,
        n_samples=10,
        device="cpu",
        code_provenance=TEST_CODE,
    )


def _tiny_result():
    return run_match(
        bid_a=HeuristicBidder(),
        bid_b=Bid30Bidder(),
        play_a=RandomPlay(seed=1),
        play_b=RandomPlay(seed=2),
        n_games=2,
        cfg=ArenaConfig(marks_to_win=1, base_seed=17),
        label_a="heuristic+random",
        label_b="bid30+random",
        fast_batching=False,
    )


def test_post_match_export_replays_every_decision_and_keeps_mechanisms_distinct():
    result = _tiny_result()
    rows = build_decision_records(
        result,
        policy_a=_policy("heuristic+random"),
        policy_b=_policy("bid30+random"),
    )

    assert len(rows) == sum(
        len(hand.plays) for game in result.games for hand in game.hands
    )
    assert len(rows) == 28 * sum(len(game.hands) for game in result.games)
    assert len({row["record_id"] for row in rows}) == len(rows)

    by_trajectory: dict[str, list[dict]] = {}
    for row in rows:
        by_trajectory.setdefault(row["trajectory"]["trajectory_id"], []).append(row)
        assert row["schema_version"] == DECISION_RECORD_SCHEMA_VERSION
        assert row["record_id_usage"] == "join_only_do_not_use_as_policy_input"
        assert row["trajectory"]["trajectory_id_usage"] == \
            "join_only_do_not_use_as_policy_input"
        assert all(section in row for section in MECHANISM_SECTIONS)
        assert row["state"]["chosen_slot"] in row["state"]["legal_slots"]
        assert row["state"]["chosen_domino"] in row["state"]["legal_dominoes"]
        assert len(row["state"]["legal_actions"]) == len(row["state"]["legal_slots"])
        assert row["identity"]["action_id"] in {
            action["action_id"] for action in row["state"]["legal_actions"]
        }
        actor = row["state"]["actor"]
        assert row["partner_coordination"]["partner_seat"] == (actor + 2) % 4
        assert row["partner_coordination"]["fixed_shuffled_condition"] == "not_run"
        assert row["action_derived_inference"]["status"] == "not_instrumented"
        assert row["action_derived_inference"]["actor_action_likelihood"] is None
        assert row["plan_persistence"]["plan_id"] is None
        assert row["uncertainty"]["q_pdf_by_domino"] is None
        assert row["identity"]["rules_state_id"].startswith(RULES_STATE_ID_VERSION)
        assert row["identity"]["actor_information_state_id"].startswith(
            ACTOR_INFORMATION_STATE_ID_VERSION
        )
        online = {key: value for key, value in row.items() if key != "offline_truth"}
        forbidden = {"seed", "game_idx", "hand_idx", "initial_hands", "remaining_hands"}

        def keys(value):
            if isinstance(value, dict):
                yield from value
                for child in value.values():
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        assert forbidden.isdisjoint(set(keys(online)))
        assert forbidden.issubset(row["offline_truth"])

    assert all(len(trajectory) == 28 for trajectory in by_trajectory.values())
    for trajectory in by_trajectory.values():
        assert trajectory[0]["role_order"]["seat_role"] == "bidder"
        assert [row["trajectory"]["decision_idx"] for row in trajectory] == list(range(28))


def test_ids_isolate_hidden_world_auction_evidence_and_match_score():
    state = new_game(seed=31)
    assert state.bidder == state.trick_leader
    actor = state.bidder
    hidden = [seat for seat in range(4) if seat != actor]
    hands = list(state.hands)
    hands[hidden[0]], hands[hidden[1]] = hands[hidden[1]], hands[hidden[0]]
    swapped = replace(state, hands=tuple(hands))

    base = canonical_state_ids(
        state,
        dealer=state.dealer,
        full_bids=state.bid_state.bids,
        marks_before=(0, 0),
        marks_to_win=7,
    )
    hidden_swap = canonical_state_ids(
        swapped,
        dealer=state.dealer,
        full_bids=state.bid_state.bids,
        marks_before=(0, 0),
        marks_to_win=7,
    )
    assert hidden_swap["rules_state_id"] == base["rules_state_id"]
    assert hidden_swap["actor_information_state_id"] == base["actor_information_state_id"]
    assert hidden_swap["decision_context_id"] == base["decision_context_id"]
    assert hidden_swap["world_state_id"] != base["world_state_id"]

    losing_bid_changed = list(state.bid_state.bids)
    losing_seat = next(seat for seat in range(4) if seat != state.bidder)
    losing_bid_changed[losing_seat] = max(1, state.bid_state.high_bid - 1)
    auction_arm = canonical_state_ids(
        state,
        dealer=state.dealer,
        full_bids=losing_bid_changed,
        marks_before=(0, 0),
        marks_to_win=7,
    )
    assert auction_arm["actor_information_state_id"] == base["actor_information_state_id"]
    assert auction_arm["decision_context_id"] != base["decision_context_id"]

    score_arm = canonical_state_ids(
        state,
        dealer=state.dealer,
        full_bids=state.bid_state.bids,
        marks_before=(6, 6),
        marks_to_win=7,
    )
    assert score_arm["actor_information_state_id"] == base["actor_information_state_id"]
    assert score_arm["decision_context_id"] != base["decision_context_id"]


def test_artifact_fingerprint_reports_available_missing_and_unspecified(tmp_path):
    artifact = tmp_path / "head.pt"
    artifact.write_bytes(b"policy-head")
    available = artifact_fingerprint(artifact, role="player_value_head")
    assert available["status"] == "available"
    assert available["sha256"] == hashlib.sha256(b"policy-head").hexdigest()
    assert available["size_bytes"] == len(b"policy-head")

    missing = artifact_fingerprint(tmp_path / "missing.pt", role="player_value_head")
    assert missing["status"] == "missing"
    assert missing["sha256"] is None

    unspecified = artifact_fingerprint(None, role="training_corpus")
    assert unspecified["status"] == "unspecified"
    assert unspecified["path"] is None


def test_policy_fingerprint_does_not_invent_sampler_or_corpus():
    policy = _policy("heuristic+random")
    assert policy["sampler"]["status"] == "not_used_by_player"
    assert policy["sampler"]["algorithm"] is None
    assert policy["corpus"] == {"status": "not_declared_by_runtime", "id": None}
    assert policy["artifacts"] == []
    assert policy["bidder"]["spec"] == "heuristic"
    assert policy["player"]["spec"] == "random"


def test_c0_fingerprint_hashes_only_selected_r8_and_oracle_without_model_load(monkeypatch):
    from forge.zeb.eval.loading import DEFAULT_ORACLE

    margin = REPO_ROOT / "champion/margin_net_r8.pt"
    oracle = REPO_ROOT / DEFAULT_ORACLE
    missing = [str(path) for path in (margin, oracle) if not path.is_file()]
    if missing:
        pytest.skip(f"canonical C0 artifact absent: {', '.join(missing)}")

    import torch

    def forbid_model_load(*args, **kwargs):
        raise AssertionError("fingerprinting must not load model weights")

    monkeypatch.setattr(torch, "load", forbid_model_load)
    policy = policy_fingerprint_from_spec(
        "margin:wp,model=champion/margin_net_r8.pt+lens:ev",
        n_samples=10,
        device="cuda",
        oracle_checkpoint=oracle,
        code_provenance=TEST_CODE,
    )
    artifacts = {artifact["role"]: artifact for artifact in policy["artifacts"]}
    assert set(artifacts) == {"bidder_value_head", "player_oracle"}
    assert all(artifact["status"] == "available" for artifact in artifacts.values())
    assert artifacts["bidder_value_head"]["sha256"] == hashlib.sha256(
        margin.read_bytes()
    ).hexdigest()
    assert artifacts["player_oracle"]["sha256"] == hashlib.sha256(
        oracle.read_bytes()
    ).hexdigest()
    assert policy["sampler"] == {
        "status": "declared",
        "algorithm": "uniform-completion-dp-v1",
        "n_worlds": 10,
        "device": "cuda",
    }
    assert policy["bidder"]["utility"] == {"kind": "marks_to_seven", "name": "wp"}
    assert policy["player"]["utility"] == {
        "kind": "fixed_e_q_collapse", "name": "ev",
    }
    assert policy["bidder"]["component_id"] != policy["player"]["component_id"]
    assert all("margin_net.pt" != artifact["requested_path"] for artifact in policy["artifacts"])


def test_judsearch_fingerprint_records_hardcoded_cpu_sampler_device():
    policy = policy_fingerprint_from_spec(
        "heuristic+judsearch:n20",
        n_samples=99,
        device="mps",
        code_provenance=TEST_CODE,
    )
    assert policy["sampler"]["n_worlds"] == 20
    assert policy["sampler"]["device"] == "cpu"
    assert policy["player"]["execution_device"] == "cpu"


def test_git_provenance_distinguishes_dirty_tracked_and_untracked_content(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    tracked = repo / "policy.py"
    tracked.write_text("VALUE = 1\n")
    subprocess.run(["git", "add", "policy.py"], cwd=repo, check=True)
    subprocess.run([
        "git", "-c", "user.name=Arena Test", "-c", "user.email=arena@test.invalid",
        "commit", "-qm", "initial",
    ], cwd=repo, check=True)

    clean = git_provenance(repo)
    assert clean["status"] == "clean_commit"
    assert clean["worktree_fingerprint"]

    tracked.write_text("VALUE = 2\n")
    dirty_a = git_provenance(repo)
    tracked.write_text("VALUE = 3\n")
    dirty_b = git_provenance(repo)
    assert dirty_a["status"] == dirty_b["status"] == "dirty_worktree"
    assert dirty_a["worktree_fingerprint"] != dirty_b["worktree_fingerprint"]
    policy_a = policy_fingerprint_from_spec(
        "heuristic+random", n_samples=10, device="cpu", code_provenance=dirty_a,
    )
    policy_b = policy_fingerprint_from_spec(
        "heuristic+random", n_samples=10, device="cpu", code_provenance=dirty_b,
    )
    assert policy_a["policy_id"] != policy_b["policy_id"]

    extra = repo / "new_policy.py"
    extra.write_text("EXTRA = 1\n")
    untracked_a = git_provenance(repo)
    extra.write_text("EXTRA = 2\n")
    untracked_b = git_provenance(repo)
    assert untracked_a["untracked_file_count"] == 1
    assert untracked_a["untracked_manifest_sha256"] != \
        untracked_b["untracked_manifest_sha256"]
    assert untracked_a["worktree_fingerprint"] != untracked_b["worktree_fingerprint"]


def test_writer_fails_closed_on_count_or_duplicate_ids(tmp_path):
    result = _tiny_result()
    policy_a = _policy("heuristic+random")
    policy_b = _policy("bid30+random")
    rows = build_decision_records(result, policy_a=policy_a, policy_b=policy_b)

    short_path = tmp_path / "short.jsonl"
    with pytest.raises(ValueError, match="count"):
        write_decision_records(
            short_path, rows[:-1], result=result,
            policy_a=policy_a, policy_b=policy_b,
        )
    assert not short_path.exists()

    duplicate = list(rows)
    duplicate[-1] = {**duplicate[-1], "record_id": duplicate[0]["record_id"]}
    duplicate_path = tmp_path / "duplicate.jsonl"
    with pytest.raises(ValueError, match="duplicate"):
        write_decision_records(
            duplicate_path, duplicate, result=result,
            policy_a=policy_a, policy_b=policy_b,
        )
    assert not duplicate_path.exists()


def test_deterministic_gzip_checksum_and_round_trip(tmp_path):
    result = _tiny_result()
    policy_a = _policy("heuristic+random")
    policy_b = _policy("bid30+random")
    rows = build_decision_records(result, policy_a=policy_a, policy_b=policy_b)
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    first_manifest = write_decision_records(
        first, rows, result=result, policy_a=policy_a, policy_b=policy_b,
    )
    write_decision_records(
        second, rows, result=result, policy_a=policy_a, policy_b=policy_b,
    )

    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, "rt", encoding="utf-8") as fh:
        round_trip = [json.loads(line) for line in fh]
    assert round_trip == rows
    manifest = json.loads(first_manifest.read_text())
    assert manifest["record_encoding"] == "gzip-jsonl-mtime0"
    assert manifest["record_sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert manifest["leakage_boundary"]["join_only_ids"] == [
        "record_id", "trajectory_id",
    ]


def test_cli_emits_jsonl_and_checksum_manifest_cpu_only(tmp_path, monkeypatch):
    from arena import cli
    from arena import decision_records

    decisions = tmp_path / "decisions.jsonl"
    out_dir = tmp_path / "result"
    provenance_calls = []

    def capture_provenance(repo_root):
        provenance_calls.append({
            "out_dir_exists": out_dir.exists(),
            "decisions_exist": decisions.exists(),
        })
        return TEST_CODE

    monkeypatch.setattr(decision_records, "git_provenance", capture_provenance)
    monkeypatch.setattr("sys.argv", [
        "arena.cli",
        "--team-a", "heuristic+random",
        "--team-b", "bid30+random",
        "--n-games", "2",
        "--marks-to-win", "1",
        "--base-seed", "19",
        "--device", "cpu",
        "--no-fast-batching",
        "--out-dir", str(out_dir),
        "--emit-decisions", str(decisions),
    ])
    assert cli.main() == 0
    assert provenance_calls == [{
        "out_dir_exists": False,
        "decisions_exist": False,
    }]

    manifest_path = decisions.with_name(decisions.name + ".manifest.json")
    assert decisions.exists() and manifest_path.exists()
    rows = [json.loads(line) for line in decisions.read_text().splitlines()]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["record_count"] == len(rows) == manifest["expected_record_count"]
    assert len(rows) == 56
    assert manifest["record_sha256"] == hashlib.sha256(decisions.read_bytes()).hexdigest()
    assert set(manifest["mechanism_sections"]) == set(MECHANISM_SECTIONS)
    assert manifest["leakage_boundary"]["offline_only_ids"] == ["world_state_id"]
    assert manifest["leakage_boundary"]["offline_provenance_fields"] == [
        "match.base_seed",
    ]
