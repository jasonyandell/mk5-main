"""Canonical, architecture-neutral decision records for Arena matches.

The Arena already retains enough information to replay every completed hand.
This module turns that retained trajectory into one JSON record per play
decision without changing policy execution.  It deliberately records what the
current harness can observe and marks unavailable mechanisms as unavailable;
it does not infer likelihoods, plans, partnership conventions, or Q values
from proxies.

Identity is split so causal controls do not accidentally become new states:

* ``rules_state_id``: public play/rules state, excluding losing auction bids
  and match score;
* ``actor_information_state_id``: rules state plus the actor's remaining hand;
* ``decision_context_id``: actor information plus the full public auction and
  pre-hand match score;
* ``world_state_id``: decision context plus every remaining hand.  This ID and
  the corresponding deal are offline-only truth.

All IDs are content hashes with an explicit schema version.  Runtime/public
IDs never receive an opponent's hidden hand.
"""
from __future__ import annotations

import hashlib
import gzip
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from forge.zeb.game import apply_action, current_player, legal_actions
from forge.zeb.types import BidState, GamePhase, ZebGameState

from .engine import HandRecord
from .match import MatchResult


DECISION_RECORD_SCHEMA_VERSION = "arena.decision-record.v1"
RULES_STATE_ID_VERSION = "arena.rules-state.v1"
ACTOR_INFORMATION_STATE_ID_VERSION = "arena.actor-information-state.v1"
DECISION_CONTEXT_ID_VERSION = "arena.decision-context.v1"
WORLD_STATE_ID_VERSION = "arena.world-state.v1"
ACTION_ID_VERSION = "arena.action.v1"
POLICY_FINGERPRINT_VERSION = "arena.policy-fingerprint.v1"
ARTIFACT_FINGERPRINT_VERSION = "arena.artifact-fingerprint.v1"
MANIFEST_SCHEMA_VERSION = "arena.decision-manifest.v1"

REPO_ROOT = Path(__file__).resolve().parents[1]

SEAT_ROLES = {
    0: "bidder",
    1: "left_setter",
    2: "bidder_partner",
    3: "right_setter",
}

MECHANISM_SECTIONS = (
    "uncertainty",
    "role_order",
    "partner_coordination",
    "action_derived_inference",
    "plan_persistence",
    "distributional_utility",
    "bidding",
    "match_score",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _content_id(version: str, payload: Any) -> str:
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    return f"{version}:{digest}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def artifact_fingerprint(
    path: str | Path | None,
    *,
    role: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Hash one declared artifact without pretending a missing path exists."""
    base: dict[str, Any] = {
        "schema_version": ARTIFACT_FINGERPRINT_VERSION,
        "role": role,
        "requested_path": None if path is None else str(path),
        "path": None,
        "status": "unspecified",
        "sha256": None,
        "size_bytes": None,
    }
    if path is None:
        return base

    requested = Path(path).expanduser()
    resolved = requested if requested.is_absolute() else repo_root / requested
    base["path"] = _display_path(resolved, repo_root)
    if not resolved.exists():
        base["status"] = "missing"
        return base
    if not resolved.is_file():
        base["status"] = "not_regular_file"
        return base

    base.update({
        "status": "available",
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    })
    return base


def _git_output(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=repo_root, check=True,
        capture_output=True,
    ).stdout


def _git_stream_hash(repo_root: Path, *args: str) -> str:
    """Hash git stdout without retaining a potentially large binary diff."""
    proc = subprocess.Popen(
        ["git", *args], cwd=repo_root,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    digest = hashlib.sha256()
    for chunk in iter(lambda: proc.stdout.read(1024 * 1024), b""):
        digest.update(chunk)
    _, stderr = proc.communicate()
    if proc.returncode:
        raise subprocess.CalledProcessError(
            proc.returncode, proc.args, stderr=stderr,
        )
    return digest.hexdigest()


def _untracked_manifest(repo_root: Path) -> tuple[str, int]:
    """Hash path, mode, type, and content for every Git-visible untracked file."""
    raw = _git_output(
        repo_root, "ls-files", "--others", "--exclude-standard", "-z",
    )
    paths = sorted(part for part in raw.split(b"\0") if part)
    digest = hashlib.sha256()
    for raw_path in paths:
        relative = os.fsdecode(raw_path)
        path = repo_root / relative
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            kind = "symlink"
            content_sha = _sha256_bytes(os.fsencode(os.readlink(path)))
        elif stat.S_ISREG(info.st_mode):
            kind = "file"
            content_sha = _sha256_file(path)
        else:
            kind = f"special:{stat.S_IFMT(info.st_mode)}"
            content_sha = None
        entry = {
            "path": relative,
            "kind": kind,
            "mode": stat.S_IMODE(info.st_mode),
            "size_bytes": info.st_size,
            "sha256": content_sha,
        }
        encoded = _canonical_json(entry)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest(), len(paths)


def git_provenance(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return an exact, bounded-memory fingerprint of Git-visible source state.

    The dirty fingerprint covers the combined staged/unstaged binary patch from
    ``HEAD`` plus the path, mode, type, and streamed content hash of every
    non-ignored untracked file. Ignored caches and build products are outside
    this code-provenance scope.
    """
    try:
        head = _git_output(repo_root, "rev-parse", "HEAD").decode().strip()
        status_output = _git_output(
            repo_root, "status", "--porcelain=v1", "--untracked-files=all",
        )
        tracked_patch_sha = _git_stream_hash(
            repo_root, "diff", "--binary", "--no-ext-diff", "--no-textconv",
            "HEAD", "--",
        )
        untracked_sha, untracked_count = _untracked_manifest(repo_root)
        dirty = bool(status_output)
        worktree_payload = {
            "head": head,
            "status_sha256": _sha256_bytes(status_output),
            "tracked_patch_sha256": tracked_patch_sha,
            "untracked_manifest_sha256": untracked_sha,
            "untracked_file_count": untracked_count,
        }
        worktree_fingerprint = _content_id(
            "arena.git-worktree.v1", worktree_payload,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {
            "status": "unavailable",
            "head": None,
            "dirty": None,
            "worktree_fingerprint": None,
            "scope": "git_visible_files",
            "error": type(exc).__name__,
        }
    return {
        "status": "dirty_worktree" if dirty else "clean_commit",
        "head": head,
        "dirty": dirty,
        "worktree_fingerprint": worktree_fingerprint,
        "scope": "tracked_patch_plus_nonignored_untracked_files",
        "tracked_patch_sha256": tracked_patch_sha,
        "untracked_manifest_sha256": untracked_sha,
        "untracked_file_count": untracked_count,
        "error": None,
    }


def _component_spec(spec: str) -> tuple[str, str]:
    name, sep, args = spec.partition(":")
    return name, args if sep else ""


def _tag_value(args: str, prefix: str, default: str | None = None) -> str | None:
    for tag in (part for part in args.split(",") if part):
        if tag.startswith(prefix):
            return tag[len(prefix):]
    return default


def _belief_adapter_from_args(args: str) -> str | None:
    """Match ``arena.cli.parse_bidder``'s path-vs-tuning-tag split."""
    adapter = None
    for part in (part for part in args.split(",") if part):
        is_tuning_tag = False
        if part[:1] in {"s", "m"}:
            try:
                float(part[1:])
                is_tuning_tag = True
            except ValueError:
                pass
        if not is_tuning_tag:
            adapter = part
    return adapter


def _play_utility(play_name: str, play_args: str) -> dict[str, Any]:
    if play_name == "lens":
        return {"kind": "fixed_e_q_collapse", "name": play_args or "ev"}
    if play_name == "belieflens":
        return {"kind": "belief_weighted_e_q_collapse", "name": play_args or "ev"}
    if play_name == "scorelens":
        return {"kind": "score_conditioned_e_q_collapse", "name": "scorelens",
                "band": float(play_args) if play_args else 0.15}
    if play_name in {"judplay", "judsearch"}:
        return {"kind": "realized_points_value", "name": play_name}
    if play_name == "random":
        return {"kind": "uniform_legal", "name": "random"}
    return {"kind": "unknown", "name": None}


def _bid_utility(bid_name: str, bid_args: str) -> dict[str, Any]:
    tags = [part for part in bid_args.split(",") if part]
    if bid_name in {"margin", "jud", "belief"}:
        return {"kind": "marks_to_seven", "name": "wp"}
    if bid_name in {"gus", "net"}:
        if any(tag.startswith("pass") for tag in tags):
            return {"kind": "marks_to_seven_with_pass_baseline", "name": "wp_pass"}
        if "wp" in tags:
            return {"kind": "marks_to_seven", "name": "wp"}
        return {"kind": "mark_ev", "name": "mark_ev"}
    if bid_name == "bid30":
        return {"kind": "fixed_bid", "name": "bid30"}
    if bid_name in {"heuristic", "random"}:
        return {"kind": "rule_policy", "name": bid_name}
    return {"kind": "unknown", "name": None}


def policy_fingerprint_from_spec(
    team_spec: str,
    *,
    n_samples: int,
    device: str,
    oracle_checkpoint: str | Path | None = None,
    gus_adapter: str | Path | None = None,
    belief_bidder_adapter: str | Path | None = None,
    repo_root: Path = REPO_ROOT,
    code_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fingerprint the policy declared by the Arena CLI player spec.

    This describes the executable consumer, not its training corpus.  A corpus
    is therefore explicitly ``not_declared_by_runtime`` unless a future caller
    supplies a stronger provenance source.
    """
    if "+" not in team_spec:
        raise ValueError(f"team_spec needs <bidder>+<play>, got {team_spec!r}")
    bid_spec, play_spec = team_spec.split("+", 1)
    bid_name, bid_args = _component_spec(bid_spec)
    play_name, play_args = _component_spec(play_spec)

    artifact_requests: list[tuple[str, str | Path | None]] = []
    if bid_name == "net":
        artifact_requests.append(("bidder_value_head", "champion/bid_net.pt"))
    elif bid_name == "margin":
        artifact_requests.append((
            "bidder_value_head",
            _tag_value(bid_args, "model=", "champion/margin_net.pt"),
        ))
    elif bid_name == "jud":
        artifact_requests.append((
            "bidder_value_head",
            _tag_value(bid_args, "model=", "champion/jud_net.pt"),
        ))
    elif bid_name == "gus":
        if gus_adapter is None:
            from gus.bidding.evaluate import ADAPTER_DEFAULT
            gus_adapter = ADAPTER_DEFAULT
        artifact_requests.append(("bidder_model", gus_adapter))
    elif bid_name == "belief":
        belief_bidder_adapter = (
            _belief_adapter_from_args(bid_args) or belief_bidder_adapter
        )
        if belief_bidder_adapter is None:
            from gus.bidding.evaluate import ADAPTER_DEFAULT
            belief_bidder_adapter = ADAPTER_DEFAULT
        artifact_requests.append(("bidder_belief_model", belief_bidder_adapter))
        artifact_requests.append(("bidder_oracle", oracle_checkpoint))

    if play_name in {"lens", "scorelens", "belieflens"}:
        artifact_requests.append(("player_oracle", oracle_checkpoint))
    if play_name == "belieflens":
        if gus_adapter is None:
            from gus.bidding.evaluate import ADAPTER_DEFAULT
            gus_adapter = ADAPTER_DEFAULT
        artifact_requests.append(("player_belief_model", gus_adapter))
    if play_name in {"judplay", "judsearch"}:
        artifact_requests.append((
            "player_value_head",
            _tag_value(play_args, "model=", "champion/jud_net.pt"),
        ))

    artifacts = [
        artifact_fingerprint(path, role=role, repo_root=repo_root)
        for role, path in artifact_requests
    ]

    if play_name in {"lens", "scorelens", "belieflens", "judsearch"}:
        from forge.eq.sampling_mrv_gpu import SAMPLER_ALGORITHM
        sampler: dict[str, Any] = {
            "status": "declared",
            "algorithm": SAMPLER_ALGORITHM,
            "n_worlds": (
                int(_tag_value(play_args, "n", "10"))
                if play_name == "judsearch" else int(n_samples)
            ),
            # parse_play deliberately pins JudSearch to CPU; Lens families use
            # the resolved CLI device.
            "device": "cpu" if play_name == "judsearch" else device,
        }
    else:
        sampler = {
            "status": "not_used_by_player",
            "algorithm": None,
            "n_worlds": None,
            "device": device,
        }

    bidder_artifacts = [
        a for a in artifacts if a["role"].startswith("bidder")
    ]
    player_artifacts = [
        a for a in artifacts if a["role"].startswith("player")
    ]
    bidder = {
        "spec": bid_spec,
        "family": bid_name,
        "execution_device": (
            device if bid_name in {"gus", "belief"} else "cpu"
        ),
        "utility": _bid_utility(bid_name, bid_args),
        "artifacts": bidder_artifacts,
    }
    bidder["component_id"] = _content_id(
        POLICY_FINGERPRINT_VERSION + ".bidder", bidder,
    )
    player = {
        "spec": play_spec,
        "family": play_name,
        "execution_device": (
            device if play_name in {"lens", "scorelens", "belieflens"} else "cpu"
        ),
        "utility": _play_utility(play_name, play_args),
        "artifacts": player_artifacts,
    }
    player["component_id"] = _content_id(
        POLICY_FINGERPRINT_VERSION + ".player", player,
    )

    body: dict[str, Any] = {
        "schema_version": POLICY_FINGERPRINT_VERSION,
        "team_spec": team_spec,
        "bidder": bidder,
        "player": player,
        "sampler": sampler,
        "artifacts": artifacts,
        "corpus": {"status": "not_declared_by_runtime", "id": None},
        "code": dict(code_provenance or git_provenance(repo_root)),
    }
    body["policy_id"] = _content_id(POLICY_FINGERPRINT_VERSION, body)
    return body


def _remaining_hand(state: ZebGameState, seat: int) -> list[int]:
    return sorted(int(d) for d in state.hands[seat] if d not in state.played)


def canonical_state_ids(
    state: ZebGameState,
    *,
    dealer: int,
    full_bids: Sequence[int],
    marks_before: Sequence[int],
    marks_to_win: int,
) -> dict[str, str]:
    """Compute the four versioned identities for one play decision."""
    actor = current_player(state)
    rules_payload = {
        "actor": actor,
        "bidder": state.bidder,
        "bid_value": state.bid_state.high_bid,
        "decl_id": state.decl_id,
        "play_history": [list(p) for p in state.play_history],
        "team_points": list(state.team_points),
        "trick_leader": state.trick_leader,
        "current_trick": list(state.current_trick),
    }
    rules_state_id = _content_id(RULES_STATE_ID_VERSION, rules_payload)
    actor_payload = {
        "rules_state_id": rules_state_id,
        "own_remaining_dominoes": _remaining_hand(state, actor),
    }
    actor_information_state_id = _content_id(
        ACTOR_INFORMATION_STATE_ID_VERSION, actor_payload,
    )
    context_payload = {
        "actor_information_state_id": actor_information_state_id,
        "auction": {"dealer": int(dealer), "bids": [int(v) for v in full_bids]},
        "match_score": {
            "marks": [int(v) for v in marks_before],
            "marks_to_win": int(marks_to_win),
        },
    }
    decision_context_id = _content_id(DECISION_CONTEXT_ID_VERSION, context_payload)
    world_payload = {
        "decision_context_id": decision_context_id,
        "remaining_hands": [_remaining_hand(state, seat) for seat in range(4)],
    }
    world_state_id = _content_id(WORLD_STATE_ID_VERSION, world_payload)
    return {
        "rules_state_id": rules_state_id,
        "actor_information_state_id": actor_information_state_id,
        "decision_context_id": decision_context_id,
        "world_state_id": world_state_id,
    }


def _initial_state(hand: HandRecord) -> ZebGameState:
    return ZebGameState(
        hands=hand.hands,
        dealer=hand.dealer,
        phase=GamePhase.PLAYING,
        bid_state=BidState(
            bids=hand.bids,
            high_bidder=hand.bidder,
            high_bid=hand.bid_value,
        ),
        decl_id=hand.decl_id,
        bidder=hand.bidder,
        played=frozenset(),
        play_history=(),
        current_trick=(),
        trick_leader=hand.bidder,
        team_points=(0, 0),
    )


def _side_for_seat(hand: HandRecord, seat: int) -> str:
    return "A" if seat % 2 == hand.a_team else "B"


def _assignment_id(hand: HandRecord, policy_a: Mapping[str, Any],
                   policy_b: Mapping[str, Any]) -> str:
    return _content_id("arena.partner-assignment.v1", {
        "a_team": hand.a_team,
        "team0_policy_id": policy_a["policy_id"] if hand.a_team == 0 else policy_b["policy_id"],
        "team1_policy_id": policy_a["policy_id"] if hand.a_team == 1 else policy_b["policy_id"],
        "seat_pairs": [[0, 2], [1, 3]],
        "assignment_mode": "arena_static_team_policy",
        "fixed_shuffled_condition": "not_run",
    })


def _records_for_hand(
    hand: HandRecord,
    *,
    marks_to_win: int,
    policy_a: Mapping[str, Any],
    policy_b: Mapping[str, Any],
    label_a: str,
    label_b: str,
) -> Iterable[dict[str, Any]]:
    marks_before = tuple(
        int(hand.marks_after[i] - hand.marks_delta[i]) for i in range(2)
    )
    state = _initial_state(hand)
    assignment_id = _assignment_id(hand, policy_a, policy_b)
    trajectory_id = _content_id("arena.trajectory.v1", {
        "a_team": hand.a_team,
        "game_idx": hand.game_idx,
        "hand_idx": hand.hand_idx,
        "seed": hand.seed,
        "dealer": hand.dealer,
        "bids": list(hand.bids),
        "bidder": hand.bidder,
        "bid_value": hand.bid_value,
        "decl_id": hand.decl_id,
        "marks_before": list(marks_before),
        "plays": [list(play) for play in hand.plays],
        "policy_a": policy_a["policy_id"],
        "policy_b": policy_b["policy_id"],
    })
    auction_order = [(hand.dealer + 1 + i) % 4 for i in range(4)]

    for decision_idx, (recorded_actor, chosen_domino) in enumerate(hand.plays):
        actor = current_player(state)
        if actor != recorded_actor:
            raise ValueError(
                f"hand {(hand.a_team, hand.game_idx, hand.hand_idx)} decision "
                f"{decision_idx}: replay actor {actor} != recorded {recorded_actor}"
            )
        legal_slots = list(legal_actions(state))
        chosen_slot = state.hands[actor].index(chosen_domino)
        if chosen_slot not in legal_slots:
            raise ValueError(
                f"hand {(hand.a_team, hand.game_idx, hand.hand_idx)} decision "
                f"{decision_idx}: chosen domino {chosen_domino} is not legal"
            )

        ids = canonical_state_ids(
            state, dealer=hand.dealer, full_bids=hand.bids,
            marks_before=marks_before, marks_to_win=marks_to_win,
        )
        action_id = _content_id(ACTION_ID_VERSION, {
            "actor_information_state_id": ids["actor_information_state_id"],
            "domino_id": chosen_domino,
        })
        contextual_action_id = _content_id(ACTION_ID_VERSION + ".context", {
            "decision_context_id": ids["decision_context_id"],
            "domino_id": chosen_domino,
        })
        actor_side = _side_for_seat(hand, actor)
        partner = (actor + 2) % 4
        opponent_seats = [(actor + 1) % 4, (actor + 3) % 4]
        actor_policy = policy_a if actor_side == "A" else policy_b
        opposing_policy = policy_b if actor_side == "A" else policy_a
        relative_to_bidder = (actor - hand.bidder) % 4
        trick_position = len(state.current_trick)
        legal_dominoes = [int(state.hands[actor][slot]) for slot in legal_slots]
        legal_candidates = [
            {
                "slot": slot,
                "domino_id": int(state.hands[actor][slot]),
                "action_id": _content_id(ACTION_ID_VERSION, {
                    "actor_information_state_id": ids["actor_information_state_id"],
                    "domino_id": int(state.hands[actor][slot]),
                }),
                "contextual_action_id": _content_id(ACTION_ID_VERSION + ".context", {
                    "decision_context_id": ids["decision_context_id"],
                    "domino_id": int(state.hands[actor][slot]),
                }),
            }
            for slot in legal_slots
        ]

        record: dict[str, Any] = {
            "schema_version": DECISION_RECORD_SCHEMA_VERSION,
            "record_id": _content_id("arena.decision-record-id.v1", {
                "trajectory_id": trajectory_id,
                "decision_idx": decision_idx,
            }),
            "record_id_usage": "join_only_do_not_use_as_policy_input",
            "trajectory": {
                "trajectory_id": trajectory_id,
                "trajectory_id_usage": "join_only_do_not_use_as_policy_input",
                "a_team": hand.a_team,
                "decision_idx": decision_idx,
                "labels": {"A": label_a, "B": label_b},
            },
            "identity": {
                **ids,
                "action_id": action_id,
                "contextual_action_id": contextual_action_id,
                "leakage_boundary": {
                    "rules_state_id": "online_public_play_and_contract_only",
                    "actor_information_state_id": "online_safe_adds_actor_hand_only",
                    "decision_context_id": "online_safe_adds_public_auction_and_score",
                    "world_state_id": "offline_only_contains_hidden_deal_truth",
                },
            },
            "state": {
                "actor": actor,
                "own_remaining_dominoes": _remaining_hand(state, actor),
                "play_history": [list(p) for p in state.play_history],
                "current_trick": list(state.current_trick),
                "trick_leader": state.trick_leader,
                "team_points": list(state.team_points),
                "legal_slots": legal_slots,
                "legal_dominoes": legal_dominoes,
                "legal_actions": legal_candidates,
                "chosen_slot": chosen_slot,
                "chosen_domino": chosen_domino,
                "choice_kind": "forced" if len(legal_slots) == 1 else "voluntary",
            },
            "provenance": {
                "actor_side": actor_side,
                "actor_policy_id": actor_policy["policy_id"],
                "actor_bidder_component_id": actor_policy["bidder"]["component_id"],
                "actor_player_component_id": actor_policy["player"]["component_id"],
                "opposing_policy_id": opposing_policy["policy_id"],
                "sampler": actor_policy["sampler"],
                "corpus": actor_policy["corpus"],
            },
            "uncertainty": {
                "status": (
                    "policy_used_sampler_but_values_not_emitted"
                    if actor_policy["sampler"]["status"] == "declared"
                    else "not_emitted_by_policy"
                ),
                "sampler": actor_policy["sampler"],
                "belief_posterior": None,
                "e_q_by_domino": None,
                "q_pdf_by_domino": None,
                "q_per_world": None,
            },
            "role_order": {
                "seat": actor,
                "team": actor % 2,
                "relative_to_bidder": relative_to_bidder,
                "seat_role": SEAT_ROLES[relative_to_bidder],
                "trick_index": decision_idx // 4,
                "trick_position": trick_position,
                "trick_order_role": (
                    "leader" if trick_position == 0
                    else "closer" if trick_position == 3
                    else "follower"
                ),
                "auction_position": auction_order.index(actor),
                "dealer": hand.dealer,
            },
            "partner_coordination": {
                "status": "assignment_observed_convention_not_instrumented",
                "assignment_id": assignment_id,
                "assignment_mode": "arena_static_team_policy",
                "fixed_shuffled_condition": "not_run",
                "partner_seat": partner,
                "partner_policy_id": actor_policy["policy_id"],
                "opponent_seats": opponent_seats,
                "opponent_policy_id": opposing_policy["policy_id"],
                "sender_convention_id": None,
                "receiver_decoder_id": None,
            },
            "action_derived_inference": {
                "status": "not_instrumented",
                "choice_kind": "forced" if len(legal_slots) == 1 else "voluntary",
                "actor_action_likelihood": None,
                "posterior_before": None,
                "posterior_after": None,
            },
            "plan_persistence": {
                "status": "not_instrumented",
                "plan_id": None,
                "plan_phase": None,
                "plan_age_decisions": None,
                "continuation_of_record_id": None,
            },
            "distributional_utility": {
                "status": "consumer_fingerprinted_values_not_emitted",
                "consumer": actor_policy["player"]["utility"],
                "policy_uses_q_distribution": actor_policy["player"]["family"] in {
                    "lens", "scorelens", "belieflens",
                },
                "full_pdf_emitted": False,
                "contextual_transform": (
                    "match_score"
                    if actor_policy["player"]["family"] == "scorelens"
                    else None
                ),
            },
            "bidding": {
                "status": "complete_public_auction",
                "dealer": hand.dealer,
                "auction_order": auction_order,
                "actor_auction_position": auction_order.index(actor),
                "bids": list(hand.bids),
                "actor_bid": hand.bids[actor],
                "bidder": hand.bidder,
                "bid_value": hand.bid_value,
                "decl_id": hand.decl_id,
                "forced_open": hand.forced,
                "redeals": hand.redeals,
            },
            "match_score": {
                "status": "observed_pre_hand",
                "marks": list(marks_before),
                "marks_to_win": marks_to_win,
                "actor_team_marks": marks_before[actor % 2],
                "opponent_team_marks": marks_before[1 - actor % 2],
            },
            "offline_truth": {
                "availability": "eval_only_do_not_use_as_policy_input",
                "game_idx": hand.game_idx,
                "hand_idx": hand.hand_idx,
                "seed": hand.seed,
                "initial_hands": [list(h) for h in hand.hands],
                "remaining_hands": [_remaining_hand(state, seat) for seat in range(4)],
                "final_team_points": list(hand.team_points),
                "contract_made": hand.made,
                "marks_delta": list(hand.marks_delta),
                "marks_after": list(hand.marks_after),
            },
        }
        if tuple(section for section in MECHANISM_SECTIONS if section not in record):
            raise AssertionError("decision record omitted a mechanism section")
        yield record
        state = apply_action(state, chosen_slot)

    if len(state.played) != 28 or state.phase != GamePhase.TERMINAL:
        raise ValueError(
            f"hand {(hand.a_team, hand.game_idx, hand.hand_idx)} did not replay to terminal"
        )
    if state.team_points != hand.team_points:
        raise ValueError(
            f"hand {(hand.a_team, hand.game_idx, hand.hand_idx)} replay points "
            f"{state.team_points} != recorded {hand.team_points}"
        )


def build_decision_records(
    result: MatchResult,
    *,
    policy_a: Mapping[str, Any],
    policy_b: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Replay a completed match into one canonical row per play decision."""
    rows: list[dict[str, Any]] = []
    for game in result.games:
        for hand in game.hands:
            rows.extend(_records_for_hand(
                hand,
                marks_to_win=result.cfg.marks_to_win,
                policy_a=policy_a,
                policy_b=policy_b,
                label_a=result.label_a,
                label_b=result.label_b,
            ))
    return rows


def write_decision_records(
    path: str | Path,
    records: Sequence[Mapping[str, Any]],
    *,
    result: MatchResult,
    policy_a: Mapping[str, Any],
    policy_b: Mapping[str, Any],
    run_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write deterministic JSONL plus a checksum-bearing sibling manifest.

    A ``.gz`` suffix selects deterministic gzip (``mtime=0``, no embedded
    filename). Count and identity checks fail closed before either file is
    created.
    """
    out = Path(path)
    expected_count = sum(
        len(hand.plays) for game in result.games for hand in game.hands
    )
    if len(records) != expected_count:
        raise ValueError(
            f"decision record count {len(records)} != expected {expected_count}"
        )
    record_ids = [record.get("record_id") for record in records]
    if any(not isinstance(record_id, str) or not record_id for record_id in record_ids):
        raise ValueError("every decision record needs a non-empty string record_id")
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("decision records contain duplicate record_id values")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix == ".gz":
        with out.open("wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0,
            ) as fh:
                for record in records:
                    fh.write(_canonical_json(record) + b"\n")
        encoding = "gzip-jsonl-mtime0"
    else:
        with out.open("wb") as fh:
            for record in records:
                fh.write(_canonical_json(record) + b"\n")
        encoding = "jsonl"

    manifest_path = out.with_name(out.name + ".manifest.json")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "decision_schema_version": DECISION_RECORD_SCHEMA_VERSION,
        "identity_versions": {
            "rules_state_id": RULES_STATE_ID_VERSION,
            "actor_information_state_id": ACTOR_INFORMATION_STATE_ID_VERSION,
            "decision_context_id": DECISION_CONTEXT_ID_VERSION,
            "world_state_id": WORLD_STATE_ID_VERSION,
            "action_id": ACTION_ID_VERSION,
        },
        "record_path": out.name,
        "record_encoding": encoding,
        "record_sha256": _sha256_file(out),
        "record_count": len(records),
        "expected_record_count": expected_count,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "match": {
            "label_a": result.label_a,
            "label_b": result.label_b,
            "n_games": result.n_games,
            "n_hands": sum(len(game.hands) for game in result.games),
            "marks_to_win": result.cfg.marks_to_win,
            "base_seed": result.cfg.base_seed,
        },
        "run_metadata": dict(run_metadata or {}),
        "mechanism_sections": list(MECHANISM_SECTIONS),
        "leakage_boundary": {
            "online_ids": [
                "rules_state_id", "actor_information_state_id", "decision_context_id",
            ],
            "join_only_ids": ["record_id", "trajectory_id"],
            "offline_only_ids": ["world_state_id"],
            "offline_only_fields": ["offline_truth"],
            "offline_provenance_fields": ["match.base_seed"],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return manifest_path
