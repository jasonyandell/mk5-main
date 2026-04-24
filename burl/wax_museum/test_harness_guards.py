"""Phase A guard tests — exercise both the turn-budget extension path and
the forced-commit-on-cap path with scripted stub models."""

from __future__ import annotations

import random
from typing import Any

from burl.harness.agent_runner import _current_player
from burl.tools import engine as engine_tools
from burl.wax_museum.harness import run_decision_waxed
from forge.zeb.game import apply_action, legal_actions, new_game


def _mid_game_state(seed: int = 2026):
    state = new_game(seed=seed, skip_bidding=True)
    rng = random.Random(seed)
    while len(state.play_history) < 20:
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))
    return state


def _legal_play(state) -> int:
    me = _current_player(state)
    for d in state.hands[me]:
        if d not in state.played and engine_tools.is_legal(state, d)[0]:
            return int(d)
    raise RuntimeError("no legal play")


def _illegal_play(state) -> int:
    """Pick a domino_id that's definitely NOT in my hand (so it will reject
    with "not in current player's hand")."""
    me = _current_player(state)
    mine = set(state.hands[me])
    for d in range(28):
        if d not in mine:
            return int(d)
    raise RuntimeError("no illegal play available")


def test_illegal_commits_extend_budget_then_succeed():
    state = _mid_game_state(seed=2026)
    target = _legal_play(state)
    bad = _illegal_play(state)

    # Turns: explore, probe, BAD commit (reject, +2 budget), BAD commit
    # (reject, +2 budget), good commit.
    script = iter([
        "Plan.\n"
        + "x" * 300
        + f"\n<|tool_call>call:explore_game{{play:{target}}}<tool_call|>",
        f"Probe worst.\n<|tool_call>call:probe_worst_case{{play:{target}}}<tool_call|>",
        f"Try illegal first.\n<|tool_call>call:commit_play{{domino_id:{bad}}}<tool_call|>",
        f"Try again (wrong).\n<|tool_call>call:commit_play{{domino_id:{bad}}}<tool_call|>",
        f"Now legal.\n<|tool_call>call:commit_play{{domino_id:{target}}}<tool_call|>",
    ])

    def stub(messages, tool_schemas):
        return next(script)

    events: list[dict] = []
    result = run_decision_waxed(
        state, stub, max_turns=4, max_retries=5,
        on_event=lambda e: events.append(e),
    )

    assert result.trace.final_play == target, (
        f"expected committed={target}, got {result.trace.final_play}"
    )
    # Two budget extensions fired.
    exts = [e for e in events if e.get("evt") == "turn_budget_extended"]
    assert len(exts) == 2, f"expected 2 extensions, got {len(exts)}: {exts}"
    assert result.max_turns_extensions == 2
    assert not result.forced_commit
    print("[test_illegal_commits_extend_budget_then_succeed] OK "
          f"final={result.trace.final_play} extensions={result.max_turns_extensions}")


def test_cap_exhaustion_force_commits_without_probe():
    """Model exhausts turn budget without ever committing; forced-commit fires
    with oracle fallback (no probes were cached).

    We force this by scripting the model to only call ask_rule repeatedly —
    no explore, no probe, no commit."""
    state = _mid_game_state(seed=2026)

    # All turns: just return thought + ask_rule (free side-call, doesn't
    # advance the gate). Since commit_play never appears in the schema and
    # native_commit is never emitted, the loop exhausts.
    def stub(messages, tool_schemas):
        return (
            "Stalling.\n"
            + "x" * 300
            + "\n<|tool_call>call:ask_rule{topic:\"trump\"}<tool_call|>"
        )

    events: list[dict] = []
    result = run_decision_waxed(
        state, stub, max_turns=3, max_retries=2,
        on_event=lambda e: events.append(e),
        oracle=None,  # oracle=None triggers the "first legal" fallback
    )

    # Forced commit fired via fallback path (no probes, no oracle).
    assert result.forced_commit, f"expected forced_commit=True, got False. events={[e for e in events if e.get('evt','').startswith('forced')]}"
    assert result.trace.final_play != -1
    # Resulting play must be legal.
    ok, reason = engine_tools.is_legal(state, result.trace.final_play)
    assert ok, f"forced commit was illegal: {result.trace.final_play} — {reason}"
    print("[test_cap_exhaustion_force_commits_without_probe] OK "
          f"final={result.trace.final_play} "
          f"reason={result.forced_commit_reason!r}")


def test_cap_exhaustion_force_commits_with_probe():
    """Model probes a play but never commits → force-commit should choose the
    probed play (single-candidate case)."""
    state = _mid_game_state(seed=2026)
    target = _legal_play(state)

    # Turn 1: explore. Turn 2: probe. Turn 3: stall with ask_rule.
    # Oracle=None; probed play should still win priority (1).
    turns_script = [
        "Plan.\n"
        + "x" * 300
        + f"\n<|tool_call>call:explore_game{{play:{target}}}<tool_call|>",
        f"Probe.\n<|tool_call>call:probe_worst_case{{play:{target}}}<tool_call|>",
        "Stall.\n<|tool_call>call:ask_rule{topic:\"trump\"}<tool_call|>",
    ]
    it = iter(turns_script)

    def stub(messages, tool_schemas):
        return next(it)

    events: list[dict] = []
    result = run_decision_waxed(
        state, stub, max_turns=3, max_retries=2,
        on_event=lambda e: events.append(e),
        oracle=None,
    )

    assert result.forced_commit, "expected forced_commit=True"
    assert result.trace.final_play == target, (
        f"expected probed play {target}, got {result.trace.final_play}"
    )
    print("[test_cap_exhaustion_force_commits_with_probe] OK "
          f"final={result.trace.final_play} "
          f"reason={result.forced_commit_reason!r}")


if __name__ == "__main__":
    test_illegal_commits_extend_budget_then_succeed()
    test_cap_exhaustion_force_commits_without_probe()
    test_cap_exhaustion_force_commits_with_probe()
    print("\nALL PHASE A GUARD TESTS PASS")
