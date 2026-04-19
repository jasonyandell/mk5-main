"""Native-format Burl runner — parallel to ``agent_runner.py``.

Drops the XML protocol block from the user turn. Tool declarations are passed
as JSON-schema dicts to ``apply_chat_template(tools=[...])`` so Gemma 4 sees
them in the native ``<|tool>...<tool|>`` shape it was post-trained on.

Public API mirrors the XML path closely so ``run_move4_spike.py`` is a thin
translation of ``run_move3.py``:

    run_decision_native(game_state, native_model, max_turns=8, max_retries=3) -> BurlTrace

where ``native_model(messages, tools) -> str``.

Tool wrappers are reused verbatim from ``agent_runner.py`` — the tools do the
same thing regardless of how the model is asked to call them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from burl.harness.agent_runner import (
    _DOMINO_LABELS,
    _current_player,
    _fmt_hand,
    _fmt_history,
    _state_key,
    _visible_history,
    build_tool_registry,
)
from burl.harness.tool_loop_native import NativeHarness, NativeModelCallable
from burl.harness.trace import BurlTrace
from burl.tools import engine as engine_tools
from forge.oracle.declarations import DOUBLES_SUIT, NOTRUMP
from forge.oracle.tables import DOMINO_COUNT_POINTS, is_in_called_suit


# --------------------------------------------------------------------------- #
# Tool schemas (JSON Schema). Rendered by the chat template into Gemma's       #
# native <|tool>…<tool|> block. Keep param descriptions terse — the schema     #
# already encodes types.                                                       #
# --------------------------------------------------------------------------- #


TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "is_legal",
            "description": "Check whether playing domino_id is legal for me right now.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domino_id": {"type": "integer", "description": "0..27"},
                },
                "required": ["domino_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "is_trump",
            "description": "Return True if domino_id belongs to the declared trump suit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "domino_id": {"type": "integer", "description": "0..27"},
                },
                "required": ["domino_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unseen",
            "description": "Return the sorted list of domino_ids not yet played and not in my hand.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "void_audit",
            "description": (
                "Return True if player_seat is known to be void in this suit "
                "based on public play history."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_seat": {
                        "type": "integer",
                        "description": "0=me, 1=left opp, 2=partner, 3=right opp",
                    },
                    "suit": {
                        "type": "integer",
                        "description": "0..6 pip suit, 7 = called suit",
                    },
                },
                "required": ["player_seat", "suit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trump_declared",
            "description": (
                "Return the declared trump suit: one of blanks, ones, twos, "
                "threes, fours, fives, sixes, doubles-trump, doubles-suit, notrump."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "eq_outcome_distribution",
            "description": (
                "Run a rollout with N random assumptions for hidden hands and "
                "return the outcome distribution (mean, stdev, p_make, pdf_bins, "
                "quantiles) for playing `play`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "play": {"type": "integer", "description": "domino_id to evaluate"},
                    "n_samples": {"type": "integer", "default": 10},
                },
                "required": ["play"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "conditional_outcome",
            "description": (
                "Like eq_outcome_distribution but constraining rollouts to an "
                "assumption about a hidden player's hand. "
                "assumption shapes: "
                '{"player":abs_seat_int,"holds":domino_id_int} or '
                '{"player":abs_seat_int,"void_in_suit":suit_int}.'
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "play": {"type": "integer"},
                    "assumption": {
                        "type": "object",
                        "description": (
                            "{player, holds} or {player, void_in_suit} — abs seats 0..3, suits 0..7."
                        ),
                    },
                    "n_samples": {"type": "integer", "default": 10},
                },
                "required": ["play", "assumption"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_play",
            "description": (
                "Commit to your final play and end the decision. "
                "domino_id must be an integer from your hand (not a label "
                "like '6-0'). If the engine rejects it as illegal, you will "
                "get another turn with the rejection shown and may call "
                "commit_play again with a different domino."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "domino_id": {
                        "type": "integer",
                        "description": "0..27, and must be present in your hand.",
                    },
                },
                "required": ["domino_id"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Prompt construction — system + user split, native format.                    #
# --------------------------------------------------------------------------- #


_SYSTEM_PREAMBLE = (
    "You are Burl, a Texas 42 dominoes agent. Pick the next play.\n"
    "You have tools that describe the game state; call them as needed. "
    "The state tools answer WHAT IS the state — they never tell you WHAT TO "
    "DO. Reason with what they return.\n\n"
    "When you are ready to commit, call the `commit_play` tool with the "
    "integer domino_id from your hand. That ends the decision. If the engine "
    "rejects your commit as illegal, you get another turn with the rejection "
    "shown back to you and may call commit_play again."
)


# Authoritative rules reference — shared with LEM. Loaded once at import.
_PRIMER_PATH = (
    Path(__file__).resolve().parent.parent.parent / "lem" / "rules" / "primer.md"
)
_PRIMER = _PRIMER_PATH.read_text()


_COMMIT_INSTRUCTION = (
    "You have not called a tool or committed. "
    "Call `commit_play` with the integer domino_id from your hand to end the "
    "decision (for example: commit_play(domino_id=21))."
)


def _fmt_domino(d: int) -> str:
    return f"{d}({_DOMINO_LABELS[d]})"


def _fmt_domino_list(dominoes: Iterable[int]) -> str:
    parts = [_fmt_domino(int(d)) for d in dominoes]
    return ", ".join(parts) if parts else "(none)"


def _trumps_under_declaration(decl_id: int) -> list[int]:
    """Return the domino ids that are trumps under this declaration.

    Empty for notrump (no trump power) and doubles-suit (doubles are their
    own suit but do not rank above other suits).
    """
    if decl_id in (NOTRUMP, DOUBLES_SUIT):
        return []
    return [d for d in range(28) if is_in_called_suit(d, decl_id)]


def _count_dominoes_remaining(played: frozenset[int]) -> tuple[list[int], list[int]]:
    """Return (five_pointers, ten_pointers) count dominoes not yet played.

    The double-six set has three 5-pt dominoes (5-0, 4-1, 3-2) and two 10-pt
    dominoes (5-5, 6-4) — 35 count points total across the five pieces.
    """
    fives: list[int] = []
    tens: list[int] = []
    for d in range(28):
        if d in played:
            continue
        pts = DOMINO_COUNT_POINTS[d]
        if pts == 5:
            fives.append(d)
        elif pts == 10:
            tens.append(d)
    return fives, tens


def _render_42_framing(
    game_state: Any,
    me_abs: int,
    hand: list[int],
) -> str:
    """42-specific framing block for the system prompt.

    Surfaces partnership, offensive/defensive role, the bid target, count
    dominoes still in play, trump membership, and the in-hand capture score.
    The goal is to prime Gemma to reason in 42 terms ('my partner', 'set
    the bidders', 'count dominoes loose') rather than generic card-game
    terms ('draw', 'establish a high card').
    """
    partner_abs = (me_abs + 2) % 4
    left_opp_abs = (me_abs + 1) % 4
    right_opp_abs = (me_abs + 3) % 4
    my_team = me_abs % 2
    opp_team = 1 - my_team

    decl_id = int(game_state.decl_id)
    decl_name = engine_tools.trump_declared(game_state)

    bidder = getattr(game_state, "bidder", -1)
    bid_state = getattr(game_state, "bid_state", None)
    high_bid = getattr(bid_state, "high_bid", 0) if bid_state is not None else 0
    target = int(high_bid) if high_bid and high_bid >= 30 else 30

    if bidder is not None and bidder >= 0:
        bidder_team = int(bidder) % 2
        if bidder_team == my_team:
            role_line = (
                f"You are on OFFENSE. Your team (Team {my_team}, seats "
                f"{me_abs} and {partner_abs}) bid {target} and must capture at "
                f"least {target} count to make the bid."
            )
        else:
            role_line = (
                f"You are on DEFENSE. Team {bidder_team} (seats "
                f"{int(bidder)} and {(int(bidder) + 2) % 4}) bid {target}. "
                f"Your job is to SET them — keep them below {target} count."
            )
        bidder_line = (
            f"Bidder: seat {int(bidder)} (Team {bidder_team}). Bid: {target} count."
        )
    else:
        role_line = (
            "Bid information unavailable; assume offense is whichever team "
            "declared trump."
        )
        bidder_line = "Bidder: unknown."

    team_points = getattr(game_state, "team_points", None)
    if team_points is not None and len(team_points) == 2:
        my_score = int(team_points[my_team])
        opp_score = int(team_points[opp_team])
        score_line = (
            f"Hand score so far: your team has captured {my_score} count; "
            f"Team {opp_team} has captured {opp_score} count. "
            f"Total count in a hand is 42 (35 in dominoes + 7 trick points)."
        )
    else:
        score_line = ""

    fives, tens = _count_dominoes_remaining(game_state.played)
    remaining_count_pts = 5 * len(fives) + 10 * len(tens)
    count_line = (
        f"Count dominoes (still in play): "
        f"5-pointers {_fmt_domino_list(fives)}; "
        f"10-pointers {_fmt_domino_list(tens)}. "
        f"{remaining_count_pts} count points loose; each trick is also worth 1."
    )

    if decl_id == NOTRUMP:
        trump_line = (
            "Declaration is NOTRUMP — no suit has trump power. Each trick is "
            "won by the highest domino in the led suit; doubles rank highest "
            "of their pip. Lead strategy centers on forcing sluffs and "
            "capturing count cleanly."
        )
    elif decl_id == DOUBLES_SUIT:
        trump_line = (
            "Declaration is DOUBLES-SUIT — the seven doubles form their own "
            "suit with no trump power. Leading a double pulls doubles; any "
            "non-double leads its normal pip suit. No suit overpowers another."
        )
    else:
        trumps = _trumps_under_declaration(decl_id)
        trumps_in_hand = [d for d in trumps if d in hand]
        trumps_played = [d for d in trumps if d in game_state.played]
        trumps_unseen = [
            d for d in trumps
            if d not in hand and d not in game_state.played
        ]
        trump_line = (
            f"Trumps ({decl_name}) — 7 dominoes total: {_fmt_domino_list(trumps)}. "
            f"In your hand: {_fmt_domino_list(trumps_in_hand)}. "
            f"Already played: {_fmt_domino_list(trumps_played)}. "
            f"Still unseen (with partner or opponents): "
            f"{_fmt_domino_list(trumps_unseen)}."
        )

    lines = [
        "=== Texas 42 framing ===",
        (
            f"Teams: seats 0 and 2 are Team 0; seats 1 and 3 are Team 1. "
            f"You sit at seat {me_abs} on Team {my_team}. Your partner is at "
            f"seat {partner_abs}. Opponents sit at seats {left_opp_abs} "
            f"(left, plays after you) and {right_opp_abs} (right, plays "
            f"before you on your lead). Partner's count is your count — "
            f"you win and lose as a pair."
        ),
        bidder_line,
        role_line,
    ]
    if score_line:
        lines.append(score_line)
    lines.extend([count_line, trump_line])
    return "\n".join(lines)


def render_native_messages(
    game_state: Any,
    hand: Iterable[int],
    visible_history: Iterable[tuple[int, ...]],
) -> tuple[str, str]:
    """Return (system_content, user_content) for the native chat template."""
    decl = engine_tools.trump_declared(game_state)
    leader = getattr(game_state, "trick_leader", None)
    if leader is None:
        leader = getattr(game_state, "leader", 0)
    current_trick = game_state.current_trick
    me_abs = (int(leader) + len(current_trick)) % 4
    trick_no = len(game_state.play_history) // 4 + 1
    position_in_trick = len(current_trick) + 1

    trick_ids: list[int] = []
    if current_trick:
        first = current_trick[0]
        if isinstance(first, int):
            trick_ids = list(current_trick)
        else:
            trick_ids = [d for _p, d in current_trick]

    trick_txt = (
        "current trick so far: "
        + ", ".join(f"{d}({_DOMINO_LABELS[d]})" for d in trick_ids)
        if trick_ids
        else "current trick: you are leading"
    )
    hand_list = list(hand)

    system = (
        _SYSTEM_PREAMBLE
        + "\n\n# Texas 42 rules reference (authoritative)\n\n"
        + _PRIMER
        + "\n# Current decision — 42-aware context\n\n"
        + _render_42_framing(game_state, me_abs, hand_list)
    )

    user = (
        f"declaration: {decl}\n"
        f"your seat (absolute): {me_abs}\n"
        f"trick: {trick_no}   position in trick: {position_in_trick}/4\n"
        f"your hand: {_fmt_hand(hand_list)}\n"
        f"{trick_txt}\n"
        f"visible history: {_fmt_history(visible_history)}\n\n"
        "Decide what to play. Use tools as needed, then commit."
    )
    return system, user


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #


def run_decision_native(
    game_state: Any,
    native_model: NativeModelCallable,
    max_turns: int = 8,
    max_retries: int = 3,
) -> BurlTrace:
    """Native-format counterpart to ``agent_runner.run_decision``.

    ``native_model`` is ``(messages, tools) -> str``; the Modal adapter wraps
    ``GemmaServerNative.generate_native.remote``.
    """
    me_abs = _current_player(game_state)
    hand_remaining = [d for d in game_state.hands[me_abs] if d not in game_state.played]
    history = _visible_history(game_state)

    system_content, user_content = render_native_messages(
        game_state, hand_remaining, history,
    )
    tools = build_tool_registry(game_state_provider=lambda: game_state)

    harness = NativeHarness(
        model_callable=native_model,
        tools=tools,
        tool_schemas=TOOL_SCHEMAS,
        is_legal_fn=lambda s, d: engine_tools.is_legal(s, int(d)),
        commit_instruction=_COMMIT_INSTRUCTION,
        max_turns=max_turns,
        max_retries=max_retries,
    )
    return harness.run(
        game_state=game_state,
        system_content=system_content,
        user_content=user_content,
        state_key=_state_key(game_state),
    )


# --------------------------------------------------------------------------- #
# Self-test: stub native model.                                                #
# --------------------------------------------------------------------------- #


def _selftest(seed: int = 2026) -> None:
    import random as _r

    from forge.zeb.game import apply_action, legal_actions, new_game

    # Build the same trick-6 state agent_runner._selftest uses.
    state = new_game(seed=seed, skip_bidding=True)
    rng = _r.Random(seed)
    while len(state.play_history) < 20:
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))
    while state.current_trick and len(state.play_history) < 28:
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))

    me = _current_player(state)
    remaining = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in remaining if engine_tools.is_legal(state, d)[0]]
    assert legal, "no legal plays at trick-6 state"
    target = legal[0]
    illegal = next(d for d in range(28) if d not in remaining)

    # Prompt-framing sanity: system must carry 42 vocabulary.
    system_text, user_text = render_native_messages(
        state, remaining, _visible_history(state),
    )
    for token in (
        "Texas 42 rules reference",
        "count dominoes",
        "Texas 42 framing",
        "Team",
        "partner",
        "Bidder",
        "Count dominoes",
    ):
        assert token in system_text, f"missing {token!r} in system prompt"
    assert "your hand:" in user_text, "user prompt lost hand line"

    completions = iter([
        '<|tool_call>{"name":"trump_declared","arguments":{}}<tool_call|>\n'
        '<|tool_call>{"name":"unseen","arguments":{}}<tool_call|>\n'
        f'<|tool_call>{{"name":"is_legal","arguments":{{"domino_id":{target}}}}}<tool_call|>',
        f'<|tool_call>call:commit_play{{domino_id:{illegal}}}<tool_call|>',
        f'<|tool_call>call:commit_play{{domino_id:{target}}}<tool_call|>',
    ])

    def stub_native(messages: list[dict], tools: list[dict]) -> str:
        assert tools, "schemas must be passed through"
        return next(completions)

    trace = run_decision_native(state, stub_native, max_turns=8, max_retries=3)
    tool_calls = [tc for turn in trace.turns for tc in turn.tool_calls]
    assert trace.final_play == target, (trace.final_play, target)
    assert trace.n_retries == 1, trace.n_retries
    assert len(tool_calls) >= 3, len(tool_calls)
    tool_names = {tc.tool_name for tc in tool_calls}
    assert {"trump_declared", "unseen", "is_legal"}.issubset(tool_names), tool_names

    rt = BurlTrace.from_json(trace.to_json())
    assert rt.final_play == trace.final_play

    print(f"[agent_runner_native selftest seed={seed}]")
    print(f"  me(abs)={me} remaining={remaining} target={target} illegal={illegal}")
    print(f"  turns={len(trace.turns)} tool_calls={len(tool_calls)} "
          f"retries={trace.n_retries} final={trace.final_play}")
    print(f"  tool names: {sorted(tool_names)}")
    print(f"  system prompt ({len(system_text)} chars):")
    for line in system_text.splitlines():
        print(f"    {line}")
    print("[agent_runner_native selftest] OK: 42-framed prompt renders, "
          "native schemas threaded through, retry + trace round-trip.")


if __name__ == "__main__":
    _selftest()
