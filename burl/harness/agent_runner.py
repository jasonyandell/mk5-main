"""Burl agent runner — stitch tools + model + retry loop into `run_decision`.

Model-agnostic: `run_decision` accepts a `Callable[[str], str]`, so a stub,
a local llama.cpp binding, and a remote Modal endpoint all plug in unchanged.
Gemma chat-template wrapping (if any) happens in the caller; this file stays
provider-neutral so the Move 3 grader can swap adapters without touching it.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable, Iterable

from burl.harness.tool_loop import Harness, ModelCallable, ToolProtocol
from burl.harness.trace import BurlTrace
from burl.tools import engine as engine_tools
from burl.tools import eq_distribution as eq_tools
from burl.tools import meta_tools as meta_tool_mod
from burl.tools import rules as rule_tools
from burl.tools.eq_distribution import OutcomeDistribution
from forge.oracle.tables import DOMINO_HIGH, DOMINO_LOW


# --------------------------------------------------------------------------- #
# Tool wrappers                                                                #
# --------------------------------------------------------------------------- #


class _Tool:
    """Minimal ToolProtocol shim — binds a name to a `(game_state, **kw)` fn."""

    def __init__(self, name: str, fn: Callable[..., Any]):
        self.name = name
        self._fn = fn

    def __call__(self, game_state: Any, **kwargs: Any) -> Any:
        return self._fn(game_state, **kwargs)


def _outcome_to_dict(od: OutcomeDistribution) -> dict[str, Any]:
    """JSON-safe view of OutcomeDistribution.

    The 85-bin PDF is the point of the tool — keep it, but round to 6 decimals
    so numeric noise doesn't explode trace diffs during STaR filtering.
    """
    return {
        "play": int(od.play),
        "mean": round(float(od.mean), 3),
        "stdev": round(float(od.stdev), 3),
        "p_make": round(float(od.p_make), 4),
        "n_samples": int(od.n_samples),
        "min_q": round(float(od.min_q), 1),
        "max_q": round(float(od.max_q), 1),
        "percentiles": {str(k): round(float(v), 1) for k, v in od.percentiles.items()},
        "is_offense": bool(od.is_offense),
        "pdf_bins": [round(float(x), 6) for x in od.pdf_bins.tolist()],
        "distribution_shape": str(od.distribution_shape),
        "modes": [
            {"center": round(float(m["center"]), 2), "mass": round(float(m["mass"]), 4)}
            for m in od.modes
        ],
        "gap_between_modes": round(float(od.gap_between_modes), 2),
        "suggested_counterfactuals": list(od.suggested_counterfactuals),
        "sampling_mode": str(od.sampling_mode),
    }


def _is_legal_tool(game_state: Any, domino_id: int) -> dict[str, Any]:
    legal, reason = engine_tools.is_legal(game_state, int(domino_id))
    return {"legal": bool(legal), "reason": reason}


def _is_trump_tool(game_state: Any, domino_id: int) -> bool:
    return bool(engine_tools.is_trump(game_state, int(domino_id)))


def _unseen_tool(game_state: Any) -> set[int]:
    return engine_tools.unseen(game_state)


def _void_audit_tool(game_state: Any, player_seat: int, suit: int) -> bool:
    return bool(engine_tools.void_audit(game_state, int(player_seat), int(suit)))


def _trump_declared_tool(game_state: Any) -> str:
    return engine_tools.trump_declared(game_state)


def _eq_outcome_tool(
    game_state: Any, play: int, n_samples: int = 10,
) -> dict[str, Any]:
    od = eq_tools.eq_outcome_distribution(
        game_state, int(play), n_samples=int(n_samples),
    )
    return _outcome_to_dict(od)


def _conditional_outcome_tool(
    game_state: Any,
    play: int,
    assumption: dict[str, Any],
    n_samples: int = 10,
) -> dict[str, Any]:
    od = eq_tools.conditional_outcome(
        game_state, int(play), assumption, n_samples=int(n_samples),
    )
    return _outcome_to_dict(od)


def _what_would_change_my_mind_tool(
    game_state: Any,
    play: int,
    n_samples_per_probe: int = 5,
    top_k: int = 5,
) -> dict[str, Any]:
    return meta_tool_mod.what_would_change_my_mind(
        game_state,
        int(play),
        n_samples_per_probe=int(n_samples_per_probe),
        top_k=int(top_k),
    )


# Rules-as-tools — gated by `enable_rules_tools`. Each wrapper just coerces
# the domino ids to int; rule_tools already returns JSON primitives.


def _count_dominoes_remaining_tool(game_state: Any) -> dict[str, Any]:
    return rule_tools.count_dominoes_remaining(game_state)


def _trick_winner_if_tool(game_state: Any, domino_id: int) -> dict[str, Any]:
    return rule_tools.trick_winner_if(game_state, int(domino_id))


def _what_beats_what_tool(
    game_state: Any,
    domino_a: int,
    domino_b: int,
    lead_domino: int | None = None,
) -> dict[str, Any]:
    lead = int(lead_domino) if lead_domino is not None else None
    return rule_tools.what_beats_what(
        game_state, int(domino_a), int(domino_b), lead_domino=lead,
    )


def _contract_progress_tool(game_state: Any) -> dict[str, Any]:
    return rule_tools.contract_progress(game_state)


def build_tool_registry(
    game_state_provider: Callable[[], Any] | None = None,
    enable_rules_tools: bool = False,
) -> dict[str, ToolProtocol]:
    """Return the allow-list tools as `ToolProtocol` callables.

    The Harness already threads `game_state` to each tool via the
    `ToolProtocol.__call__(state, **kwargs)` contract, so the registry itself
    holds no state. `game_state_provider` is kept on the signature as the
    obvious seam for callers that want late binding (e.g. the Move 3 grader
    snapshotting state per decision) without changing the tool functions.

    When ``enable_rules_tools`` is True, the four rules-as-tools
    (``count_dominoes_remaining``, ``trick_winner_if``, ``what_beats_what``,
    ``contract_progress``) are registered alongside the existing seven. This
    is the iter-3 lever; iter-2 keeps it False so the verbosity-blend
    experiment stays single-variable.
    """
    del game_state_provider  # see docstring
    registry: dict[str, ToolProtocol] = {
        "is_legal": _Tool("is_legal", _is_legal_tool),
        "is_trump": _Tool("is_trump", _is_trump_tool),
        "unseen": _Tool("unseen", _unseen_tool),
        "void_audit": _Tool("void_audit", _void_audit_tool),
        "trump_declared": _Tool("trump_declared", _trump_declared_tool),
        "eq_outcome_distribution": _Tool("eq_outcome_distribution", _eq_outcome_tool),
        "conditional_outcome": _Tool("conditional_outcome", _conditional_outcome_tool),
        "what_would_change_my_mind": _Tool(
            "what_would_change_my_mind", _what_would_change_my_mind_tool,
        ),
    }
    if enable_rules_tools:
        registry.update(
            {
                "count_dominoes_remaining": _Tool(
                    "count_dominoes_remaining", _count_dominoes_remaining_tool,
                ),
                "trick_winner_if": _Tool(
                    "trick_winner_if", _trick_winner_if_tool,
                ),
                "what_beats_what": _Tool(
                    "what_beats_what", _what_beats_what_tool,
                ),
                "contract_progress": _Tool(
                    "contract_progress", _contract_progress_tool,
                ),
            }
        )
    return registry


# --------------------------------------------------------------------------- #
# Prompt rendering                                                             #
# --------------------------------------------------------------------------- #


_DOMINO_LABELS = tuple(f"{DOMINO_HIGH[i]}-{DOMINO_LOW[i]}" for i in range(28))


def _fmt_domino(d: int) -> str:
    return f"{d}({_DOMINO_LABELS[d]})"


def _fmt_hand(hand: Iterable[int]) -> str:
    parts = [f"id={d} ({_DOMINO_LABELS[d]})" for d in hand]
    return ", ".join(parts) or "(empty)"


def _fmt_history(history: Iterable[tuple[int, ...]]) -> str:
    parts = [f"seat{int(entry[0])}:{_fmt_domino(int(entry[1]))}" for entry in history]
    return ", ".join(parts) if parts else "(no plays yet)"


_TOOL_MENU = """\
Available tools. `domino_id` is an integer 0..27; `play` is also an integer
domino_id. The "(6-0)" style is a human-readable label — never pass the label
as an argument. Emit each call as `<tool>{"name":"NAME","args":{...}}</tool>`:

  is_legal(domino_id: int)                 -> {legal: bool, reason: str}
  is_trump(domino_id: int)                 -> bool
  unseen()                                 -> sorted domino_ids not in your hand, not played
  void_audit(player_seat: int, suit: int)  -> bool
      player_seat: 0=you 1=leftopp 2=partner 3=rightopp
      suit: 0..6 pip, 7 called-suit
  trump_declared()                         -> one of: blanks ones twos threes fours fives
                                              sixes doubles-trump doubles-suit notrump
  eq_outcome_distribution(play: int, n_samples: int = 10)        -> outcome dict
  conditional_outcome(play: int, assumption: dict, n_samples: int = 10) -> outcome dict
      assumption shapes:
        {"player": abs_seat_int, "holds": domino_id_int}
        {"player": abs_seat_int, "void_in_suit": suit_int}
  what_would_change_my_mind(play: int, n_samples_per_probe: int = 5, top_k: int = 5)
      -> {play, unconditional_mean, assumptions: [{player, holds,
         conditional_mean, shift, rationale}, ...]} — ranks unseen-world
         assumptions by how much they swing E[Q] of `play`.

Example call: <tool>{"name":"is_legal","args":{"domino_id":21}}</tool>

outcome dict fields: play, mean, stdev, p_make, n_samples, min_q, max_q,
percentiles{10,25,50,75,90}, is_offense, pdf_bins[85].
"""

_PROTOCOL = """\
Output protocol (one turn per generation):
  <think>...reasoning, optional, may repeat...</think>
  <tool>{"name":"NAME","args":{...}}</tool>    # zero or more
  <commit>21</commit>                           # integer domino_id from your hand

The value inside <commit>...</commit> MUST be an actual integer domino_id from
your hand — for example `<commit>21</commit>`, NOT `<commit>21(6-0)</commit>`
and NOT the literal `<commit>INT</commit>`. Exactly one <commit> ends the
decision. If the engine rejects your commit as illegal, you get another turn
with the rejection visible in the next prompt. Tools answer WHAT IS the state;
they never tell you WHAT TO DO. Reason with what they return.
"""


def render_system_prompt(
    game_state: Any,
    hand: Iterable[int],
    visible_history: Iterable[tuple[int, ...]],
) -> str:
    """Describe game state + tool menu + XML emission protocol.

    Model-agnostic text prompt — no chat template. Callers wrap with whatever
    template the backend wants (Gemma ChatML, llama inst, etc.).
    """
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
        f"current trick so far: {', '.join(_fmt_domino(d) for d in trick_ids)}"
        if trick_ids else "current trick: you are leading"
    )

    hand_list = list(hand)

    return (
        "You are Burl, a Texas 42 dominoes agent. Decide the next play.\n\n"
        f"declaration: {decl}\n"
        f"your seat (absolute): {me_abs}\n"
        f"trick: {trick_no}   position in trick: {position_in_trick}/4\n"
        f"your hand: {_fmt_hand(hand_list)}\n"
        f"{trick_txt}\n"
        f"visible history: {_fmt_history(visible_history)}\n\n"
        + _TOOL_MENU
        + "\n"
        + _PROTOCOL
        + "\nBegin."
    )


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #


def _current_player(game_state: Any) -> int:
    leader = getattr(game_state, "trick_leader", None)
    if leader is None:
        leader = getattr(game_state, "leader", 0)
    return (int(leader) + len(game_state.current_trick)) % 4


def _visible_history(game_state: Any) -> list[tuple[int, int]]:
    return [(int(entry[0]), int(entry[1])) for entry in game_state.play_history]


def _state_key(game_state: Any) -> str:
    """Stable shortish id — good enough for trace dedup / cache keys."""
    leader = getattr(game_state, "trick_leader", None)
    if leader is None:
        leader = getattr(game_state, "leader", 0)
    parts = (
        int(game_state.decl_id),
        int(leader),
        int(getattr(game_state, "bidder", 0)),
        len(game_state.played),
        len(game_state.play_history),
    )
    prefix = "-".join(str(p) for p in parts)
    hands_repr = repr(tuple(tuple(h) for h in game_state.hands)).encode()
    h = hashlib.sha1(hands_repr).hexdigest()[:8]
    return f"{prefix}:{h}"


def run_decision(
    game_state: Any,
    model_callable: ModelCallable,
    max_turns: int = 8,
    max_retries: int = 3,
) -> BurlTrace:
    """Run one Burl decision end-to-end: prompt -> tool loop -> legal commit.

    Returns a populated `BurlTrace`. Raises `burl.harness.retry.RetryExhausted`
    if no legal play appears within `max_retries` (Move 3 grades loud failures
    rather than silently substituting a fallback).
    """
    me_abs = _current_player(game_state)
    hand_remaining = [d for d in game_state.hands[me_abs] if d not in game_state.played]
    history = _visible_history(game_state)

    prompt = render_system_prompt(game_state, hand_remaining, history)
    tools = build_tool_registry(game_state_provider=lambda: game_state)

    harness = Harness(
        model_callable=model_callable,
        tools=tools,
        is_legal_fn=lambda s, d: engine_tools.is_legal(s, int(d)),
        max_turns=max_turns,
        max_retries=max_retries,
    )
    return harness.run(
        game_state=game_state,
        decision_prompt=prompt,
        state_key=_state_key(game_state),
    )


# --------------------------------------------------------------------------- #
# Self-test: no network, no GPU, no Gemma. Uses a scripted stub model.         #
# --------------------------------------------------------------------------- #


def _build_trick6_state(seed: int) -> Any:
    """Deterministic random-legal rollout to the sixth trick's lead."""
    import random as _r

    from forge.zeb.game import apply_action, legal_actions, new_game

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
    return state


def _selftest(seed: int = 2026) -> None:
    state = _build_trick6_state(seed)
    me = _current_player(state)
    remaining = [d for d in state.hands[me] if d not in state.played]
    legal = [d for d in remaining if engine_tools.is_legal(state, d)[0]]
    assert legal, f"no legal plays at trick-6 state for seed={seed}"
    target = legal[0]
    illegal = next(d for d in range(28) if d not in remaining)

    # Scripted, deterministic stub model. 3 turns: tool-use, illegal, legal.
    completions = iter([
        "<think>Probe the state: declaration, unseen, legality of a candidate.</think>"
        '<tool>{"name":"trump_declared","args":{}}</tool>'
        '<tool>{"name":"unseen","args":{}}</tool>'
        f'<tool>{{"name":"is_legal","args":{{"domino_id":{target}}}}}</tool>',

        f"<think>Probe engine with {illegal} — expected illegal.</think>"
        f"<commit>{illegal}</commit>",

        f"<think>Rejected as expected. Commit the legal target {target}.</think>"
        f"<commit>{target}</commit>",
    ])

    def stub_model(prompt: str) -> str:
        return next(completions)

    registry = build_tool_registry()
    assert len(registry) == 8, f"expected 8 tools, got {len(registry)}: {list(registry)}"

    trace = run_decision(state, stub_model, max_turns=8, max_retries=3)

    tool_calls = [tc for turn in trace.turns for tc in turn.tool_calls]
    assert trace.final_play == target, (trace.final_play, target)
    assert trace.n_retries == 1, trace.n_retries
    assert len(tool_calls) >= 2, len(tool_calls)
    assert all(tc.ok for tc in tool_calls), [tc.error for tc in tool_calls if not tc.ok]

    roundtrip = BurlTrace.from_json(trace.to_json())
    assert roundtrip.final_play == target
    assert roundtrip.n_retries == trace.n_retries
    assert len(roundtrip.turns) == len(trace.turns)

    print(f"[agent_runner selftest seed={seed}]")
    print(f"  registry tools ({len(registry)}): {sorted(registry)}")
    print(f"  me(abs)={me}  remaining={remaining}")
    print(f"  legal_plays={legal}  target={target}  illegal_probe={illegal}")
    print(f"  turns={len(trace.turns)}  tool_calls={len(tool_calls)}"
          f"  retries={trace.n_retries}  final_play={trace.final_play}")
    print(f"  state_key={trace.game_state_key}")
    print(f"  tokens_in={trace.tokens_in}  tokens_out={trace.tokens_out}")
    print(f"  tool call trace:")
    for tc in tool_calls:
        result_preview = repr(tc.result)
        if len(result_preview) > 70:
            result_preview = result_preview[:67] + "..."
        print(f"    - {tc.tool_name}({tc.args}) -> {result_preview}")
    print(f"  trace JSON length: {len(trace.to_json())} bytes")
    print("[agent_runner selftest] OK: 7 tools wired, retry counted, trace roundtrips")


if __name__ == "__main__":
    _selftest()
