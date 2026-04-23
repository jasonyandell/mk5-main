"""Hard-gated HATEOAS harness for wax_museum.

Drives one decision end-to-end:

    1. Render system + user prompts (reuse agent_runner_native.render_native_messages).
    2. Start with GateState.INITIAL (only explore_game visible).
    3. Each turn: call model(messages, tools=menu_for(state)), parse completion,
       execute tool calls, advance state.
    4. Gate enforcement: commit_play is only in the schema when state==AFTER_PROBE.
       If the model emits commit_play earlier (native leak), we reject softly and
       nudge.
    5. Bail detection: turn 1 produces zero tool calls AND <200 chars of thought
       => decision is marked bailed and the harness returns early.
    6. Engine validation of commit_play via legality check + retry loop.

Events are emitted through an optional ``on_event`` callback so the pilot can
build tail-able logs without the harness owning filesystem policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from burl.harness.agent_runner import _current_player, _state_key, _visible_history
from burl.harness.agent_runner_native import render_native_messages
from burl.harness.tool_loop_native import parse_native_completion
from burl.harness.trace import BurlTrace, ToolCall, TurnStep
from burl.tools import engine as engine_tools
from burl.wax_museum.schemas import (
    COMMIT_PLAY,
    GateState,
    advance,
    menu_for,
    menu_names,
    next_actions_unchanged,
)
from burl.wax_museum.tools import WaxContext, build_registry


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


BAIL_MIN_THOUGHT_CHARS = 200
"""Turn 1 with no tool call and <N chars of thought is a bail signal."""


NativeModelCallable = Callable[[list[dict], list[dict]], str]
"""(messages, tool_schemas) -> raw completion text."""

EventCallback = Callable[[dict], None]
ToolResultCallback = Callable[[str, dict, Any], None]
"""(tool_name, args, full_result) — fires immediately after a tool call
succeeds, BEFORE the harness emits its truncated ``tool_call`` event.
Callers use this to stream full payloads into live logs without monkey-patching."""


# --------------------------------------------------------------------------- #
# Result                                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class WaxResult:
    trace: BurlTrace
    bailed: bool = False
    bail_reason: str = ""
    n_turns: int = 0
    tool_call_sequence: list[str] = field(default_factory=list)
    probed: bool = False
    gated_commit_leaks: int = 0   # commit_play emitted before a probe ran


# --------------------------------------------------------------------------- #
# Harness                                                                      #
# --------------------------------------------------------------------------- #


def run_decision_waxed(
    game_state: Any,
    native_model: NativeModelCallable,
    max_turns: int = 8,
    max_retries: int = 3,
    on_event: EventCallback | None = None,
    on_tool_result: ToolResultCallback | None = None,
    oracle: Any = None,
    parse_completion: Callable[[str], tuple[str, list[tuple[str, dict]], int | None]] | None = None,
    tool_response_style: str = "gemma_native",
    system_prompt_transform: Callable[[str], str] | None = None,
    preload_tool_calls: list[tuple[str, dict]] | None = None,
    menu_override: Callable[[Any], list[dict]] | None = None,
) -> WaxResult:
    """Run one decision through the gated HATEOAS loop.

    ``parse_completion`` overrides the default Gemma parser. Plug in
    ``burl.wax_museum.qwen_parser.parse_qwen_completion`` for Qwen3.6.

    ``tool_response_style`` controls how tool responses are serialized into
    the message list, because chat templates are model-specific:

    - ``"gemma_native"`` (default): responses live on the assistant message as
      ``tool_responses=[{name, response}]``. Gemma 4's chat template drops
      ``role="tool"`` entirely; this is the native shape it was trained on.
    - ``"role_tool"``: each response is a ``{"role": "tool", ...}`` message
      following the assistant turn. Qwen3.6 and most OpenAI-template clones
      accept this; Gemma drops it silently.
    """
    if on_event is None:
        on_event = lambda _e: None
    if parse_completion is None:
        parse_completion = parse_native_completion

    me_abs = _current_player(game_state)
    hand_remaining = [
        d for d in game_state.hands[me_abs] if d not in game_state.played
    ]
    history = _visible_history(game_state)

    # Prompts: reuse iter-3-rules shape (primer + 42 framing) so the gate is the
    # only changed variable vs the last known-working baseline.
    system_content, user_content = render_native_messages(
        game_state,
        hand_remaining,
        history,
        enable_rules_tools=False,
        enable_primer=True,
    )
    system_content = _append_gate_instructions(system_content)
    if system_prompt_transform is not None:
        system_content = system_prompt_transform(system_content)

    ctx = WaxContext(game_state=game_state, me_abs=me_abs, oracle=oracle)
    registry = build_registry(ctx)

    messages: list[dict] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    # Optional preload: run one or more tool calls BEFORE the model's first
    # turn, and inject the results as a synthetic assistant message using the
    # same tool_calls / tool_responses shape the harness uses for real turns.
    # Use-case: "Here, I already ran belief_trajectory for you — reason over
    # this." Preload tools MUST be in the INITIAL menu OR be explicitly
    # approved by the caller via ``menu_override`` (see variant F).
    if preload_tool_calls:
        preload_calls_structured: list[dict] = []
        preload_resps_structured: list[dict] = []
        for (tname, targs) in preload_tool_calls:
            fn = registry.get(tname)
            if fn is None:
                on_event({"evt": "preload_skip", "tool": tname,
                          "reason": "unknown tool"}) if on_event else None
                continue
            try:
                presult = fn(**targs)
                if on_tool_result is not None:
                    try:
                        on_tool_result(tname, dict(targs), presult)
                    except Exception:
                        pass
                if isinstance(presult, dict) and "prose" in presult:
                    ptext = presult["prose"]
                elif isinstance(presult, dict):
                    ptext = json.dumps(presult, default=_json_default, separators=(",", ":"))
                else:
                    ptext = str(presult)
            except Exception as e:
                ptext = f"ERROR: {e}"
            preload_calls_structured.append({
                "type": "function",
                "function": {"name": tname, "arguments": dict(targs)},
            })
            preload_resps_structured.append({"name": tname, "response": ptext})
        if preload_calls_structured:
            preload_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": preload_calls_structured,
            }
            if tool_response_style == "gemma_native":
                preload_msg["tool_responses"] = preload_resps_structured
                messages.append(preload_msg)
            else:
                messages.append(preload_msg)
                for tr in preload_resps_structured:
                    messages.append({
                        "role": "tool",
                        "name": tr.get("name", "unknown"),
                        "content": tr.get("response", ""),
                    })
    trace = BurlTrace(
        game_state_key=_state_key(game_state),
        decision_prompt=f"[SYSTEM]\n{system_content}\n\n[USER]\n{user_content}",
    )
    result = WaxResult(trace=trace)

    state = GateState.INITIAL
    last_explored_play: int | None = None
    committed: int | None = None
    attempted_commits: list[int] = []

    for turn_idx in range(1, max_turns + 1):
        schemas = (
            menu_override(state) if menu_override is not None else menu_for(state)
        )
        on_event({
            "evt": "turn_start",
            "turn": turn_idx,
            "menu": menu_names(state),
            "state": state.value,
        })

        trace.tokens_in += len(json.dumps(messages, default=str))
        completion = native_model(messages, schemas)
        trace.tokens_out += len(completion)

        thought, tool_specs, native_commit = parse_completion(completion)

        on_event({
            "evt": "completion",
            "turn": turn_idx,
            "n_chars": len(completion),
            "thought_preview": thought[:400],
            "full_completion": completion,   # persisted to file by the pilot
            "tool_calls_parsed": [(n, a) for n, a in tool_specs],
            "native_commit": native_commit,
        })

        # We will build ONE assistant message carrying the thought + structured
        # tool_calls + structured tool_responses for this turn. Gemma 4's chat
        # template (a) ignores role='tool' messages entirely, and (b) expects
        # tool responses as `tool_responses=[{name, response}]` siblings of
        # `tool_calls` on the assistant turn. Anything else renders as
        # `{value:None}` and the model hallucinates because the real response
        # literally never arrived. This is the native shape the model was
        # post-trained on.
        assistant_turn: dict[str, Any] = {
            "role": "assistant",
            "content": thought,
        }
        turn_tool_calls: list[dict] = []
        turn_tool_responses: list[dict] = []

        executed: list[ToolCall] = []

        # --- Bail check (turn 1 only) --- #
        if turn_idx == 1 and not tool_specs and native_commit is None:
            if len(thought.strip()) < BAIL_MIN_THOUGHT_CHARS:
                result.bailed = True
                result.bail_reason = (
                    f"turn 1: no tool call + thought {len(thought.strip())} chars "
                    f"(< {BAIL_MIN_THOUGHT_CHARS}); model not engaging with the gate"
                )
                on_event({"evt": "bail", "reason": result.bail_reason})
                break

        # --- Commit_play gate enforcement --- #
        # Gemma may emit commit_play even when the schema doesn't declare it
        # (native-format muscle memory). If state != AFTER_PROBE, reject softly.
        if native_commit is not None and state != GateState.AFTER_PROBE:
            result.gated_commit_leaks += 1
            nudge = (
                f"`commit_play` is not available yet. You must call "
                f"`probe_best_case(play=<int>)` or `probe_worst_case(play=<int>)` "
                f"at least once before committing. Current state: {state.value}. "
                f"Currently available tools: {menu_names(state)}."
            )
            # Record the rejected commit as a tool_call + tool_response on this
            # turn's assistant message so the model's next prompt shows
            # "I tried commit_play and the gate rejected it" in native shape.
            turn_tool_calls.append({
                "type": "function",
                "function": {
                    "name": "commit_play",
                    "arguments": {"domino_id": int(native_commit)},
                },
            })
            turn_tool_responses.append({
                "name": "commit_play", "response": f"REJECTED: {nudge}",
            })
            if tool_response_style == "gemma_native":
                if turn_tool_calls:
                    assistant_turn["tool_calls"] = turn_tool_calls
                if turn_tool_responses:
                    assistant_turn["tool_responses"] = turn_tool_responses
                messages.append(assistant_turn)
            else:  # role_tool
                if turn_tool_calls:
                    assistant_turn["tool_calls"] = turn_tool_calls
                messages.append(assistant_turn)
                for tr in turn_tool_responses:
                    messages.append({
                        "role": "tool",
                        "name": tr.get("name", "unknown"),
                        "content": tr.get("response", ""),
                    })
            on_event({
                "evt": "gate_reject",
                "turn": turn_idx,
                "tool": "commit_play",
                "reason": "commit_play pre-probe",
            })
            native_commit = None  # swallow it
            # No state transition; next turn will see the same menu.
            on_event({"evt": "turn_end", "turn": turn_idx})
            continue

        # --- Tool call filtering + execution --- #
        # Each parsed tool_call becomes a (tool_calls[i], tool_responses[i])
        # pair attached to this turn's assistant message. Gemma's template
        # renders that as `<|tool_call>call:NAME{args}<tool_call|>
        # <|tool_response>response:NAME{value:<|"|>prose<|"|>}<tool_response|>`
        # — the native shape the model was trained on. Anything else gets
        # silently dropped and the model hallucinates a response.
        allowed = {s["function"]["name"] for s in schemas}
        state_transitioned_this_turn = False

        def _record_call(
            name: str, args: dict, tc: ToolCall, response_text: str,
        ) -> None:
            turn_tool_calls.append({
                "type": "function",
                "function": {"name": name, "arguments": dict(args)},
            })
            turn_tool_responses.append({"name": name, "response": response_text})
            executed.append(tc)

        for name, args in tool_specs:
            if name not in allowed:
                err = (
                    f"{name} is not available in state {state.value}. "
                    f"Available: {sorted(allowed)}"
                )
                tc = ToolCall(
                    tool_name=name, args=args, result=None, ok=False, error=err,
                )
                _record_call(name, args, tc, response_text=f"ERROR: {err}")
                on_event({
                    "evt": "tool_call", "turn": turn_idx,
                    "tool": name, "args": args, "ok": False,
                    "error": err, "state_allowed": False,
                })
                continue

            tool_fn = registry.get(name)
            if tool_fn is None:
                err = f"unknown tool: {name}"
                tc = ToolCall(
                    tool_name=name, args=args, result=None, ok=False, error=err,
                )
                _record_call(name, args, tc, response_text=f"ERROR: {err}")
                on_event({
                    "evt": "tool_call", "turn": turn_idx,
                    "tool": name, "args": args, "ok": False, "error": err,
                })
                continue

            try:
                payload = tool_fn(**args)
                tc = ToolCall(tool_name=name, args=args, result=payload, ok=True)
                if on_tool_result is not None:
                    # Caller opts into a full-payload stream (transcripts,
                    # debugging). Failures in the callback should never kill
                    # the decision loop.
                    try:
                        on_tool_result(name, dict(args), payload)
                    except Exception as _cb_err:   # noqa: BLE001
                        pass
            except Exception as e:
                tc = ToolCall(
                    tool_name=name, args=args, result=None, ok=False, error=str(e),
                )

            ctx.call_log.append((name, args))
            result.tool_call_sequence.append(name)

            if name == "explore_game" and tc.ok:
                last_explored_play = int(args.get("play", -1))
            if name in ("probe_best_case", "probe_worst_case") and tc.ok:
                result.probed = True

            # ask_rule leaves next_actions empty; refill with the current
            # state's advertised actions so the model knows what's still callable.
            if (
                tc.ok and isinstance(tc.result, dict)
                and not tc.result.get("next_actions")
            ):
                extra_actions = next_actions_unchanged(state, play=last_explored_play)
                if extra_actions:
                    tail = "\n\nNext actions:\n" + "\n".join(
                        f"  - {a['tool']}: {a['when']}" for a in extra_actions
                    )
                    tc.result = {
                        **tc.result,
                        "next_actions": extra_actions,
                        "prose": tc.result.get("prose", "") + tail,
                    }

            # The `response` field of a tool_response is the string the model
            # will see inside `<|tool_response>response:NAME{value:<|"|>...<|"|>}
            # <tool_response|>`. Use prose when the tool produced it; fall back
            # to an error line or JSON for legacy payloads.
            if not tc.ok:
                response_text = f"ERROR: {tc.error}"
            elif isinstance(tc.result, dict) and "prose" in tc.result:
                response_text = tc.result["prose"]
            else:
                response_text = json.dumps(
                    tc.result, default=_json_default, separators=(",", ":"),
                )
            _record_call(name, args, tc, response_text=response_text)

            on_event({
                "evt": "tool_call", "turn": turn_idx,
                "tool": name, "args": args, "ok": tc.ok,
                "error": tc.error,
                "result_preview": _preview(tc.result) if tc.ok else None,
            })

            # State transition — first advancing call wins the turn.
            if not state_transitioned_this_turn:
                new_state = advance(state, name)
                if new_state != state:
                    on_event({
                        "evt": "menu_change",
                        "turn": turn_idx,
                        "before": state.value, "after": new_state.value,
                        "before_menu": menu_names(state),
                        "after_menu": menu_names(new_state),
                    })
                    state = new_state
                    state_transitioned_this_turn = True

        # --- Commit handling (only reachable in AFTER_PROBE) --- #
        commit_outcome: str | None = None
        if native_commit is not None:
            attempted_commits.append(int(native_commit))
            ok, reason = engine_tools.is_legal(game_state, int(native_commit))
            on_event({
                "evt": "commit_attempt",
                "turn": turn_idx,
                "domino_id": int(native_commit),
                "legal": ok,
                "reason": reason if not ok else "",
            })
            turn_tool_calls.append({
                "type": "function",
                "function": {
                    "name": "commit_play",
                    "arguments": {"domino_id": int(native_commit)},
                },
            })
            if ok:
                turn_tool_responses.append({
                    "name": "commit_play",
                    "response": f"LEGAL: committed domino {int(native_commit)}.",
                })
                committed = int(native_commit)
                trace.final_play = committed
                commit_outcome = "ok"
            else:
                turn_tool_responses.append({
                    "name": "commit_play",
                    "response": f"REJECTED: illegal play — {reason}",
                })
                trace.n_retries += 1
                if trace.n_retries > max_retries:
                    commit_outcome = "retry_exhausted"
                else:
                    commit_outcome = "retry"

        # --- If turn produced nothing at all, append the assistant message as-is
        # and nudge via the *next* user turn. After turn 1 we add the nudge to
        # the message log so the model can recover.
        if not turn_tool_calls and commit_outcome is None and turn_idx > 1:
            on_event({"evt": "empty_nudge", "turn": turn_idx})

        # --- Finalize assistant message for this turn --- #
        # Gemma's template reads tool_responses off the assistant; Qwen /
        # OpenAI-style templates read them from separate role='tool' messages.
        if tool_response_style == "gemma_native":
            if turn_tool_calls:
                assistant_turn["tool_calls"] = turn_tool_calls
            if turn_tool_responses:
                assistant_turn["tool_responses"] = turn_tool_responses
            messages.append(assistant_turn)
        elif tool_response_style == "role_tool":
            if turn_tool_calls:
                assistant_turn["tool_calls"] = turn_tool_calls
            messages.append(assistant_turn)
            for tr in turn_tool_responses:
                messages.append({
                    "role": "tool",
                    "name": tr.get("name", "unknown"),
                    "content": tr.get("response", ""),
                })
        else:
            raise ValueError(
                f"unknown tool_response_style={tool_response_style!r}"
            )

        # Post-turn: append user-role nudges for recovery scenarios.
        if commit_outcome == "ok":
            trace.turns.append(TurnStep(
                thought=thought, tool_calls=executed,
                committed_play=committed, raw_completion=completion,
            ))
            on_event({"evt": "commit_ok", "turn": turn_idx, "domino_id": committed})
            break
        if commit_outcome == "retry_exhausted":
            on_event({"evt": "retry_exhausted", "turn": turn_idx})
            trace.turns.append(TurnStep(
                thought=thought, tool_calls=executed,
                committed_play=int(native_commit),
                raw_completion=completion,
                engine_rejection=reason if not ok else None,
            ))
            break
        if commit_outcome == "retry":
            # Assistant message already has the REJECTED tool_response; no user
            # nudge needed — the model reads it inline next turn.
            pass
        elif not turn_tool_calls and turn_idx > 1:
            messages.append({
                "role": "user",
                "content": (
                    "You emitted no tool call. Call one of the available tools: "
                    f"{menu_names(state)}."
                ),
            })

        trace.turns.append(TurnStep(
            thought=thought, tool_calls=executed,
            committed_play=None, raw_completion=completion,
        ))
        on_event({"evt": "turn_end", "turn": turn_idx})

    result.n_turns = len(trace.turns)
    return result


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _append_gate_instructions(system_content: str) -> str:
    """Add a one-paragraph note about the hard-gated tool surface."""
    note = (
        "\n\n# Decision protocol (wax_museum)\n\n"
        "Your tool menu is STATE-DEPENDENT. On turn 1 `explore_game(play)` is "
        "the main gated tool; `belief_trajectory()` is a free side-call "
        "available from any state. Pick a candidate play and examine its "
        "outcome distribution. The response will name which probes are "
        "worthwhile. After exploring, `probe_best_case(play)` and "
        "`probe_worst_case(play)` become available: pick the branch you most "
        "need to resolve and run it. `ask_rule` and `belief_trajectory` remain "
        "free side-calls at every state. Only after at least one probe has "
        "run does `commit_play` appear.\n\n"
        "Form a plan on turn 1 (which play are you examining, what's the "
        "shape of its distribution, which branch worries you), then execute. "
        "You have plenty of context — reason explicitly."
    )
    return system_content + note


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _preview(result: Any, max_len: int = 200) -> str:
    s = json.dumps(result, default=_json_default, separators=(",", ":"))
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


# --------------------------------------------------------------------------- #
# Self-test — scripted stub model; no Modal.                                   #
# --------------------------------------------------------------------------- #


def _selftest(seed: int = 2026) -> None:
    """End-to-end smoke: stub model routes through the gate correctly."""
    import random as _r

    from forge.zeb.game import apply_action, legal_actions, new_game

    state = new_game(seed=seed, skip_bidding=True)
    rng = _r.Random(seed)
    while len(state.play_history) < 20:
        slots = legal_actions(state)
        if not slots:
            break
        state = apply_action(state, rng.choice(slots))

    me_abs = _current_player(state)
    legal = [
        d for d in state.hands[me_abs]
        if d not in state.played and engine_tools.is_legal(state, d)[0]
    ]
    assert legal, "no legal plays"
    target = legal[0]

    # Scripted completions — tests gate transitions end-to-end.
    script = iter([
        # Turn 1: thought + explore_game
        (
            "I'll look at my top candidate before planning.\n"
            + "x" * 300  # ensure > BAIL threshold
            + f"\n<|tool_call>call:explore_game{{play:{target}}}<tool_call|>"
        ),
        # Turn 2: probe_worst_case (should advance to AFTER_PROBE)
        f"Check the downside.\n<|tool_call>call:probe_worst_case{{play:{target}}}<tool_call|>",
        # Turn 3: commit
        f"Commit.\n<|tool_call>call:commit_play{{domino_id:{target}}}<tool_call|>",
    ])

    calls_seen: list[tuple[tuple, tuple]] = []

    def stub(messages: list[dict], tool_schemas: list[dict]) -> str:
        names = tuple(s["function"]["name"] for s in tool_schemas)
        calls_seen.append((names, (len(messages),)))
        return next(script)

    events: list[dict] = []

    result = run_decision_waxed(
        state, stub, max_turns=5, max_retries=2,
        on_event=lambda e: events.append(e),
    )

    assert not result.bailed, f"unexpected bail: {result.bail_reason}"
    assert result.probed, "probe did not run"
    assert result.trace.final_play == target, result.trace.final_play
    assert result.tool_call_sequence == [
        "explore_game", "probe_worst_case",
    ], result.tool_call_sequence

    # Menu progression witnessed by the stub:
    # INITIAL menu includes explore_game + belief_trajectory; probes come later.
    assert "explore_game" in calls_seen[0][0] and "probe_best_case" not in calls_seen[0][0]
    assert "belief_trajectory" in calls_seen[0][0], calls_seen[0][0]
    assert "probe_best_case" in calls_seen[1][0] and "commit_play" not in calls_seen[1][0]
    assert "commit_play" in calls_seen[2][0]

    print(f"[wax_museum.harness selftest seed={seed}]")
    print(f"  tool seq: {result.tool_call_sequence}")
    print(f"  final_play={result.trace.final_play} turns={result.n_turns}")
    print(f"  events: {len(events)} emitted")
    print("[wax_museum.harness selftest] OK")


if __name__ == "__main__":
    _selftest()
