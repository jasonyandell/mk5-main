"""Native Gemma 4 tool-use loop — parallel to ``tool_loop.py``.

Differences from the XML path:

1. The model does NOT see a hand-rolled tool menu or output protocol. Tools
   are rendered by the chat template via the ``tools=[...]`` kwarg. Gemma 4
   was post-trained to emit its native shape in response.
2. Conversation state is a list of ``messages`` (HF chat format), NOT a
   concatenated text log. Tool observations come back as role=``tool``
   messages, so the server's chat template can emit them in Gemma's native
   ``<|tool_response>...<tool_response|>`` wrapper.
3. Committing a play is a **native tool call** named ``commit_play``:
   ``<|tool_call>call:commit_play{domino_id:N}<tool_call|>``. The parser
   sieves it out of the regular tool-call list and treats it as the
   terminal action — the harness runs the legality check and exits. This
   is the "go with Gemma's grain" fix: the model's reflex is to wrap every
   answer in a tool-call envelope, so we give it a tool that means "this
   is my final answer." No more XML ``<commit>`` tag anywhere.

Parser is permissive on purpose. The ergonomics doc describes the canonical
token shape as::

    <|tool_call>call:NAME{param:<|"|>VALUE<|"|>}<tool_call|>

but vLLM's detokenizer can render the special tokens in several equivalent
surface forms depending on ``skip_special_tokens`` and the tokenizer version.
We try, in order:

* The canonical ``<|tool_call>...<tool_call|>`` envelope with ``call:NAME(args)``
  or ``call:NAME{args}`` inside it.
* A JSON envelope ``<|tool_call>{"name":"NAME","arguments":{...}}<tool_call|>``
  (the shape HF's generic chat-template often emits).
* A bare ``call:NAME(args)`` line (observed in ergo probe 4 — the model
  emitting the native syntax even when the special tokens got stripped).

If none match, the turn is treated as pure reasoning and the loop continues.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, Callable

from burl.harness.retry import RetryExhausted, retry_on_illegal
from burl.harness.tool_loop import ToolProtocol
from burl.harness.trace import BurlTrace, ToolCall, TurnStep

NativeModelCallable = Callable[[list[dict], list[dict]], str]
"""`(messages, tools) -> raw completion text`. The adapter wraps Modal."""


# --------------------------------------------------------------------------- #
# Parser                                                                       #
# --------------------------------------------------------------------------- #


COMMIT_TOOL_NAME = "commit_play"
"""Native tool that means 'this is my final answer'. Sieved out of the
regular tool-call list by the parser; sets trace.final_play instead."""

# Canonical: <|tool_call>...body...<tool_call|>
_NATIVE_ENVELOPE_RE = re.compile(
    r"<\|tool_call>(.*?)<tool_call\|>", re.DOTALL,
)
# Inside an envelope OR as a bare line: call:NAME ( args ) or call:NAME { args }
_CALL_SIG_RE = re.compile(
    r"call\s*:\s*(\w+)\s*[({](.*?)[)}]", re.DOTALL,
)
# Strip Gemma's string-delimiter special token if present.
_STRDELIM_RE = re.compile(r"<\|\"\|>")


def _parse_args_blob(blob: str) -> dict[str, Any]:
    """Best-effort parse of the args payload inside a tool call.

    Handles the shapes we've seen / expect:
      - ``{"domino_id": 21}``                    (JSON)
      - ``{domino_id: 21}``                      (JS-ish)
      - ``domino_id=21, n_samples=10``           (Python kwargs)
      - ``domino_id:21``                         (Gemma native: braces already
        stripped, Gemma's string-delim tokens peeled)
      - ``play=21, assumption={"player":1,"holds":5}`` (nested)

    Returns ``{}`` on parse failure — the caller records this as a tool_call
    with error, so the model sees it in the next turn and can recover.
    """
    blob = _STRDELIM_RE.sub('"', blob).strip()
    if not blob:
        return {}

    # JSON shape (outer).
    try:
        parsed = json.loads(blob)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # JSON shape with wrapping braces stripped.
    try:
        parsed = json.loads("{" + blob + "}")
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # Python literal shape, e.g. {'a': 1}.
    try:
        parsed = ast.literal_eval(blob)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    # kwargs shape: name=value, name=value.
    result: dict[str, Any] = {}
    for part in _split_top_level(blob, ","):
        if "=" not in part and ":" not in part:
            continue
        sep = "=" if "=" in part else ":"
        k, v = part.split(sep, 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            result[k] = ast.literal_eval(v)
        except Exception:
            try:
                result[k] = json.loads(v)
            except Exception:
                # Last resort: plain string.
                result[k] = v.strip('"').strip("'")
    return result


def _split_top_level(text: str, sep: str) -> list[str]:
    """Split on ``sep`` at brace/paren depth 0 only."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in text:
        if ch in "({[":
            depth += 1
        elif ch in ")}]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf))
    return [p.strip() for p in parts if p.strip()]


def parse_native_completion(
    text: str,
) -> tuple[str, list[tuple[str, dict]], int | None]:
    """Return (thought, tool_calls, commit) parsed from a native-shaped reply.

    ``commit`` is ``domino_id`` extracted from the first ``commit_play`` tool
    call if one appears; it does NOT appear in ``tool_calls`` (the harness
    runs the legality check on ``commit`` and exits — commit_play has no
    registry entry to execute). Malformed commit_play calls (missing or
    non-integer ``domino_id``) are dropped silently; the harness nudges the
    model to try again on the next turn.

    ``thought`` is whatever free-text survives after stripping the envelope
    blocks; only used for logging.
    """
    tool_calls: list[tuple[str, dict]] = []

    # Canonical envelope form.
    remainder = text
    for envelope in _NATIVE_ENVELOPE_RE.findall(text):
        body = envelope.strip()
        # Try JSON inside envelope first (HF's generic shape).
        parsed_json: dict | None = None
        try:
            maybe = json.loads(body)
            if (
                isinstance(maybe, dict)
                and "name" in maybe
                and ("arguments" in maybe or "args" in maybe or "parameters" in maybe)
            ):
                parsed_json = maybe
        except Exception:
            parsed_json = None

        if parsed_json is not None:
            name = str(parsed_json["name"])
            args = (
                parsed_json.get("arguments")
                or parsed_json.get("args")
                or parsed_json.get("parameters")
                or {}
            )
            if not isinstance(args, dict):
                args = {}
            tool_calls.append((name, args))
            continue

        # Fall back to call:NAME(...) inside envelope.
        m = _CALL_SIG_RE.search(body)
        if m:
            name = m.group(1)
            args = _parse_args_blob(m.group(2))
            tool_calls.append((name, args))

    # Drop envelopes from the remainder so we can see the rest.
    remainder = _NATIVE_ENVELOPE_RE.sub("", text)

    # Bare `call:NAME(...)` lines (no envelope — this catches the leaky
    # "only emitted call:trump_declared()" shape the ergo probe saw).
    # Only scan lines that weren't already inside an envelope.
    if not tool_calls:
        for m in _CALL_SIG_RE.finditer(remainder):
            name = m.group(1)
            args = _parse_args_blob(m.group(2))
            tool_calls.append((name, args))

    # Sieve commit_play out of the tool list — it's the terminal action.
    commit: int | None = None
    filtered: list[tuple[str, dict]] = []
    for name, args in tool_calls:
        if name == COMMIT_TOOL_NAME:
            if commit is None:
                raw = args.get("domino_id", args.get("play"))
                try:
                    if raw is not None:
                        commit = int(raw)
                except (TypeError, ValueError):
                    pass
            continue  # never forward commit_play to the executor
        filtered.append((name, args))
    tool_calls = filtered

    # Thought = whatever survives after stripping envelopes and bare calls.
    thought = _NATIVE_ENVELOPE_RE.sub("", text)
    thought = _CALL_SIG_RE.sub("", thought)
    # Strip Gemma's internal channel markers if they leak through.
    thought = re.sub(r"<\|channel>|<channel\|>|<\|think\|>", "", thought).strip()

    return thought, tool_calls, commit


# --------------------------------------------------------------------------- #
# Harness                                                                      #
# --------------------------------------------------------------------------- #


class NativeHarness:
    """Like ``Harness`` but keeps a ``messages`` list and renders tool outputs
    as HF-shaped tool messages (``role="tool"``) instead of inline XML blobs.
    """

    def __init__(
        self,
        model_callable: NativeModelCallable,
        tools: dict[str, ToolProtocol],
        tool_schemas: list[dict],
        is_legal_fn: Callable[[Any, int], tuple[bool, str]],
        commit_instruction: str,
        max_turns: int = 8,
        max_retries: int = 5,
    ):
        self._model = model_callable
        self._tools = tools
        self._tool_schemas = tool_schemas
        self._is_legal = is_legal_fn
        self._commit_instruction = commit_instruction
        self._max_turns = max_turns
        self._max_retries = max_retries

    def run(
        self,
        game_state: Any,
        system_content: str,
        user_content: str,
        state_key: str = "",
        *,
        extra_user_messages: list[str] | None = None,
    ) -> BurlTrace:
        # decision_prompt is the concatenated view we record for STaR-style
        # trace analysis — matches what the XML harness writes. Extra user
        # messages (EQ-gate nudges) are deliberately NOT folded into
        # decision_prompt: the STaR SFT target is "decision → correction
        # trace" with the nudge stripped, so the adapter learns the fixed
        # behavior without inference-time scaffolding.
        trace = BurlTrace(
            game_state_key=state_key,
            decision_prompt=f"[SYSTEM]\n{system_content}\n\n[USER]\n{user_content}",
        )

        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        if extra_user_messages:
            for extra in extra_user_messages:
                messages.append({"role": "user", "content": extra})

        turn_idx = [0]

        def step_fn(t: BurlTrace, rejection: str | None) -> TurnStep:
            if turn_idx[0] >= self._max_turns:
                raise RetryExhausted(
                    f"max_turns={self._max_turns} exceeded", trace=t
                )
            turn_idx[0] += 1

            if rejection is not None:
                # Feed engine rejection back as a synthetic tool-style message.
                messages.append(
                    {
                        "role": "tool",
                        "name": "engine.commit",
                        "content": json.dumps(
                            {"error": f"illegal: {rejection}"},
                            separators=(",", ":"),
                        ),
                    }
                )

            # Char count of the serialized messages list — comparable to the
            # XML path's tokens_in (both are char-level approximations).
            serialized = json.dumps(messages, default=str)
            t.tokens_in += len(serialized)
            completion = self._model(messages, self._tool_schemas)
            t.tokens_out += len(completion)

            thought, tool_specs, commit = parse_native_completion(completion)

            # Append the assistant turn verbatim so the next round has the full
            # chat history. We keep the raw text in content so the chat template
            # has *something* to render; native tool_calls are split out below.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": completion,
            }
            messages.append(assistant_msg)

            executed: list[ToolCall] = []
            for name, args in tool_specs:
                tc = self._execute(game_state, name, args)
                executed.append(tc)
                tool_body = (
                    {"error": tc.error} if not tc.ok else {"result": tc.result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": tc.tool_name,
                        "content": json.dumps(
                            tool_body, default=_json_default,
                            sort_keys=True, separators=(",", ":"),
                        ),
                    }
                )

            # Nudge the model back toward commit_play if the turn produced
            # neither real tools nor a commit (rare, but possible when an
            # envelope is malformed or empty).
            if commit is None and not tool_specs:
                messages.append(
                    {
                        "role": "user",
                        "content": self._commit_instruction,
                    }
                )

            return TurnStep(
                thought=thought,
                tool_calls=executed,
                committed_play=commit,
                raw_completion=completion,
            )

        def legality(dom: int) -> tuple[bool, str]:
            return self._is_legal(game_state, dom)

        retry_on_illegal(step_fn, legality, trace, max_retries=self._max_retries)
        return trace

    def _execute(self, game_state: Any, name: str, args: dict) -> ToolCall:
        tool = self._tools.get(name)
        if tool is None:
            return ToolCall(
                tool_name=name, args=args, result=None, ok=False,
                error=f"unknown tool: {name}",
            )
        try:
            result = tool(game_state, **args)
            return ToolCall(tool_name=name, args=args, result=result, ok=True)
        except Exception as e:
            return ToolCall(
                tool_name=name, args=args, result=None, ok=False, error=str(e),
            )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


# --------------------------------------------------------------------------- #
# Self-test — pure parser + scripted stub model.                               #
# --------------------------------------------------------------------------- #


def _selftest() -> None:
    # Parser — canonical envelope with JSON body.
    thought, calls, commit = parse_native_completion(
        'pre-text <|tool_call>{"name":"is_legal","arguments":{"domino_id":21}}<tool_call|> post'
    )
    assert calls == [("is_legal", {"domino_id": 21})], calls
    assert commit is None

    # Parser — envelope with call:NAME(...) body.
    thought, calls, commit = parse_native_completion(
        "<|tool_call>call: trump_declared()<tool_call|>"
    )
    assert calls == [("trump_declared", {})], calls

    # Parser — bare leaky call with kwargs (no envelope).
    thought, calls, commit = parse_native_completion(
        "reasoning here. call:is_legal(domino_id=21) more prose."
    )
    assert calls == [("is_legal", {"domino_id": 21})], calls
    assert commit is None

    # Parser — commit_play is sieved out of tool_calls and sets commit.
    thought, calls, commit = parse_native_completion(
        '<|tool_call>call:is_legal{domino_id:21}<tool_call|>'
        '<|tool_call>call:commit_play{domino_id:21}<tool_call|>'
    )
    assert calls == [("is_legal", {"domino_id": 21})], calls
    assert commit == 21, commit

    # Parser — commit_play-only turn.
    thought, calls, commit = parse_native_completion(
        '<|tool_call>call:commit_play{domino_id:14}<tool_call|>'
    )
    assert calls == [], calls
    assert commit == 14, commit

    # Parser — Gemma native braces with string-delim token.
    thought, calls, commit = parse_native_completion(
        '<|tool_call>call:void_audit{player_seat:<|"|>2<|"|>, suit:3}<tool_call|>'
    )
    assert calls and calls[0][0] == "void_audit"
    assert calls[0][1].get("suit") == 3, calls[0][1]

    # Harness round-trip with scripted native model.
    from burl.harness.tool_loop import ToolProtocol as _TP  # noqa: F401

    class Stub:
        def __init__(self, name: str, fn: Callable):
            self.name = name
            self._fn = fn

        def __call__(self, state: Any, **kw: Any) -> Any:
            return self._fn(state, **kw)

    def _trump(_s: Any) -> str:
        return "fives"

    def _legal(_s: Any, dom: int) -> tuple[bool, str]:
        return (dom == 0, "" if dom == 0 else f"{dom} not in hand")

    tools_reg: dict[str, Any] = {"trump_declared": Stub("trump_declared", _trump)}
    tool_schemas: list[dict] = [
        {
            "type": "function",
            "function": {
                "name": "trump_declared",
                "description": "...",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    script = iter([
        '<|tool_call>{"name":"trump_declared","arguments":{}}<tool_call|>',
        '<|tool_call>call:commit_play{domino_id:5}<tool_call|>',
        '<|tool_call>call:commit_play{domino_id:0}<tool_call|>',
    ])

    def stub_model(messages: list[dict], _tools: list[dict]) -> str:
        # sanity: harness is feeding along a growing chat history
        assert isinstance(messages, list) and messages, messages
        return next(script)

    harness = NativeHarness(
        model_callable=stub_model,
        tools=tools_reg,
        tool_schemas=tool_schemas,
        is_legal_fn=_legal,
        commit_instruction=(
            "Call commit_play with your final domino_id to end the decision."
        ),
        max_turns=8,
        max_retries=3,
    )
    trace = harness.run(
        game_state={"hand": [0]},
        system_content="You are Burl.",
        user_content="Decide.",
        state_key="selftest",
    )

    assert trace.final_play == 0, trace.final_play
    assert trace.n_retries == 1, trace.n_retries
    tool_calls = [tc for turn in trace.turns for tc in turn.tool_calls]
    assert len(tool_calls) == 1 and tool_calls[0].tool_name == "trump_declared"
    rt = BurlTrace.from_json(trace.to_json())
    assert rt.final_play == 0 and rt.n_retries == 1

    print(f"[tool_loop_native selftest] turns={len(trace.turns)} "
          f"retries={trace.n_retries} final={trace.final_play}")
    print("[tool_loop_native selftest] OK: parser sieves commit_play, "
          "harness round-trips with retry.")


if __name__ == "__main__":
    _selftest()
