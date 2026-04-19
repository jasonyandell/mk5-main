"""ReAct-style think/act/observe loop for Burl.

Parser format: XML-ish tags (`<think>`, `<tool>`, `<commit>`, `<observation>`).
Chosen over pure JSON fences because tool args are themselves JSON; nesting
JSON inside a JSON-fenced envelope either requires escaping or a second parser
pass. XML tags sidestep that: the envelope is text, the payload is JSON, each
parsed with its own tool. Swap this whole module out by replacing the three
regexes and `format_observation` if Gemma 4's native tool format wins.

Model contract:
    model_callable(prompt: str) -> str

The model produces one turn per call. A turn is any mix of `<think>`, one or
more `<tool>` calls, OR exactly one `<commit>`. A turn with a commit ends the
loop (subject to legality check in `retry.py`).

Engine adapter:
    is_legal_fn(domino_id: int) -> (bool, reason)  (passed through to retry)

Tools are a dict of `name -> ToolProtocol`. Stubs in __main__ below.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Protocol

from burl.harness.retry import RetryExhausted, retry_on_illegal
from burl.harness.trace import BurlTrace, ToolCall, TurnStep

ModelCallable = Callable[[str], str]


class ToolProtocol(Protocol):
    name: str

    def __call__(self, game_state: Any, **kwargs: Any) -> Any: ...


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)
_COMMIT_RE = re.compile(r"<commit>\s*(-?\d+)\s*</commit>")


def format_observation(call: ToolCall) -> str:
    """One place, so prompt serialization stays deterministic and diffable."""
    if not call.ok:
        payload = {"error": call.error}
    else:
        payload = {"result": call.result}
    # Sorted keys + compact separators so identical tool calls produce identical
    # observation strings. Matters for trace dedup and cache keys during STaR.
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return f'<observation tool="{call.tool_name}">{body}</observation>'


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")


def parse_completion(text: str) -> tuple[str, list[tuple[str, dict]], int | None]:
    thoughts = _THINK_RE.findall(text)
    thought = "\n".join(t.strip() for t in thoughts)

    tool_calls: list[tuple[str, dict]] = []
    for blob in _TOOL_RE.findall(text):
        spec = json.loads(blob.strip())
        tool_calls.append((spec["name"], spec.get("args", {})))

    m = _COMMIT_RE.search(text)
    commit = int(m.group(1)) if m else None
    return thought, tool_calls, commit


class Harness:
    def __init__(
        self,
        model_callable: ModelCallable,
        tools: dict[str, ToolProtocol],
        is_legal_fn: Callable[[Any, int], tuple[bool, str]],
        max_turns: int = 8,
        max_retries: int = 5,
    ):
        self._model = model_callable
        self._tools = tools
        self._is_legal = is_legal_fn
        self._max_turns = max_turns
        self._max_retries = max_retries

    def run(self, game_state: Any, decision_prompt: str, state_key: str = "") -> BurlTrace:
        trace = BurlTrace(
            game_state_key=state_key,
            decision_prompt=decision_prompt,
        )

        turn_idx = [0]
        prompt_log: list[str] = [decision_prompt]

        def step_fn(t: BurlTrace, rejection: str | None) -> TurnStep:
            if turn_idx[0] >= self._max_turns:
                raise RetryExhausted(f"max_turns={self._max_turns} exceeded")
            turn_idx[0] += 1

            if rejection is not None:
                prompt_log.append(
                    f'<observation tool="engine.commit">'
                    f'{{"error":"illegal: {rejection}"}}</observation>'
                )

            prompt = "\n".join(prompt_log)
            t.tokens_in += len(prompt)
            completion = self._model(prompt)
            t.tokens_out += len(completion)

            thought, tool_specs, commit = parse_completion(completion)

            executed: list[ToolCall] = []
            for name, args in tool_specs:
                tc = self._execute(game_state, name, args)
                executed.append(tc)
                prompt_log.append(format_observation(tc))

            if commit is not None:
                prompt_log.append(f"<commit>{commit}</commit>")

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


# ---------------------------------------------------------------------------
# Self-test. Stub tools simulate what burl/tools/{zeb,engine}.py will provide.
# Real imports slot in at the marked lines inside the `if __name__` block.
# ---------------------------------------------------------------------------


def _selftest() -> None:
    calls_log: list[str] = []

    class StubTool:
        def __init__(self, name: str, fn: Callable):
            self.name = name
            self._fn = fn

        def __call__(self, state: Any, **kwargs: Any) -> Any:
            calls_log.append(f"{self.name}({kwargs})")
            return self._fn(state, **kwargs)

    def stub_trump_declared(_state: Any) -> str:
        return "fives"

    # Real import when Move 3 lands:
    #     from burl.tools.engine import is_legal, trump_declared, is_trump, unseen, void_audit
    #     from burl.tools.zeb import get_belief
    def stub_is_legal(_state: Any, dom: int) -> tuple[bool, str]:
        if dom == 0:
            return (True, "")
        return (False, f"domino {dom} not in hand")

    tools: dict[str, ToolProtocol] = {
        "trump_declared": StubTool("trump_declared", stub_trump_declared),
    }

    # Scripted model: asks trump_declared, commits illegal 5, then legal 0.
    script = iter([
        "<think>Who set trump?</think>"
        '<tool>{"name":"trump_declared","args":{}}</tool>',

        "<think>Fives are trump. I'll try 5.</think>"
        "<commit>5</commit>",

        "<think>Rejected. Try 0.</think>"
        "<commit>0</commit>",
    ])

    last_prompt: list[str] = [""]

    def stub_model(prompt: str) -> str:
        last_prompt[0] = prompt
        return next(script)

    game_state = {"hand": [0], "seat": "S"}
    harness = Harness(
        model_callable=stub_model,
        tools=tools,
        is_legal_fn=stub_is_legal,
        max_turns=8,
        max_retries=5,
    )

    trace = harness.run(game_state, "Your move. Seat S, trick 6.", state_key="test-0001")

    assert trace.final_play == 0, f"expected final_play=0, got {trace.final_play}"
    assert trace.n_retries == 1, f"expected n_retries=1, got {trace.n_retries}"

    roundtrip = BurlTrace.from_json(trace.to_json())
    assert roundtrip.final_play == 0
    assert roundtrip.n_retries == 1
    assert len(roundtrip.turns) == len(trace.turns)
    assert roundtrip.turns[0].tool_calls[0].tool_name == "trump_declared"

    # The third turn's prompt must show the observation from turn 1 AND the
    # engine rejection from turn 2.
    final_prompt = last_prompt[0]
    assert '<observation tool="trump_declared">' in final_prompt, \
        "tool observation missing from subsequent prompt"
    assert "fives" in final_prompt, "observation body missing"
    assert "illegal" in final_prompt, "engine rejection missing from prompt"

    print(f"[selftest] turns={len(trace.turns)} retries={trace.n_retries} final={trace.final_play}")
    print(f"[selftest] tool calls executed: {calls_log}")
    print(f"[selftest] trace JSON len={len(trace.to_json())}")
    print(f"[selftest] final prompt ({len(final_prompt)} chars):")
    for line in final_prompt.splitlines():
        print(f"    {line}")
    print("[selftest] OK: trace roundtrips, retry counted, observations forwarded")


if __name__ == "__main__":
    _selftest()
