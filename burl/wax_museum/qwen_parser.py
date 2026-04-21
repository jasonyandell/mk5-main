"""Parser for Qwen3.6's native tool-call shape.

Qwen emits completions in this form::

    <think>
    reasoning...
    </think>
    <tool_call>
    <function=explore_game>
    <parameter=play>
    13
    </parameter>
    </function>
    </tool_call>

Also supports the Hermes JSON fallback (``<tool_call>{"name":...}</tool_call>``).
Returns the same ``(thought, tool_calls, commit)`` tuple as
``burl.harness.tool_loop_native.parse_native_completion`` so harnesses can
swap parsers without touching downstream logic.
"""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_FN_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>", re.DOTALL)

COMMIT_TOOL_NAME = "commit_play"


def _coerce(val: str) -> Any:
    """Best-effort: int / float / bool / string."""
    v = val.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_tool_call_body(body: str) -> tuple[str, dict] | None:
    """Parse one ``<tool_call>...</tool_call>`` body into (name, args)."""
    m = _FN_RE.search(body)
    if m is not None:
        name = m.group(1).strip()
        args: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(m.group(2)):
            args[pm.group(1).strip()] = _coerce(pm.group(2))
        return name, args
    # Hermes JSON fallback.
    try:
        parsed = json.loads(body.strip())
        if isinstance(parsed, dict) and "name" in parsed:
            args = parsed.get("arguments") or parsed.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            return str(parsed["name"]), args
    except Exception:
        pass
    return None


def parse_qwen_completion(
    text: str,
) -> tuple[str, list[tuple[str, dict]], int | None]:
    """Return ``(thought, tool_calls, commit_domino_id)``.

    - ``thought``: whatever survives after stripping ``<think>`` / ``<tool_call>``
      blocks. Used for logging.
    - ``tool_calls``: list of ``(name, args)`` pairs, with ``commit_play`` sieved
      out — the harness treats that as the terminal action.
    - ``commit_domino_id``: if the completion contained a ``commit_play`` call,
      its ``domino_id`` (int) or ``None``.
    """
    tool_calls: list[tuple[str, dict]] = []
    for body in _TOOL_CALL_RE.findall(text):
        parsed = _parse_tool_call_body(body)
        if parsed is not None:
            tool_calls.append(parsed)

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
            continue
        filtered.append((name, args))
    tool_calls = filtered

    # Extract the first <think> block as thought if present; otherwise strip
    # the tool-call blocks and use the remainder.
    think_match = _THINK_RE.search(text)
    if think_match is not None:
        thought = think_match.group(1).strip()
    else:
        thought = _TOOL_CALL_RE.sub("", text)
        thought = _THINK_RE.sub("", thought)
        thought = thought.strip()

    return thought, tool_calls, commit


def _selftest() -> None:
    sample = """<think>
Let me think about this. I have 13 and 21.
</think>
<tool_call>
<function=explore_game>
<parameter=play>
13
</parameter>
</function>
</tool_call>"""
    thought, calls, commit = parse_qwen_completion(sample)
    assert calls == [("explore_game", {"play": 13})], calls
    assert commit is None
    assert "Let me think" in thought

    commit_sample = """<think>ok</think><tool_call>
<function=commit_play><parameter=domino_id>21</parameter></function>
</tool_call>"""
    _, calls, commit = parse_qwen_completion(commit_sample)
    assert calls == [], calls
    assert commit == 21, commit

    # Hermes JSON fallback.
    json_sample = '<tool_call>{"name":"probe_best_case","arguments":{"play":21}}</tool_call>'
    _, calls, _ = parse_qwen_completion(json_sample)
    assert calls == [("probe_best_case", {"play": 21})], calls

    print("[qwen_parser selftest] OK")


if __name__ == "__main__":
    _selftest()
