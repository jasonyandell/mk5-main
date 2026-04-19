"""Structured trace of one Burl decision.

Shape matters: STaR iteration filters traces by E[Q] and replays them as SFT
corpus, so fields here must survive JSON roundtrip losslessly and stay stable
across Burl versions. Anything experiment-specific goes in `metadata`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ToolCall:
    tool_name: str
    args: dict[str, Any]
    result: Any
    ok: bool
    error: str | None = None


@dataclass
class TurnStep:
    thought: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    committed_play: int | None = None
    engine_rejection: str | None = None
    raw_completion: str = ""


@dataclass
class BurlTrace:
    game_state_key: str
    decision_prompt: str
    turns: list[TurnStep] = field(default_factory=list)
    final_play: int = -1
    n_retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), default=_default)

    @classmethod
    def from_json(cls, s: str) -> "BurlTrace":
        d = json.loads(s)
        turns = [
            TurnStep(
                thought=t["thought"],
                tool_calls=[ToolCall(**tc) for tc in t["tool_calls"]],
                committed_play=t["committed_play"],
                engine_rejection=t["engine_rejection"],
                raw_completion=t.get("raw_completion", ""),
            )
            for t in d["turns"]
        ]
        return cls(
            game_state_key=d["game_state_key"],
            decision_prompt=d["decision_prompt"],
            turns=turns,
            final_play=d["final_play"],
            n_retries=d["n_retries"],
            tokens_in=d["tokens_in"],
            tokens_out=d["tokens_out"],
            metadata=d.get("metadata", {}),
        )


def _default(obj: Any) -> Any:
    # Tool results may be sets / tuples; keep JSON stable and order-independent.
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")
