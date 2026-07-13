"""Decision-loader for the wax_museum harvest.

Reads a harvest dir produced by ``scratch/belief_trajectory_rollout/harvest.py``
and exposes individual decisions as ``(meta, messages)`` pairs the chat client
can use as a prefix.

Phase 1 reconstruction strategy: build a single assistant message that
narrates Burl's full reasoning trace (per-turn thoughts, tool calls, tool
results, final commit). Not byte-identical to the original rendered prompt,
but high-fidelity enough that the model can answer questions about the
position with the actual context in scope.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

_TOOL_CALL_RE = re.compile(r"<\|tool_call>.*?<tool_call\|>", re.DOTALL)
_THOUGHT_RE = re.compile(r"<\|channel>thought\s*(.*?)\s*<channel\|>", re.DOTALL)
_ORPHAN_MARKERS_RE = re.compile(
    r"<\|channel>thought\b|<channel\|>|<turn\|>|<\|turn>",
)

SCRATCH_ROOT = Path(
    os.environ.get(
        "BURL_CHAT_HARVEST_ROOT",
        "/Users/jason/code/mk5-main/scratch/belief_trajectory_rollout",
    )
)


def list_harvests() -> list[dict]:
    """Return harvest dirs under SCRATCH_ROOT, newest first.

    Each entry: ``{name, n_decisions, buckets: {name: count}}``.
    """
    if not SCRATCH_ROOT.exists():
        return []
    out: list[dict] = []
    for path in sorted(SCRATCH_ROOT.glob("harvest_batched_*"), reverse=True):
        idx = path / "corpus_index.jsonl"
        if not idx.exists():
            continue
        buckets: Counter = Counter()
        n = 0
        with idx.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                buckets[d.get("bucket", "?")] += 1
                n += 1
        out.append({
            "name": path.name,
            "n_decisions": n,
            "buckets": dict(buckets.most_common()),
        })
    return out


def list_decisions(
    harvest: str,
    bucket: str | None = None,
    limit: int = 40,
    offset: int = 0,
) -> list[dict]:
    """List decisions in a harvest, optionally filtered by bucket."""
    idx_path = SCRATCH_ROOT / harvest / "corpus_index.jsonl"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)
    rows: list[dict] = []
    with idx_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if bucket and d.get("bucket") != bucket:
                continue
            rows.append(d)
    return rows[offset : offset + limit]


def _decision_dir(harvest: str, idx: dict) -> Path:
    return SCRATCH_ROOT / harvest / idx["transcript_path"].split("/transcript")[0]


def load_decision(harvest: str, global_idx: int) -> dict:
    """Load one decision as ``{meta, messages, events}``.

    ``messages`` is a 3-message OpenAI/Gemma-shaped list (system + user +
    assistant narrative), suitable for use as a chat-template prefix.
    """
    idx_path = SCRATCH_ROOT / harvest / "corpus_index.jsonl"
    matched: dict | None = None
    with idx_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("global_idx") == global_idx:
                matched = d
                break
    if matched is None:
        raise KeyError(f"global_idx {global_idx} not found in {harvest}")

    dec_dir = _decision_dir(harvest, matched)
    events_path = dec_dir / "events.jsonl"
    events: list[dict] = []
    with events_path.open() as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    messages = _reconstruct_messages(events)
    segments = _build_prefix_segments(events)
    return {
        "meta": matched,
        "messages": messages,
        "segments": segments,
        "events": events,
    }


def _build_prefix_segments(events: list[dict]) -> list[dict]:
    """Parse events.jsonl into typed display segments for the chat UI."""
    sys_event = next((e for e in events if e["kind"] == "prompt_system"), None)
    usr_event = next((e for e in events if e["kind"] == "prompt_user"), None)
    if sys_event is None or usr_event is None:
        return []

    out: list[dict] = [
        {"kind": "system", "content": sys_event["content"]},
        {"kind": "user", "content": usr_event["content"]},
    ]

    turns: dict[int, list[dict]] = {}
    for e in events:
        t = e.get("turn")
        if t is None:
            continue
        turns.setdefault(int(t), []).append(e)

    for turn_idx in sorted(turns):
        for e in turns[turn_idx]:
            kind = e["kind"]
            if kind == "thinking":
                out.append({
                    "kind": "thinking",
                    "turn": turn_idx,
                    "content": e["content"],
                })
            elif kind == "tool_call":
                out.append({
                    "kind": "tool_call",
                    "turn": turn_idx,
                    "tool": e["tool"],
                    "args": e.get("args") or {},
                })
            elif kind == "tool_result":
                out.append({
                    "kind": "tool_result",
                    "turn": turn_idx,
                    "tool": e["tool"],
                    "content": e["content"],
                })
            elif kind == "assistant_text":
                # Strip tool_call markup + any orphan channel/turn markers
                # that leak through; if any narrative remains, surface it.
                cleaned = _TOOL_CALL_RE.sub("", e["content"])
                cleaned = _ORPHAN_MARKERS_RE.sub("", cleaned).strip()
                if cleaned:
                    out.append({
                        "kind": "assistant_text",
                        "turn": turn_idx,
                        "content": cleaned,
                    })

    commit = next((e for e in events if e["kind"] == "commit"), None)
    if commit is not None:
        out.append({
            "kind": "commit",
            "final_play": commit["final_play"],
            "legal": commit.get("legal"),
        })
    return out


def _reconstruct_messages(events: list[dict]) -> list[dict]:
    sys_event = next((e for e in events if e["kind"] == "prompt_system"), None)
    usr_event = next((e for e in events if e["kind"] == "prompt_user"), None)
    if sys_event is None or usr_event is None:
        raise ValueError("missing prompt_system or prompt_user event")

    # Group per-turn events
    turns: dict[int, dict] = {}
    for e in events:
        t = e.get("turn")
        if t is None:
            continue
        bucket = turns.setdefault(int(t), {})
        bucket.setdefault(e["kind"], []).append(e)

    parts: list[str] = []
    for turn_idx in sorted(turns):
        bucket = turns[turn_idx]
        thinking = bucket.get("thinking", [])
        if thinking:
            parts.append(
                f"━━━ turn {turn_idx} — thought ━━━\n{thinking[0]['content'].strip()}"
            )
        for ar in bucket.get("assistant_text", []):
            parts.append(
                f"━━━ turn {turn_idx} — action ━━━\n{ar['content'].strip()}"
            )
        for tc in bucket.get("tool_call", []):
            args = json.dumps(tc.get("args") or {})
            parts.append(f"  → tool_call: {tc['tool']}({args})")
        for tr in bucket.get("tool_result", []):
            parts.append(
                f"━━━ turn {turn_idx} — tool result ({tr['tool']}) ━━━\n"
                f"{tr['content'].strip()}"
            )

    commit = next((e for e in events if e["kind"] == "commit"), None)
    if commit is not None:
        parts.append(
            f"━━━ commit ━━━\nfinal_play: {commit['final_play']}  "
            f"legal: {commit.get('legal')}"
        )

    assistant_narrative = "\n\n".join(parts)

    return [
        {"role": "system", "content": sys_event["content"]},
        {"role": "user", "content": usr_event["content"]},
        {"role": "assistant", "content": assistant_narrative},
    ]
