"""Tests for burl.corpus.blender."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from burl.corpus.blender import (
    SHORT_ASST_CHARS_MAX,
    SHORT_THOUGHT_CHARS_MAX,
    blend,
    build_blended_corpus,
    classify,
    classify_row,
    has_commit_play,
    load_corpus,
    shorten_assistant,
    shorten_row,
    thought_chars,
    write_corpus,
)


# --- Fixtures: minimal rows that mirror the real corpus shape -----------------

_LONG_THOUGHT_PROSE = (
    "Partnership, bid, trump, count, offense/defense. " * 40
)  # ~1.9 KB of narrative — well above SHORT_THOUGHT_CHARS_MAX.

LONG_ASSISTANT = (
    "<|channel>thought\n"
    f"{_LONG_THOUGHT_PROSE}\n"
    "I will commit to play 14.\n"
    "<channel|>"
    "<|tool_call>call:is_legal{domino_id:14}<tool_call|>"
    "<|tool_response>{\"result\":{\"legal\":true}}<tool_response|>"
    "<|tool_call>call:commit_play{domino_id:14}<tool_call|>"
)

SHORT_ASSISTANT = (
    "<|tool_call>call:is_legal{domino_id:4}<tool_call|>"
    "<|tool_response>{\"result\":{\"legal\":true}}<tool_response|>"
    "<|tool_call>call:commit_play{domino_id:4}<tool_call|>"
)

USER_MSG = "[SYSTEM] You are Burl...\n[USER] pick a play"


def _row(asst: str, source: str = "rollout_win", **extra) -> dict:
    return {
        "messages": [
            {"role": "user", "content": USER_MSG},
            {"role": "assistant", "content": asst},
        ],
        "source": source,
        **extra,
    }


# --- Helper metrics ------------------------------------------------------------

def test_thought_chars_counts_inside_thought_block():
    asst = "<|channel>thought\nhello world<channel|><|tool_call>commit<tool_call|>"
    # chars between "<|channel>thought" and "<channel|>"
    expected = len("\nhello world")
    assert thought_chars(asst) == expected


def test_thought_chars_multiple_blocks():
    asst = (
        "<|channel>thought\nfirst<channel|>"
        "<|tool_call>t<tool_call|>"
        "<|channel>thought\nsecond<channel|>"
    )
    assert thought_chars(asst) == len("\nfirst") + len("\nsecond")


def test_thought_chars_no_block():
    assert thought_chars(SHORT_ASSISTANT) == 0


def test_has_commit_play():
    assert has_commit_play(SHORT_ASSISTANT)
    assert has_commit_play(LONG_ASSISTANT)
    assert not has_commit_play("<|tool_call>call:is_legal{domino_id:1}<tool_call|>")


# --- Classifier ---------------------------------------------------------------

def test_classify_short_row():
    cr = classify_row(_row(SHORT_ASSISTANT))
    assert cr.verbosity == "short"
    assert cr.thought_chars == 0
    assert cr.asst_chars <= SHORT_ASST_CHARS_MAX


def test_classify_long_row():
    cr = classify_row(_row(LONG_ASSISTANT))
    # The fixture has >SHORT_THOUGHT_CHARS_MAX? Let's assert long by construction.
    # LONG_ASSISTANT has ~130 chars of thought, which is below 200; but we pad.
    long_asst = (
        "<|channel>thought\n" + ("x " * 500) + "<channel|>"
        "<|tool_call>call:commit_play{domino_id:1}<tool_call|>"
    )
    cr = classify_row(_row(long_asst))
    assert cr.verbosity == "long"
    assert cr.thought_chars > SHORT_THOUGHT_CHARS_MAX


def test_classify_long_when_no_commit():
    # Even if tiny, without commit_play it's treated as long (not a valid short).
    a = "<|tool_call>call:is_legal{domino_id:1}<tool_call|>"
    cr = classify_row(_row(a))
    assert cr.verbosity == "long"


def test_classify_partitions_rows():
    rows = [_row(LONG_ASSISTANT), _row(SHORT_ASSISTANT), _row(SHORT_ASSISTANT)]
    longs, shorts = classify(rows)
    assert len(longs) == 1
    assert len(shorts) == 2


# --- Shortener ---------------------------------------------------------------

def test_shorten_assistant_strips_thought():
    out = shorten_assistant(LONG_ASSISTANT)
    assert "<|channel>thought" not in out
    assert "<channel|>" not in out
    assert "I will commit to play 14" not in out


def test_shorten_preserves_tool_calls_and_commit():
    out = shorten_assistant(LONG_ASSISTANT)
    assert "call:is_legal{domino_id:14}" in out
    assert "call:commit_play{domino_id:14}" in out
    assert '"legal":true' in out


def test_shorten_is_idempotent():
    once = shorten_assistant(LONG_ASSISTANT)
    twice = shorten_assistant(once)
    assert once == twice


def test_shorten_strips_orphan_channel_close():
    # Gemma sometimes emits an extra <channel|> without a matching open.
    asst = (
        "<|channel>thought\nhello<channel|>"
        "<|tool_call>call:commit_play{domino_id:1}<tool_call|><channel|>"
    )
    out = shorten_assistant(asst)
    assert "<channel|>" not in out
    assert "call:commit_play{domino_id:1}" in out


def test_shorten_output_shorter_than_input():
    assert len(shorten_assistant(LONG_ASSISTANT)) < len(LONG_ASSISTANT)


def test_shorten_row_marks_synthetic():
    row = _row(LONG_ASSISTANT)
    short = shorten_row(row)
    assert short["synthetic"] is True
    assert short["verbosity"] == "short_synthetic"
    # Original untouched.
    assert "synthetic" not in row or row.get("synthetic") is False or row["synthetic"] is not True
    # Assistant content is shortened.
    assert "<|channel>thought" not in short["messages"][1]["content"]
    # User message preserved verbatim.
    assert short["messages"][0]["content"] == USER_MSG


def test_shorten_row_preserves_metadata():
    row = _row(LONG_ASSISTANT, seed=900010, declaration=3)
    short = shorten_row(row)
    assert short["seed"] == 900010
    assert short["declaration"] == 3
    assert short["source"] == "rollout_win"


# --- Blender ----------------------------------------------------------------

def test_blend_hits_target_ratio_within_one():
    longs = [_row(LONG_ASSISTANT, seed=i) for i in range(60)]
    shorts = [_row(SHORT_ASSISTANT, seed=1000 + i) for i in range(100)]
    out = blend(longs, shorts, target_short_ratio=0.33, seed=1)
    n_short = sum(1 for r in out if r["messages"][1]["content"] == SHORT_ASSISTANT)
    # 0.33 * 60 / (1 - 0.33) ≈ 29.55 → 30 short.  30 / 90 = 0.333
    assert abs(n_short / len(out) - 0.33) < 0.02
    assert len(out) == 60 + 30


def test_blend_uses_all_shorts_when_short_pool_insufficient():
    longs = [_row(LONG_ASSISTANT)] * 100
    shorts = [_row(SHORT_ASSISTANT)] * 5
    out = blend(longs, shorts, target_short_ratio=0.5, seed=1)
    # Would need 100 shorts for 50%, only 5 available.
    assert len(out) == 105


def test_blend_deterministic_with_seed():
    longs = [_row(LONG_ASSISTANT, seed=i) for i in range(10)]
    shorts = [_row(SHORT_ASSISTANT, seed=1000 + i) for i in range(10)]
    a = blend(longs, shorts, 0.33, seed=42)
    b = blend(longs, shorts, 0.33, seed=42)
    assert [r["seed"] for r in a] == [r["seed"] for r in b]


def test_blend_zero_ratio_drops_shorts():
    longs = [_row(LONG_ASSISTANT)] * 5
    shorts = [_row(SHORT_ASSISTANT)] * 5
    out = blend(longs, shorts, target_short_ratio=0.0, seed=1)
    assert len(out) == 5
    assert all(r["messages"][1]["content"] == LONG_ASSISTANT for r in out)


def test_blend_rejects_bad_ratio():
    with pytest.raises(ValueError):
        blend([], [], target_short_ratio=1.5)
    with pytest.raises(ValueError):
        blend([], [], target_short_ratio=-0.1)


# --- End-to-end over a temp file --------------------------------------------

def test_build_and_write(tmp_path: Path):
    src = tmp_path / "src.jsonl"
    rows = [_row(LONG_ASSISTANT, seed=i) for i in range(12)]
    rows.append(_row(SHORT_ASSISTANT, seed=999))  # one natural short
    write_corpus(src, rows)

    blended, stats = build_blended_corpus(
        [src], target_short_ratio=0.33, seed=7
    )
    assert stats["n_raw"] == 13
    assert stats["n_natural_long"] == 12
    assert stats["n_natural_short"] == 1
    assert stats["n_synthetic_short"] == 12  # one per long
    # 0.33 * 12 / 0.67 ≈ 5.91 → 6 shorts.
    assert stats["n_blended_short"] in (5, 6)
    assert stats["n_blended_long"] == 12

    out = tmp_path / "blended.jsonl"
    n = write_corpus(out, blended)
    assert n == len(blended)
    # Round-trip.
    loaded = load_corpus([out])
    assert len(loaded) == n
    # Synthetic rows carry the flag.
    synthetic = [r for r in loaded if r.get("synthetic") is True]
    assert len(synthetic) >= 1
    assert all(
        "<|channel>thought" not in r["messages"][1]["content"]
        for r in synthetic
    )
