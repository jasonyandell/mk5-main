from __future__ import annotations

import json

from burl.train.star_corpus import build_filter_only_corpus, per_turn_rows


def test_per_turn_rows_preserves_thoughts_and_tool_history():
    events = [
        {"kind": "prompt_user", "content": "Pick a domino."},
        {
            "kind": "tool_call",
            "turn": 0,
            "tool": "belief_trajectory",
            "args": {},
        },
        {
            "kind": "tool_result",
            "turn": 0,
            "tool": "belief_trajectory",
            "content": "posterior text",
        },
        {"kind": "thinking", "turn": 1, "content": "I should inspect 6-6."},
        {
            "kind": "assistant_text",
            "turn": 1,
            "content": "<|tool_call>call:explore_game{play:27}<tool_call|>",
        },
        {
            "kind": "tool_call",
            "turn": 1,
            "tool": "explore_game",
            "args": {"play": 27},
        },
        {
            "kind": "tool_result",
            "turn": 1,
            "tool": "explore_game",
            "content": "eq +1.2",
        },
        {
            "kind": "assistant_text",
            "turn": 2,
            "content": "<|tool_call>call:commit_play{domino_id:27}<tool_call|>",
        },
    ]

    rows = per_turn_rows(events)

    assert len(rows) == 2
    assert "posterior text" in rows[0]["messages"][0]["content"]
    assert "<|channel>thought\nI should inspect 6-6.<channel|>" in (
        rows[0]["messages"][1]["content"]
    )
    assert "eq +1.2" in rows[1]["messages"][0]["content"]
    assert "call:commit_play{domino_id:27}" in rows[1]["messages"][1]["content"]


def test_build_filter_only_corpus_uses_manifest_and_min_chars(tmp_path):
    harvest = tmp_path / "harvest"
    decision_dir = harvest / "D_required_first" / "decision_0"
    decision_dir.mkdir(parents=True)
    (harvest / "corpus_index.jsonl").write_text(json.dumps({
        "global_idx": 0,
        "bucket": "ALL_AGREE_CORRECT",
        "transcript_path": "D_required_first/decision_0/transcript.live",
    }) + "\n")
    (decision_dir / "events.jsonl").write_text("\n".join([
        json.dumps({"kind": "prompt_user", "content": "Pick."}),
        json.dumps({"kind": "assistant_text", "turn": 1, "content": "tiny"}),
        json.dumps({
            "kind": "assistant_text",
            "turn": 2,
            "content": "<|tool_call>call:commit_play{domino_id:1}<tool_call|>",
        }),
    ]) + "\n")

    result = build_filter_only_corpus(
        harvest_dir=harvest,
        min_assistant_chars=10,
        val_frac=0.0,
        seed=0,
    )

    assert len(result.train_rows) == 1
    assert result.train_rows[0]["messages"][1]["content"].startswith("<|tool_call>")
    assert result.manifest["n_decisions_total"] == 1
    assert result.manifest["skipped"]["rows_filtered_short"] == 1
    assert result.manifest["bucket_decision_counts"]["ALL_AGREE_CORRECT"] == 1
