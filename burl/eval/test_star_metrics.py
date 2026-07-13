from __future__ import annotations

import json

from burl.eval.star_metrics import (
    aggregate_rescore,
    classify_bucket,
    load_eval_rows,
    rescore_rows,
    summarize_eval_rows,
)


def test_summarize_eval_rows_reports_signed_delta_and_regret():
    rows = [
        {"eq_delta_vs_bot": 2.0, "matches_bot": False, "final_play": 1},
        {"eq_delta_vs_bot": 0.0, "matches_bot": True, "final_play": 2},
        {
            "eq_delta_vs_bot": -5.0,
            "matches_bot": False,
            "final_play": 3,
            "forced_commit": True,
            "thought_block_present": True,
        },
    ]

    summary = summarize_eval_rows(
        rows,
        adapter=None,
        variant_name="D_required_first",
        indices=[0, 1, 2],
        wall_s=12.34,
        batch_size=6,
        max_tokens=2048,
        turn_cap=12,
    )

    assert summary["mean_signed_eq_delta"] == -1.0
    assert summary["mean_abs_eq_delta"] == 7.0 / 3.0
    assert summary["mean_oracle_regret"] == 5.0 / 3.0
    assert summary["n_delta_win"] == 1
    assert summary["n_delta_tie"] == 1
    assert summary["n_delta_loss"] == 1
    assert summary["n_forced_commit"] == 1
    assert summary["thought_block_rate"] == 1 / 3


def test_load_eval_rows_marks_actual_thinking_events_not_belief_calls(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "decision_0").mkdir()
    (eval_dir / "decision_1").mkdir()
    (eval_dir / "decision_0" / "events.jsonl").write_text(
        json.dumps({"kind": "tool_call", "tool": "belief_trajectory"}) + "\n"
    )
    (eval_dir / "decision_1" / "events.jsonl").write_text(
        json.dumps({"kind": "thinking", "content": "real thought"}) + "\n"
    )
    summary_path = eval_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "global_indices": [0, 1],
        "rows": [
            {"global_idx": 0, "belief_called_turns": [1], "final_play": 1},
            {"global_idx": 1, "belief_called_turns": [], "final_play": 2},
        ],
    }))

    rows, _ = load_eval_rows(eval_summary=summary_path)

    assert rows[0]["thought_block_present"] is False
    assert rows[1]["thought_block_present"] is True


def test_rescore_uses_oracle_regret_and_real_thought_flag():
    eval_rows = [
        {
            "global_idx": 0,
            "final_play": 10,
            "bot_play": 11,
            "burl_eq": 3.0,
            "eq_delta_vs_bot": -2.0,
            "legal_final": True,
            "matches_bot": False,
            "belief_called_turns": [1],
            "thought_block_present": False,
        },
        {
            "global_idx": 1,
            "final_play": 21,
            "bot_play": 20,
            "burl_eq": 7.0,
            "eq_delta_vs_bot": 1.0,
            "legal_final": True,
            "matches_bot": False,
            "thought_block_present": True,
        },
    ]
    corpus = {
        0: {
            "pi_play": 11,
            "qmean_play": 11,
            "oracle_play": 11,
            "burl_play": 10,
            "bucket": "BURL_BREAKS_CONSENSUS",
        },
        1: {
            "pi_play": 20,
            "qmean_play": 21,
            "oracle_play": 21,
            "burl_play": 20,
            "bucket": "QMEAN_ALONE_FIXES",
        },
    }
    per_decision = {
        0: {"oracle_best_eq": 5.0, "e_q": [3.0, 5.0], "legal_mask": [1, 1]},
        1: {"oracle_best_eq": 7.0, "e_q": [6.0, 7.0], "legal_mask": [1, 1]},
    }
    slot_to_dom = {0: [10, 11], 1: [20, 21]}

    rows, diag = rescore_rows(
        eval_rows=eval_rows,
        corpus=corpus,
        per_decision=per_decision,
        slot_to_dom=slot_to_dom,
    )
    aggregate = aggregate_rescore(rows)

    assert diag == {"n_rescored": 2, "n_skipped": 0}
    assert [r["oracle_regret"] for r in rows] == [2.0, 0.0]
    assert aggregate["mean_oracle_regret"] == 1.0
    assert aggregate["mean_signed_delta"] == -0.5
    assert aggregate["thought_block_rate"] == 0.5
    assert rows[0]["adapter_bucket"] == "BURL_BREAKS_CONSENSUS"
    assert rows[1]["adapter_bucket"] == "BOTH_FIX"


def test_classify_bucket_forced_commit_overrides_everything():
    assert classify_bucket(
        pi_play=1,
        qmean_play=1,
        burl_play=1,
        oracle_play=1,
        forced_commit=True,
        legal_final=True,
    ) == "FORCED_COMMIT"
