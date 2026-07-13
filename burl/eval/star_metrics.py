"""STaR eval metrics and post-hoc oracle-regret rescoring for Burl.

This module promotes the load-bearing metric logic from the experimental
``scratch/belief_trajectory_rollout/star`` scripts into tracked code.  It is
pure Python and read-only: no model load, no GPU, no mutation of harvest dirs.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


BUCKETS = [
    "ALL_AGREE_CORRECT",
    "ALL_AGREE_WRONG",
    "BURL_ALONE_FIXES",
    "QMEAN_ALONE_FIXES",
    "BOTH_FIX",
    "BURL_PARROTS_PI_WRONG",
    "BURL_INDEPENDENT_WRONG",
    "BURL_INDEPENDENT_RIGHT",
    "BURL_FOLLOWS_PI_RIGHT",
    "BURL_BREAKS_CONSENSUS",
    "BURL_PARROTS_QMEAN_WRONG",
    "BURL_DRIFTS_FROM_PI",
    "FORCED_COMMIT",
    "ILLEGAL",
    "OTHER",
]


def mean_or_none(values: list[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def rounded_mean(values: list[float | None], digits: int = 4) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), digits) if vals else None


def classify_bucket(
    *,
    pi_play: int | None,
    qmean_play: int | None,
    burl_play: int | None,
    oracle_play: int | None,
    forced_commit: bool,
    legal_final: bool = True,
) -> str:
    """Classify one Burl decision against pi/Q-mean/oracle references."""
    if forced_commit:
        return "FORCED_COMMIT"
    if burl_play is None or burl_play == -1 or not legal_final:
        return "ILLEGAL"

    burl_ok = burl_play == oracle_play
    pi_ok = pi_play == oracle_play
    qm_ok = qmean_play == oracle_play
    bu_eq_pi = burl_play == pi_play
    bu_eq_qm = burl_play == qmean_play
    pi_eq_qm = pi_play == qmean_play

    if burl_ok and pi_ok and qm_ok:
        return "ALL_AGREE_CORRECT"
    if not burl_ok and not pi_ok and not qm_ok and burl_play == pi_play == qmean_play:
        return "ALL_AGREE_WRONG"

    if burl_ok:
        if pi_ok and not qm_ok:
            return "BURL_FOLLOWS_PI_RIGHT"
        if qm_ok and not pi_ok:
            return "BOTH_FIX"
        if pi_eq_qm:
            return "BURL_INDEPENDENT_RIGHT"
        return "BURL_ALONE_FIXES"

    if pi_ok:
        if qm_ok:
            return "BURL_BREAKS_CONSENSUS"
        if bu_eq_qm:
            return "BURL_PARROTS_QMEAN_WRONG"
        return "BURL_DRIFTS_FROM_PI"

    if qm_ok:
        if bu_eq_pi:
            return "BURL_PARROTS_PI_WRONG"
        return "QMEAN_ALONE_FIXES"

    if bu_eq_pi:
        return "BURL_PARROTS_PI_WRONG"
    return "BURL_INDEPENDENT_WRONG"


def summarize_eval_rows(
    rows: list[dict[str, Any]],
    *,
    adapter: Path | None,
    variant_name: str,
    indices: list[int],
    wall_s: float,
    batch_size: int,
    max_tokens: int,
    turn_cap: int | None = None,
) -> dict[str, Any]:
    """Aggregate per-decision eval rows with regret-facing metrics.

    ``eq_delta_vs_bot`` is signed as model E[Q] minus bot/oracle E[Q].
    Oracle-relative regret for the rollout bot reference is therefore
    ``max(0, -delta)``.  Keeping signed, absolute, and regret views prevents
    bot-match from hiding a lossy disagreement split.
    """
    n = len(rows)
    deltas = [
        float(r["eq_delta_vs_bot"])
        for r in rows
        if isinstance(r.get("eq_delta_vs_bot"), (int, float))
    ]
    wins = [d for d in deltas if d > 0]
    ties = [d for d in deltas if d == 0]
    losses = [d for d in deltas if d < 0]
    abs_deltas = [abs(d) for d in deltas]
    regrets = [max(0.0, -d) for d in deltas]

    n_match = sum(1 for r in rows if r.get("matches_bot"))
    n_legal = sum(1 for r in rows if r.get("final_play") is not None)
    n_error = sum(1 for r in rows if r.get("error"))
    n_belief = sum(1 for r in rows if r.get("belief_called_turns"))
    n_forced = sum(1 for r in rows if r.get("forced_commit"))
    n_bailed = sum(1 for r in rows if r.get("bailed"))
    n_thought = sum(1 for r in rows if thought_block_present(r))
    n_token_cap = sum(1 for r in rows if r.get("hit_token_cap"))

    mean_signed = mean_or_none(deltas)
    return {
        "adapter": str(adapter) if adapter else None,
        "variant": variant_name,
        "batch_size": batch_size,
        "max_tokens": max_tokens,
        "turn_cap": turn_cap,
        "n_decisions": n,
        "global_indices": indices,
        "n_match": n_match,
        "n_legal": n_legal,
        "n_error": n_error,
        "n_belief_used": n_belief,
        "n_thought_block": n_thought,
        "n_forced_commit": n_forced,
        "n_bailed": n_bailed,
        "n_hit_token_cap": n_token_cap,
        "match_rate": n_match / n if n else None,
        "legal_rate": n_legal / n if n else None,
        "error_rate": n_error / n if n else None,
        "belief_rate": n_belief / n if n else None,
        "thought_block_rate": n_thought / n if n else None,
        "forced_commit_rate": n_forced / n if n else None,
        "hit_token_cap_rate": n_token_cap / n if n else None,
        "mean_signed_eq_delta": mean_signed,
        "mean_eq_delta": mean_signed,
        "mean_abs_eq_delta": mean_or_none(abs_deltas),
        "mean_oracle_regret": mean_or_none(regrets),
        "n_delta_win": len(wins),
        "n_delta_tie": len(ties),
        "n_delta_loss": len(losses),
        "delta_win_rate": len(wins) / len(deltas) if deltas else None,
        "delta_tie_rate": len(ties) / len(deltas) if deltas else None,
        "delta_loss_rate": len(losses) / len(deltas) if deltas else None,
        "rows": rows,
        "wall_s": round(wall_s, 1),
    }


def events_have_thought_block(events_path: Path) -> bool:
    """Return True when a decision event log contains a thinking event."""
    if not events_path.exists():
        return False
    try:
        with events_path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("kind") == "thinking":
                    return True
    except (OSError, json.JSONDecodeError):
        return False
    return False


def thought_block_present(row: dict[str, Any]) -> bool:
    """Read thought presence from explicit thought metadata only.

    Belief-tool usage is intentionally not a proxy for a thought block.  Run-3c
    exposed this footgun: almost every decision called ``belief_trajectory``,
    while the actual thought-block count had to be read from ``events.jsonl``.
    """
    if "thought_block_present" in row:
        return bool(row["thought_block_present"])
    if "n_thinking_events" in row:
        return int(row["n_thinking_events"] or 0) > 0
    return False


def load_eval_rows(
    *,
    eval_summary: Path | None = None,
    eval_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load eval rows from ``summary.json`` or reconstruct from decision dirs."""
    if eval_summary is not None and eval_summary.exists():
        payload = json.loads(eval_summary.read_text())
        header = {k: v for k, v in payload.items() if k != "rows"}
        rows = list(payload.get("rows", []))
        idxs = payload.get("global_indices") or []
        if rows and "global_idx" not in rows[0] and idxs:
            for row, gi in zip(rows, idxs):
                row["global_idx"] = int(gi)
        base_dir = eval_dir or eval_summary.parent
        _annotate_thought_blocks(rows, base_dir)
        return rows, header

    if eval_dir is None:
        raise ValueError("eval_dir is required when eval_summary is absent")

    rows: list[dict[str, Any]] = []
    for sub in sorted(eval_dir.glob("decision_*"), key=_decision_dir_sort_key):
        trace_summary = sub / "trace_summary.json"
        if not trace_summary.exists() or trace_summary.stat().st_size == 0:
            continue
        try:
            row = json.loads(trace_summary.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        gi = int(sub.name.split("_", 1)[1])
        row["global_idx"] = gi
        row["thought_block_present"] = events_have_thought_block(sub / "events.jsonl")
        rows.append(row)

    header = {
        "adapter": str(eval_dir),
        "n_decisions_dirs": len(list(eval_dir.glob("decision_*"))),
        "n_decisions_parsed": len(rows),
        "_reconstructed_from_decision_dirs": True,
    }
    return rows, header


def _annotate_thought_blocks(rows: list[dict[str, Any]], eval_dir: Path) -> None:
    for row in rows:
        gi = row.get("global_idx")
        if gi is None:
            continue
        row["thought_block_present"] = events_have_thought_block(
            eval_dir / f"decision_{int(gi)}" / "events.jsonl"
        )


def _decision_dir_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except Exception:
        return 10**12


def load_jsonl_by_global_idx(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["global_idx"])] = row
    return out


def eq_at_play(
    *,
    play: int | None,
    slot_to_dom: list[int],
    e_q: list[float],
    legal_mask: list[int],
) -> float | None:
    if play is None:
        return None
    for slot, dom_id in enumerate(slot_to_dom):
        if int(dom_id) == int(play) and bool(legal_mask[slot]):
            return float(e_q[slot])
    return None


def rescore_rows(
    *,
    eval_rows: list[dict[str, Any]],
    corpus: dict[int, dict[str, Any]],
    per_decision: dict[int, dict[str, Any]],
    slot_to_dom: dict[int, list[int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rescored: list[dict[str, Any]] = []
    skipped = 0
    for eval_row in eval_rows:
        gi = int(eval_row["global_idx"])
        corpus_row = corpus.get(gi)
        per_row = per_decision.get(gi)
        slots = slot_to_dom.get(gi)
        if corpus_row is None or per_row is None or slots is None:
            skipped += 1
            continue

        adapter_play = eval_row.get("final_play")
        adapter_eq = eval_row.get("burl_eq")
        oracle_best_eq = float(per_row["oracle_best_eq"])
        adapter_eq_from_oracle = eq_at_play(
            play=adapter_play,
            slot_to_dom=slots,
            e_q=per_row["e_q"],
            legal_mask=per_row["legal_mask"],
        )
        if adapter_eq is None:
            adapter_eq = adapter_eq_from_oracle

        eq_delta = eval_row.get("eq_delta_vs_bot")
        forced_commit = bool(eval_row.get("forced_commit", False))
        legal_final = bool(eval_row.get("legal_final", False))
        oracle_play = int(corpus_row["oracle_play"])
        pi_play = int(corpus_row["pi_play"])
        qmean_play = int(corpus_row["qmean_play"])
        base_burl_play = corpus_row.get("burl_play")

        adapter_bucket = classify_bucket(
            pi_play=pi_play,
            qmean_play=qmean_play,
            burl_play=int(adapter_play) if adapter_play is not None else None,
            oracle_play=oracle_play,
            forced_commit=forced_commit,
            legal_final=legal_final,
        )
        oracle_regret = (
            oracle_best_eq - float(adapter_eq)
            if isinstance(adapter_eq, (int, float)) else None
        )

        rescored.append({
            "global_idx": gi,
            "adapter_play": adapter_play,
            "base_burl_play": base_burl_play,
            "pi_play": pi_play,
            "qmean_play": qmean_play,
            "oracle_play": oracle_play,
            "bot_play": eval_row.get("bot_play"),
            "k1_pass": (
                isinstance(eq_delta, (int, float)) and float(eq_delta) >= 0.0
            ),
            "signed_delta": (
                float(eq_delta) if isinstance(eq_delta, (int, float)) else None
            ),
            "abs_delta": (
                abs(float(eq_delta)) if isinstance(eq_delta, (int, float)) else None
            ),
            "adapter_eq": adapter_eq,
            "adapter_eq_from_oracle": adapter_eq_from_oracle,
            "oracle_best_eq": oracle_best_eq,
            "oracle_regret": oracle_regret,
            "matches_bot": bool(eval_row.get("matches_bot")),
            "matches_oracle": (
                int(adapter_play) == oracle_play if adapter_play is not None else False
            ),
            "matches_pi": (
                int(adapter_play) == pi_play if adapter_play is not None else False
            ),
            "matches_base_burl": (
                int(adapter_play) == int(base_burl_play)
                if adapter_play is not None and base_burl_play is not None
                else False
            ),
            "forced_commit": forced_commit,
            "legal_final": legal_final,
            "base_bucket": corpus_row.get("bucket", "OTHER"),
            "adapter_bucket": adapter_bucket,
            "thought_block_present": thought_block_present(eval_row),
        })

    return rescored, {"n_rescored": len(rescored), "n_skipped": skipped}


def aggregate_rescore(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    regrets = [r["oracle_regret"] for r in rows if r["oracle_regret"] is not None]
    near_tie = [
        r for r in rows
        if r["oracle_regret"] is not None and r["oracle_regret"] <= 0.5
    ]
    return {
        "n": n,
        "k1_pass_rate": sum(1 for r in rows if r["k1_pass"]) / n if n else None,
        "match_bot_rate": sum(1 for r in rows if r["matches_bot"]) / n if n else None,
        "match_oracle_rate": (
            sum(1 for r in rows if r["matches_oracle"]) / n if n else None
        ),
        "match_pi_rate": sum(1 for r in rows if r["matches_pi"]) / n if n else None,
        "match_base_burl_rate": (
            sum(1 for r in rows if r["matches_base_burl"]) / n if n else None
        ),
        "mean_signed_delta": rounded_mean([r["signed_delta"] for r in rows]),
        "mean_abs_delta": rounded_mean([r["abs_delta"] for r in rows]),
        "mean_oracle_regret": rounded_mean(regrets),
        "near_tie_rate": len(near_tie) / n if n else None,
        "bucket_dist_adapter": dict(Counter(r["adapter_bucket"] for r in rows)),
        "bucket_dist_base": dict(Counter(r["base_bucket"] for r in rows)),
        "thought_block_rate": (
            sum(1 for r in rows if r["thought_block_present"]) / n if n else None
        ),
        "n_forced_commit": sum(1 for r in rows if r["forced_commit"]),
        "n_illegal": sum(1 for r in rows if not r["legal_final"]),
    }


def bucket_flip_matrix(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        base_bucket = row["base_bucket"]
        adapter_bucket = row["adapter_bucket"]
        matrix.setdefault(base_bucket, {})
        matrix[base_bucket][adapter_bucket] = (
            matrix[base_bucket].get(adapter_bucket, 0) + 1
        )
    return matrix


def match_by_base_bucket(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for bucket in sorted({r["base_bucket"] for r in rows}):
        subset = [r for r in rows if r["base_bucket"] == bucket]
        out[bucket] = aggregate_rescore(subset)
    return out


def write_rescore_outputs(
    *,
    out_json: Path,
    out_rows_jsonl: Path | None,
    label: str,
    eval_header: dict[str, Any],
    diag: dict[str, int],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "label": label,
        "eval_header": eval_header,
        "diag": diag,
        "aggregate": aggregate_rescore(rows),
        "bucket_flip_matrix": bucket_flip_matrix(rows),
        "match_by_base_bucket": match_by_base_bucket(rows),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    if out_rows_jsonl is not None:
        out_rows_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with out_rows_jsonl.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Post-hoc STaR eval rescore with oracle-regret metrics."
    )
    ap.add_argument("--eval-summary", type=Path, default=None)
    ap.add_argument("--eval-dir", type=Path, default=None)
    ap.add_argument("--corpus-index", type=Path, required=True)
    ap.add_argument("--per-decision", type=Path, required=True)
    ap.add_argument("--slot-to-dom", type=Path, required=True)
    ap.add_argument("--label", default="star-eval")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-rows-jsonl", type=Path, default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rows, header = load_eval_rows(eval_summary=args.eval_summary, eval_dir=args.eval_dir)
    corpus = load_jsonl_by_global_idx(args.corpus_index)
    per_decision = load_jsonl_by_global_idx(args.per_decision)
    slot_to_dom_raw = json.loads(args.slot_to_dom.read_text())
    slot_to_dom = {int(k): [int(x) for x in v] for k, v in slot_to_dom_raw.items()}
    rescored, diag = rescore_rows(
        eval_rows=rows,
        corpus=corpus,
        per_decision=per_decision,
        slot_to_dom=slot_to_dom,
    )
    payload = write_rescore_outputs(
        out_json=args.out_json,
        out_rows_jsonl=args.out_rows_jsonl,
        label=args.label,
        eval_header=header,
        diag=diag,
        rows=rescored,
    )
    print(json.dumps(payload["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
