"""STaR-style filter over candlewax spike traces.

Reads all ``events.jsonl`` files under a ``live_dir`` produced by
``qwen_batch.py`` and emits a SFT corpus JSONL containing only "good"
traces. "Good" is defined by three criteria (AND):

1. The commit landed — a ``commit`` event exists with ``final_play >= 0``.
2. K1 soft-margin — ``meta.per_play_eq[str(final_play)]`` is within
   ``--k1-epsilon`` (default 1.0) of ``meta.bot_eq``.
3. Prediction correctness — if a ``prediction_check`` event exists with
   ``all_correct=True`` it passes; ``False`` fails; ``None`` or missing is
   a warning (passes unless ``--require-prediction`` is set).

The completion field is the model's own good output replayed back as an
SFT target: thinking (if nonempty) + assistant_text + the final
``<tool_call>`` block. The ``<prediction>`` tags emitted in the
assistant_text are preserved verbatim.

Run:
    python -u -m burl.candlewax_spike.star_filter \\
        --live-dir scratch/candlewax_spike/live_qwen36 \\
        --output scratch/candlewax_spike/star_corpus_v1.jsonl \\
        --k1-epsilon 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_SNAPSHOTS_MANIFEST = Path("scratch/candlewax_spike/snapshots/index.jsonl")


@dataclass
class TraceEvents:
    """Parsed bag of events from one events.jsonl file."""
    meta: dict | None = None
    prompt_system: dict | None = None
    prompt_user: dict | None = None
    thinking: list[dict] = field(default_factory=list)
    assistant_text: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    commit: dict | None = None
    prediction_check: dict | None = None
    other: list[dict] = field(default_factory=list)


def _parse_events(path: Path) -> TraceEvents:
    """Parse one events.jsonl. Malformed lines are skipped with a warning."""
    events = TraceEvents()
    with path.open() as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[star_filter] WARN {path}:{lineno} skipping malformed "
                      f"JSON: {e}", file=sys.stderr)
                continue
            kind = rec.get("kind")
            if kind == "meta":
                events.meta = rec
            elif kind == "prompt_system":
                events.prompt_system = rec
            elif kind == "prompt_user":
                events.prompt_user = rec
            elif kind == "thinking":
                events.thinking.append(rec)
            elif kind == "assistant_text":
                events.assistant_text.append(rec)
            elif kind == "tool_call":
                events.tool_calls.append(rec)
            elif kind == "commit":
                events.commit = rec
            elif kind == "prediction_check":
                events.prediction_check = rec
            else:
                events.other.append(rec)
    return events


def _load_snapshots_manifest(path: Path) -> dict[int, str]:
    """Map decision idx -> png path. Missing manifest is fine — returns {}."""
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "idx" in rec and "png" in rec:
                out[int(rec["idx"])] = str(rec["png"])
    return out


def _format_tool_call_block(tc: dict) -> str:
    """Re-serialize a tool_call event as an OpenAI-style <tool_call> block.

    Qwen's Hermes chat template emits tool calls inside
    ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``. We keep the
    same shape so the SFT target matches the model's native output format.
    """
    payload = {"name": tc.get("tool"), "arguments": tc.get("args")}
    return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"


def _build_completion(events: TraceEvents) -> str:
    """Concatenate thinking + assistant_text + tool_call block(s).

    Preserves any ``<prediction>`` tags already inside assistant_text.
    """
    parts: list[str] = []
    for t in events.thinking:
        content = (t.get("content") or "").strip()
        if content:
            parts.append(f"<think>\n{content}\n</think>")
    for at in events.assistant_text:
        content = at.get("content") or ""
        if content:
            parts.append(content)
    for tc in events.tool_calls:
        parts.append(_format_tool_call_block(tc))
    return "\n".join(parts)


@dataclass
class FilterResult:
    accepted: bool
    reasons: list[str]
    bot_match: bool
    eq_delta: float | None
    prediction_correct: bool | None
    final_play: int
    bot_play: int


def _apply_filter(
    events: TraceEvents,
    *,
    k1_epsilon: float,
    require_prediction: bool,
    prediction_strictness: str = "winner",
) -> FilterResult:
    reasons: list[str] = []

    meta = events.meta or {}
    bot_play = int(meta.get("bot_play", -1)) if meta else -1

    # 1. Commit landed
    final_play = -1
    if events.commit is None:
        reasons.append("no_commit")
    else:
        final_play = int(events.commit.get("final_play", -1))
        if final_play < 0:
            reasons.append("no_commit")

    # 2. K1 soft-margin
    eq_delta: float | None = None
    if final_play >= 0:
        per_play_eq = (meta.get("per_play_eq") or {}) if meta else {}
        bot_eq = meta.get("bot_eq") if meta else None
        chosen_eq = per_play_eq.get(str(final_play))
        if bot_eq is None or chosen_eq is None:
            reasons.append("k1_missing_eq")
        else:
            eq_delta = float(chosen_eq) - float(bot_eq)
            # Pass if chosen_eq >= bot_eq - epsilon  <=>  eq_delta >= -epsilon.
            if eq_delta < -float(k1_epsilon):
                reasons.append("k1_failed")

    # 3. Prediction correctness — controllable strictness.
    #
    # Qwen is good at naming WHO wins a trick (correct-by-symmetry reasoning:
    # "highest trump wins") but bad at the count arithmetic (summing 5s/10s).
    # Strict "all three fields match" rejects everything. Use strictness:
    #  - "all": winner + mine + theirs exact.
    #  - "winner" (default): only winner_seat must match. Count-math errors
    #    are still flagged but don't reject.
    #  - "none": no prediction gate at all — just K1 + commit.
    prediction_correct: bool | None = None
    if events.prediction_check is not None:
        if prediction_strictness == "none":
            # Report for the record but don't gate.
            ac = events.prediction_check.get("all_correct")
            prediction_correct = bool(ac) if ac is not None else None
        elif prediction_strictness == "winner":
            wc = events.prediction_check.get("winner_correct")
            prediction_correct = bool(wc) if wc is not None else None
            if wc is False:
                reasons.append("prediction_winner_wrong")
            elif wc is None and require_prediction:
                reasons.append("prediction_missing")
        else:  # "all"
            ac = events.prediction_check.get("all_correct")
            if ac is True:
                prediction_correct = True
            elif ac is False:
                prediction_correct = False
                reasons.append("prediction_wrong")
            else:
                prediction_correct = None
                if require_prediction:
                    reasons.append("prediction_missing")
    else:
        prediction_correct = None
        if require_prediction:
            reasons.append("prediction_missing")

    bot_match = (final_play >= 0 and bot_play >= 0 and final_play == bot_play)

    return FilterResult(
        accepted=not reasons,
        reasons=reasons,
        bot_match=bot_match,
        eq_delta=eq_delta,
        prediction_correct=prediction_correct,
        final_play=final_play,
        bot_play=bot_play,
    )


def _corpus_row(
    events: TraceEvents,
    result: FilterResult,
    snapshots: dict[int, str],
) -> dict:
    meta = events.meta or {}
    idx = int(meta.get("idx", -1))
    mode = meta.get("mode", "")
    seed = meta.get("seed")
    image_path: str | None = None
    if mode == "multimodal":
        image_path = snapshots.get(idx)

    system_prompt = (events.prompt_system or {}).get("content", "")
    user_prompt = (events.prompt_user or {}).get("content", "")
    completion = _build_completion(events)

    return {
        "decision_idx": idx,
        "mode": mode,
        "seed": seed,
        "image_path": image_path,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "completion": completion,
        "filter_metadata": {
            "bot_match": bool(result.bot_match),
            "eq_delta": (None if result.eq_delta is None
                         else float(result.eq_delta)),
            "prediction_correct": result.prediction_correct,
            "final_play": int(result.final_play),
            "bot_play": int(result.bot_play),
        },
    }


def _format_example(
    run_id: str, events: TraceEvents, result: FilterResult
) -> str:
    meta = events.meta or {}
    idx = meta.get("idx")
    mode = meta.get("mode")
    return (
        f"  run_id={run_id} idx={idx} mode={mode} "
        f"final_play={result.final_play} bot_play={result.bot_play} "
        f"eq_delta={result.eq_delta} "
        f"prediction_correct={result.prediction_correct} "
        f"reasons={result.reasons or ['ok']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-dir", type=Path, required=True,
                        help="Directory containing <id>_<mode>/events.jsonl")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to emit SFT corpus JSONL")
    parser.add_argument("--k1-epsilon", type=float, default=1.0,
                        help="K1 soft-margin: pass if chosen_eq >= bot_eq - eps")
    parser.add_argument("--require-prediction", action="store_true",
                        help="Treat missing/None prediction_check as a reject "
                             "(default: warn + pass)")
    parser.add_argument("--prediction-strictness",
                        choices=["all", "winner", "none"], default="winner",
                        help="'all' = all three fields match; 'winner' = only "
                             "winner_seat must match (default, counts are "
                             "reported but not gated); 'none' = no prediction "
                             "gate at all.")
    parser.add_argument("--snapshots-manifest", type=Path,
                        default=DEFAULT_SNAPSHOTS_MANIFEST,
                        help="Manifest mapping idx->png (used for image_path)")
    args = parser.parse_args()

    live_dir: Path = args.live_dir
    if not live_dir.is_dir():
        print(f"[star_filter] ERROR live-dir not a directory: {live_dir}",
              file=sys.stderr)
        sys.exit(2)

    snapshots = _load_snapshots_manifest(args.snapshots_manifest)

    # Collect (run_id, events_path) pairs in sorted order for stable output.
    run_dirs = sorted(
        p for p in live_dir.iterdir()
        if p.is_dir() and (p / "events.jsonl").exists()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    accepted = 0
    rejected = 0
    reason_counts: Counter[str] = Counter()
    warnings = 0
    first_accepted: tuple[str, TraceEvents, FilterResult] | None = None
    first_rejected: tuple[str, TraceEvents, FilterResult] | None = None

    with args.output.open("w") as out_fh:
        for run_dir in run_dirs:
            total += 1
            run_id = run_dir.name
            events_path = run_dir / "events.jsonl"
            try:
                events = _parse_events(events_path)
            except OSError as e:
                print(f"[star_filter] WARN could not read {events_path}: {e}",
                      file=sys.stderr)
                rejected += 1
                reason_counts["read_error"] += 1
                continue

            if events.meta is None:
                rejected += 1
                reason_counts["no_meta"] += 1
                continue

            result = _apply_filter(
                events,
                k1_epsilon=args.k1_epsilon,
                require_prediction=args.require_prediction,
                prediction_strictness=args.prediction_strictness,
            )

            # "warning" = prediction missing but not required
            if (events.prediction_check is None
                    or events.prediction_check.get("all_correct") is None):
                warnings += 1

            if result.accepted:
                accepted += 1
                row = _corpus_row(events, result, snapshots)
                out_fh.write(json.dumps(row) + "\n")
                if first_accepted is None:
                    first_accepted = (run_id, events, result)
            else:
                rejected += 1
                for r in result.reasons:
                    reason_counts[r] += 1
                if first_rejected is None:
                    first_rejected = (run_id, events, result)

    # ---------- Report ----------
    print(f"[star_filter] live_dir         : {live_dir}", file=sys.stderr)
    print(f"[star_filter] output           : {args.output}", file=sys.stderr)
    print(f"[star_filter] k1_epsilon       : {args.k1_epsilon}", file=sys.stderr)
    print(f"[star_filter] require_prediction: {args.require_prediction}",
          file=sys.stderr)
    print(f"[star_filter] total traces    : {total}", file=sys.stderr)
    print(f"[star_filter] accepted        : {accepted}", file=sys.stderr)
    print(f"[star_filter] rejected        : {rejected}", file=sys.stderr)
    print(f"[star_filter] prediction warnings (missing/None): {warnings}",
          file=sys.stderr)
    if reason_counts:
        print("[star_filter] rejection reasons:", file=sys.stderr)
        for reason, count in reason_counts.most_common():
            print(f"                {reason}: {count}", file=sys.stderr)
    else:
        print("[star_filter] rejection reasons: (none)", file=sys.stderr)

    if first_accepted is not None:
        run_id, events, result = first_accepted
        print("[star_filter] example accepted:", file=sys.stderr)
        print(_format_example(run_id, events, result), file=sys.stderr)
    else:
        print("[star_filter] example accepted: (none)", file=sys.stderr)

    if first_rejected is not None:
        run_id, events, result = first_rejected
        print("[star_filter] example rejected:", file=sys.stderr)
        print(_format_example(run_id, events, result), file=sys.stderr)
    else:
        print("[star_filter] example rejected: (none)", file=sys.stderr)


if __name__ == "__main__":
    main()
