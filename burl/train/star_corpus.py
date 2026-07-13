"""Build Burl STaR SFT corpora from harvested wax-museum traces.

This is the tracked home for the filter-only corpus builder that originated
under ``scratch/belief_trajectory_rollout/star``.  It converts harvested
``events.jsonl`` files into one user/assistant training row per model turn,
preserving Gemma thought blocks for ``burl.train.star_mlx``.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STRICT_BUCKETS = {
    "ALL_AGREE_CORRECT",
    "BURL_ALONE_FIXES",
    "BOTH_FIX",
    "BURL_INDEPENDENT_RIGHT",
    "BURL_FOLLOWS_PI_RIGHT",
}


@dataclass(frozen=True)
class CorpusBuildResult:
    train_rows: list[dict[str, Any]]
    val_rows: list[dict[str, Any]]
    manifest: dict[str, Any]


def load_events(events_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with events_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def per_turn_rows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one SFT row per assistant turn.

    The row's user side is the initial user prompt plus prior tool history in
    Gemma-native text form.  The assistant side is the exact completion for
    the current turn, including ``<|channel>thought ... <channel|>`` when it
    was emitted.
    """
    user0 = next((e for e in events if e.get("kind") == "prompt_user"), None)
    if user0 is None:
        return []
    base_user = str(user0["content"])

    by_turn: dict[int, dict[str, list[dict[str, Any]] | list[str]]] = {}
    for event in events:
        turn = event.get("turn")
        if turn is None:
            continue
        bucket = by_turn.setdefault(
            int(turn),
            {"thinking": [], "assistant_text": [], "tool_calls": [], "tool_results": []},
        )
        kind = event.get("kind")
        if kind == "thinking":
            bucket["thinking"].append(str(event.get("content", "")))  # type: ignore[index]
        elif kind == "assistant_text":
            bucket["assistant_text"].append(str(event.get("content", "")))  # type: ignore[index]
        elif kind == "tool_call":
            bucket["tool_calls"].append(event)  # type: ignore[index]
        elif kind == "tool_result":
            bucket["tool_results"].append(event)  # type: ignore[index]

    rows: list[dict[str, Any]] = []
    history: list[str] = []
    for turn in sorted(by_turn):
        turn_events = by_turn[turn]
        if turn == 0:
            _append_tool_history(history, turn_events)
            continue

        assistant_parts: list[str] = []
        thinking = turn_events["thinking"]
        if thinking:
            assistant_parts.append(
                "<|channel>thought\n" + str(thinking[0]) + "<channel|>"
            )
        assistant_parts.extend(str(x) for x in turn_events["assistant_text"])
        assistant_text = "".join(assistant_parts).strip()
        if not assistant_text:
            continue

        user_text = base_user
        if history:
            user_text = base_user + "\n\n" + "\n".join(history)
        rows.append({
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ]
        })
        _append_tool_history(history, turn_events)
    return rows


def _append_tool_history(
    history: list[str],
    turn_events: dict[str, list[dict[str, Any]] | list[str]],
) -> None:
    for tool_call in turn_events["tool_calls"]:
        args_str = json.dumps(tool_call.get("args", {}), separators=(",", ":"))
        history.append(
            f"<|tool_call>call:{tool_call.get('tool')}{{{args_str}}}<tool_call|>"
        )
    for tool_result in turn_events["tool_results"]:
        history.append(
            f'<|tool_response>response:{tool_result.get("tool")}'
            f'{{value:<|"|>{tool_result.get("content", "")}<|"|>}}'
            f"<tool_response|>"
        )


def load_corpus_index(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_filter_only_corpus(
    *,
    harvest_dir: Path,
    include_buckets: set[str] | None = None,
    min_assistant_chars: int = 0,
    val_frac: float = 0.2,
    seed: int = 0,
) -> CorpusBuildResult:
    index_path = harvest_dir / "corpus_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"missing corpus index: {index_path}")
    keep_buckets = set(include_buckets or STRICT_BUCKETS)
    min_chars = max(0, int(min_assistant_chars))

    skipped = {
        "missing_events": 0,
        "empty_rows": 0,
        "wrong_bucket": 0,
        "rows_filtered_short": 0,
        "decisions_emptied_by_filter": 0,
    }
    bucket_counts = {bucket: 0 for bucket in keep_buckets}
    kept: list[tuple[int, str, list[dict[str, Any]]]] = []

    for index_row in load_corpus_index(index_path):
        bucket = index_row.get("bucket")
        if bucket not in keep_buckets:
            skipped["wrong_bucket"] += 1
            continue
        events_path = harvest_dir / Path(index_row["transcript_path"]).parent / "events.jsonl"
        if not events_path.exists():
            skipped["missing_events"] += 1
            continue
        decision_rows = per_turn_rows(load_events(events_path))
        if not decision_rows:
            skipped["empty_rows"] += 1
            continue
        if min_chars:
            before = len(decision_rows)
            decision_rows = [
                row for row in decision_rows
                if len(row["messages"][1]["content"]) >= min_chars
            ]
            skipped["rows_filtered_short"] += before - len(decision_rows)
            if not decision_rows:
                skipped["decisions_emptied_by_filter"] += 1
                continue

        global_idx = int(index_row["global_idx"])
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        kept.append((global_idx, str(bucket), decision_rows))

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_val = max(1, int(round(len(kept) * val_frac))) if len(kept) > 1 else 0
    val_decisions = kept[:n_val]
    train_decisions = kept[n_val:]

    train_rows = [row for _, _, rows in train_decisions for row in rows]
    val_rows = [row for _, _, rows in val_decisions for row in rows]
    manifest = {
        "harvest_dir": str(harvest_dir),
        "include_buckets": sorted(keep_buckets),
        "min_assistant_chars": min_chars,
        "skipped": skipped,
        "bucket_decision_counts": bucket_counts,
        "n_decisions_total": len(kept),
        "n_decisions_train": len(train_decisions),
        "n_decisions_val": len(val_decisions),
        "n_rows_train": len(train_rows),
        "n_rows_val": len(val_rows),
        "val_frac": val_frac,
        "seed": seed,
        "train_global_idx": [gi for gi, _, _ in train_decisions],
        "val_global_idx": [gi for gi, _, _ in val_decisions],
    }
    return CorpusBuildResult(train_rows=train_rows, val_rows=val_rows, manifest=manifest)


def write_corpus(result: CorpusBuildResult, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    with train_path.open("w") as f:
        for row in result.train_rows:
            f.write(json.dumps(row) + "\n")
    with val_path.open("w") as f:
        for row in result.val_rows:
            f.write(json.dumps(row) + "\n")
    manifest = {
        **result.manifest,
        "train_path": str(train_path),
        "val_path": str(val_path),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-buckets", type=str, nargs="*", default=None)
    ap.add_argument("--min-assistant-chars", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = build_filter_only_corpus(
        harvest_dir=args.harvest,
        include_buckets=set(args.include_buckets) if args.include_buckets else None,
        min_assistant_chars=args.min_assistant_chars,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    manifest = write_corpus(result, args.out_dir)
    printable = {
        k: v for k, v in manifest.items()
        if k not in {"train_global_idx", "val_global_idx"}
    }
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
