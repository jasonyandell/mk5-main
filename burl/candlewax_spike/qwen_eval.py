"""Evaluation harness for local Qwen3.6 on candlewax decisions.

Runs a range of held-out decisions through ``qwen_batch._run_one`` with an
optional LoRA adapter, then computes a summary of agreement with the bot and
prediction-correctness, and writes a markdown table to
``{live_dir}/eval_summary.md``.

Typical use::

    python -m burl.candlewax_spike.qwen_eval \
        --base-model mlx-community/Qwen3.6-35B-A3B-4bit \
        --decisions 20-29 \
        --modes plain multimodal \
        --image-variant existing \
        --live-dir scratch/candlewax_spike/eval_qwen_base

To compare against a LoRA-tuned variant, pass ``--adapter``::

    python -m burl.candlewax_spike.qwen_eval \
        --base-model mlx-community/Qwen3.6-35B-A3B-4bit \
        --adapter scratch/candlewax_spike/qwen_adapter \
        --decisions 20-29 \
        --live-dir scratch/candlewax_spike/eval_qwen_lora

Run both (base + LoRA) with matching ``--decisions`` and compare the two
``eval_summary.md`` files side by side. A future flag could also write a
combined compare-table, but for now keeping the two invocations symmetric
makes the harness simpler and less opinionated about A/B structure.

This CLI only evaluates one variant per invocation. The ``eval_summary.md``
it writes is a single-row table that documents the variant under test —
the caller stitches the comparison together.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from burl.candlewax_spike.qwen_batch import (
    DEFAULT_SNAPSHOTS_DIR,
    _load_manifest,
    _run_one,
)
from burl.candlewax_spike.qwen_local import DEFAULT_MODEL, load_qwen


# --------------------------------------------------------------------------- #
# Config                                                                       #
# --------------------------------------------------------------------------- #


DEFAULT_LIVE_DIR = Path("scratch/candlewax_spike/eval_qwen")
DEFAULT_DECISIONS = "20-29"                      # held-out, no overlap w/ 0-3
IMAGE_VARIANTS: dict[str, Path] = {
    "existing": Path("scratch/candlewax_spike/snapshots"),
    "minimal":  Path("scratch/candlewax_spike/snapshots_minimal"),
}

# K1 soft-margin threshold (E[Q] points). A Qwen play is "within margin"
# if its per-play E[Q] is within ``EPSILON`` of the bot's pick. This is a
# softer target than exact-match since several plays often tie within rounding.
EPSILON = 1.0


def _parse_range(s: str) -> list[int]:
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


# --------------------------------------------------------------------------- #
# Per-run event parsing                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class RunMetrics:
    run_id: str
    idx: int
    mode: str
    committed: bool = False
    committed_play: int | None = None
    bot_play: int | None = None
    bot_match: bool = False
    soft_match: bool = False              # within EPSILON E[Q] of bot
    had_illegal_call: bool = False
    prediction_present: bool = False
    prediction_correct: bool | None = None   # None if not checked
    gen_tokens: int = 0
    tokens_per_sec: float | None = None
    error: str | None = None


def _parse_events(run_dir: Path) -> RunMetrics | None:
    """Collapse a run's events.jsonl into summary metrics.

    Returns ``None`` if the events file is missing (e.g. the run errored
    before opening it).
    """
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return None

    run_id = run_dir.name
    idx = int(run_id.split("_", 1)[0])
    mode = run_id.split("_", 1)[1] if "_" in run_id else ""
    m = RunMetrics(run_id=run_id, idx=idx, mode=mode)

    # `per_play_eq` in meta is a dict {str(domino_id): float} emitted by
    # qwen_batch. Normalize to int→float for direct lookup regardless of
    # whether the producer used string or int keys.
    meta_per_play_eq: dict[int, float] | None = None
    meta_bot_eq: float | None = None

    with events_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            kind = ev.get("kind")
            if kind == "meta":
                m.bot_play = (
                    int(ev["bot_play"]) if ev.get("bot_play") is not None else None
                )
                if ev.get("bot_eq") is not None:
                    meta_bot_eq = float(ev["bot_eq"])
                per_play = ev.get("per_play_eq")
                if isinstance(per_play, dict):
                    meta_per_play_eq = {
                        int(k): float(v) for k, v in per_play.items()
                    }
                elif isinstance(per_play, list):
                    # Legacy format: list positionally aligned with legal_plays.
                    legal = [int(p) for p in ev.get("legal_plays", [])]
                    if len(legal) == len(per_play):
                        meta_per_play_eq = {
                            legal[i]: float(per_play[i])
                            for i in range(len(legal))
                        }
            elif kind == "commit":
                m.committed = True
                m.committed_play = int(ev.get("final_play", -1))
            elif kind == "tool_result" and ev.get("is_error"):
                # We only flag as illegal if Qwen actually tried to commit_play.
                # Non-commit tool errors would also land here if we ever add
                # more tools, so keep this forgiving.
                m.had_illegal_call = True
            elif kind == "prediction":
                m.prediction_present = True
            elif kind == "prediction_check":
                all_ok = ev.get("all_correct")
                if all_ok is None:
                    m.prediction_correct = None
                else:
                    m.prediction_correct = bool(all_ok)
            elif kind == "result":
                usage = ev.get("usage") or {}
                m.gen_tokens = int(usage.get("generation_tokens") or 0)
                tps = usage.get("tokens_per_sec")
                m.tokens_per_sec = float(tps) if tps is not None else None
                if ev.get("is_error"):
                    m.error = "result.is_error=True"
            elif kind == "error":
                m.error = str(ev.get("message"))

    # Bot-match + K1 soft-margin.
    # K1: chosen_eq >= bot_eq - EPSILON (i.e. within EPSILON of bot play).
    # Bot-match trivially satisfies K1 (delta = 0).
    if m.committed and m.bot_play is not None and m.committed_play is not None:
        m.bot_match = (m.committed_play == m.bot_play)
        if meta_per_play_eq is not None and meta_bot_eq is not None:
            chosen_eq = meta_per_play_eq.get(m.committed_play)
            if chosen_eq is not None and chosen_eq >= meta_bot_eq - EPSILON:
                m.soft_match = True
    return m


# --------------------------------------------------------------------------- #
# Summary                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class Summary:
    total: int = 0
    committed: int = 0
    bot_matches: int = 0
    soft_matches: int = 0
    predictions_checked: int = 0
    prediction_correct: int = 0
    no_commit: int = 0
    illegal: int = 0
    prediction_mismatch: int = 0
    tps_samples: list[float] = field(default_factory=list)

    def ingest(self, m: RunMetrics) -> None:
        self.total += 1
        if m.committed:
            self.committed += 1
        else:
            self.no_commit += 1
        if m.bot_match:
            self.bot_matches += 1
        if m.soft_match:
            self.soft_matches += 1
        if m.had_illegal_call:
            self.illegal += 1
        if m.prediction_correct is True:
            self.predictions_checked += 1
            self.prediction_correct += 1
        elif m.prediction_correct is False:
            self.predictions_checked += 1
            self.prediction_mismatch += 1
        if m.tokens_per_sec is not None:
            self.tps_samples.append(m.tokens_per_sec)

    def as_row(self, label: str) -> list[str]:
        def pct(n: int, d: int) -> str:
            if d == 0:
                return "—"
            return f"{n}/{d} ({100.0 * n / d:.0f}%)"

        avg_tps = (
            f"{sum(self.tps_samples) / len(self.tps_samples):.1f}"
            if self.tps_samples else "—"
        )
        pred_rate = (
            pct(self.prediction_correct, self.predictions_checked)
            if self.predictions_checked else "—"
        )
        return [
            label,
            str(self.total),
            pct(self.bot_matches, self.total),
            pct(self.soft_matches, self.total),
            pred_rate,
            pct(self.committed, self.total),
            avg_tps,
            str(self.no_commit),
            str(self.illegal),
            str(self.prediction_mismatch),
        ]


_HEADER = [
    "variant",
    "n",
    "bot-match",
    f"within {EPSILON} E[Q]",
    "prediction-correct",
    "commit-rate",
    "avg tok/s",
    "no-commit",
    "illegal",
    "pred-mismatch",
]


def _render_markdown(rows: list[list[str]]) -> str:
    widths = [max(len(str(r[i])) for r in [_HEADER, *rows]) for i in range(len(_HEADER))]

    def fmt_row(r: list[str]) -> str:
        return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines = [fmt_row(_HEADER), sep]
    lines.extend(fmt_row(r) for r in rows)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_MODEL,
                        help="HF repo or local path for the Qwen base.")
    parser.add_argument("--adapter", type=Path, default=None,
                        help="Optional LoRA adapter directory (mlx-vlm format).")
    parser.add_argument("--decisions", type=_parse_range, default=DEFAULT_DECISIONS,
                        help="Decision indices, e.g. '20-29' or '20,22,24'.")
    parser.add_argument("--modes", nargs="+",
                        default=["plain", "multimodal"],
                        choices=["plain", "multimodal"])
    parser.add_argument("--image-variant", choices=list(IMAGE_VARIANTS),
                        default="existing",
                        help="Which snapshot directory to pull manifest+PNGs from.")
    parser.add_argument("--snapshots-dir", type=Path, default=None,
                        help=(
                            "Override the snapshot directory directly. If set, "
                            "takes precedence over --image-variant."
                        ))
    parser.add_argument("--live-dir", type=Path, default=DEFAULT_LIVE_DIR,
                        help="Where per-run events.jsonl and eval_summary.md land.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--thinking", action="store_true",
                        help="Enable Qwen <think> (off by default — forced prefix "
                             "can blow max-tokens).")
    parser.add_argument("--label", default=None,
                        help="Row label in the summary table; defaults to "
                             "'base' or 'adapter:<name>'.")
    args = parser.parse_args()

    # Resolve snapshots dir.
    snapshots_dir: Path
    if args.snapshots_dir is not None:
        snapshots_dir = args.snapshots_dir
    else:
        snapshots_dir = IMAGE_VARIANTS[args.image_variant]
    manifest_path = snapshots_dir / "index.jsonl"
    if not manifest_path.exists():
        print(
            f"[qwen-eval] WARNING: manifest not found at {manifest_path}; "
            f"falling back to default {DEFAULT_SNAPSHOTS_DIR}/index.jsonl",
            file=sys.stderr, flush=True,
        )
        snapshots_dir = DEFAULT_SNAPSHOTS_DIR
        manifest_path = snapshots_dir / "index.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no manifest at {manifest_path} — snapshots missing?"
            )

    args.live_dir.mkdir(parents=True, exist_ok=True)

    # Filter requested decisions against what's actually in the manifest;
    # skip-and-warn on misses rather than crashing.
    manifest = _load_manifest(manifest_path)
    available = {int(e["idx"]) for e in manifest}
    requested = list(args.decisions)
    runnable = [i for i in requested if i in available]
    missing = [i for i in requested if i not in available]
    if missing:
        print(
            f"[qwen-eval] WARNING: {len(missing)} idx not in manifest "
            f"({manifest_path}): {missing[:10]}{'…' if len(missing) > 10 else ''}",
            file=sys.stderr, flush=True,
        )
    if not runnable:
        print("[qwen-eval] nothing to do — no requested idx in manifest.",
              file=sys.stderr, flush=True)
        return

    label = args.label or (
        f"adapter:{Path(args.adapter).name}" if args.adapter else "base"
    )

    print(
        f"[qwen-eval] variant={label} model={args.base_model} "
        f"adapter={args.adapter} snapshots={snapshots_dir} "
        f"decisions={runnable} modes={args.modes}",
        file=sys.stderr, flush=True,
    )

    # Preload once with the right adapter; the cache hands the same tuple
    # back to every ``_run_one`` call.
    print("[qwen-eval] preloading model ...", file=sys.stderr, flush=True)
    load_qwen(args.base_model, adapter_path=args.adapter)

    runs: list[RunMetrics] = []
    t_start = time.time()
    for idx in runnable:
        for mode in args.modes:
            print(f"\n[qwen-eval] ▶ d{idx:03d} mode={mode}",
                  file=sys.stderr, flush=True)
            t0 = time.time()
            try:
                _run_one(
                    idx=idx, mode=mode, root=args.live_dir,
                    model_path=args.base_model, max_tokens=args.max_tokens,
                    enable_thinking=args.thinking,
                    adapter_path=args.adapter,
                    manifest_path=manifest_path,
                )
            except Exception as e:
                # Record the failure and keep going — one bad run shouldn't
                # torpedo the whole eval.
                print(f"[qwen-eval] ! d{idx:03d} mode={mode} failed: "
                      f"{type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                continue
            run_id = f"{idx:03d}_{mode}"
            m = _parse_events(args.live_dir / run_id)
            if m is not None:
                runs.append(m)
            print(
                f"[qwen-eval] ✓ d{idx:03d} mode={mode} in {time.time() - t0:.1f}s",
                file=sys.stderr, flush=True,
            )

    total_wall = time.time() - t_start
    print(
        f"\n[qwen-eval] DONE — {len(runs)} runs in {total_wall:.1f}s",
        file=sys.stderr, flush=True,
    )

    # Aggregate. We emit one overall row and — if both modes ran — a per-mode
    # breakdown, since plain vs multimodal usually diverge enough to care.
    overall = Summary()
    for m in runs:
        overall.ingest(m)
    rows = [overall.as_row(label)]
    if len(args.modes) > 1:
        for mode in args.modes:
            per = Summary()
            for m in runs:
                if m.mode == mode:
                    per.ingest(m)
            if per.total:
                rows.append(per.as_row(f"{label} · {mode}"))

    md = _render_markdown(rows)
    out_path = args.live_dir / "eval_summary.md"
    header_block = (
        f"# Qwen eval summary\n\n"
        f"- **variant**: {label}\n"
        f"- **base**: `{args.base_model}`\n"
        f"- **adapter**: `{args.adapter}`\n"
        f"- **snapshots**: `{snapshots_dir}`\n"
        f"- **decisions**: `{runnable}` (skipped {missing})\n"
        f"- **modes**: `{args.modes}`\n"
        f"- **wall**: {total_wall:.1f}s\n\n"
    )
    out_path.write_text(header_block + md)
    print(f"[qwen-eval] wrote {out_path}", file=sys.stderr, flush=True)
    # Also echo the table to stdout so it's captured by default.
    print(md)


if __name__ == "__main__":
    main()
