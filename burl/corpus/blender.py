"""Verbosity-blend a STaR corpus (LS-Mixture style).

Background — iter-0 / iter-1 adapters inherit ~3 KB rambly thought blocks from
the Layer-1 primer+framing prompt. LS-Mixture SFT (arxiv 2505.03469) reports
that mixing a minority of structure-preserved-short traces with the full
reasoning traces cuts response length ~47% while preserving (or mildly
improving) accuracy. This module implements that transform on Burl corpora.

Flow:
  1. `load_corpus(paths)` — read one or more `{"messages":[...]}` JSONL files.
  2. `classify(rows)` — split into `long` (full thought block) and `short`
     (tool-call-only, no narrative thought).
  3. `shorten(row)` — synthesize a short variant from a long row by stripping
     `<|channel>thought ... <channel|>` regions, preserving tool call /
     response envelopes and the terminal `commit_play`. Synthetic rows are
     marked `verbosity="short_synthetic"`, originals `verbosity="long"`.
  4. `blend(long, short, target_short_ratio, seed)` — deterministic shuffled
     mix at the requested ratio.
  5. `write_corpus(path, rows)` — emit JSONL for the trainer.

CLI:
  python -m burl.corpus.blender \\
    --in burl/data/star_iter0_corpus.jsonl burl/data/star_iter1_corpus.jsonl \\
    --out scratch/burl_p5_iter2_prep/star_iter2_blended_preview.jsonl \\
    --target-short-ratio 0.33 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


# --- Structural markers in the assistant channel (Gemma 4 native) ---

THOUGHT_BLOCK_RE = re.compile(
    r"<\|channel>thought.*?<channel\|>",
    flags=re.DOTALL,
)
ORPHAN_CHANNEL_OPEN_RE = re.compile(r"<\|channel>thought\b")
ORPHAN_CHANNEL_CLOSE_RE = re.compile(r"<channel\|>")
TOOL_CALL_RE = re.compile(r"<\|tool_call>.*?<tool_call\|>", flags=re.DOTALL)
COMMIT_PLAY_RE = re.compile(r"<\|tool_call>call:commit_play\{[^}]*\}<tool_call\|>")

# Length thresholds for the classifier. See wiki/topics/ls-mixture.md for choice.
SHORT_ASST_CHARS_MAX = 1500  # total assistant content cap
SHORT_THOUGHT_CHARS_MAX = 200  # cumulative chars inside <|channel>thought blocks


# --- Row shape ----------------------------------------------------------------

def assistant_content(row: dict) -> str:
    msgs = row.get("messages") or []
    for m in msgs:
        if m.get("role") == "assistant":
            return m.get("content", "")
    return ""


def thought_chars(asst: str) -> int:
    """Cumulative chars inside all `<|channel>thought ... <channel|>` blocks."""
    total = 0
    for m in THOUGHT_BLOCK_RE.finditer(asst):
        total += len(m.group()) - len("<|channel>thought") - len("<channel|>")
    return total


def has_commit_play(asst: str) -> bool:
    return COMMIT_PLAY_RE.search(asst) is not None


# --- Classifier ---------------------------------------------------------------

@dataclass
class ClassifiedRow:
    row: dict
    verbosity: str  # "short" | "long"
    asst_chars: int
    thought_chars: int


def classify_row(row: dict) -> ClassifiedRow:
    a = assistant_content(row)
    tc = thought_chars(a)
    is_short = (
        len(a) <= SHORT_ASST_CHARS_MAX
        and tc <= SHORT_THOUGHT_CHARS_MAX
        and has_commit_play(a)
    )
    return ClassifiedRow(
        row=row,
        verbosity="short" if is_short else "long",
        asst_chars=len(a),
        thought_chars=tc,
    )


def classify(rows: Iterable[dict]) -> tuple[list[ClassifiedRow], list[ClassifiedRow]]:
    longs: list[ClassifiedRow] = []
    shorts: list[ClassifiedRow] = []
    for r in rows:
        cr = classify_row(r)
        (shorts if cr.verbosity == "short" else longs).append(cr)
    return longs, shorts


# --- Shortener ----------------------------------------------------------------

def shorten_assistant(asst: str) -> str:
    """Strip thought blocks; keep tool_call / tool_response envelopes + commit.

    Idempotent: running twice gives the same result.
    Preserves ordering of tool calls and responses verbatim.
    """
    # Strip well-formed <|channel>thought ... <channel|> blocks.
    out = THOUGHT_BLOCK_RE.sub("", asst)
    # Strip orphans (Gemma occasionally emits an extra <channel|> or a bare
    # <|channel>thought header without a matching close).
    out = ORPHAN_CHANNEL_CLOSE_RE.sub("", out)
    out = ORPHAN_CHANNEL_OPEN_RE.sub("", out)
    # Collapse runs of blank lines and trim.
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def shorten_row(row: dict) -> dict:
    """Return a new row with the assistant content shortened.

    The new row copies metadata from the source and adds:
      verbosity="short_synthetic", synthetic=True, source_verbosity=<original>
    """
    new_row = json.loads(json.dumps(row))  # deep copy via JSON
    for m in new_row["messages"]:
        if m.get("role") == "assistant":
            m["content"] = shorten_assistant(m["content"])
            break
    new_row["verbosity"] = "short_synthetic"
    new_row["synthetic"] = True
    new_row["source_verbosity"] = row.get("verbosity", "long")
    return new_row


# --- Blender ------------------------------------------------------------------

def blend(
    long_rows: Sequence[dict],
    short_rows: Sequence[dict],
    target_short_ratio: float,
    seed: int = 42,
) -> list[dict]:
    """Produce a shuffled corpus with roughly `target_short_ratio` short rows.

    If `short_rows` < required count, use all of them (do not upsample).
    If `short_rows` > required count, randomly sample (seeded).
    """
    if not 0.0 <= target_short_ratio <= 1.0:
        raise ValueError(f"target_short_ratio must be in [0, 1], got {target_short_ratio}")

    rng = random.Random(seed)
    n_long = len(long_rows)
    # Solve: n_short / (n_long + n_short) == target_short_ratio
    # → n_short == target_short_ratio * n_long / (1 - target_short_ratio)
    if target_short_ratio >= 1.0:
        n_short_target = len(short_rows)
    elif target_short_ratio <= 0.0:
        n_short_target = 0
    else:
        n_short_target = round(target_short_ratio * n_long / (1 - target_short_ratio))

    if n_short_target > len(short_rows):
        n_short_actual = len(short_rows)
        chosen_short = list(short_rows)
    else:
        n_short_actual = n_short_target
        chosen_short = rng.sample(list(short_rows), n_short_actual)

    out = list(long_rows) + chosen_short
    rng.shuffle(out)
    return out


# --- I/O ----------------------------------------------------------------------

def load_corpus(paths: Sequence[str | Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    return rows


def write_corpus(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


# --- CLI ----------------------------------------------------------------------

def build_blended_corpus(
    inputs: Sequence[str | Path],
    target_short_ratio: float,
    seed: int = 42,
    shorten_all_longs: bool = True,
) -> tuple[list[dict], dict]:
    """Load → classify → synthesize shorts → blend.

    When `shorten_all_longs` is True (default), every `long` row is shortened
    to grow the `short` pool. When False, only naturally-short rows are used.
    Returns (blended_rows, stats).
    """
    raw = load_corpus(inputs)
    longs, shorts = classify(raw)
    # Tag originals.
    for cr in longs:
        cr.row["verbosity"] = "long"
        cr.row["synthetic"] = False
    for cr in shorts:
        cr.row.setdefault("verbosity", "short")
        cr.row.setdefault("synthetic", False)

    long_rows = [cr.row for cr in longs]
    short_rows = [cr.row for cr in shorts]
    if shorten_all_longs:
        short_rows = short_rows + [shorten_row(cr.row) for cr in longs]

    blended = blend(long_rows, short_rows, target_short_ratio, seed=seed)
    stats = {
        "inputs": [str(p) for p in inputs],
        "n_raw": len(raw),
        "n_natural_long": len(longs),
        "n_natural_short": len(shorts),
        "n_synthetic_short": len(short_rows) - len(shorts),
        "target_short_ratio": target_short_ratio,
        "n_blended": len(blended),
        "n_blended_long": sum(1 for r in blended if r.get("verbosity") == "long"),
        "n_blended_short": sum(1 for r in blended if r.get("verbosity") != "long"),
        "seed": seed,
    }
    stats["actual_short_ratio"] = (
        stats["n_blended_short"] / stats["n_blended"] if stats["n_blended"] else 0.0
    )
    return blended, stats


def _cli() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inputs", nargs="+", required=True)
    ap.add_argument("--out", dest="output", required=True)
    ap.add_argument("--target-short-ratio", type=float, default=0.33)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-shorten",
        action="store_true",
        help="Skip synthesis; use only naturally-short rows from the inputs.",
    )
    args = ap.parse_args()

    blended, stats = build_blended_corpus(
        args.inputs,
        target_short_ratio=args.target_short_ratio,
        seed=args.seed,
        shorten_all_longs=not args.no_shorten,
    )
    n_written = write_corpus(args.output, blended)
    stats["output"] = args.output
    stats["n_written"] = n_written

    stats_path = Path(args.output).with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    _cli()
