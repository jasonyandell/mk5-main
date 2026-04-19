"""Local sanity checks for iter-2 corpus ingestion.

These tests confirm the blended corpus survives the exact preprocessing
`burl/train/star.py::train_iter0` applies — no Modal spend, no training
launch. If any of these fail, iter-2 training would fail upstream.

Covers:
  - schema shape (two messages, user then assistant, non-empty content)
  - Gemma 4 tokenizer preserves `<|tool_call>` / `<|tool_response>` /
    `<|channel>` as single tokens (the property SPIKE_REPORT Phase 3
    validated for iter-0)
  - `apply_chat_template` round-trips the assistant content without
    dropping the tool-call envelopes
  - tokenized label mask isn't all -100 (every row has trainable tokens)
  - the shortener's synthetic rows still contain an assistant-side commit

Requires the Gemma 4 tokenizer cache (~/.cache/huggingface/hub/
models--google--gemma-4-E2B-it). If absent, tokenizer-dependent tests
skip; schema tests still run.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PREVIEW_CORPUS = (
    REPO_ROOT / "scratch/burl_p5_iter2_prep/star_iter2_blended_preview.jsonl"
)
MODEL_ID = "google/gemma-4-E2B-it"

SPECIAL_MARKERS = [
    "<|channel>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<channel|>",
]


# --- Fixtures ---------------------------------------------------------------

@pytest.fixture(scope="session")
def corpus_rows() -> list[dict]:
    if not PREVIEW_CORPUS.exists():
        pytest.skip(f"preview corpus missing: {PREVIEW_CORPUS}")
    rows = []
    with open(PREVIEW_CORPUS) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@pytest.fixture(scope="session")
def tokenizer():
    pytest.importorskip("transformers")
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    except (OSError, ValueError) as e:
        pytest.skip(f"Gemma 4 tokenizer not in HF cache: {e}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# --- Schema shape -----------------------------------------------------------

def test_corpus_nonempty(corpus_rows):
    assert len(corpus_rows) > 0, "blended preview corpus is empty"


def test_every_row_has_user_then_assistant(corpus_rows):
    for i, r in enumerate(corpus_rows):
        msgs = r.get("messages")
        assert msgs is not None, f"row {i} missing 'messages'"
        assert len(msgs) == 2, f"row {i} has {len(msgs)} messages, expected 2"
        assert msgs[0]["role"] == "user", f"row {i} msg[0] role={msgs[0]['role']}"
        assert msgs[1]["role"] == "assistant", f"row {i} msg[1] role={msgs[1]['role']}"
        assert msgs[0]["content"], f"row {i} user content empty"
        assert msgs[1]["content"], f"row {i} assistant content empty"


def test_every_row_commits_a_play(corpus_rows):
    for i, r in enumerate(corpus_rows):
        asst = r["messages"][1]["content"]
        assert (
            "call:commit_play" in asst
        ), f"row {i} (verbosity={r.get('verbosity')}) has no commit_play"


def test_synthetic_rows_have_no_thought_blocks(corpus_rows):
    synthetic = [r for r in corpus_rows if r.get("synthetic") is True]
    assert synthetic, "no synthetic short rows in preview — blend likely mis-ran"
    for i, r in enumerate(synthetic):
        asst = r["messages"][1]["content"]
        assert (
            "<|channel>thought" not in asst
        ), f"synthetic row {i} retained thought block"


def test_blend_ratio_within_expected_band(corpus_rows):
    """0.33 target with 79 longs → ~30-40% short is acceptable."""
    short = sum(1 for r in corpus_rows if r.get("verbosity") != "long")
    ratio = short / len(corpus_rows)
    assert 0.25 <= ratio <= 0.40, f"short ratio {ratio:.3f} outside [0.25, 0.40]"


# --- Tokenizer: special tokens are atomic -----------------------------------

@pytest.mark.parametrize("marker", SPECIAL_MARKERS)
def test_special_markers_are_single_tokens(tokenizer, marker):
    ids = tokenizer.encode(marker, add_special_tokens=False)
    assert len(ids) == 1, (
        f"marker {marker!r} tokenized to {len(ids)} ids={ids} — "
        f"SFT will see fragmented tool-call envelopes"
    )


# --- Chat template + tokenization round-trip -------------------------------

def _apply_chat_template(tokenizer, messages):
    """Mirror what `SFTTrainer` does under apply_chat_template."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )


def test_chat_template_renders_every_row(tokenizer, corpus_rows):
    # Sample first 20 + last 5 rows for speed; at 118 rows the full sweep is
    # also cheap but this keeps the test quick.
    sample = corpus_rows[:20] + corpus_rows[-5:]
    for i, r in enumerate(sample):
        text = _apply_chat_template(tokenizer, r["messages"])
        assert isinstance(text, str) and text, f"empty render at sample idx {i}"


_TOOL_CALL_RE = __import__("re").compile(
    r"<\|tool_call>.*?<tool_call\|>", flags=__import__("re").DOTALL
)


def test_rendered_text_preserves_tool_calls(tokenizer, corpus_rows):
    """The chat template must pass through every `<|tool_call>...<tool_call|>`
    envelope unmodified — that is the training signal. Thought blocks are
    expected to be stripped (see `test_chat_template_strips_thought_blocks`),
    so we deliberately do not require the full assistant content to survive.
    """
    for i, r in enumerate(corpus_rows[:15]):
        text = _apply_chat_template(tokenizer, r["messages"])
        asst = r["messages"][1]["content"]
        asst_tool_calls = _TOOL_CALL_RE.findall(asst)
        assert asst_tool_calls, f"row {i} has no tool calls in assistant"
        for tc in asst_tool_calls:
            assert tc in text, (
                f"row {i}: tool call {tc[:60]}... not preserved by chat template"
            )


def test_chat_template_strips_thought_blocks(tokenizer):
    """Pin the Gemma 4 chat-template behavior that iter-2 training inherits.

    SCHEMA SURPRISE — documented here because it changes what the corpus
    shortener achieves. Gemma 4 E2B's `chat_template.jinja` contains a
    `strip_thinking()` macro (line ~148) invoked on any `role == 'model'`
    string content (line ~307). It removes every `<|channel>thought ... <channel|>`
    region before tokenization, meaning SFTTrainer NEVER sees the thought prose
    in iter-0/iter-1 corpora. The long `<|channel>thought` blocks we observe
    in live rollouts are base-model (Gemma 4) thinking reflexes; the adapter
    only trained on the tool-call chain + commit. Implication: the verbosity
    blend reduces observed corpus file size but the trainer sees ~identical
    sequences for long and synthetic-short rows. Read
    scratch/burl_p5_iter2_prep/launch_iter2.md for the full writeup.
    """
    msgs = [
        {"role": "user", "content": "test"},
        {
            "role": "assistant",
            "content": (
                "<|channel>thought\nTHIS_PROSE_SHOULD_DISAPPEAR\n<channel|>"
                "<|tool_call>call:commit_play{domino_id:1}<tool_call|>"
            ),
        },
    ]
    text = _apply_chat_template(tokenizer, msgs)
    assert "THIS_PROSE_SHOULD_DISAPPEAR" not in text, (
        "Chat template no longer strips thoughts — the whole iter-2 launch "
        "plan assumed it does. Re-read launch_iter2.md and reconsider."
    )
    assert "<|tool_call>call:commit_play{domino_id:1}<tool_call|>" in text


def test_commit_play_survives_tokenize_decode_roundtrip(tokenizer, corpus_rows):
    """Ensure `commit_play` tool calls are not fragmented by the tokenizer."""
    for i, r in enumerate(corpus_rows[:10]):
        asst = r["messages"][1]["content"]
        ids = tokenizer.encode(asst, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        assert "call:commit_play" in decoded, (
            f"row {i}: commit_play lost in decode round-trip"
        )


# --- Labels aren't all -100 -------------------------------------------------

def test_labels_not_all_masked(tokenizer, corpus_rows):
    """SFTTrainer builds labels = input_ids with pad → -100. Confirm that for
    our rows the non-pad token count is comfortably positive — i.e. the
    trainer will actually see gradient signal per example."""
    pad_id = tokenizer.pad_token_id
    assert pad_id is not None, "tokenizer has no pad_token_id"

    for i, r in enumerate(corpus_rows[:10]):
        text = _apply_chat_template(tokenizer, r["messages"])
        ids = tokenizer.encode(text, add_special_tokens=False)
        non_pad = sum(1 for tid in ids if tid != pad_id)
        assert non_pad > 50, (
            f"row {i}: only {non_pad} non-pad tokens — label mask would "
            f"leave nothing to train on"
        )


def test_assistant_share_of_tokens_is_reasonable(tokenizer, corpus_rows):
    """Sanity: the assistant span should be meaningful relative to the user
    span. If assistant is 1% of the sequence the model won't learn much per
    row. For our corpus we expect ~20-60% assistant share on long rows."""
    shares = []
    for r in corpus_rows[:30]:
        user_ids = tokenizer.encode(
            r["messages"][0]["content"], add_special_tokens=False
        )
        asst_ids = tokenizer.encode(
            r["messages"][1]["content"], add_special_tokens=False
        )
        total = len(user_ids) + len(asst_ids)
        if total == 0:
            continue
        shares.append(len(asst_ids) / total)
    assert shares, "no rows tokenized"
    # Allow short synthetic rows to drag the min down, but the mean should
    # still be healthy.
    assert 0.03 < (sum(shares) / len(shares)) < 0.95, (
        f"mean assistant share {sum(shares)/len(shares):.3f} looks wrong"
    )
