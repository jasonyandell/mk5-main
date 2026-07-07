# stage-0-v4-comprehension-eval — audit 2026-07-07

## Corrections

- Page said the adapter was `jasonyandell/gemma-4-e2b-texas42-stage0-v4`; the 67% result was on the 3-epoch final adapter `jasonyandell/gemma-4-e2b-texas42-stage0-v4-full3ep` (evidence: `lem/OVERVIEW.md @ 2f11f32`, "Adapters on HuggingFace" list in the same section as the results table; the plain `-v4` name is only the stale default in `lem/gemma_star/eval_comprehension.py:21`).

## Verified

- Per-category numbers (100/90/70/60/15, overall 67%) match both the 3c33e86 commit message and `lem/OVERVIEW.md @ 2f11f32`.
- Prior-eval bug story (~40% overall, legal_moves 0% at 4729dad; EOS-token + left-pad-slicing fixes in 1d3e1b7) matches the 1d3e1b7 commit message and `wiki/sources/4729dad.md`.
- Superseded numbers (v9 83%, v10-maskfix 86%) match `wiki/experiments/v10-maskfix-breakthrough.md`.
- Referenced paths exist: `lem/gemma_star/eval_comprehension.py`, `lem/gemma_star/grade_offline.py`, `lem/rules/generate_comprehension.py`.

## Follow-ups

- HF adapter repos themselves not verified (private HuggingFace); name taken from OVERVIEW at the results commit.
- `eval_comprehension.py`'s ADAPTER_REPO default still points at the non-full3ep name — a one-line cleanup if that script is ever run again (it's superseded by the qwen variants).

## Review (second pass, 2026-07-07)

- Core correction stands: `lem/OVERVIEW.md @ 2f11f32` (lines 387–389, "Adapters on HuggingFace") names `jasonyandell/gemma-4-e2b-texas42-stage0-v4-full3ep` as the final adapter for the 67% result; the plain `-v4` name appears only as the ADAPTER_REPO default in `lem/gemma_star/eval_comprehension.py:21` (still unfixed in the worktree, so the second follow-up stays valid).
- Amended one detail on the page: OVERVIEW names the per-epoch checkpoints `-full3ep-ep1` through `ep3`, not bare `-ep1`…`-ep3`; tightened the Setup line to match (evidence: `git show 2f11f32:lem/OVERVIEW.md`, line 389).
- Re-verified the "Verified" bullets independently: per-category numbers match both the 3c33e86 commit message and OVERVIEW @ 2f11f32 (lines 376–381); the ~40%-overall / legal_moves-0% prior eval matches the 4729dad commit message; EOS + left-pad bugs match the 1d3e1b7 commit message; v9 83% / v10-maskfix 86% match `wiki/experiments/v10-maskfix-breakthrough.md`; all three referenced script paths exist.
