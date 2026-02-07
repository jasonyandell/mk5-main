"""Regression tests to prevent accidental GPU sync points in hot paths.

These tests are intentionally lightweight and run on CPU-only machines.
"""

from __future__ import annotations

import inspect

from forge.zeb import gpu_game_state, gpu_mcts, gpu_training_pipeline


def _assert_no_sync_primitives_in_if_guards(fn) -> None:
    """Disallow tensor.any()/item() used as Python control-flow in GPU hot loops.

    We allow tensor.any() in an `if` only when it is explicitly CPU-guarded, e.g.:
        if device.type == "cpu" and not active.any():
            ...
    """
    source = inspect.getsource(fn)
    assert ".item(" not in source, f"{fn.__name__} contains .item(), which forces device sync"

    for line in source.splitlines():
        stripped = line.strip()
        if not stripped.startswith("if "):
            continue
        if ".any(" in stripped:
            assert (
                'device.type == "cpu"' in stripped or "device.type == 'cpu'" in stripped
            ), f"{fn.__name__} uses .any() in a Python if without a CPU guard: {stripped}"


def test_gpu_mcts_hot_paths_have_no_sync_primitives():
    _assert_no_sync_primitives_in_if_guards(gpu_mcts.select_leaves_gpu)
    _assert_no_sync_primitives_in_if_guards(gpu_mcts.backprop_gpu)
    _assert_no_sync_primitives_in_if_guards(gpu_mcts.expand_gpu)
    _assert_no_sync_primitives_in_if_guards(gpu_mcts.run_mcts_search)


def test_apply_action_gpu_has_no_tensor_any_branch():
    source = inspect.getsource(gpu_game_state.apply_action_gpu)
    assert "if trick_complete.any()" not in source


def test_tokenizer_has_no_tensor_any_break():
    source = inspect.getsource(gpu_training_pipeline.GPUObservationTokenizer.tokenize_batch)
    assert "if not has_play.any()" not in source

