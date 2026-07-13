"""Smoke test for burl.lab.core.engine.MlxEngine.

Slow on first run because it pulls the Gemma 4 E2B BF16 weights (~5GB).
After the first run the model is cached locally and the test takes ~5-10s
on an M-series Mac.
"""

from __future__ import annotations

import asyncio

import pytest

mlx_lm = pytest.importorskip("mlx_lm")

from burl.lab.core.engine import (  # noqa: E402
    EngineDone,
    EngineStart,
    EngineToken,
    MlxEngine,
)


@pytest.mark.slow
def test_mlx_engine_smoke() -> None:
    engine = MlxEngine(model_repo="mlx-community/gemma-4-e2b-it-bf16")

    async def run() -> list:
        events: list = []
        async for ev in engine.step(
            messages=[{"role": "user", "content": "Say hi in 3 words"}],
            tools=[],
            max_tokens=32,
        ):
            events.append(ev)
        return events

    events = asyncio.run(run())

    assert len(events) >= 3, f"expected >=3 events, got {len(events)}"

    start = events[0]
    assert isinstance(start, EngineStart)
    assert start.n_messages == 1
    assert start.n_tools == 0
    assert start.stamp.tok_in > 0
    assert start.stamp.tok_cum_in == start.stamp.tok_in

    token_events = [e for e in events if isinstance(e, EngineToken)]
    assert len(token_events) >= 1, "expected at least one EngineToken"
    assert all(e.text for e in token_events), "EngineToken.text must be non-empty"
    # ms_ttft is set on the first EngineToken stamp and forward.
    assert token_events[0].stamp.ms_ttft is not None
    assert token_events[0].stamp.ms_ttft >= 0
    # Cumulative output tokens should be non-decreasing across token events.
    cum = [e.stamp.tok_cum_out for e in token_events]
    assert cum == sorted(cum), f"tok_cum_out not monotonic: {cum}"

    done = events[-1]
    assert isinstance(done, EngineDone)
    assert done.reason in {"done", "budget"}
    assert done.stamp.ms_ttft is not None and done.stamp.ms_ttft >= 0
    assert done.stamp.ms_decode is not None and done.stamp.ms_decode >= 0
    assert done.stamp.tok_per_s is not None and done.stamp.tok_per_s > 0.0
    assert done.stamp.tok_cum_in > 0
    assert done.stamp.tok_cum_out > 0
