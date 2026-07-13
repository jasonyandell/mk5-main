"""Concurrency tests for ``burl.eval.run_move4_star_rollout``.

Verifies the ``--concurrency N`` refactor:

- At concurrency=1, Phase A is sequential (wall ~= N * per_decision).
- At concurrency=4, Phase A parallelizes (wall ~= ceil(N/4) * per_decision).
- Record ordering is preserved (record[i] matches dataset[i]).
- Cost-cap interrupts scheduling cleanly after the batch that tripped it.

Strategy: monkey-patch ``_run_one_rollout`` with a stub that sleeps in a
worker thread (``time.sleep(0.1)``), returns a minimal ``BurlTrace``, and
records the ``decision.seed`` so we can assert ordering. The Modal
``gemma_app.run()`` context manager is also stubbed so tests never touch
the network.

All deterministic; no Modal, no Gemma, no disk writes outside tmp_path.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from burl.eval import run_move4_star_rollout as r4
from burl.eval.decision_dataset import BurlDecision
from burl.harness.trace import BurlTrace


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def _fake_decision(seed: int) -> BurlDecision:
    """A minimal BurlDecision stand-in.

    We never run anything that actually needs ``game_state`` to be a real
    ZebGameState — the stubbed ``_run_one_rollout`` ignores it — so a bare
    namespace object is enough. ``per_play_eq`` is empty so the record's
    ``burl_eq`` stays ``None`` (legal_loss path not taken in these tests).
    """

    class _FakeState:
        hands = [[], [], [], []]
        played = set()

    return BurlDecision(
        seed=seed,
        declaration=0,
        narrator_seat=0,
        bidder=0,
        trick_idx=0,
        play_history=[],
        legal_plays=[0, 1],
        bot_play=0,
        bot_eq=0.0,
        per_play_eq={},
        eq_gap=0.0,
        game_state=_FakeState(),
    )


def _stub_run_one_rollout_factory(sleep_s: float):
    """Return a stub for ``_run_one_rollout`` that sleeps then returns a
    trivial trace whose ``final_play`` is decision.seed % 32.

    We sleep with ``time.sleep`` (not ``asyncio.sleep``) because the
    production code wraps this function with ``asyncio.to_thread``, so
    blocking work here parallelizes across threads exactly the way the
    real Modal network call will.
    """
    def _stub(decision, model_fn, max_turns, max_retries,
              enable_rules_tools=False, enable_primer=True):
        time.sleep(sleep_s)
        trace = BurlTrace(
            game_state_key=f"seed{decision.seed}",
            decision_prompt="",
        )
        trace.final_play = int(decision.seed) % 32
        trace.metadata["seed"] = int(decision.seed)
        trace.metadata["declaration"] = int(decision.declaration)
        trace.metadata["narrator_seat"] = int(decision.narrator_seat)
        return trace, False
    return _stub


def _stub_gate_one(decision, model_fn, nudge, max_turns, max_retries,
                   enable_rules_tools=False, enable_primer=True):
    """Gate-phase stub — not exercised in these tests (all decisions are
    stubbed to ``win`` so Phase B has zero losses to gate)."""
    trace = BurlTrace(
        game_state_key=f"seed{decision.seed}",
        decision_prompt="",
    )
    trace.final_play = int(decision.bot_play)
    return trace, False


@contextmanager
def _stub_modal_app():
    """Stand-in for ``gemma_app.run()``; yields a dummy server object."""
    class _DummyServer:
        class _Method:
            def remote(self, *a, **kw):
                return {"text": ""}
        generate_native = _Method()
    yield _DummyServer()


@pytest.fixture
def stub_modal(monkeypatch):
    """Prevent any real Modal import by monkey-patching both the app and
    the server class inside the target module."""
    class _FakeApp:
        def run(self):
            return _stub_modal_app()

    # The target module does `from burl.modal.gemma_serve_native import ...`
    # at call time (inside run_star_rollout). We intercept the import by
    # pre-installing a fake module in sys.modules.
    import sys
    import types
    fake_mod = types.ModuleType("burl.modal.gemma_serve_native")
    fake_mod.GemmaServerNative = lambda: _stub_modal_app().__enter__()
    fake_mod.app = _FakeApp()
    monkeypatch.setitem(sys.modules, "burl.modal.gemma_serve_native", fake_mod)
    # Also block the modal parent package from executing real imports.
    if "burl.modal" not in sys.modules:
        monkeypatch.setitem(
            sys.modules, "burl.modal", types.ModuleType("burl.modal")
        )
    yield


def _common_kwargs(tmp_path: Path, dataset_path: Path, n: int) -> dict:
    return dict(
        dataset_path=dataset_path,
        out_dir=tmp_path / "out",
        corpus_path=tmp_path / "corpus.jsonl",
        n_decisions=n,
        max_turns=4,
        max_retries=1,
        cost_cap_usd=1000.0,  # effectively uncapped
        gate_variant="tool-nudge",
        max_gate_retries=1,
        eq_epsilon=0.25,
    )


def _write_dataset(path: Path, n: int) -> None:
    """Write N synthetic dataset rows readable by load_dataset."""
    import json
    with path.open("w") as f:
        for i in range(n):
            f.write(json.dumps({
                "seed": 900000 + i,
                "declaration": 0,
                "narrator_seat": i % 4,
                "bidder": 0,
                "trick_idx": 0,
                "play_history": [],
                "legal_plays": [0, 1],
                "bot_play": 0,
                "bot_eq": 0.0,
                "per_play_eq": {"0": 0.0, "1": 0.0},
                "eq_gap": 0.0,
            }) + "\n")


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #


def _patch_rollouts(monkeypatch, sleep_s: float = 0.1):
    monkeypatch.setattr(
        r4, "_run_one_rollout", _stub_run_one_rollout_factory(sleep_s)
    )
    monkeypatch.setattr(r4, "_gate_one", _stub_gate_one)
    # Also patch _lookup_decision to not blow up if called — all of our
    # stubbed rollouts come back as "win" (burl_eq >= bot_eq because both
    # zero), so Phase B shouldn't fire.
    # Also patch load_dataset to read our scratch dataset.


def test_concurrency_preserves_ordering(tmp_path, monkeypatch, stub_modal):
    """At concurrency=4 over N=8 decisions, record[i] must correspond to
    dataset[i] — i.e., gather() ordering preserves dataset order."""
    _patch_rollouts(monkeypatch, sleep_s=0.01)
    ds = tmp_path / "ds.jsonl"
    _write_dataset(ds, n=8)

    kwargs = _common_kwargs(tmp_path, ds, n=8)
    kwargs["concurrency"] = 4

    asyncio.run(r4.run_star_rollout(**kwargs))

    traces_path = tmp_path / "out" / "rollout_traces.jsonl"
    assert traces_path.exists(), "rollout_traces.jsonl not written"
    import json
    lines = traces_path.read_text().strip().splitlines()
    assert len(lines) == 8
    seeds_in_order = [json.loads(line)["metadata"]["seed"] for line in lines]
    assert seeds_in_order == [900000 + i for i in range(8)], (
        f"ordering broken: {seeds_in_order}"
    )


def test_concurrency_speedup(tmp_path, monkeypatch, stub_modal):
    """At concurrency=4 with 0.1s per decision × 8 decisions, wall time
    should be ~2 × 0.1s (two batches of four), vs ~0.8s sequential.

    Tolerance: concurrent wall must be at most 60% of sequential wall.
    (Generous to absorb asyncio.run overhead + thread spawn on the CI box.)
    """
    _patch_rollouts(monkeypatch, sleep_s=0.1)
    ds = tmp_path / "ds.jsonl"
    _write_dataset(ds, n=8)

    # Baseline: sequential.
    kw_seq = _common_kwargs(tmp_path / "seq", ds, n=8)
    kw_seq["concurrency"] = 1
    (tmp_path / "seq").mkdir()
    t0 = time.perf_counter()
    asyncio.run(r4.run_star_rollout(**kw_seq))
    wall_seq = time.perf_counter() - t0

    # Concurrent.
    kw_par = _common_kwargs(tmp_path / "par", ds, n=8)
    kw_par["concurrency"] = 4
    (tmp_path / "par").mkdir()
    t0 = time.perf_counter()
    asyncio.run(r4.run_star_rollout(**kw_par))
    wall_par = time.perf_counter() - t0

    assert wall_seq > 0.7, f"sequential wall unexpectedly short: {wall_seq:.3f}s"
    assert wall_par < 0.6 * wall_seq, (
        f"concurrency=4 wall ({wall_par:.3f}s) not meaningfully faster "
        f"than concurrency=1 ({wall_seq:.3f}s)"
    )
    # Print for benchmark capture — pytest -s surfaces this.
    print(
        f"\n[bench] N=8 seq={wall_seq:.3f}s par4={wall_par:.3f}s "
        f"speedup={wall_seq/wall_par:.2f}x"
    )


def test_concurrency_one_is_sequential(tmp_path, monkeypatch, stub_modal):
    """concurrency=1 must still behave as a plain sequential loop: the
    records list matches dataset order, and wall scales linearly."""
    _patch_rollouts(monkeypatch, sleep_s=0.05)
    ds = tmp_path / "ds.jsonl"
    _write_dataset(ds, n=4)

    kwargs = _common_kwargs(tmp_path, ds, n=4)
    kwargs["concurrency"] = 1

    t0 = time.perf_counter()
    asyncio.run(r4.run_star_rollout(**kwargs))
    wall = time.perf_counter() - t0

    # 4 × 0.05s = 0.2s minimum (plus harness overhead).
    assert 0.15 < wall < 2.0, f"unexpected wall {wall:.3f}s at concurrency=1"

    traces_path = tmp_path / "out" / "rollout_traces.jsonl"
    import json
    lines = traces_path.read_text().strip().splitlines()
    assert len(lines) == 4
    seeds = [json.loads(line)["metadata"]["seed"] for line in lines]
    assert seeds == [900000, 900001, 900002, 900003]


def test_cost_cap_stops_scheduling(tmp_path, monkeypatch, stub_modal):
    """A cost cap that trips after the first batch should prevent any
    further batches from being scheduled. With per-decision sleep=0.1s
    and concurrency=4, one batch takes ~0.1s of wall; setting the cap to
    trip immediately (1e-9 USD) stops us after exactly one batch.
    """
    _patch_rollouts(monkeypatch, sleep_s=0.1)
    ds = tmp_path / "ds.jsonl"
    _write_dataset(ds, n=12)  # 3 batches at concurrency=4

    kwargs = _common_kwargs(tmp_path, ds, n=12)
    kwargs["concurrency"] = 4
    kwargs["cost_cap_usd"] = 1e-9  # effectively zero

    asyncio.run(r4.run_star_rollout(**kwargs))

    traces_path = tmp_path / "out" / "rollout_traces.jsonl"
    import json
    lines = traces_path.read_text().strip().splitlines()
    # We should have processed the first batch (4) and then stopped,
    # not the full 12. Anything ≤ 4 means the cap tripped as intended.
    assert 1 <= len(lines) <= 4, (
        f"cost-cap failed to interrupt: got {len(lines)} traces, "
        f"expected ≤ 4 (one batch)"
    )
