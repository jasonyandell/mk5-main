"""Per-decision latency bench for Burl's eval/harvest inference path.

Drives the production batched eval code path (``GemmaLocalNativeBatched``
from ``burl/modal/gemma_local_batched.py`` plus the ``harvest_batched``
lockstep loop primitives) against the frozen 5-decision subset, capturing
``BatchResponse.stats`` per generate call so we can report:

  * ``wall_s_total / wall_s_p50 / wall_s_p95`` — per-decision walls.
  * ``prefill_tok_s`` — aggregated prompt-processing throughput, weighted
    by prompt-tokens-per-step (``sum(prompt_tokens) / sum(prompt_time)``).
  * ``decode_tok_s`` — generation throughput, weighted by generated tokens
    (``sum(generation_tokens) / sum(generation_time)``).
  * ``peak_mem_gb`` — ``max(stats.peak_memory)`` from MLX-LM, which reads
    ``mx.get_peak_memory()`` per step.
  * ``k1_grade_match_pct`` — vs the latest baseline-bf16 row in the ledger.
    K1 grade per decision = ``eq_delta_vs_bot >= 0``; we compare per-gi.
  * ``regret_delta_pct`` — signed mean-regret diff vs baseline as percent
    of baseline mean.

The bench is hermetic in that it pins the subset (sha256 of corpus.pt +
per-decision fingerprint) and asserts on mismatch.  It is not bit-pinned
because Gemma 4 E2B at temp=0.6 is sampling-driven; the documented
"K1 match within tolerance" definition is "perfect match expected on
deterministic input distribution; per-decision flips occur but should
average <5% noise".  See ``wiki/experiments/burl-perf-phase0.md``.

Output:
  * ``burl/eval/results/perf_<timestamp>_<variant>.json`` — full per-step
    detail.
  * ``burl/eval/results/perf_ledger.csv`` — one row appended.

Usage::

    PYTHONPATH=. .venv/bin/python -u -m burl.eval.bench_decision_latency \\
        --variant baseline-bf16 --subset 5

    # Sweep, capturing a delta vs the latest baseline-bf16 row:
    PYTHONPATH=. .venv/bin/python -u -m burl.eval.bench_decision_latency \\
        --variant prefix-cache --batch 8 --notes "lever 1"
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
# When the bench runs inside a git worktree, scratch/ lives in the main
# checkout (gitignored).  Fall back to the canonical absolute path so the
# harvest_batched + sweep modules resolve regardless of where the bench
# is invoked from.  If/when the eval harness is promoted to tracked code,
# this fallback can drop.
_SCRATCH_CANDIDATES = [
    REPO_ROOT / "scratch" / "belief_trajectory_rollout",
    Path("/Users/jason/code/mk5-main/scratch/belief_trajectory_rollout"),
]
SCRATCH_HARVEST = next(
    (p for p in _SCRATCH_CANDIDATES if p.exists()),
    _SCRATCH_CANDIDATES[0],
)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRATCH_HARVEST))
sys.path.insert(0, str(SCRATCH_HARVEST / "diagnostic"))


SUBSET_PATH = REPO_ROOT / "burl" / "eval" / "data" / "perf_subset_5.jsonl"
LEDGER_PATH = REPO_ROOT / "burl" / "eval" / "results" / "perf_ledger.csv"
LEDGER_HEADER = [
    "timestamp",
    "sha",
    "branch",
    "variant_label",
    "subset",
    "batch",
    "max_tokens_policy",
    "n_decisions",
    "wall_s_total",
    "wall_s_p50",
    "wall_s_p95",
    "prefill_tok_s",
    "decode_tok_s",
    "peak_mem_gb",
    "k1_grade_match_pct",
    "regret_delta_pct",
    "notes",
]
DEFAULT_MAX_TOKENS = 8192    # matches run-3c eval reference
DEFAULT_TURN_CAP = 8         # matches harvest_batched.MAX_TURNS_PER_DECISION
DEFAULT_VARIANT_NAME_RUNTIME = "D_required_first"  # the eval variant
DEFAULT_MODEL_REPO = "mlx-community/gemma-4-e2b-it-bf16"


# --------------------------------------------------------------------------- #
# Subset I/O                                                                   #
# --------------------------------------------------------------------------- #


def load_subset(path: Path) -> tuple[dict, list[dict]]:
    """Return ``(header, rows)`` for the frozen subset JSONL."""
    with path.open() as fid:
        lines = [json.loads(line) for line in fid if line.strip()]
    if not lines or not lines[0].get("_meta"):
        raise ValueError(f"{path}: missing header line (must be first line, _meta=true)")
    return lines[0], lines[1:]


def fingerprint_decision(bd) -> str:
    """Match ``freeze_perf_subset.fingerprint_decision``."""
    payload = {
        "seed": int(bd.seed),
        "declaration": int(bd.declaration),
        "narrator_seat": int(bd.narrator_seat),
        "trick_idx": int(bd.trick_idx),
        "legal_plays": sorted(int(x) for x in bd.legal_plays),
        "bot_play": int(bd.bot_play),
        "bot_eq": round(float(bd.bot_eq), 6),
        "per_play_eq": {
            str(k): round(float(v), 6)
            for k, v in sorted(bd.per_play_eq.items())
        },
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def sha256_of(path: Path, chunk_bytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fid:
        while chunk := fid.read(chunk_bytes):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# Stats-tracking model wrapper                                                  #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Phase timer — wall-time fingerprint of the per-decision loop                 #
# --------------------------------------------------------------------------- #


class PhaseTimer:
    """Cumulative ``time.perf_counter()`` accounting keyed by phase name.

    Used as a context manager: ``with PHASES.phase("prefill_decode"): ...``.
    Sub-microsecond overhead per enter/exit. Phases nest cleanly because each
    enter/exit pair only touches its own bucket — caller-side outer phases
    over-count if they wrap an inner phase, so we keep the instrumentation
    flat (no phase wraps another timed phase).
    """

    def __init__(self) -> None:
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def add(self, name: str, dt: float) -> None:
        self._totals[name] = self._totals.get(name, 0.0) + dt
        self._counts[name] = self._counts.get(name, 0) + 1

    def phase(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner.t0 = time.perf_counter()
                return self_inner

            def __exit__(self_inner, *_):
                timer.add(name, time.perf_counter() - self_inner.t0)
                return False

        return _Ctx()

    def report(self, *, total_wall_s: float) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for name, total in sorted(
            self._totals.items(), key=lambda kv: -kv[1],
        ):
            count = self._counts.get(name, 0)
            out[name] = {
                "total_s": round(total, 4),
                "pct_of_wall": round(100.0 * total / total_wall_s, 2)
                if total_wall_s > 0 else 0.0,
                "n_calls": count,
                "mean_ms": round(1000.0 * total / count, 3)
                if count else 0.0,
            }
        return out


PHASES = PhaseTimer()


# --------------------------------------------------------------------------- #
# Kernel-class call counter (lever #7 — sub-fused-kernel audit)               #
# --------------------------------------------------------------------------- #


class KernelCounter:
    """Counts kernel-class invocations on the Gemma 4 hot path.

    Instrumented by ``install_kernel_hooks`` which monkey-patches the loaded
    model's bound methods + the imported ``mx.fast.rms_norm`` /
    ``scaled_dot_product_attention`` symbols inside ``mlx_lm.models.gemma4_text``
    to increment counters before delegating. Zero perf cost (counter increment
    plus original call); does NOT call ``mx.synchronize`` so the lazy-async
    pipeline is preserved.

    Cumulative time per kernel-class is computed analytically from call counts
    and the per-call read bytes (decode is bandwidth-bound at B=5, L=1 — every
    matmul is a vector-matrix gemv whose wall is dominated by reading the
    weight tile from unified memory). M5 Max bf16 effective bandwidth is
    measured at ~400 GB/s achieved (≈50% of 800 GB/s peak; standard Apple
    Silicon utilization for streaming gemv).
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        # bytes_read per call, populated lazily on first call so we don't have
        # to know the model config up front.
        self.bytes_per_call: dict[str, int] = {}

    def bump(self, name: str, bytes_read: int = 0) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1
        if name not in self.bytes_per_call and bytes_read > 0:
            self.bytes_per_call[name] = bytes_read

    def report(
        self, *, gen_tokens: int, decode_wall_s: float, achieved_bw_gbs: float = 400.0,
    ) -> dict[str, Any]:
        """Build the audited 5-row table.

        achieved_bw_gbs: M5 Max measured effective bandwidth (~400 GB/s on bf16
        gemv from prior benches). Per-call wall = bytes / (achieved_bw * 1e9).
        Cumulative wall = wall_per_call × calls.
        """
        rows = []
        for name, count in self.counts.items():
            bytes_pc = self.bytes_per_call.get(name, 0)
            est_wall = (bytes_pc * count) / (achieved_bw_gbs * 1e9)
            rows.append({
                "kernel_class": name,
                "calls_total": count,
                "calls_per_token": round(count / max(gen_tokens, 1), 2),
                "bytes_per_call": bytes_pc,
                "est_cum_time_s": round(est_wall, 3),
                "pct_of_decode_wall": round(
                    100.0 * est_wall / decode_wall_s, 2,
                ) if decode_wall_s > 0 else 0.0,
            })
        rows.sort(key=lambda r: -r["est_cum_time_s"])
        return {
            "model_decode_bandwidth_assumed_gbs": achieved_bw_gbs,
            "gen_tokens": gen_tokens,
            "decode_wall_s": round(decode_wall_s, 3),
            "kernels": rows,
        }


KERNELS = KernelCounter()


def install_kernel_hooks(loaded_model) -> None:
    """Monkey-patch Gemma 4 hot-path entry points to increment KERNELS counters.

    The patch is applied to (a) the imported ``mx.fast.rms_norm`` /
    ``scaled_dot_product_attention`` symbols in ``mlx_lm.models.gemma4_text``,
    (b) ``BatchKVCache.update_and_fetch`` on the cache class, and (c) every
    ``nn.Linear`` instance reachable from the loaded model — wrappers cache
    the read-bytes-per-call from the weight shape on first invocation.

    Reversible: the patch installs ``_kernel_audit_orig_*`` attributes and the
    bench tears them down at run end. Idempotent: re-installing is a no-op.
    """
    import mlx.core as mx
    from mlx_lm.models import gemma4_text as g4t
    from mlx_lm.models import cache as mcache
    from mlx.nn.layers.linear import Linear as _Linear

    if getattr(install_kernel_hooks, "_installed", False):
        return

    # 1. Wrap mx.fast.rms_norm at the gemma4_text module's lookup site.
    _orig_rms = g4t.mx.fast.rms_norm
    def _rms_norm_hooked(x, scale, eps, **kw):
        # rms_norm reads x once + scale once; 2*B*L*D bytes for x + scale.
        # In bf16 (2B). Counted per call; cumulative is small.
        nbytes = 0
        try:
            nbytes = 2 * x.size * x.dtype.size
        except Exception:
            pass
        KERNELS.bump("rms_norm_fused", nbytes)
        return _orig_rms(x, scale, eps, **kw)
    g4t.mx.fast.rms_norm = _rms_norm_hooked

    # 2. Wrap scaled_dot_product_attention.
    _orig_sdpa = g4t.scaled_dot_product_attention
    def _sdpa_hooked(queries, keys, values, **kw):
        # SDPA bytes = read Q + K + V; all three are
        # B * n_heads * L * head_dim. At decode L_q=1, L_kv=cache_len.
        try:
            kv_b, kv_h, kv_l, kv_d = keys.shape
            q_b, q_h, q_l, q_d = queries.shape
            nbytes_q = q_b * q_h * q_l * q_d * queries.dtype.size
            nbytes_kv = 2 * kv_b * kv_h * kv_l * kv_d * keys.dtype.size
            nbytes = nbytes_q + nbytes_kv
        except Exception:
            nbytes = 0
        KERNELS.bump("scaled_dot_product_attention_fused", nbytes)
        return _orig_sdpa(queries, keys, values, **kw)
    g4t.scaled_dot_product_attention = _sdpa_hooked

    # 3. Wrap nn.Linear.__call__ at the CLASS level (instance-level override
    # of __call__ does not work — Python skips instance dict for dunder method
    # lookup). We tag each Linear we want to count with `_audit_kernel_name`
    # + `_audit_bytes_per_call`; the class-level wrapper increments only for
    # tagged instances and falls through for everything else.
    text_model = loaded_model.language_model.model if hasattr(
        loaded_model, "language_model",
    ) else loaded_model.model
    _orig_linear_call = _Linear.__call__
    def _wrap_linear(layer, kernel_name):
        weight = layer.weight  # shape (out, in)
        out_dim, in_dim = weight.shape
        weight_bytes = out_dim * in_dim * weight.dtype.size
        layer._audit_kernel_name = kernel_name
        layer._audit_bytes_per_call = weight_bytes

    def _linear_call_hooked(self, x):
        name = getattr(self, "_audit_kernel_name", None)
        if name is not None:
            KERNELS.bump(name, getattr(self, "_audit_bytes_per_call", 0))
        return _orig_linear_call(self, x)
    _Linear.__call__ = _linear_call_hooked

    # Walk the decoder layers and bind kernel-class names to projection sites.
    for li, layer in enumerate(text_model.layers):
        attn = layer.self_attn
        _wrap_linear(attn.q_proj, "matmul_q_proj")
        if hasattr(attn, "k_proj") and not getattr(attn, "use_k_eq_v", False):
            _wrap_linear(attn.k_proj, "matmul_k_proj")
            if hasattr(attn, "v_proj") and attn.v_proj is not None:
                _wrap_linear(attn.v_proj, "matmul_v_proj")
        _wrap_linear(attn.o_proj, "matmul_o_proj")
        if hasattr(layer.mlp, "gate_proj"):
            _wrap_linear(layer.mlp.gate_proj, "matmul_mlp_gate_proj")
            _wrap_linear(layer.mlp.up_proj, "matmul_mlp_up_proj")
            _wrap_linear(layer.mlp.down_proj, "matmul_mlp_down_proj")
        if getattr(layer, "per_layer_input_gate", None) is not None:
            _wrap_linear(layer.per_layer_input_gate, "matmul_per_layer_gate")
            _wrap_linear(layer.per_layer_projection, "matmul_per_layer_proj")

    # 4. lm_head — reachable from the outer Model. The output head is the
    # single largest weight tile in the forward (vocab × hidden_size).
    if hasattr(loaded_model, "language_model") and hasattr(
        loaded_model.language_model, "lm_head",
    ):
        _wrap_linear(loaded_model.language_model.lm_head, "matmul_lm_head")
    elif hasattr(loaded_model, "lm_head"):
        _wrap_linear(loaded_model.lm_head, "matmul_lm_head")

    # 5. BatchKVCache.update_and_fetch — count via class wrap.
    _orig_uaf = mcache.BatchKVCache.update_and_fetch
    def _uaf_hooked(self, keys, values):
        # update copies (B * n_kv_heads * L_new * head_dim) for K and V each.
        try:
            B, n_kv, L_new, d = keys.shape
            nbytes = 2 * B * n_kv * L_new * d * keys.dtype.size
        except Exception:
            nbytes = 0
        KERNELS.bump("kv_cache_update_and_fetch", nbytes)
        return _orig_uaf(self, keys, values)
    mcache.BatchKVCache.update_and_fetch = _uaf_hooked

    install_kernel_hooks._installed = True
    install_kernel_hooks._teardown = (
        ("g4t.mx.fast.rms_norm", g4t.mx.fast, "rms_norm", _orig_rms),
        ("g4t.scaled_dot_product_attention", g4t,
         "scaled_dot_product_attention", _orig_sdpa),
        ("BatchKVCache.update_and_fetch", mcache.BatchKVCache,
         "update_and_fetch", _orig_uaf),
        ("Linear.__call__", _Linear, "__call__", _orig_linear_call),
    )


def teardown_kernel_hooks() -> None:
    if not getattr(install_kernel_hooks, "_installed", False):
        return
    for _label, mod, name, original in install_kernel_hooks._teardown:
        setattr(mod, name, original)
    install_kernel_hooks._installed = False


@dataclass
class StepStats:
    """One BatchResponse.stats record + the wall it cost."""
    wall_s: float
    n_active: int
    prompt_tokens: int
    prompt_time: float
    prompt_tps: float
    generation_tokens: int
    generation_time: float
    generation_tps: float
    peak_memory_gb: float


@dataclass
class StatsTracker:
    steps: list[StepStats] = field(default_factory=list)

    def record(self, *, wall_s: float, n_active: int, batch_stats: Any) -> None:
        self.steps.append(StepStats(
            wall_s=float(wall_s),
            n_active=int(n_active),
            prompt_tokens=int(batch_stats.prompt_tokens),
            prompt_time=float(batch_stats.prompt_time),
            prompt_tps=float(batch_stats.prompt_tps),
            generation_tokens=int(batch_stats.generation_tokens),
            generation_time=float(batch_stats.generation_time),
            generation_tps=float(batch_stats.generation_tps),
            peak_memory_gb=float(batch_stats.peak_memory),
        ))

    def aggregate(self) -> dict[str, Any]:
        if not self.steps:
            return {
                "n_steps": 0,
                "prefill_tok_s": None,
                "decode_tok_s": None,
                "peak_mem_gb": None,
            }
        total_prompt_tokens = sum(s.prompt_tokens for s in self.steps)
        total_prompt_time = sum(s.prompt_time for s in self.steps)
        total_gen_tokens = sum(s.generation_tokens for s in self.steps)
        total_gen_time = sum(s.generation_time for s in self.steps)
        return {
            "n_steps": len(self.steps),
            "prefill_tok_s": (
                total_prompt_tokens / total_prompt_time
                if total_prompt_time > 0 else None
            ),
            "decode_tok_s": (
                total_gen_tokens / total_gen_time
                if total_gen_time > 0 else None
            ),
            "peak_mem_gb": max(s.peak_memory_gb for s in self.steps),
            "total_prompt_tokens": total_prompt_tokens,
            "total_generation_tokens": total_gen_tokens,
            "total_prompt_time_s": total_prompt_time,
            "total_generation_time_s": total_gen_time,
        }


def make_tracking_model(
    *,
    adapter_path: str | None,
    max_tokens: int,
    temperature: float,
    model_repo: str,
    tracker: StatsTracker,
    enable_prompt_cache: bool = False,
    prune_lm_head: bool = False,
    prune_freq_tsv: str | None = None,
    prune_keep_n: int = 8192,
    log_argmax_winners: str | None = None,
):
    """Build a ``GemmaLocalNativeBatched`` whose ``step_batch`` records stats.

    We instrument at runtime (rather than editing the model class) so the
    bench's instrumentation never leaks into production code paths.  The
    instrumented step_batch wraps the production path in a
    ``BatchStats``-capturing ``stats()`` block; the production class owns
    the cache-management decision (``enable_prompt_cache``) so the bench
    treats both code paths uniformly.
    """
    from burl.modal.gemma_local_batched import GemmaLocalNativeBatched
    from mlx_lm.generate import BatchGenerator, BatchStats
    from mlx_lm.models.cache import make_prompt_cache

    class _Tracking(GemmaLocalNativeBatched):
        def step_batch(self, active: list[dict]) -> list[str]:
            not_done = [e for e in active if not e.get("done", False)]
            if not not_done:
                return []
            with PHASES.phase("prompt_build_inner"):
                prompts = [
                    self._render_prompt_ids(e["messages"], e.get("tools"))
                    for e in not_done
                ]
            # Prefix-cache fetch (or fresh caches if disabled).
            cache_hit_before = self.cache_hit_tokens_total
            processed_before = self.cache_processed_tokens_total
            if self._prompt_cache is not None:
                prompt_caches: list = []
                suffix_prompts: list[list[int]] = []
                for full_prompt in prompts:
                    cache, rest = self._prompt_cache.fetch_nearest_cache(
                        self._cache_model_key, full_prompt,
                    )
                    n_hit = len(full_prompt) - len(rest)
                    self.cache_hit_tokens_total += n_hit
                    self.cache_processed_tokens_total += len(rest)
                    if cache is None or not rest:
                        cache = make_prompt_cache(self.model)
                        suffix_prompts.append(list(full_prompt))
                        # Roll back the misleading hit counter when we fall
                        # back to a fresh cache (entire prompt is processed).
                        if not rest:
                            self.cache_hit_tokens_total -= n_hit
                            self.cache_processed_tokens_total += n_hit
                    else:
                        suffix_prompts.append(list(rest))
                    prompt_caches.append(cache)
            else:
                prompt_caches = None
                suffix_prompts = prompts

            # Drive BatchGenerator directly so we get a single BatchStats
            # block per step_batch call (the public batch_generate creates a
            # fresh generator each call too, so this is functionally identical
            # but exposes stats() to the tracker).
            t0 = time.time()
            with PHASES.phase("prefill_decode"):
                gen = BatchGenerator(
                    self.model,
                    stop_tokens=[[t] for t in self.tokenizer.eos_token_ids],
                    completion_batch_size=len(suffix_prompts),
                    prefill_batch_size=2,
                )
                uids = gen.insert(
                    suffix_prompts,
                    [self.max_tokens] * len(suffix_prompts),
                    caches=prompt_caches,
                    samplers=[self._sampler] * len(suffix_prompts),
                )
                results: dict[int, list[int]] = {uid: [] for uid in uids}
                post_caches: dict[int, list] = {}
                stats = BatchStats()
                with gen.stats(stats):
                    while responses := gen.next_generated():
                        for r in responses:
                            if r.finish_reason is not None:
                                if self._prompt_cache is not None:
                                    post_caches[r.uid] = r.prompt_cache
                            if r.finish_reason != "stop":
                                results[r.uid].append(r.token)
                gen.close()
            wall = time.time() - t0
            tracker.record(
                wall_s=wall,
                n_active=len(not_done),
                batch_stats=stats,
            )
            with PHASES.phase("tokenizer_decode_output"):
                texts = [self.tokenizer.decode(results[uid]) for uid in uids]

            # Insert post-decode caches.
            if self._prompt_cache is not None:
                for full_prompt, text, uid in zip(prompts, texts, uids):
                    cache = post_caches.get(uid)
                    if cache is None:
                        continue
                    completion_ids = self.tokenizer.encode(text) if text else []
                    key = list(full_prompt) + list(completion_ids)
                    try:
                        self._prompt_cache.insert_cache(self._cache_model_key, key, cache)
                    except Exception:
                        pass

            return texts

    return _Tracking(
        model_repo=model_repo,
        adapter_path=adapter_path,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_prompt_cache=enable_prompt_cache,
        prune_lm_head=prune_lm_head,
        prune_freq_tsv=prune_freq_tsv,
        prune_keep_n=prune_keep_n,
        log_argmax_winners=log_argmax_winners,
    )


# --------------------------------------------------------------------------- #
# Bench loop — wraps the harvest_batched lockstep primitives                   #
# --------------------------------------------------------------------------- #


def run_bench_continuous(
    *,
    decisions: list,
    global_indices: list[int],
    out_dir: Path,
    model: Any,
    variant: Any,
    oracle: Any,
    batch: int,
    turn_cap: int,
    tracker: "StatsTracker",
    heartbeat_label: str = "",
    heartbeat_interval_s: float = 30.0,
) -> dict:
    """Continuous-batching dispatcher — Phase 2 lever 2.

    Replaces the sync-wave loop in run_bench with a single long-lived
    ``BatchGenerator``.  All decisions submit their first turn at once;
    as a stream finishes (hits EOS or runs to max_tokens), tools dispatch
    + state transitions for that decision happen on CPU and its next-turn
    prompt is immediately re-submitted to the same generator while the
    other streams keep decoding on the GPU.

    The straggler-tail savings: a decision finishing in 5 turns no longer
    holds up the GPU waiting for a decision running 8 turns; the freed
    slot in the dispatcher pool fills with the next decision's next turn
    in the same step.
    """
    import harvest_batched as hb  # type: ignore
    from burl.harness.tool_loop_native import parse_native_completion
    from mlx_lm.generate import BatchGenerator, BatchStats

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    bench_t0 = time.time()
    finish_wall_by_gi: dict[int, float] = {}

    # Initialize all per-decision state up front; they'll all submit their
    # first turn into the same dispatcher pool.
    states: list[hb._DecisionState] = []
    for gi, dec in zip(global_indices, decisions):
        dec_dir = out_dir / f"decision_{gi}"
        st = hb._init_decision_state(
            gi, dec, variant, oracle, dec_dir, max_turns=turn_cap,
        )
        states.append(st)

    # Pool size = configured batch.  At batch=5 with 5 decisions this is one
    # cohort, but the dispatcher still wins because turns N+1 of fast
    # finishers can issue while slow turn N's finish.
    gen = BatchGenerator(
        model.model,
        stop_tokens=[[t] for t in model.tokenizer.eos_token_ids],
        completion_batch_size=max(batch, 1),
        prefill_batch_size=2,
    )
    uid_to_state: dict[int, hb._DecisionState] = {}
    uid_to_token_buf: dict[int, list[int]] = {}

    def submit(state: hb._DecisionState) -> int:
        msgs, schemas = hb._prepare_step(state)
        ptext = model.tokenizer.apply_chat_template(
            state.messages, tools=state.schemas,
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        prompt_ids = model.tokenizer.encode(ptext)
        (uid,) = gen.insert(
            [prompt_ids],
            [model.max_tokens],
            samplers=[model._sampler],
        )
        uid_to_state[uid] = state
        uid_to_token_buf[uid] = []
        # Stash the prompt text on the state so apply_step can mirror the
        # transcript bookkeeping that the wave path does.
        state._continuous_prompt_text = ptext
        state._continuous_started_at = time.time()
        return uid

    # Submit initial turns.
    for st in states:
        submit(st)

    # Heartbeat state: never go more than ~heartbeat_interval_s without
    # emitting *something*, so silent-jetsam (process gone but log frozen)
    # is detectable by the orchestrator within 30s instead of 14min.
    n_total = len(states)
    n_finished = 0
    hb_label = heartbeat_label or "[bench]"
    last_emit_t = time.monotonic()
    last_emit_gen_tokens = 0

    # Pump until all states are done.
    bench_stats = BatchStats()
    with gen.stats(bench_stats):
        while uid_to_state:
            responses = gen.next_generated()
            now_mono = time.monotonic()
            elapsed_since_emit = now_mono - last_emit_t
            if elapsed_since_emit >= heartbeat_interval_s:
                cur_gen = int(bench_stats.generation_tokens)
                delta_tok = cur_gen - last_emit_gen_tokens
                last_30s_tps = (
                    delta_tok / elapsed_since_emit
                    if elapsed_since_emit > 0 else 0.0
                )
                print(
                    f"{hb_label}: alive ({n_finished}/{n_total} done, "
                    f"n_active={len(uid_to_state)}, "
                    f"gen_tokens_total={cur_gen}, "
                    f"last_{int(elapsed_since_emit)}s_tok_s="
                    f"{last_30s_tps:.1f})",
                    flush=True,
                )
                last_emit_t = now_mono
                last_emit_gen_tokens = cur_gen
            if not responses:
                break
            for r in responses:
                uid = r.uid
                buf = uid_to_token_buf.get(uid)
                if buf is None:
                    continue
                if r.finish_reason != "stop":
                    buf.append(r.token)
                if r.finish_reason is None:
                    continue
                # Stream finished — apply tool dispatch + state transition.
                state = uid_to_state.pop(uid)
                buf = uid_to_token_buf.pop(uid)
                completion = model.tokenizer.decode(buf)
                t_after_gen = time.time()
                tracker.steps.append(StepStats(
                    wall_s=t_after_gen - state._continuous_started_at,
                    n_active=len(uid_to_state) + 1,
                    prompt_tokens=0,
                    prompt_time=0.0,
                    prompt_tps=0.0,
                    generation_tokens=len(buf),
                    generation_time=t_after_gen - state._continuous_started_at,
                    generation_tps=len(buf) / max(
                        t_after_gen - state._continuous_started_at, 1e-9,
                    ),
                    peak_memory_gb=0.0,
                ))
                was_done = state.done
                try:
                    hb._apply_step(
                        state, completion,
                        getattr(state, "_continuous_prompt_text", ""),
                        parse_completion=parse_native_completion,
                    )
                except Exception as exc:
                    state.done = True
                    state.result_meta["bailed"] = True
                    state.result_meta["bail_reason"] = (
                        f"apply_step failed (continuous): "
                        f"{type(exc).__name__}: {exc}"
                    )
                if state.done and not was_done:
                    finish_wall_by_gi[int(state.gi)] = (
                        t_after_gen - state.wall_t0
                    )
                    n_finished += 1
                    dec_wall = t_after_gen - state.wall_t0
                    cur_gen = int(bench_stats.generation_tokens)
                    decode_tps = (
                        cur_gen / float(bench_stats.generation_time)
                        if bench_stats.generation_time > 0 else 0.0
                    )
                    peak_gb = float(bench_stats.peak_memory)
                    print(
                        f"{hb_label}: decision_{state.gi} finished "
                        f"({n_finished}/{n_total}, wall={dec_wall:.2f}s, "
                        f"decode_tok_s={decode_tps:.1f}, "
                        f"peak_gb={peak_gb:.2f})",
                        flush=True,
                    )
                    last_emit_t = time.monotonic()
                    last_emit_gen_tokens = cur_gen
                if not state.done:
                    submit(state)
    gen.close()

    # Finalize each decision (preserves wave-loop's _finalize semantics).
    for s in states:
        wall_s = finish_wall_by_gi.get(int(s.gi), time.time() - s.wall_t0)
        row = hb._finalize(s, oracle, wall_s)
        row["global_idx"] = s.gi
        row["bench_per_decision_wall_s"] = round(wall_s, 3)
        rows.append(row)

    bench_wall = time.time() - bench_t0
    # Promote BatchStats into a single tracker entry covering the run so
    # prefill / decode tok/s aggregates show up in the ledger row.
    tracker.steps.insert(0, StepStats(
        wall_s=bench_wall,
        n_active=len(states),
        prompt_tokens=int(bench_stats.prompt_tokens),
        prompt_time=float(bench_stats.prompt_time),
        prompt_tps=float(getattr(bench_stats, "prompt_tps", 0.0)),
        generation_tokens=int(bench_stats.generation_tokens),
        generation_time=float(bench_stats.generation_time),
        generation_tps=float(getattr(bench_stats, "generation_tps", 0.0)),
        peak_memory_gb=float(bench_stats.peak_memory),
    ))
    return {
        "rows": rows,
        "bench_wall_s": bench_wall,
        "wave_walls_s": [bench_wall],
    }


def run_bench_continuous_cohorts_subprocess(
    *,
    n_decisions: int,
    out_dir: Path,
    cohort_size: int,
    parent_argv: list[str],
) -> dict:
    """Lever #6 closure: each cohort runs in a fresh python subprocess.

    The parent process resolves the decision list and out_dir, then
    spawns one ``subprocess.run`` per cohort. Each child re-enters
    ``__main__`` with ``--subprocess-cohort-range LO:HI`` (a slice into
    the parent's resolved decision list, identical because subset
    resolution is deterministic) plus ``--subprocess-cohort-result``
    (sidecar JSON path). The child runs ``run_bench_continuous`` on
    just its slice, writes ``{rows, bench_wall_s, wave_walls_s,
    step_stats}`` to the sidecar, and exits — releasing all Metal
    cache + KV state + ALL cumulative session pressure. The parent
    aggregates per-cohort sidecars into the same ``bench_result`` dict
    the in-process path produces.

    Eliminates the iter 17/18/22/31 silent-jetsam ceiling drift
    permanently: each cohort starts from a fresh python interpreter
    state, so cumulative pressure cannot accumulate across cohort
    boundaries.
    """
    n_cohorts = (n_decisions + cohort_size - 1) // cohort_size
    print(
        f"[bench] subprocess-isolation cohort mode: N={n_decisions} "
        f"cohort_size={cohort_size} -> {n_cohorts} cohort(s)",
        flush=True,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    all_wave_walls: list[float] = []
    all_step_stats: list[dict] = []
    cohort_walls: list[float] = []
    bench_t0 = time.time()
    for ci in range(n_cohorts):
        lo = ci * cohort_size
        hi = min(lo + cohort_size, n_decisions)
        sidecar_path = out_dir / f"cohort_{ci:03d}_result.json"
        print(
            f"[bench] subproc-cohort {ci+1}/{n_cohorts}: decisions "
            f"{lo}..{hi-1} (n={hi-lo}) -> sidecar={sidecar_path.name}",
            flush=True,
        )
        # Build child argv: same as parent, with isolation/cohort-size
        # stripped (so child does not recurse) and cohort-range +
        # cohort-result added.
        child_argv = list(parent_argv)
        # Drop --subprocess-isolation (flag) and --cohort-size N (key+val)
        # so the child runs the in-process continuous path on its slice.
        cleaned: list[str] = []
        i = 0
        while i < len(child_argv):
            tok = child_argv[i]
            if tok == "--subprocess-isolation":
                i += 1
                continue
            if tok == "--cohort-size":
                i += 2
                continue
            if tok.startswith("--cohort-size="):
                i += 1
                continue
            cleaned.append(tok)
            i += 1
        child_argv = cleaned + [
            "--subprocess-cohort-range", f"{lo}:{hi}",
            "--subprocess-cohort-result", str(sidecar_path),
        ]
        child_cmd = [sys.executable, "-u", "-m",
                     "burl.eval.bench_decision_latency"] + child_argv
        c_t0 = time.time()
        # Inherit parent's stdout/stderr so the child's [bench] logs
        # stream into the parent log in real time.
        rc = subprocess.run(
            child_cmd, cwd=str(REPO_ROOT), check=False,
        ).returncode
        c_wall = time.time() - c_t0
        cohort_walls.append(c_wall)
        if rc != 0:
            raise RuntimeError(
                f"subproc-cohort {ci+1}/{n_cohorts} exited rc={rc} "
                f"(see child stderr above; sidecar may be absent)"
            )
        if not sidecar_path.exists():
            raise RuntimeError(
                f"subproc-cohort {ci+1}/{n_cohorts} exited cleanly but "
                f"sidecar {sidecar_path} was not written"
            )
        with sidecar_path.open() as fh:
            sidecar = json.load(fh)
        all_rows.extend(sidecar["rows"])
        all_wave_walls.extend(sidecar["wave_walls_s"])
        all_step_stats.extend(sidecar["step_stats"])
        print(
            f"[bench] subproc-cohort {ci+1}/{n_cohorts} done in "
            f"{c_wall:.1f}s (per-decision wall ~"
            f"{c_wall / max(hi - lo, 1):.2f}s)",
            flush=True,
        )

    bench_wall = time.time() - bench_t0
    print(
        f"[bench] all subproc-cohorts done; total "
        f"bench_wall={bench_wall:.1f}s per-cohort walls="
        f"{[round(w, 1) for w in cohort_walls]}",
        flush=True,
    )
    return {
        "rows": all_rows,
        "bench_wall_s": bench_wall,
        "wave_walls_s": all_wave_walls,
        "cohort_walls_s": cohort_walls,
        "subprocess_step_stats": all_step_stats,
    }


def run_bench_continuous_cohorts(
    *,
    decisions: list,
    global_indices: list[int],
    out_dir: Path,
    model: Any,
    variant: Any,
    oracle: Any,
    batch: int,
    turn_cap: int,
    tracker: "StatsTracker",
    cohort_size: int,
) -> dict:
    """Lever #6 cohort wrapper around ``run_bench_continuous``.

    Splits ``decisions`` into chunks of ``cohort_size`` and runs each
    chunk as an independent BatchGenerator session, with explicit
    KV/heap teardown between cohorts. Hypothesis: cumulative KV cache +
    heap from a single long-running session is what trips macOS jetsam
    at N>=52 on M5 Max bf16 (iter 17 silent-SIGKILL). Releasing the
    generator + forcing gc + clearing the metal cache between cohorts
    keeps peak transient pressure bounded.
    """
    import gc
    import mlx.core as mx

    n = len(decisions)
    n_cohorts = (n + cohort_size - 1) // cohort_size
    print(
        f"[bench] cohort mode: N={n} cohort_size={cohort_size} "
        f"-> {n_cohorts} cohort(s)",
        flush=True,
    )

    all_rows: list[dict] = []
    all_wave_walls: list[float] = []
    cohort_walls: list[float] = []
    bench_t0 = time.time()
    for ci in range(n_cohorts):
        lo = ci * cohort_size
        hi = min(lo + cohort_size, n)
        c_decisions = decisions[lo:hi]
        c_gis = global_indices[lo:hi]
        c_t0 = time.time()
        print(
            f"[bench] cohort {ci+1}/{n_cohorts}: decisions {lo}..{hi-1} "
            f"(n={len(c_decisions)})",
            flush=True,
        )
        result = run_bench_continuous(
            decisions=c_decisions,
            global_indices=c_gis,
            out_dir=out_dir,
            model=model,
            variant=variant,
            oracle=oracle,
            batch=batch,
            turn_cap=turn_cap,
            tracker=tracker,
            heartbeat_label=f"[bench] cohort {ci+1}/{n_cohorts}",
        )
        all_rows.extend(result["rows"])
        all_wave_walls.extend(result["wave_walls_s"])
        c_wall = time.time() - c_t0
        cohort_walls.append(c_wall)
        print(
            f"[bench] cohort {ci+1}/{n_cohorts} done in {c_wall:.1f}s "
            f"(per-decision wall ~{c_wall / max(len(c_decisions), 1):.2f}s)",
            flush=True,
        )

        # Inter-cohort teardown: gen.close() already ran inside
        # run_bench_continuous, so the BatchGenerator + its KV cache
        # are released by name. Force gc + clear metal cache so the
        # OS sees the freed working set before the next cohort starts.
        gc.collect()
        try:
            mx.metal.clear_cache()
        except Exception:
            pass
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass
        print(
            f"[bench] post-cohort teardown: gc.collect + "
            f"mx.metal.clear_cache",
            flush=True,
        )

    bench_wall = time.time() - bench_t0
    print(
        f"[bench] all cohorts done; total bench_wall={bench_wall:.1f}s "
        f"per-cohort walls={[round(w,1) for w in cohort_walls]}",
        flush=True,
    )
    return {
        "rows": all_rows,
        "bench_wall_s": bench_wall,
        "wave_walls_s": all_wave_walls,
        "cohort_walls_s": cohort_walls,
    }


def run_bench(
    *,
    decisions: list,
    global_indices: list[int],
    out_dir: Path,
    model: Any,
    variant: Any,
    oracle: Any,
    batch: int,
    turn_cap: int,
) -> dict:
    """Drive ``decisions`` through the production tool loop, return per-decision rows.

    Imports the harvest_batched primitives at call time — they are not
    public modules, but they are the canonical eval inner loop.  When/if
    they get promoted, this bench updates with them.
    """
    import harvest_batched as hb  # type: ignore
    from burl.harness.tool_loop_native import parse_native_completion

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    bench_t0 = time.time()

    # Process in waves of ``batch``; each wave runs lockstep until all
    # decisions in the wave finish (mirrors ``eval_adapter_smoke._run_batched``).
    wave_walls: list[float] = []
    for wave_start in range(0, len(decisions), batch):
        wave_decisions = decisions[wave_start : wave_start + batch]
        wave_gis = global_indices[wave_start : wave_start + batch]
        wave_t0 = time.time()

        states: list[hb._DecisionState] = []
        with PHASES.phase("init_decision_states"):
            for gi, dec in zip(wave_gis, wave_decisions):
                dec_dir = out_dir / f"decision_{gi}"
                st = hb._init_decision_state(
                    gi, dec, variant, oracle, dec_dir, max_turns=turn_cap,
                )
                states.append(st)

        # Per-decision finish-wall: records the moment ``s.done`` first
        # flipped to True (or wave end if it never did, which can't happen
        # post-finalize but we guard).  At batch=N with one wave, decisions
        # that finish in fewer turns get a shorter wall — what we care
        # about for p50/p95.  Note: at batch>1 the GPU is shared, so this
        # under-reports the marginal cost of an additional decision in
        # the batch.  Single-stream (batch=1) gives the cleanest per-
        # decision number; the bench's job is to report both.
        finish_wall_by_gi: dict[int, float] = {}

        step_idx = 0
        while any(not s.done for s in states):
            step_idx += 1
            active = [s for s in states if not s.done]
            active_payload: list[dict] = []
            prompt_texts: list[str] = []
            with PHASES.phase("prompt_build_outer"):
                for s in active:
                    msgs, schemas = hb._prepare_step(s)
                    active_payload.append({
                        "messages": msgs, "tools": schemas, "done": False,
                    })
                    prompt_texts.append(model.tokenizer.apply_chat_template(
                        s.messages, tools=s.schemas,
                        tokenize=False, add_generation_prompt=True,
                        enable_thinking=False,
                    ))

            completions = model.step_batch(active_payload)
            assert len(completions) == len(active)
            t_after_gen = time.time()
            with PHASES.phase("apply_step"):
                for s, comp, ptext in zip(active, completions, prompt_texts):
                    was_done = s.done
                    try:
                        hb._apply_step(
                            s, comp, ptext,
                            parse_completion=parse_native_completion,
                        )
                    except Exception as exc:
                        s.done = True
                        s.result_meta["bailed"] = True
                        s.result_meta["bail_reason"] = (
                            f"apply_step failed (sync-wave): "
                            f"{type(exc).__name__}: {exc}"
                        )
                    if s.done and not was_done:
                        finish_wall_by_gi[int(s.gi)] = t_after_gen - s.wall_t0

        # Finalize each decision; preserves the per-decision wall.
        with PHASES.phase("finalize"):
            for s in states:
                # Use the recorded finish-wall (when s.done flipped); fall back
                # to wave-end for any decision that completed via the
                # post-loop finalize forced-commit path.
                wall_s = finish_wall_by_gi.get(int(s.gi), time.time() - s.wall_t0)
                row = hb._finalize(s, oracle, wall_s)
                row["global_idx"] = s.gi
                row["bench_per_decision_wall_s"] = round(wall_s, 3)
                rows.append(row)

        wave_walls.append(time.time() - wave_t0)

    bench_wall = time.time() - bench_t0
    return {
        "rows": rows,
        "bench_wall_s": bench_wall,
        "wave_walls_s": wave_walls,
    }


# --------------------------------------------------------------------------- #
# K1 + regret aggregation                                                      #
# --------------------------------------------------------------------------- #


def per_decision_grade(row: dict) -> dict:
    """Extract K1-pass and regret from one eval row.

    K1 = ``eq_delta_vs_bot >= 0`` (model E[Q] meets or beats the bot's E[Q]).
    Regret = ``max(0, -eq_delta_vs_bot)`` (oracle-relative since bot is
    E[Q]-greedy on the same Q-tensor).
    """
    delta = row.get("eq_delta_vs_bot")
    if delta is None:
        return {
            "global_idx": int(row["global_idx"]),
            "k1_pass": False,
            "signed_delta": None,
            "regret": None,
            "final_play": row.get("final_play"),
            "matches_bot": bool(row.get("matches_bot", False)),
        }
    return {
        "global_idx": int(row["global_idx"]),
        "k1_pass": float(delta) >= 0.0,
        "signed_delta": float(delta),
        "regret": max(0.0, -float(delta)),
        "final_play": row.get("final_play"),
        "matches_bot": bool(row.get("matches_bot", False)),
    }


def compare_to_baseline(
    *,
    grades: list[dict],
    baseline_run: dict | None,
) -> dict:
    """Return ``(k1_grade_match_pct, regret_delta_pct)`` keyed dict.

    ``k1_grade_match_pct``: fraction of decisions whose K1 pass/fail
    classification matches the baseline run's, joined by ``global_idx``.
    Returns None if ``baseline_run`` is missing or no overlap.

    ``regret_delta_pct``: ``(this_mean - baseline_mean) / baseline_mean *
    100``.  Negative is better.  Returns None if baseline mean is 0
    (degenerate but possible) or baseline run is missing.
    """
    if baseline_run is None:
        return {
            "k1_grade_match_pct": None,
            "regret_delta_pct": None,
            "baseline_label": None,
            "baseline_timestamp": None,
            "baseline_n": 0,
        }

    base_grades = {
        int(g["global_idx"]): g for g in baseline_run["per_decision_grades"]
    }
    overlap = [g for g in grades if int(g["global_idx"]) in base_grades]
    if not overlap:
        return {
            "k1_grade_match_pct": None,
            "regret_delta_pct": None,
            "baseline_label": baseline_run.get("variant_label"),
            "baseline_timestamp": baseline_run.get("timestamp"),
            "baseline_n": len(base_grades),
        }
    n_match = sum(
        1 for g in overlap
        if bool(g["k1_pass"]) == bool(base_grades[int(g["global_idx"])]["k1_pass"])
    )
    k1_pct = 100.0 * n_match / len(overlap)

    base_regrets = [
        base_grades[int(g["global_idx"])]["regret"] for g in overlap
        if base_grades[int(g["global_idx"])]["regret"] is not None
        and g["regret"] is not None
    ]
    this_regrets = [
        g["regret"] for g in overlap
        if g["regret"] is not None
        and base_grades[int(g["global_idx"])]["regret"] is not None
    ]
    if base_regrets and statistics.mean(base_regrets) > 0:
        regret_pct = 100.0 * (
            statistics.mean(this_regrets) - statistics.mean(base_regrets)
        ) / statistics.mean(base_regrets)
    else:
        regret_pct = (
            0.0
            if (this_regrets and statistics.mean(this_regrets) == 0)
            else None
        )

    return {
        "k1_grade_match_pct": round(k1_pct, 2),
        "regret_delta_pct": (
            round(regret_pct, 2) if regret_pct is not None else None
        ),
        "baseline_label": baseline_run.get("variant_label"),
        "baseline_timestamp": baseline_run.get("timestamp"),
        "baseline_n": len(overlap),
    }


def latest_baseline_run(
    *,
    ledger_path: Path,
    subset_label: str,
    results_dir: Path,
) -> dict | None:
    """Return the per-run JSON for the latest baseline-bf16 row at ``subset_label``.

    Returns None if no prior baseline-bf16 row exists.
    """
    if not ledger_path.exists():
        return None
    prior_baseline_ts: str | None = None
    with ledger_path.open() as fid:
        reader = csv.DictReader(fid)
        for row in reader:
            if row.get("variant_label") == "baseline-bf16" and row.get("subset") == subset_label:
                prior_baseline_ts = row["timestamp"]
    if prior_baseline_ts is None:
        return None
    candidate = results_dir / f"perf_{prior_baseline_ts}_baseline-bf16.json"
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text())


# --------------------------------------------------------------------------- #
# Ledger I/O                                                                   #
# --------------------------------------------------------------------------- #


def append_ledger_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=LEDGER_HEADER)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LEDGER_HEADER})


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _git_rev() -> tuple[str, str]:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
    except Exception:
        sha = ""
    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
    except Exception:
        branch = ""
    return sha, branch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--variant", required=True,
        help=(
            "Free-form label for this perf row (e.g. 'baseline-bf16', "
            "'prefix-cache', 'turn-aware-tokens'). Phase 1+ scribes "
            "use this to thread their lever through ``--variant <name>``."
        ),
    )
    ap.add_argument(
        "--subset", default="5",
        help="'5' (frozen 5-decision subset) or '560' (full run-3c eval set).",
    )
    ap.add_argument(
        "--subset-limit", type=int, default=None,
        help=(
            "Optional row cap: truncate the resolved decision list to the first "
            "N rows. Used by sub-lever 5c to slice subset 560 down to N=40-60 "
            "where N >> batch_width is true but iter wall stays under ~15min."
        ),
    )
    ap.add_argument(
        "--model-path", default=None,
        help=(
            "Optional MLX adapter path. The bench feeds this through to "
            "GemmaLocalNativeBatched(adapter_path=...). bf16 base model "
            "if omitted."
        ),
    )
    ap.add_argument(
        "--max-tokens-policy", choices=("default", "turn-aware"), default="default",
        help=(
            "default = constant max_tokens=8192 per turn (run-3c parity). "
            "turn-aware reserved for Phase 1's small-budget-on-early-turns "
            "lever; not implemented in Phase 0."
        ),
    )
    ap.add_argument(
        "--batch", type=int, default=5,
        help=(
            "Lockstep batch width. Default 5 (matches the frozen subset "
            "size — one wave). Phase 2's continuous-batching scribe will "
            "thread alternative widths through here."
        ),
    )
    ap.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help="Per-turn max generation tokens. Default matches run-3c reference.",
    )
    ap.add_argument(
        "--turn-cap", type=int, default=DEFAULT_TURN_CAP,
        help="Per-decision turn cap. Default 8 (harvest_batched).",
    )
    ap.add_argument(
        "--temperature", type=float, default=0.6,
        help=(
            "Sampler temperature. 0.6 matches run-3c. Setting to 0 makes "
            "MLX nondeterminism still possible at the kernel level — "
            "K1 grade match across runs at temp=0 is documented as "
            "'expected to round to 100%%, with rare per-decision flips' "
            "in wiki/experiments/burl-perf-phase0."
        ),
    )
    ap.add_argument(
        "--model-repo", default=DEFAULT_MODEL_REPO,
        help="MLX-LM model repo. bf16 baseline by default.",
    )
    ap.add_argument(
        "--notes", default="",
        help="Free-form notes column written to the ledger.",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help=(
            "Optional override for per-run output directory. Default: "
            "burl/eval/results/perf_<timestamp>_<variant>/"
        ),
    )
    ap.add_argument(
        "--enable-prompt-cache", action="store_true",
        help=(
            "Phase 2 lever 1: thread mlx_lm's LRUPromptCache through "
            "batch_generate so growing message histories share their KV "
            "across turns. Off by default (Phase 0 baseline parity). "
            "Note: in this shape it loses on M5 Max — see "
            "wiki/experiments/burl-perf-phase2.md for the negative-result "
            "writeup."
        ),
    )
    ap.add_argument(
        "--continuous", action="store_true",
        help=(
            "Phase 2 lever 2: drive the bench through a continuous-batching "
            "dispatcher built on mlx-lm's BatchGenerator. Decisions submit "
            "their first turn at once; as a stream finishes, tools dispatch "
            "and the next turn's prompt re-submits to the same generator "
            "while other streams keep decoding. Replaces the sync-wave loop."
        ),
    )
    ap.add_argument(
        "--cohort-size", type=int, default=0,
        help=(
            "Lever #6 cohort abstraction (continuous mode only). 0 = no "
            "cohorts (current behavior; one BatchGenerator session for all "
            "N decisions). N>0 = split decisions into chunks of size N; "
            "each chunk runs through a fresh BatchGenerator with explicit "
            "KV/heap teardown between chunks (gc.collect + "
            "mx.metal.clear_cache). Workaround for the iter 17 silent-jetsam "
            "ceiling — sidesteps the cumulative KV-cache + heap pressure "
            "from a single long-running session."
        ),
    )
    ap.add_argument(
        "--subprocess-isolation", action="store_true",
        help=(
            "Lever #6 closure (continuous + cohort-size>0 only). Each "
            "cohort runs in a fresh python subprocess that loads the "
            "model from scratch and exits, so cumulative session "
            "pressure (Metal cache, KV state, heap fragmentation) "
            "cannot accumulate across cohort boundaries. Eliminates "
            "the silent-jetsam ceiling drift documented in iters "
            "17/18/22/31. Pays ~3s warmup tax per cohort, negligible "
            "vs cohort wall of ~7-14min."
        ),
    )
    ap.add_argument(
        "--subprocess-cohort-range", default=None,
        help=(
            "INTERNAL — set by the parent when --subprocess-isolation "
            "is on. Format LO:HI; restricts decisions to this slice "
            "of the resolved (and subset-limit-applied) decision list."
        ),
    )
    ap.add_argument(
        "--subprocess-cohort-result", default=None,
        help=(
            "INTERNAL — set by the parent when --subprocess-isolation "
            "is on. Path where the child writes its sidecar JSON "
            "(rows + bench_wall_s + wave_walls_s + step_stats); the "
            "child exits before parent-only ledger aggregation."
        ),
    )
    ap.add_argument(
        "--prune-lm-head", action="store_true",
        help=(
            "Lever #16 phase 2: slice the (tied) LM head output projection "
            "down to the iter28 keep-set (top-N most-frequent emit token "
            "rows ∪ tokenizer special ids). Input embeddings are unchanged; "
            "only the output matmul shrinks. Sampler is wrapped with a "
            "pruned-index → original-vocab-id LUT so KV cache + tokenizer "
            "downstream see original IDs."
        ),
    )
    ap.add_argument(
        "--prune-keep-n", type=int, default=8192,
        help=(
            "Number of top-frequency emit token rows to keep when "
            "--prune-lm-head is set. Default 8192. Phase 1 measured only "
            "4,493 unique emit IDs across 2.2M emit tokens, so 8192 covers "
            "100%% with comfortable headroom for distribution drift."
        ),
    )
    ap.add_argument(
        "--prune-freq-tsv",
        default="/Users/jason/code/mk5-main/scratch/burl-perf-2/iter28_emit_token_freqs.tsv",
        help=(
            "Path to iter28's emit-frequency TSV. Default: the canonical "
            "main-checkout artifact built by iter28_probe.py."
        ),
    )
    ap.add_argument(
        "--log-argmax-winners", default=None,
        help=(
            "Lever #16 phase 2-redo calibration mode (iter 30). When set, "
            "wrap the sampler with a shim that logs strict-argmax winners "
            "of the FULL-vocab logprobs at every decode step to this path "
            "(one line per step, comma-separated ids per active stream). "
            "Run with --prune-lm-head OFF so the logged ids are real "
            "argmax winners, not LUT-mapped pruned-space winners. Used to "
            "derive a greedy-decode keep_set that complements the "
            "temp=0.6-sampled iter28 emit set."
        ),
    )
    ap.add_argument(
        "--kernel-audit", action="store_true",
        help=(
            "Lever #7 sub-fused-kernel audit: install KernelCounter hooks on "
            "the loaded Gemma 4 hot path before run; emit a kernel_audit "
            "block in the per-run JSON ranking kernel-class call counts and "
            "estimated cumulative GPU wall (decode is bandwidth-bound at B=5 "
            "L=1; per-call cost is dominated by reading the weight tile from "
            "unified memory). If MTL_CAPTURE_ENABLED=1, also writes a "
            ".gputrace file under the out_dir as a research artifact."
        ),
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_tokens_policy == "turn-aware":
        raise NotImplementedError(
            "turn-aware max-tokens policy is reserved for Phase 1; "
            "the Phase 0 bench measures only --max-tokens-policy default."
        )

    # Convert silent macOS jetsam SIGKILL into a visible RuntimeError when
    # the bench's working set exceeds Apple's recommended 40.2 GB Metal
    # working-set on M5 Max. See wiki/playbooks/perf-sprint-traps.md
    # ("Silent SIGKILL right after [bench] out_dir= print at N>=52 ...").
    import mlx.core as mx
    mx.set_memory_limit(40 * 1024**3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sha, branch = _git_rev()

    # ----- subset resolution -----
    if args.subset == "5":
        header, rows = load_subset(SUBSET_PATH)
        subset_label = "5"
    elif args.subset == "560":
        # Full-eval mode: reuse the run-3c eval directory's global_indices
        # as the "frozen 560" set.  The bench resolves them through the
        # same gus_eval_bridge path.
        header = {"_meta": True, "subset_name": "perf_subset_560",
                  "corpus_path": str(REPO_ROOT.parent.parent.parent / "gus" / "data" / "corpus_eval_20.pt")
                  if False else "/Users/jason/code/mk5-main/gus/data/corpus_eval_20.pt"}
        rows = [{"global_idx": gi} for gi in range(560)]
        subset_label = "560"
    else:
        raise ValueError(f"--subset must be '5' or '560', got {args.subset!r}")

    print(
        f"[bench] variant={args.variant} subset={subset_label} batch={args.batch} "
        f"max_tokens={args.max_tokens} turn_cap={args.turn_cap} "
        f"adapter={args.model_path}",
        flush=True,
    )

    # ----- corpus load + fingerprint check -----
    from burl.eval.gus_eval_bridge import build_burl_decision, load_corpus

    corpus_path = Path(header["corpus_path"])
    if subset_label == "5":
        observed_sha = sha256_of(corpus_path)
        if observed_sha != header["corpus_sha256"]:
            raise RuntimeError(
                f"corpus.pt SHA256 changed: frozen={header['corpus_sha256']} "
                f"observed={observed_sha} — re-freeze the subset."
            )
    print(f"[bench] loading corpus {corpus_path} ...", flush=True)
    corpus = load_corpus(corpus_path)

    decisions = []
    for row in rows:
        gi = int(row["global_idx"])
        bd = build_burl_decision(corpus, gi)
        if subset_label == "5":
            observed_fp = fingerprint_decision(bd)
            if observed_fp != row["fingerprint"]:
                raise RuntimeError(
                    f"gi={gi} fingerprint mismatch: frozen={row['fingerprint']} "
                    f"observed={observed_fp} — re-freeze the subset."
                )
        decisions.append(bd)

    global_indices = [int(r["global_idx"]) for r in rows]
    if args.subset_limit is not None and args.subset_limit < len(decisions):
        decisions = decisions[: args.subset_limit]
        global_indices = global_indices[: args.subset_limit]
        print(
            f"[bench] subset-limit applied: truncated to first "
            f"{args.subset_limit} decisions",
            flush=True,
        )
    # Subprocess-isolation child: restrict to the parent's cohort slice.
    # Subset resolution is deterministic, so the child re-resolving the
    # same subset and slicing [lo:hi] yields the same decisions the parent
    # would have passed to a single cohort of run_bench_continuous.
    if args.subprocess_cohort_range is not None:
        try:
            lo_str, hi_str = args.subprocess_cohort_range.split(":")
            lo, hi = int(lo_str), int(hi_str)
        except ValueError as exc:
            raise ValueError(
                f"--subprocess-cohort-range must be LO:HI integers, "
                f"got {args.subprocess_cohort_range!r}"
            ) from exc
        if lo < 0 or hi > len(decisions) or lo >= hi:
            raise ValueError(
                f"--subprocess-cohort-range {lo}:{hi} out of range for "
                f"resolved n={len(decisions)} decisions"
            )
        decisions = decisions[lo:hi]
        global_indices = global_indices[lo:hi]
        print(
            f"[bench] subprocess-cohort-range applied: sliced to "
            f"[{lo}:{hi}] -> n={len(decisions)} decisions",
            flush=True,
        )
    print(
        f"[bench] resolved {len(decisions)} decisions; "
        f"trick positions = "
        f"{sorted({(d.trick_idx // 4) + 1 for d in decisions})}",
        flush=True,
    )

    # ----- model + harness deps -----
    from burl.tools.eq_distribution import load_eq_oracle
    from burl.tools.belief_trajectory import load_gus
    from sweep import build_variants  # type: ignore

    print("[bench] loading E[Q] oracle ...", flush=True)
    oracle = load_eq_oracle()
    print("[bench] loading gus belief adapter ...", flush=True)
    # Resolve gus adapter explicitly so the bench works inside a worktree
    # (gus/adapters/ is gitignored and lives only in the main checkout).
    gus_candidates = [
        REPO_ROOT / "gus" / "adapters" / "v3_consistency_10000g.pt",
        Path("/Users/jason/code/mk5-main/gus/adapters/v3_consistency_10000g.pt"),
    ]
    gus_path = next((p for p in gus_candidates if p.exists()), None)
    if gus_path is None:
        raise FileNotFoundError(
            "gus belief adapter not found at any of: "
            + ", ".join(str(p) for p in gus_candidates)
        )
    _ = load_gus(adapter_path=gus_path)
    variant_obj = next(
        v for v in build_variants() if v.name == DEFAULT_VARIANT_NAME_RUNTIME
    )

    tracker = StatsTracker()
    # Subprocess-isolation parent: do NOT load the model. Each cohort
    # subprocess loads its own (and exits), which is the whole point —
    # the parent never holds the bf16 weights, so OS-level pressure
    # cannot accumulate across cohort boundaries.
    parent_isolation = (
        args.subprocess_isolation
        and args.continuous
        and int(args.cohort_size) > 0
        and args.subprocess_cohort_range is None
    )
    if parent_isolation:
        model = None
        load_wall = 0.0
        print(
            f"[bench] subprocess-isolation parent: skipping model load "
            f"(each cohort child loads its own)",
            flush=True,
        )
    else:
        print(
            f"[bench] loading Gemma 4 E2B (bf16, adapter={args.model_path}) "
            f"prompt_cache={args.enable_prompt_cache}...",
            flush=True,
        )
        t_load = time.time()
        model = make_tracking_model(
            adapter_path=args.model_path,
            max_tokens=int(args.max_tokens),
            temperature=float(args.temperature),
            model_repo=args.model_repo,
            tracker=tracker,
            enable_prompt_cache=bool(args.enable_prompt_cache),
            prune_lm_head=bool(args.prune_lm_head),
            prune_freq_tsv=args.prune_freq_tsv,
            prune_keep_n=int(args.prune_keep_n),
            log_argmax_winners=args.log_argmax_winners,
        )
        load_wall = time.time() - t_load
        print(f"[bench] model ready in {load_wall:.1f}s", flush=True)

    # ----- run -----
    out_dir = args.out_dir or (
        REPO_ROOT / "burl" / "eval" / "results" / f"perf_{timestamp}_{args.variant}"
    )
    print(f"[bench] out_dir={out_dir}", flush=True)

    # Lever #7 kernel-class audit instrumentation (pre-run install).
    gputrace_path: Path | None = None
    if args.kernel_audit and not parent_isolation:
        # The model wrapper's underlying mlx model lives on .model.
        underlying = getattr(model, "model", model)
        install_kernel_hooks(underlying)
        print(f"[bench] kernel-audit hooks installed", flush=True)
        if os.environ.get("MTL_CAPTURE_ENABLED") == "1":
            import mlx.core as _mx
            out_dir.mkdir(parents=True, exist_ok=True)
            gputrace_path = out_dir / "kernel_audit.gputrace"
            if gputrace_path.exists():
                # mx.metal.start_capture refuses to overwrite.
                import shutil
                shutil.rmtree(gputrace_path)
            try:
                _mx.metal.start_capture(str(gputrace_path))
                print(
                    f"[bench] mx.metal capture started → {gputrace_path}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[bench] WARN: mx.metal.start_capture failed "
                    f"({type(exc).__name__}: {exc}); continuing without trace",
                    flush=True,
                )
                gputrace_path = None

    if parent_isolation:
        # Strip the program name (sys.argv[0]) — we re-invoke with -m,
        # so the child receives only flags.
        parent_argv = list(sys.argv[1:])
        bench_result = run_bench_continuous_cohorts_subprocess(
            n_decisions=len(decisions),
            out_dir=out_dir,
            cohort_size=int(args.cohort_size),
            parent_argv=parent_argv,
        )
        # Reconstitute the parent tracker from each child's step_stats so
        # decode_tok_s / peak_mem_gb aggregation flows through the same
        # tracker.aggregate() path the in-process run uses. step_stats
        # were serialized in the child via __dict__; re-hydrate via
        # StepStats(**fields).
        for s in bench_result.get("subprocess_step_stats", []):
            tracker.steps.append(StepStats(**s))
    elif args.continuous and int(args.cohort_size) > 0:
        bench_result = run_bench_continuous_cohorts(
            decisions=decisions,
            global_indices=global_indices,
            out_dir=out_dir,
            model=model,
            variant=variant_obj,
            oracle=oracle,
            batch=int(args.batch),
            turn_cap=int(args.turn_cap),
            tracker=tracker,
            cohort_size=int(args.cohort_size),
        )
    elif args.continuous:
        # In subprocess-isolation child mode the cohort label includes the
        # parent's slice range so heartbeat lines disambiguate cohorts in
        # the parent log.
        if args.subprocess_cohort_range is not None:
            hb_label = f"[bench] subproc-cohort {args.subprocess_cohort_range}"
        else:
            hb_label = "[bench]"
        bench_result = run_bench_continuous(
            decisions=decisions,
            global_indices=global_indices,
            out_dir=out_dir,
            model=model,
            variant=variant_obj,
            oracle=oracle,
            batch=int(args.batch),
            turn_cap=int(args.turn_cap),
            tracker=tracker,
            heartbeat_label=hb_label,
        )
    else:
        bench_result = run_bench(
            decisions=decisions,
            global_indices=global_indices,
            out_dir=out_dir,
            model=model,
            variant=variant_obj,
            oracle=oracle,
            batch=int(args.batch),
            turn_cap=int(args.turn_cap),
        )
    bench_wall = bench_result["bench_wall_s"]

    # Subprocess-isolation child: emit the sidecar JSON and exit before
    # parent-only ledger aggregation. Parent re-aggregates across all
    # cohort sidecars from inside its own main().
    if args.subprocess_cohort_result is not None:
        sidecar = {
            "rows": bench_result["rows"],
            "bench_wall_s": float(bench_result["bench_wall_s"]),
            "wave_walls_s": list(bench_result.get("wave_walls_s", [])),
            "step_stats": [s.__dict__ for s in tracker.steps],
            "n_decisions": len(bench_result["rows"]),
        }
        sidecar_path = Path(args.subprocess_cohort_result)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(sidecar, default=str))
        print(
            f"[bench] subproc-cohort sidecar -> {sidecar_path} "
            f"(n={len(bench_result['rows'])} bench_wall="
            f"{bench_result['bench_wall_s']:.1f}s)",
            flush=True,
        )
        return 0

    # Lever #7 — stop Metal capture (if running) and tear down hooks.
    if args.kernel_audit and not parent_isolation:
        if gputrace_path is not None:
            try:
                import mlx.core as _mx
                _mx.metal.stop_capture()
                print(
                    f"[bench] mx.metal capture stopped → {gputrace_path}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[bench] WARN: mx.metal.stop_capture failed "
                    f"({type(exc).__name__}: {exc})",
                    flush=True,
                )
        teardown_kernel_hooks()

    # ----- aggregate -----
    cache_hit_tokens_total = int(getattr(model, "cache_hit_tokens_total", 0))
    cache_processed_tokens_total = int(getattr(
        model, "cache_processed_tokens_total", 0,
    ))
    print(
        f"[bench] cache_hit_tokens={cache_hit_tokens_total} "
        f"cache_processed_tokens={cache_processed_tokens_total}",
        flush=True,
    )
    eval_rows = bench_result["rows"]
    grades = [per_decision_grade(r) for r in eval_rows]
    walls = [
        float(r["bench_per_decision_wall_s"]) for r in eval_rows
    ]
    walls_sorted = sorted(walls)
    n = len(walls_sorted)
    p50 = walls_sorted[n // 2] if n else 0.0
    p95_idx = int(n * 0.95)
    if p95_idx >= n:
        p95_idx = n - 1
    p95 = walls_sorted[p95_idx] if n else 0.0
    stat_agg = tracker.aggregate()

    # K1 + regret vs baseline-bf16.
    # If no prior baseline-bf16 row exists AND this run is itself a
    # baseline-bf16 row, fall back to self-comparison so the ledger row
    # carries 100/0 instead of empty cells — exercises the rescore code
    # path the first time the bench is invoked.
    baseline_run = latest_baseline_run(
        ledger_path=LEDGER_PATH,
        subset_label=subset_label,
        results_dir=REPO_ROOT / "burl" / "eval" / "results",
    )
    if baseline_run is None and args.variant == "baseline-bf16":
        cmp = compare_to_baseline(
            grades=grades,
            baseline_run={
                "variant_label": "self",
                "timestamp": timestamp,
                "per_decision_grades": grades,
            },
        )
        cmp["baseline_label"] = "self (no prior baseline)"
    else:
        cmp = compare_to_baseline(grades=grades, baseline_run=baseline_run)

    # ----- write per-run JSON -----
    json_path = REPO_ROOT / "burl" / "eval" / "results" / (
        f"perf_{timestamp}_{args.variant}.json"
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    phase_breakdown = PHASES.report(total_wall_s=bench_wall)
    detail = {
        "timestamp": timestamp,
        "sha": sha,
        "branch": branch,
        "variant_label": args.variant,
        "subset": subset_label,
        "batch": int(args.batch),
        "max_tokens": int(args.max_tokens),
        "max_tokens_policy": args.max_tokens_policy,
        "turn_cap": int(args.turn_cap),
        "temperature": float(args.temperature),
        "adapter_path": args.model_path,
        "model_repo": args.model_repo,
        "n_decisions": len(eval_rows),
        "bench_wall_s_total": round(bench_wall, 3),
        "model_load_wall_s": round(load_wall, 3),
        "wall_s_per_decision": [round(w, 3) for w in walls],
        "wall_s_p50": round(p50, 3),
        "wall_s_p95": round(p95, 3),
        "stats_aggregate": stat_agg,
        "per_decision_grades": grades,
        "per_decision_rows": [
            {k: v for k, v in r.items() if k != "rows"}
            for r in eval_rows
        ],
        "wave_walls_s": [round(w, 3) for w in bench_result["wave_walls_s"]],
        "step_stats": [s.__dict__ for s in tracker.steps],
        "phase_breakdown": phase_breakdown,
        "comparison_vs_baseline": cmp,
        "notes": args.notes,
        "subset_header": header,
    }
    if args.kernel_audit:
        # Use BatchStats decode wall as the denominator (not bench_wall —
        # bench_wall includes apply_step + prompt_build, kernel hooks only
        # fire inside decode).
        decode_wall_s = float(stat_agg.get("total_generation_time_s") or 0.0)
        gen_tok = int(stat_agg.get("total_generation_tokens") or 0)
        detail["kernel_audit"] = KERNELS.report(
            gen_tokens=gen_tok, decode_wall_s=decode_wall_s,
        )
        if gputrace_path is not None and gputrace_path.exists():
            detail["kernel_audit"]["gputrace_path"] = str(gputrace_path)
    json_path.write_text(json.dumps(detail, indent=2, default=str))
    print(f"[bench] per-run detail -> {json_path}", flush=True)

    # ----- append ledger row -----
    ledger_row = {
        "timestamp": timestamp,
        "sha": sha,
        "branch": branch,
        "variant_label": args.variant,
        "subset": subset_label,
        "batch": int(args.batch),
        "max_tokens_policy": args.max_tokens_policy,
        "n_decisions": len(eval_rows),
        "wall_s_total": round(bench_wall, 3),
        "wall_s_p50": round(p50, 3),
        "wall_s_p95": round(p95, 3),
        "prefill_tok_s": (
            round(stat_agg["prefill_tok_s"], 1)
            if stat_agg["prefill_tok_s"] is not None else ""
        ),
        "decode_tok_s": (
            round(stat_agg["decode_tok_s"], 1)
            if stat_agg["decode_tok_s"] is not None else ""
        ),
        "peak_mem_gb": (
            round(stat_agg["peak_mem_gb"], 2)
            if stat_agg["peak_mem_gb"] is not None else ""
        ),
        "k1_grade_match_pct": (
            cmp["k1_grade_match_pct"]
            if cmp["k1_grade_match_pct"] is not None else ""
        ),
        "regret_delta_pct": (
            cmp["regret_delta_pct"]
            if cmp["regret_delta_pct"] is not None else ""
        ),
        "notes": args.notes,
    }
    append_ledger_row(LEDGER_PATH, ledger_row)
    print(f"[bench] ledger row -> {LEDGER_PATH}", flush=True)

    # ----- console summary -----
    print()
    print("=" * 72)
    print(f"[bench] {args.variant} subset={subset_label} n={len(eval_rows)}")
    print("=" * 72)
    print(f"  wall_s_total      : {bench_wall:.1f}s")
    print(f"  wall_s_p50        : {p50:.1f}s")
    print(f"  wall_s_p95        : {p95:.1f}s")
    print(f"  prefill_tok_s     : {stat_agg['prefill_tok_s']}")
    print(f"  decode_tok_s      : {stat_agg['decode_tok_s']}")
    print(f"  peak_mem_gb       : {stat_agg['peak_mem_gb']}")
    print(f"  k1_grade_match_pct: {cmp['k1_grade_match_pct']}")
    print(f"  regret_delta_pct  : {cmp['regret_delta_pct']}")
    print("=" * 72)
    print("[bench] phase breakdown (sorted by wall, sum may differ from")
    print("        total because untimed regions exist between phases):")
    print("=" * 72)
    print(f"  {'phase':<28}{'total_s':>10}{'pct':>8}{'n':>6}{'mean_ms':>12}")
    for name, info in phase_breakdown.items():
        print(
            f"  {name:<28}"
            f"{info['total_s']:>10.4f}"
            f"{info['pct_of_wall']:>7.2f}%"
            f"{info['n_calls']:>6}"
            f"{info['mean_ms']:>12.3f}"
        )
    timed_sum = sum(info['total_s'] for info in phase_breakdown.values())
    print(f"  {'(sum of timed phases)':<28}{timed_sum:>10.4f}"
          f"{100.0 * timed_sum / bench_wall if bench_wall > 0 else 0.0:>7.2f}%")
    print(f"  {'(bench_wall_s_total)':<28}{bench_wall:>10.4f}{'100.00':>8}%")
    print("=" * 72)

    if args.kernel_audit and "kernel_audit" in detail:
        ka = detail["kernel_audit"]
        print("[bench] kernel-class audit "
              f"(decode_wall={ka['decode_wall_s']}s gen_tokens="
              f"{ka['gen_tokens']} bw_assumed={ka['model_decode_bandwidth_assumed_gbs']}GB/s):")
        print("=" * 72)
        print(f"  {'kernel_class':<32}{'calls':>8}{'/tok':>7}{'MB/call':>10}{'cum_s':>8}{'pct':>7}")
        for r in ka["kernels"]:
            mb = r["bytes_per_call"] / 1e6
            print(
                f"  {r['kernel_class']:<32}"
                f"{r['calls_total']:>8}"
                f"{r['calls_per_token']:>7.2f}"
                f"{mb:>10.2f}"
                f"{r['est_cum_time_s']:>8.3f}"
                f"{r['pct_of_decode_wall']:>6.2f}%"
            )
        print("=" * 72)

    if subset_label == "560":
        # Phase-exit gate: also emit a markdown report.
        report_path = REPO_ROOT / "burl" / "eval" / "results" / (
            f"perf_full560_{timestamp}.md"
        )
        report_path.write_text(
            "# perf full-560 report\n\n"
            f"- variant: `{args.variant}`\n"
            f"- sha: `{sha}` branch: `{branch}`\n"
            f"- timestamp: `{timestamp}`\n"
            f"- batch: `{args.batch}` max_tokens: `{args.max_tokens}` "
            f"turn_cap: `{args.turn_cap}`\n\n"
            "## Aggregate\n\n"
            f"- wall_s_total: `{bench_wall:.1f}s`\n"
            f"- wall_s_p50/p95: `{p50:.1f}s / {p95:.1f}s`\n"
            f"- prefill/decode tok/s: `{stat_agg['prefill_tok_s']:.1f} / "
            f"{stat_agg['decode_tok_s']:.1f}`\n"
            f"- peak mem: `{stat_agg['peak_mem_gb']:.2f} GB`\n"
            f"- k1_grade_match_pct: `{cmp['k1_grade_match_pct']}`\n"
            f"- regret_delta_pct: `{cmp['regret_delta_pct']}`\n\n"
            "Notes: " + (args.notes or "_(none)_") + "\n"
        )
        print(f"[bench] phase-exit report -> {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
