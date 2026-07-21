#!/usr/bin/env python
"""Bounded-memory external-sampling BR gates."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import hoyt as K
from hoyt import toys as T
from hoyt.cfr import cfr_solve
from hoyt.sampled_br import SampledBR, SampledCFR, audit_sampled_gap
from walt.worlds import enumerate_worlds

_failures = []


def _gate(ok, name, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}"
          f"{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def gate_S1():
    root, worlds = T.gen_engine_root(23, 3)
    sub = K.build_subgame(root, worlds,
                          np.full(len(worlds), 1.0 / len(worlds)))
    exact = K.br_solve(sub, K.StochasticProfile(uniform_fallback=True),
                       K.payoff_points(), hero=root.me)
    solver = SampledBR(root, K.payoff_points(), capacity=1 << 18)
    solver.train(epochs=300, batch_size=128, seed=20)
    got = solver.evaluate(batches=30, batch_size=512, seed=100_000)
    z = abs(got.value - exact.value) / got.value_se
    ok = (got.best_move == exact.best_move and z < 4.0
          and abs(got.value - exact.value) <= got.value_error
          and got.key_collisions == 0
          and got.occupied_buckets < got.table_capacity)
    _gate(ok, "S1 sampled Metal BR covers exact H3 BR",
          f"move {got.best_move}/{exact.best_move}, |z| {z:.2f}, "
          f"rows {got.occupied_buckets:,}, peak {got.peak_frontier:,}")


def gate_S2():
    root = T.get_toy("t2_decl_w12").root
    solver = SampledBR(root, K.payoff_points(), capacity=1 << 14)
    solver.train(epochs=100, batch_size=64, seed=10)
    got = solver.evaluate(batches=20, batch_size=256, seed=100_000)
    exact_worlds = enumerate_worlds(root)
    exact_sub = K.build_subgame(
        root, exact_worlds, np.full(len(exact_worlds), 1.0 / len(exact_worlds)))
    exact = K.br_solve(exact_sub,
                       K.StochasticProfile(uniform_fallback=True),
                       K.payoff_points(), hero=root.me)
    ok = got.best_move == exact.best_move \
        and abs(got.value - exact.value) <= got.value_error
    _gate(ok, "S2 sampled BR reproducible on enumerable H2",
          f"value {got.value:.4f}, EB ±{got.value_error:.4f}, "
          f"exact {exact.value:.4f}")


def gate_S3():
    root, _ = T.gen_engine_root(23, 3)
    solver = SampledBR(root, K.payoff_points(), capacity=16)
    try:
        solver.train(epochs=2, batch_size=64, seed=5)
    except MemoryError:
        ok = True
    else:
        ok = False
    _gate(ok, "S3 sparse-table capacity fails closed")


def gate_S4():
    root = T.get_toy("t2_decl_w12").root
    worlds = enumerate_worlds(root)
    sub = K.build_subgame(
        root, worlds, np.full(len(worlds), 1.0 / len(worlds)))
    exact = cfr_solve(sub, K.payoff_points(), iters=300, br_every=50)
    solver = SampledCFR(root, K.payoff_points(), capacity=1 << 17)
    solver.train(epochs=300, batch_size=64, seed=30)
    got = solver.evaluate(batches=30, batch_size=512, seed=700_000)

    policy = solver.snapshot()
    mask = int(solver.sub.my0)
    legal = [tile for tile in range(28) if mask & (1 << tile)]
    moves, probs = policy.dist(root.me, mask, (), legal)
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / "policy.npz"
        policy.save(artifact)
        loaded = type(policy).load(root, artifact)
        artifact_ok = np.array_equal(loaded.keys, policy.keys) \
            and np.array_equal(loaded.regret, policy.regret)
    actors = set((policy.keys >> np.uint64(62)).astype(int).tolist())
    fork = solver.fork_best_response(root.me)
    forked = fork.snapshot()
    hero = (policy.keys >> np.uint64(62)) == int(root.me)
    frozen_ok = np.array_equal(forked.keys, policy.keys) \
        and np.all(forked.regret[hero] == 0.0) \
        and np.array_equal(forked.regret[~hero], policy.regret[~hero])
    audit = audit_sampled_gap(
        solver, br_epochs=20, train_batch=32,
        eval_batches=5, eval_batch=128, seed=800_000)
    try:
        audit_sampled_gap(
            solver, br_epochs=0, train_batch=32,
            eval_batches=1, eval_batch=2, shortfall_upper=0.0)
    except ValueError:
        split_ok = True
    else:
        split_ok = False
    z = abs(got.value - exact.value) / got.value_se
    ok = split_ok and artifact_ok \
        and np.array_equal(moves, np.asarray(legal)) \
        and abs(float(probs.sum()) - 1.0) < 1e-12 \
        and not np.allclose(probs, 1.0 / len(probs)) \
        and actors == {0, 1, 2, 3} and frozen_ok and z < 4.0 \
        and abs(got.value - exact.value) <= got.value_error \
        and audit.verdict == "unresolved" \
        and np.isfinite(audit.candidate_gap_upper) \
        and np.isinf(audit.gap_upper)
    _gate(ok, "S4 four-seat CFR and frozen-policy BR fork",
          f"value |z| {z:.2f}, rows {len(policy.keys):,}, "
          f"actors {sorted(actors)}, audit {audit.verdict}")


def main():
    gate_S1()
    gate_S2()
    gate_S3()
    gate_S4()
    if _failures:
        print(f"\n{len(_failures)} gate(s) FAILED: {_failures}")
        return 1
    print("\nall sampled-BR gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
