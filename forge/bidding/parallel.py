"""Multi-process parallel simulation.

Spawns worker processes, each with its own model, to parallelize
game simulation across CPU cores. Bypasses GIL for true parallelism.

Usage:
    with MultiProcessSimulator(n_workers=8) as sim:
        results = sim.simulate_hands(hands, trump=5, n_games=200)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.multiprocessing as mp

from .inference import PolicyModel, DEFAULT_CHECKPOINT
from .simulator import simulate_games


@dataclass
class WorkItem:
    """A unit of work for a worker process."""
    hand: List[int]
    trump: int
    n_games: int
    seed: int | None
    greedy: bool


@dataclass
class WorkResult:
    """Result from a worker process."""
    hand: List[int]
    trump: int
    points: torch.Tensor  # (n_games,) on CPU


def _worker_loop(
    work_queue: mp.Queue,
    result_queue: mp.Queue,
    model_path: str,
    worker_id: int,
    n_gpus: int,
) -> None:
    """Worker process main loop.

    Loads model once, then processes work items until shutdown.
    Each worker is assigned to a specific GPU (round-robin).
    """
    # Assign this worker to a specific GPU (round-robin across available GPUs)
    if n_gpus > 0:
        device = f"cuda:{worker_id % n_gpus}"
    else:
        device = "cpu"

    # Load model in this process
    # Note: compile_model=False because reduce-overhead mode uses CUDA graphs
    # which are incompatible with TransformerEncoder's nested tensor fast path
    model = PolicyModel(checkpoint_path=model_path, device=device, compile_model=False)

    # Signal ready
    result_queue.put(("ready", worker_id))

    while True:
        item = work_queue.get()

        if item is None:  # Poison pill = shutdown
            break

        # Run simulation
        points = simulate_games(
            model=model,
            bidder_hand=item.hand,
            decl_id=item.trump,
            n_games=item.n_games,
            seed=item.seed,
            greedy=item.greedy,
        )

        # Send result back (move to CPU first)
        result = WorkResult(
            hand=item.hand,
            trump=item.trump,
            points=points.cpu(),
        )
        result_queue.put(result)


class MultiProcessSimulator:
    """Multi-process game simulator for multi-GPU clusters.

    Spawns N worker processes, each assigned to a specific GPU (round-robin).
    Designed for multi-GPU machines where each worker gets its own GPU.

    On single-GPU machines, use single-process mode instead (no benefit from
    multi-process when all workers share one GPU).

    Example:
        # On a 4-GPU machine
        with MultiProcessSimulator(n_workers=4) as sim:
            results = sim.simulate_hands(hands, trump=5, n_games=200)
    """

    def __init__(
        self,
        n_workers: int | None = None,
        model_path: Path | str | None = None,
    ):
        """Initialize simulator and spawn workers.

        Args:
            n_workers: Number of worker processes. Defaults to GPU count.
            model_path: Path to model checkpoint. Uses default if None.
        """
        # Detect available GPUs
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if n_workers is None:
            # Default to one worker per GPU (or 1 for CPU-only)
            n_workers = max(n_gpus, 1)

        if model_path is None:
            model_path = DEFAULT_CHECKPOINT

        self.n_workers = n_workers
        self.n_gpus = n_gpus
        self.model_path = str(model_path)

        # Use spawn to avoid CUDA context issues
        ctx = mp.get_context("spawn")

        self.work_queue: mp.Queue = ctx.Queue()
        self.result_queue: mp.Queue = ctx.Queue()
        self.workers: List[mp.Process] = []

        # Spawn workers
        for i in range(n_workers):
            p = ctx.Process(
                target=_worker_loop,
                args=(self.work_queue, self.result_queue, self.model_path, i, n_gpus),
                daemon=True,
            )
            p.start()
            self.workers.append(p)

        # Wait for all workers to be ready
        ready_count = 0
        while ready_count < n_workers:
            msg = self.result_queue.get()
            if msg[0] == "ready":
                ready_count += 1

    def simulate_hands(
        self,
        hands: List[List[int]],
        trump: int,
        n_games: int,
        seed: int | None = None,
        greedy: bool = True,
    ) -> List[torch.Tensor]:
        """Simulate multiple hands in parallel.

        Args:
            hands: List of hands (each hand is list of 7 domino IDs)
            trump: Declaration/trump ID
            n_games: Number of games per hand
            seed: Base random seed (each hand gets seed + index)
            greedy: Use greedy action selection

        Returns:
            List of point tensors, one per hand, in same order as input
        """
        # Submit all work
        for i, hand in enumerate(hands):
            item = WorkItem(
                hand=hand,
                trump=trump,
                n_games=n_games,
                seed=seed + i if seed is not None else None,
                greedy=greedy,
            )
            self.work_queue.put(item)

        # Collect results (may arrive out of order)
        results_map = {}
        for _ in range(len(hands)):
            result = self.result_queue.get()
            # Use tuple of hand as key (lists aren't hashable)
            key = tuple(result.hand)
            results_map[key] = result.points

        # Return in original order
        return [results_map[tuple(hand)] for hand in hands]

    def simulate_batch(
        self,
        work_items: List[Tuple[List[int], int, int, int | None, bool]],
    ) -> List[WorkResult]:
        """Simulate arbitrary batch of (hand, trump, n_games, seed, greedy) tuples.

        More flexible than simulate_hands - allows different trumps per item.
        Results returned in submission order.
        """
        # Submit all work
        for i, (hand, trump, n_games, seed, greedy) in enumerate(work_items):
            item = WorkItem(
                hand=hand,
                trump=trump,
                n_games=n_games,
                seed=seed,
                greedy=greedy,
            )
            self.work_queue.put(item)

        # Collect results
        results = []
        for _ in range(len(work_items)):
            result = self.result_queue.get()
            results.append(result)

        # Sort back to submission order using (hand, trump) as key
        # Build index map from work_items
        order_map = {(tuple(hand), trump): i for i, (hand, trump, *_) in enumerate(work_items)}
        results.sort(key=lambda r: order_map[(tuple(r.hand), r.trump)])

        return results

    def shutdown(self) -> None:
        """Shutdown all workers."""
        # Send poison pills
        for _ in self.workers:
            self.work_queue.put(None)

        # Wait for workers to exit
        for p in self.workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()

        self.workers = []

    def __enter__(self) -> "MultiProcessSimulator":
        return self

    def __exit__(self, *args) -> None:
        self.shutdown()
