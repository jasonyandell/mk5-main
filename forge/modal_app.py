"""Modal app definition for distributed shard generation.

This module defines the Modal infrastructure for running GPU-intensive oracle
generation at scale. The app provides:

- GPU-accelerated container image with torch, pyarrow, numpy
- Persistent volume for shard storage
- Configurable GPU types (A10G default, A100 for larger workloads)

Usage:
    # Deploy the app
    modal deploy forge/modal_app.py

    # Generate val/test marginalized shards (seeds 900-999, 10 opp_seeds each)
    modal run forge/modal_app.py::generate_valtest

    # Generate specific seed range
    modal run forge/modal_app.py::generate_range --start-seed 900 --end-seed 950 --n-opp-seeds 10

    # Single shard for testing
    modal run forge/modal_app.py --base-seed 900 --opp-seed 0

See Also:
    - forge/ORIENTATION.md for ML pipeline architecture
    - forge/cli/generate_continuous.py for local shard generation
"""
from __future__ import annotations

import modal

# =============================================================================
# App Definition
# =============================================================================

app = modal.App("texas-42-forge")

# =============================================================================
# Image Configuration
# =============================================================================
#
# Image includes:
# - torch>=2.0: GPU computation for oracle solver
# - pyarrow>=14.0: Parquet I/O for shard storage
# - numpy>=1.26: Array operations (pinned for torch compatibility)
# - rich: Pretty progress output
#
# Note: We use pip_install (not uv_pip_install) for torch compatibility
# with CUDA. The debian_slim base includes CUDA drivers.

forge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        # Core ML stack (matches forge/requirements.txt)
        "torch>=2.0",
        "lightning>=2.0",
        "pyarrow>=14.0",
        "numpy>=1.26,<2",
        "pandas>=2.0",
        "rich",
        "wandb>=0.16",  # May be imported by training code
    )
    # Mount the forge package source code
    .add_local_dir(
        local_path="forge",
        remote_path="/root/forge",
        ignore=["__pycache__", "*.pyc", "venv", "*.egg-info"],
    )
)

# =============================================================================
# Volume Configuration
# =============================================================================
#
# Persistent volume for shard storage. Shards are written to:
#   /shards/{train,val,test}/seed_XXXXXXXX_*.parquet
#
# Volume persists across function invocations and can be accessed by
# multiple functions concurrently.

VOLUME_NAME = "texas-42-shards"
SHARD_MOUNT_PATH = "/shards"

shard_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# =============================================================================
# GPU Configuration
# =============================================================================
#
# Default: A10G (24GB VRAM) - sufficient for most shard generation
# Alternative: A100-40GB for seeds with >100M states
#
# Memory usage scales with state count:
#   - Typical seed: 50-70k states, ~2GB VRAM
#   - Large seed: 100-200M states, ~20GB VRAM
#   - Pathological: 400M+ states, may OOM even on A100

GPU_B200 = "B200"  # 192GB VRAM, newest, best availability?
GPU_H200 = "H200"  # 141GB VRAM, very fast
GPU_H100 = "H100"  # 80GB VRAM, fast
GPU_A10G = "A10G"  # 24GB VRAM, slower but works
GPU_A100_40GB = "A100-40GB"  # 40GB VRAM, for large seeds
GPU_A100_80GB = "A100-80GB"  # 80GB VRAM, for pathological seeds


# =============================================================================
# Helper Functions
# =============================================================================


def get_split_subdir(base_seed: int) -> str:
    """Route seed to train/val/test subdirectory."""
    bucket = base_seed % 1000
    if bucket < 900:
        return "train"
    elif bucket < 950:
        return "val"
    else:
        return "test"


# =============================================================================
# GPU Functions
# =============================================================================


@app.function(
    image=forge_image,
    gpu=GPU_A10G,  # A10 - cheap and reliable
    volumes={SHARD_MOUNT_PATH: shard_volume},
    timeout=600,  # 10 minutes per shard
)
@modal.concurrent(max_inputs=2)  # 2x concurrency - may OOM on ~2% large seeds, acceptable tradeoff
def generate_shard(
    base_seed: int,
    opp_seed: int,
    decl_id: int | None = None,
    max_states: int = 500_000_000,  # A10 has 24GB VRAM, can handle large seeds
) -> dict:
    """Generate a single marginalized shard.

    Args:
        base_seed: Base RNG seed determining P0's hand
        opp_seed: Opponent shuffle seed for marginalization
        decl_id: Declaration type (0-9). If None, uses base_seed % 10.
        max_states: Skip seeds with more states than this (OOM protection for local use)

    Returns:
        Dict with generation metadata (path, state_count, root_value, status)
    """
    import sys
    sys.path.insert(0, "/root")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Modal container")

    from pathlib import Path
    from forge.oracle.context import build_context
    from forge.oracle.declarations import DECL_ID_TO_NAME
    from forge.oracle.output import write_result
    from forge.oracle.rng import deal_from_seed
    from forge.oracle.solve import SolveConfig, build_child_index, enumerate_gpu, solve_gpu

    device = torch.device("cuda")
    if decl_id is None:
        decl_id = base_seed % 10

    split = get_split_subdir(base_seed)
    output_path = Path(SHARD_MOUNT_PATH) / split / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"
    lock_path = output_path.with_suffix(".lock")

    # Check if already exists or being generated
    if output_path.exists() or lock_path.exists():
        return {
            "base_seed": base_seed,
            "opp_seed": opp_seed,
            "decl_id": decl_id,
            "status": "skipped",
            "message": "Already exists or in progress",
            "path": str(output_path),
        }

    # Claim this shard immediately to prevent duplicate work
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch()

    print(f"[{torch.cuda.get_device_name(0)}] seed={base_seed} opp={opp_seed} decl={DECL_ID_TO_NAME.get(decl_id, str(decl_id))}")

    # Get P0's hand from base seed
    hands = deal_from_seed(base_seed)
    p0_hand = list(hands[0])

    # Build context with fixed P0 hand, opp_seed for opponent distribution
    ctx = build_context(seed=opp_seed, decl_id=decl_id, device=device, p0_hand=p0_hand)

    config = SolveConfig(
        child_index_chunk_size=1_000_000,
        solve_chunk_size=1_000_000,
        enum_chunk_size=100_000,
    )

    all_states = enumerate_gpu(ctx, config=config)
    state_count = int(all_states.shape[0])
    print(f"  States: {state_count:,}")

    if state_count > max_states:
        return {
            "base_seed": base_seed,
            "opp_seed": opp_seed,
            "decl_id": decl_id,
            "status": "skipped",
            "message": f"Too many states: {state_count:,} > {max_states:,}",
            "state_count": state_count,
        }

    child_idx = build_child_index(all_states, ctx, config=config)
    v, move_values = solve_gpu(all_states, child_idx, ctx, config=config)
    root_value = int(v[0])
    print(f"  Root value: {root_value:+d}")

    # Write result and remove lock file
    write_result(output_path, base_seed, decl_id, all_states, v, move_values, fmt="parquet")
    lock_path.unlink(missing_ok=True)

    # Skip per-shard commit - let Modal batch commits for efficiency
    # Volume auto-commits periodically and on container shutdown
    # shard_volume.commit()

    return {
        "base_seed": base_seed,
        "opp_seed": opp_seed,
        "decl_id": decl_id,
        "status": "success",
        "state_count": state_count,
        "root_value": root_value,
        "path": str(output_path),
    }


@app.function(
    image=forge_image,
    gpu=GPU_A100_80GB,  # 80GB for huge seeds
    volumes={SHARD_MOUNT_PATH: shard_volume},
    timeout=1800,  # 30 minutes for large seeds
)
def generate_shard_large(
    base_seed: int,
    opp_seed: int,
    decl_id: int | None = None,
    max_states: int = 800_000_000,  # 800M states for A100-80GB
) -> dict:
    """Generate shard on A100-80GB for seeds with high state counts."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    from pathlib import Path
    from forge.oracle.context import build_context
    from forge.oracle.declarations import DECL_ID_TO_NAME
    from forge.oracle.output import write_result
    from forge.oracle.rng import deal_from_seed
    from forge.oracle.solve import SolveConfig, build_child_index, enumerate_gpu, solve_gpu

    device = torch.device("cuda")
    if decl_id is None:
        decl_id = base_seed % 10

    split = get_split_subdir(base_seed)
    output_path = Path(SHARD_MOUNT_PATH) / split / f"seed_{base_seed:08d}_opp{opp_seed}_decl_{decl_id}.parquet"

    if output_path.exists():
        return {"status": "skipped", "message": "Already exists", "path": str(output_path)}

    print(f"[{torch.cuda.get_device_name(0)}] seed={base_seed} opp={opp_seed} decl={DECL_ID_TO_NAME.get(decl_id, str(decl_id))}")

    hands = deal_from_seed(base_seed)
    p0_hand = list(hands[0])
    ctx = build_context(seed=opp_seed, decl_id=decl_id, device=device, p0_hand=p0_hand)

    config = SolveConfig(child_index_chunk_size=1_000_000, solve_chunk_size=1_000_000, enum_chunk_size=100_000)
    all_states = enumerate_gpu(ctx, config=config)
    state_count = int(all_states.shape[0])
    print(f"  States: {state_count:,}")

    if state_count > max_states:
        return {"status": "skipped", "message": f"Too many states: {state_count:,}", "state_count": state_count}

    child_idx = build_child_index(all_states, ctx, config=config)
    v, move_values = solve_gpu(all_states, child_idx, ctx, config=config)
    root_value = int(v[0])
    print(f"  Root value: {root_value:+d}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_result(output_path, base_seed, decl_id, all_states, v, move_values, fmt="parquet")

    return {"status": "success", "state_count": state_count, "root_value": root_value, "path": str(output_path)}


# =============================================================================
# Batch Entrypoints
# =============================================================================


@app.local_entrypoint()
def main(
    base_seed: int = 900,
    opp_seed: int = 0,
    decl_id: int | None = None,
    large: bool = False,
):
    """Generate a single shard for testing.

    Usage:
        modal run forge/modal_app.py --base-seed 900 --opp-seed 0
        modal run forge/modal_app.py --base-seed 900 --large  # Use A100
    """
    if large:
        result = generate_shard_large.remote(base_seed, opp_seed, decl_id)
    else:
        result = generate_shard.remote(base_seed, opp_seed, decl_id)

    print(f"Result: {result}")


@app.function(image=forge_image, timeout=86400)  # 24h timeout for orchestration
def generate_range(
    start_seed: int = 900,
    end_seed: int = 1000,
    n_opp_seeds: int = 10,
):
    """Generate marginalized shards for a range of seeds.

    Spawns parallel GPU tasks for each (base_seed, opp_seed) combination.

    Args:
        start_seed: First base seed (inclusive)
        end_seed: Last base seed (exclusive)
        n_opp_seeds: Number of opponent seeds per base seed
    """
    from rich.console import Console
    console = Console()

    # Build list of all (base_seed, opp_seed) pairs
    tasks = []
    for base_seed in range(start_seed, end_seed):
        decl_id = base_seed % 10
        for opp_seed in range(n_opp_seeds):
            tasks.append((base_seed, opp_seed, decl_id))

    console.print(f"[bold]Generating {len(tasks)} shards[/bold]")
    console.print(f"  Seeds: {start_seed} to {end_seed-1}")
    console.print(f"  Opp seeds: 0 to {n_opp_seeds-1}")

    # Launch all tasks in parallel using starmap
    results = list(generate_shard.starmap(tasks))

    # Summarize results
    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = len(results) - success - skipped

    console.print(f"\n[bold green]Complete![/bold green]")
    console.print(f"  Success: {success}")
    console.print(f"  Skipped (exists/too large): {skipped}")
    if failed:
        console.print(f"  [red]Failed: {failed}[/red]")

    return {"success": success, "skipped": skipped, "failed": failed}


@app.local_entrypoint(name="generate_valtest")
def generate_valtest():
    """Generate val (900-949) and test (950-999) shards with 10 opp_seeds each.

    This is the primary entrypoint for generating the validation and test splits.

    Usage:
        modal run forge/modal_app.py::generate_valtest
    """
    from rich.console import Console
    console = Console()

    console.print("[bold]Generating val/test marginalized shards[/bold]")
    console.print("  Val: seeds 900-949 (50 seeds × 10 opp_seeds = 500 shards)")
    console.print("  Test: seeds 950-999 (50 seeds × 10 opp_seeds = 500 shards)")
    console.print("  Total: 1000 shards\n")

    result = generate_range.remote(start_seed=900, end_seed=1000, n_opp_seeds=10)
    console.print(f"\nFinal result: {result}")


@app.function(image=forge_image, volumes={SHARD_MOUNT_PATH: shard_volume}, timeout=3600)
def list_shards(split: str = "all") -> list[str]:
    """List all shards in the Modal volume.

    Args:
        split: "train", "val", "test", or "all"
    """
    from pathlib import Path

    shards = []
    base = Path(SHARD_MOUNT_PATH)

    splits = ["train", "val", "test"] if split == "all" else [split]
    for s in splits:
        split_dir = base / s
        if split_dir.exists():
            for f in sorted(split_dir.glob("*.parquet")):
                shards.append(f"{s}/{f.name}")

    return shards


@app.local_entrypoint(name="count_shards")
def count_shards():
    """Count shards on the Modal volume by split."""
    from rich.console import Console
    console = Console()

    shards = list_shards.remote("all")

    train = sum(1 for s in shards if s.startswith("train/"))
    val = sum(1 for s in shards if s.startswith("val/"))
    test = sum(1 for s in shards if s.startswith("test/"))

    console.print(f"[bold]Shard Counts on Modal Volume[/bold]")
    console.print(f"  Train: {train}")
    console.print(f"  Val:   {val}")
    console.print(f"  Test:  {test}")
    console.print(f"  Total: {len(shards)}")


@app.local_entrypoint(name="find_missing")
def find_missing():
    """Find missing val/test shards."""
    from rich.console import Console
    console = Console()

    shards = list_shards.remote("all")

    # Parse existing (seed, opp) pairs
    existing = set()
    for s in shards:
        if "val/" in s or "test/" in s:
            parts = s.split("/")[1]  # e.g., seed_00000900_opp0_decl_0.parquet
            seed = int(parts.split("_")[1])
            opp = int(parts.split("opp")[1].split("_")[0])
            existing.add((seed, opp))

    # Find missing
    missing = []
    for seed in range(900, 1000):
        for opp in range(10):
            if (seed, opp) not in existing:
                missing.append((seed, opp, seed % 10))

    console.print(f"[bold]Missing val/test shards: {len(missing)}[/bold]")
    for seed, opp, decl in missing:
        split = "val" if seed < 950 else "test"
        console.print(f"  {split}: seed={seed} opp={opp} decl={decl}")


@app.function(image=forge_image, volumes={SHARD_MOUNT_PATH: shard_volume}, timeout=3600)
def read_shard(path: str) -> bytes:
    """Read a shard file and return its contents.

    Args:
        path: Relative path like "val/seed_00000900_opp0_decl_0.parquet"
    """
    from pathlib import Path
    full_path = Path(SHARD_MOUNT_PATH) / path
    return full_path.read_bytes()


@app.local_entrypoint(name="download")
def download_shards(
    split: str = "all",
    output_dir: str = "data/shards-marginalized",
):
    """Download shards from Modal volume to local filesystem.

    Usage:
        modal run forge/modal_app.py::download --split val
        modal run forge/modal_app.py::download --split all --output-dir /mnt/d/shards
    """
    from pathlib import Path
    from rich.console import Console
    from rich.progress import Progress

    console = Console()
    out = Path(output_dir)

    # List shards
    shards = list_shards.remote(split)
    console.print(f"Found {len(shards)} shards to download")

    if not shards:
        console.print("[yellow]No shards found[/yellow]")
        return

    # Download each shard
    downloaded = 0
    skipped = 0

    with Progress() as progress:
        task = progress.add_task("Downloading...", total=len(shards))

        for shard_path in shards:
            local_path = out / shard_path
            if local_path.exists():
                skipped += 1
                progress.advance(task)
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            data = read_shard.remote(shard_path)
            local_path.write_bytes(data)
            downloaded += 1
            progress.advance(task)

    console.print(f"[green]Downloaded: {downloaded}[/green]")
    if skipped:
        console.print(f"[yellow]Skipped (exists): {skipped}[/yellow]")


# =============================================================================
# E[Q] Dataset Generation (t42-26dl)
# =============================================================================
#
# Generate Stage 2 E[Q] training data on Modal GPUs.
# Uses H200 for best memory bandwidth per dollar (inference is bandwidth-bound).
#
# Optimizations:
# - FP16 inference (halves bandwidth requirements)
# - torch.compile for kernel fusion
# - torch.inference_mode for reduced overhead
# - Warmup to avoid recompilation

EQ_VOLUME_NAME = "texas-42-eq-datasets"
EQ_MOUNT_PATH = "/eq_data"

eq_volume = modal.Volume.from_name(EQ_VOLUME_NAME, create_if_missing=True)


@app.cls(
    image=forge_image,
    gpu=GPU_H200,
    volumes={EQ_MOUNT_PATH: eq_volume},
    timeout=3600,  # 1 hour max
)
class EQGenerator:
    """E[Q] dataset generator with optimized inference."""

    @modal.enter()
    def setup(self):
        """Load and optimize model once per container."""
        import sys
        sys.path.insert(0, "/root")

        import time
        import torch
        import numpy as np

        from forge.eq import Stage1Oracle

        print(f"[{torch.cuda.get_device_name(0)}] Loading oracle...")
        load_start = time.time()

        # Load oracle
        checkpoint_path = "/root/forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt"
        self.oracle = Stage1Oracle(checkpoint_path, device="cuda")

        # Convert to FP16 for faster inference (bandwidth-bound workload)
        self.oracle.model.half()

        # Compile for kernel fusion
        self.oracle.model = torch.compile(self.oracle.model, mode="default")

        # Warmup compiled model for expected batch sizes to avoid runtime recompilation/module loading.
        # - current decision query: n_samples (typically 100)
        # - posterior scoring: window_k * n_samples (typically 8 * 100 = 800)
        print("Warming up compiled model...")
        with torch.inference_mode():
            for batch_size in (100, 200, 300, 400, 500, 600, 700, 800):
                dummy_tokens = torch.zeros(batch_size, 32, 12, dtype=torch.int32, device="cuda")
                dummy_mask = torch.ones(batch_size, 32, dtype=torch.int8, device="cuda")
                dummy_player = torch.zeros(batch_size, dtype=torch.long, device="cuda")
                self.oracle.model(dummy_tokens, dummy_mask, dummy_player)

        print(f"Oracle ready in {time.time() - load_start:.2f}s")

        # Store for use in methods
        self.torch = torch
        self.np = np

    @modal.method()
    def generate_games(
        self,
        n_games: int,
        n_samples: int = 100,
        seed: int = 42,
        posterior: bool = True,
        explore: bool = True,
    ) -> bytes:
        """Generate E[Q] dataset for n_games.

        Args:
            n_games: Number of games to generate
            n_samples: Samples per decision for marginalization
            seed: Random seed
            posterior: Enable posterior-weighted marginalization
            explore: Enable exploration policy

        Returns:
            Serialized dataset as bytes (torch.save format)
        """
        import sys
        sys.path.insert(0, "/root")

        import io
        import time

        from forge.eq.generate import ExplorationPolicy, PosteriorConfig
        from forge.eq.generate_dataset import generate_dataset

        torch = self.torch

        print(f"Generating {n_games} games (seed={seed}, posterior={posterior}, explore={explore})")
        start_time = time.time()

        # Build configs
        posterior_config = None
        if posterior:
            posterior_config = PosteriorConfig(
                enabled=True,
                tau=10.0,
                beta=0.10,
                window_k=8,
                delta=30.0,
            )

        exploration_policy = None
        if explore:
            exploration_policy = ExplorationPolicy.mixed_exploration(
                temperature=3.0,
                epsilon=0.05,
                blunder_rate=0.02,
                blunder_max_regret=3.0,
                seed=seed,
            )

        game_batch_size = min(32, n_games)
        progress_interval = max(1, n_games // 10)

        dataset = generate_dataset(
            oracle=self.oracle,
            n_games=n_games,
            n_samples=n_samples,
            seed=seed,
            val_fraction=0.1,
            progress_interval=progress_interval,
            posterior_config=posterior_config,
            exploration_policy=exploration_policy,
            checkpoint_path="/root/forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt",
            game_batch_size=game_batch_size,
        )

        dataset["metadata"]["gpu"] = torch.cuda.get_device_name(0)
        dataset["metadata"]["games_per_second"] = n_games / dataset["metadata"]["total_time_seconds"]

        total_time = time.time() - start_time
        print(f"Generated {dataset['metadata']['n_examples']} examples in {total_time:.1f}s")

        # Serialize to bytes
        buffer = io.BytesIO()
        torch.save(dataset, buffer)
        return buffer.getvalue()


@app.local_entrypoint(name="eq_generate")
def eq_generate(
    n_games: int = 1000,
    n_samples: int = 100,
    seed: int = 42,
    output: str = "forge/data/eq_dataset_modal.pt",
    no_posterior: bool = False,
    no_explore: bool = False,
):
    """Generate E[Q] dataset on Modal GPU.

    Usage:
        modal run forge/modal_app.py::eq_generate --n-games 10
        modal run forge/modal_app.py::eq_generate --n-games 1000 --output forge/data/eq_1k.pt
    """
    from pathlib import Path
    from rich.console import Console

    console = Console()

    console.print("[bold]E[Q] Dataset Generation on Modal[/bold]")
    console.print(f"  Games: {n_games}")
    console.print(f"  Samples/decision: {n_samples}")
    console.print(f"  Seed: {seed}")
    console.print(f"  Posterior: {not no_posterior}")
    console.print(f"  Exploration: {not no_explore}")
    console.print(f"  Output: {output}")
    console.print()

    # Run on GPU
    generator = EQGenerator()
    data_bytes = generator.generate_games.remote(
        n_games=n_games,
        n_samples=n_samples,
        seed=seed,
        posterior=not no_posterior,
        explore=not no_explore,
    )

    # Save locally
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data_bytes)

    file_size_mb = len(data_bytes) / 1024 / 1024
    console.print(f"\n[green]Saved: {output} ({file_size_mb:.1f} MB)[/green]")
