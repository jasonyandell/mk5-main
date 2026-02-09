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

# Eval image: adds huggingface_hub for downloading Zeb models from HF.
# pip_install must come before add_local_dir, so we rebuild from scratch.
eval_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "lightning>=2.0",
        "pyarrow>=14.0",
        "numpy>=1.26,<2",
        "pandas>=2.0",
        "rich",
        "huggingface_hub",
    )
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

# =============================================================================
# Training Volume Configuration
# =============================================================================
#
# Persistent volume for tokenized training data and checkpoints.
# Contains: /tokenized/{train,val,test}/*.npy, /checkpoints/

TRAINING_VOLUME_NAME = "stage1-training-data"
TRAINING_MOUNT_PATH = "/data"

training_volume = modal.Volume.from_name(TRAINING_VOLUME_NAME, create_if_missing=True)


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


# =============================================================================
# Training Functions
# =============================================================================
#
# Train DominoTransformer models on Modal GPUs using the tokenized dataset.
# Uses T4 ($0.59/hr) for cost-efficient training.
#
# Usage:
#     modal run forge/modal_app.py::train --epochs 10
#     modal run forge/modal_app.py::train --epochs 10 --no-shuffle-hands  # Disable slot shuffle
#     modal run forge/modal_app.py::train_sweep  # Parallel hyperparameter search


@app.function(
    image=forge_image,
    gpu=GPU_A100_40GB,  # A100-40 test
    volumes={TRAINING_MOUNT_PATH: training_volume},
    timeout=14400,  # 4 hours max
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_model(
    epochs: int = 10,
    batch_size: int = 32768,  # Large batch for B200 (192GB VRAM)
    lr: float = 3e-4,
    embed_dim: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    ff_dim: int = 128,
    value_weight: float = 0.5,
    loss_mode: str = "policy",
    shuffle_hands: bool = True,
    seed: int = 42,
    wandb_group: str | None = None,
    run_name: str | None = None,
    compile: bool = True,
    limit_train_batches: int | None = None,
) -> dict:
    """Train a DominoTransformer model on Modal GPU.

    Args:
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        embed_dim: Transformer embedding dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        ff_dim: Feed-forward dimension
        value_weight: Weight for value head loss
        loss_mode: 'policy' (cross-entropy) or 'qvalue' (MSE)
        shuffle_hands: Shuffle hand slots during training (fixes slot 0 bias)
        seed: Random seed
        wandb_group: Optional WandB group name
        run_name: Optional run name (auto-generated if not provided)
        compile: Whether to use torch.compile (disable for quick tests)
        limit_train_batches: Limit training to N batches per epoch (for quick tests)

    Returns:
        Dict with training results and best checkpoint path
    """
    import sys
    sys.path.insert(0, "/root")

    import os
    import time
    from datetime import datetime

    import torch
    import lightning as L
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )
    from lightning.pytorch.loggers import CSVLogger, WandbLogger

    from forge.ml.data import DominoDataModule
    from forge.ml.module import DominoLightningModule

    print(f"[{torch.cuda.get_device_name(0)}] Starting training")
    print(f"  Epochs: {epochs}, Batch size: {batch_size}, LR: {lr}")
    print(f"  Architecture: {n_layers}L-{n_heads}H-d{embed_dim}")
    print(f"  Shuffle hands: {shuffle_hands}, Compile: {compile}")
    if limit_train_batches:
        print(f"  Limit batches: {limit_train_batches} (quick perf test)")
    start_time = time.time()

    # Reproducibility
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # Model
    model = DominoLightningModule(
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        lr=lr,
        value_weight=value_weight,
        loss_mode=loss_mode,
    )

    # torch.compile for optimization (skip for quick perf tests)
    if compile:
        model.model = torch.compile(model.model)

    # Data - uses volume mount
    data = DominoDataModule(
        data_path=f"{TRAINING_MOUNT_PATH}/tokenized",
        batch_size=batch_size,
        num_workers=16,
        prefetch_factor=8,
        shuffle_hands=shuffle_hands,
    )

    # Compute model size
    total_params = sum(p.numel() for p in model.parameters())
    if total_params >= 1_000_000:
        model_size = f"{total_params / 1_000_000:.1f}M"
    else:
        model_size = f"{total_params // 1000}k"

    # Run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        prefix = "qval" if loss_mode == "qvalue" else "train"
        shuffle_tag = "-shuffle" if shuffle_hands else ""
        run_name = f"{prefix}-{timestamp}-{model_size}-{n_layers}L-{n_heads}H{shuffle_tag}"

    # Loggers
    checkpoint_dir = f"{TRAINING_MOUNT_PATH}/checkpoints/{run_name}"
    loggers = [CSVLogger(checkpoint_dir, name="logs")]

    # WandB if secret is available
    if os.environ.get("WANDB_API_KEY"):
        tags = [loss_mode, model_size]
        if shuffle_hands:
            tags.append("shuffle-hands")
        if wandb_group:
            tags.append(wandb_group.split("/")[0])

        loggers.append(WandbLogger(
            project="crystal-forge",
            name=run_name,
            group=wandb_group,
            tags=tags,
            save_dir=checkpoint_dir,
            log_model=False,
            config={
                "total_params": total_params,
                "model_size": model_size,
                "embed_dim": embed_dim,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "ff_dim": ff_dim,
                "batch_size": batch_size,
                "lr": lr,
                "value_weight": value_weight,
                "loss_mode": loss_mode,
                "shuffle_hands": shuffle_hands,
                "seed": seed,
            },
        ))
        print("  WandB: enabled")
    else:
        print("  WandB: disabled (no WANDB_API_KEY)")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val/q_gap",
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="{epoch}-{val_q_gap:.3f}",
        ),
        EarlyStopping(
            monitor="val/q_gap",
            mode="min",
            patience=5,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        logger=loggers,
        callbacks=callbacks,
        default_root_dir=checkpoint_dir,
        log_every_n_steps=25,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        limit_train_batches=limit_train_batches,
    )

    trainer.fit(model, data)

    # Commit checkpoints to volume
    training_volume.commit()

    total_time = time.time() - start_time
    best_path = trainer.checkpoint_callback.best_model_path
    best_q_gap = trainer.checkpoint_callback.best_model_score

    print(f"\nTraining complete in {total_time / 60:.1f} minutes")
    print(f"Best checkpoint: {best_path}")
    print(f"Best val/q_gap: {best_q_gap:.4f}")

    return {
        "run_name": run_name,
        "epochs_trained": trainer.current_epoch + 1,
        "best_q_gap": float(best_q_gap) if best_q_gap else None,
        "best_checkpoint": best_path,
        "total_time_seconds": total_time,
        "model_size": model_size,
        "shuffle_hands": shuffle_hands,
    }


# =============================================================================
# B200 Adaptive E[Q] Generator (t42-nk73)
# =============================================================================
#
# Run adaptive sampling on B200 (192GB VRAM) for massive parallelism.
# Uses the same adaptive config as local benchmarks but with 100-1000 games.


@app.cls(
    image=forge_image,
    gpu=GPU_B200,  # B200 - 192GB, let's see what it can do!
    volumes={EQ_MOUNT_PATH: eq_volume},
    timeout=3600,  # 1 hour max
)
class EQGeneratorB200:
    """Adaptive E[Q] generator optimized for B200's 192GB VRAM."""

    @modal.enter()
    def setup(self):
        """Load and optimize model once per container."""
        import sys
        sys.path.insert(0, "/root")

        import time
        import torch

        from forge.eq import Stage1Oracle

        print(f"[{torch.cuda.get_device_name(0)}] Loading oracle for B200...")
        load_start = time.time()

        # Load oracle - Stage1Oracle already does torch.compile with reduce-overhead
        checkpoint_path = "/root/forge/models/domino-qval-3.3M-shuffle-qgap0.074-qmae0.96.ckpt"
        self.oracle = Stage1Oracle(checkpoint_path, device="cuda", compile=True)

        # Store reference to the actual model for generate_eq_games_gpu
        self.model = self.oracle.model

        # Warmup for expected batch sizes (Stage1Oracle already does basic warmup)
        print("Additional warmup for large batches...")
        with torch.inference_mode():
            for batch_size in (5000, 10000, 50000):
                dummy_tokens = torch.zeros(batch_size, 32, 12, dtype=torch.int32, device="cuda")
                dummy_mask = torch.ones(batch_size, 32, dtype=torch.int8, device="cuda")
                dummy_player = torch.zeros(batch_size, dtype=torch.long, device="cuda")
                try:
                    self.model(dummy_tokens, dummy_mask, dummy_player)
                    print(f"  Warmup batch={batch_size}: OK")
                except Exception as e:
                    print(f"  Warmup batch={batch_size}: {e}")
                    break

        print(f"Oracle ready in {time.time() - load_start:.2f}s")
        self.torch = torch

    @modal.method()
    def generate_adaptive(
        self,
        n_games: int = 100,
        seed_start: int = 50000,
        min_samples: int = 50000,
        max_samples: int = 2000000,
        batch_size: int = 1000,
        sem_threshold: float = 0.1,
    ) -> dict:
        """Generate E[Q] data with adaptive sampling.

        Args:
            n_games: Number of games to generate
            seed_start: Starting seed for deals
            min_samples: Minimum samples before checking convergence
            max_samples: Maximum samples (hard cap)
            batch_size: Samples per iteration
            sem_threshold: Convergence threshold

        Returns:
            Dict with timing stats and summary
        """
        import sys
        sys.path.insert(0, "/root")

        import time
        from forge.eq.generate import generate_eq_games_gpu, AdaptiveConfig
        from forge.oracle.rng import deal_from_seed

        torch = self.torch

        print(f"Generating {n_games} games with adaptive sampling")
        print(f"  Seeds: {seed_start} to {seed_start + n_games - 1}")
        print(f"  Config: min={min_samples}, max={max_samples}, batch={batch_size}, sem={sem_threshold}")

        # Build hands from seeds
        hands = [deal_from_seed(seed_start + i) for i in range(n_games)]
        decl_ids = [i % 10 for i in range(n_games)]

        config = AdaptiveConfig(
            enabled=True,
            min_samples=min_samples,
            max_samples=max_samples,
            batch_size=batch_size,
            sem_threshold=sem_threshold,
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        results = generate_eq_games_gpu(
            model=self.model,
            hands=hands,
            decl_ids=decl_ids,
            n_samples=1000,  # ignored when adaptive
            device='cuda',
            adaptive_config=config,
        )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        # Collect stats
        samples_per_decision = []
        n_converged = 0
        total_decisions = 0

        for game in results:
            for dec in game.decisions:
                if dec.n_samples is not None:
                    samples_per_decision.append(dec.n_samples)
                if dec.converged:
                    n_converged += 1
                total_decisions += 1

        avg_samples = sum(samples_per_decision) / len(samples_per_decision) if samples_per_decision else 0

        stats = {
            'n_games': n_games,
            'total_time': elapsed,
            'sec_per_game': elapsed / n_games,
            'games_per_sec': n_games / elapsed,
            'total_decisions': total_decisions,
            'avg_samples_per_decision': avg_samples,
            'min_samples_per_decision': min(samples_per_decision) if samples_per_decision else 0,
            'max_samples_per_decision': max(samples_per_decision) if samples_per_decision else 0,
            'convergence_rate': n_converged / total_decisions if total_decisions > 0 else 0,
            'gpu': torch.cuda.get_device_name(0),
        }

        print(f"\n=== Results ===")
        print(f"Total time: {elapsed:.1f}s")
        print(f"Sec/game: {stats['sec_per_game']:.2f}")
        print(f"Games/sec: {stats['games_per_sec']:.2f}")
        print(f"Avg samples/decision: {avg_samples:.0f}")
        print(f"Convergence rate: {stats['convergence_rate']*100:.1f}%")

        return stats


@app.local_entrypoint(name="eq_adaptive_b200")
def eq_adaptive_b200(
    n_games: int = 500,  # Default to 500 games to saturate H200
    seed_start: int = 50000,
    batch_size: int = 2000,  # Larger batch for H200's 150GB
    sem_threshold: float = 0.1,  # Convergence threshold
    min_samples: int = 50000,  # Minimum samples before checking convergence
):
    """Run adaptive E[Q] generation on H200.

    Usage:
        modal run forge/modal_app.py::eq_adaptive_b200 --n-games 500
        modal run forge/modal_app.py::eq_adaptive_b200 --n-games 1000  # Crystal palace mode!
        modal run forge/modal_app.py::eq_adaptive_b200 --sem-threshold 0.5 --min-samples 10000  # Fast!
    """
    from rich.console import Console
    console = Console()

    console.print("[bold]🔥 H200 Adaptive E[Q] Generation 🔥[/bold]")
    console.print(f"  Games: {n_games}")
    console.print(f"  Seeds: {seed_start} to {seed_start + n_games - 1}")
    console.print(f"  Min samples: {min_samples}, SEM threshold: {sem_threshold}")
    console.print()

    generator = EQGeneratorB200()
    stats = generator.generate_adaptive.remote(
        n_games=n_games,
        seed_start=seed_start,
        min_samples=min_samples,
        max_samples=2000000,
        batch_size=batch_size,
        sem_threshold=sem_threshold,
    )

    console.print(f"\n[bold green]🏰 Crystal Palace Complete! 🏰[/bold green]")
    console.print(f"  GPU: {stats['gpu']}")
    console.print(f"  Total time: {stats['total_time']:.1f}s")
    console.print(f"  Games/sec: {stats['games_per_sec']:.2f}")
    console.print(f"  Sec/game: {stats['sec_per_game']:.2f}")
    console.print(f"  Avg samples/decision: {stats['avg_samples_per_decision']:.0f}")
    console.print(f"  Convergence: {stats['convergence_rate']*100:.1f}%")

    # Compare to laptop
    laptop_sec_per_game = 102.8
    speedup = laptop_sec_per_game / stats['sec_per_game']
    console.print(f"\n  [bold]Speedup vs RTX 3050 Ti: {speedup:.1f}x[/bold]")


# =============================================================================
# Zeb Self-Play Training (AlphaZero-style MCTS)
# =============================================================================
#
# True self-play training: model's value head evaluates MCTS leaves.
# Uses B200 for fast parallel MCTS game generation.
#
# Key features:
# - GPU-native MCTS with model evaluation (not oracle)
# - GPU replay buffer for stable training
# - Checkpoint resume support
# - W&B logging

ZEB_VOLUME_NAME = "zeb-training-runs"
ZEB_MOUNT_PATH = "/zeb_runs"

zeb_volume = modal.Volume.from_name(ZEB_VOLUME_NAME, create_if_missing=True)


@app.function(
    image=forge_image,
    gpu=GPU_B200,
    volumes={ZEB_MOUNT_PATH: zeb_volume},
    timeout=14400,  # 4 hours max
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_zeb(
    checkpoint_bytes: bytes,
    epochs: int = 10,
    games_per_epoch: int = 256,
    n_simulations: int = 200,
    n_parallel_games: int = 256,
    max_mcts_nodes: int = 256,
    batch_size: int = 64,
    lr: float = 1e-4,
    temperature: float = 1.0,
    replay_buffer_size: int = 50000,
    training_steps: int = 1000,
    eval_every: int = 1,
    save_every: int = 10,
    seed: int = 42,
    run_name: str | None = None,
    wandb_enabled: bool = True,
    profile_mode: bool = False,
    freeze_data: bool = False,
    model_size: str | None = None,
) -> dict:
    """Train Zeb via true self-play MCTS on Modal B200.

    AlphaZero-style training where the model's value head evaluates MCTS leaves.
    This enables bootstrap improvement through self-play.

    Args:
        checkpoint_bytes: Serialized checkpoint to resume from
        epochs: Number of training epochs
        games_per_epoch: Games to generate per epoch
        n_simulations: MCTS simulations per move
        n_parallel_games: Concurrent MCTS games
        max_mcts_nodes: Max nodes per MCTS tree
        batch_size: Training batch size
        lr: Learning rate
        temperature: Action sampling temperature
        replay_buffer_size: GPU replay buffer capacity
        training_steps: Training steps per epoch (0 = one pass over examples)
        eval_every: Evaluate vs random every N epochs
        save_every: Save checkpoint every N epochs
        seed: Random seed
        run_name: Run name for W&B and checkpoints
        wandb_enabled: Whether to log to W&B
        profile_mode: Add deliberate gaps between phases for GPU profiling
        freeze_data: Skip game generation, train on frozen replay buffer
        model_size: Override model with fresh random weights ('small', 'medium', 'large')

    Returns:
        Dict with training results and final checkpoint bytes
    """
    import sys
    sys.path.insert(0, "/root")

    import io
    import os
    import random
    import time
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import torch

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    from forge.zeb.gpu_training_pipeline import create_selfplay_pipeline, GPUReplayBuffer
    from forge.zeb.model import ZebModel, get_model_config
    from forge.zeb.evaluate import evaluate_vs_random

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)

    mode_str = "MEMORIZATION (frozen data)" if freeze_data else "self-play training"
    print(f"[{gpu_name}] Starting Zeb {mode_str}")

    # Load checkpoint
    print("Loading checkpoint...")
    buffer = io.BytesIO(checkpoint_bytes)
    ckpt = torch.load(buffer, map_location='cpu', weights_only=False)

    # Override model with fresh random weights if model_size specified
    if model_size:
        print(f"  Creating fresh {model_size} model (random weights)...")
        model_config = get_model_config(model_size)
        model = ZebModel(**model_config)
        model.to(device)
        start_epoch = 0  # Fresh model starts at epoch 0
        wandb_run_id = None  # Force new W&B run
    else:
        model_config = ckpt['model_config']
        model = ZebModel(**model_config)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        start_epoch = ckpt.get('epoch', 0) + 1
        wandb_run_id = ckpt.get('wandb_run_id')

    total_games_prior = ckpt.get('total_games', 0)

    total_params = sum(p.numel() for p in model.parameters())
    model_size_str = f"{total_params / 1_000_000:.1f}M" if total_params >= 1_000_000 else f"{total_params // 1000}k"

    print(f"  Model: {model_size_str} ({total_params:,} params)")
    if model_size:
        print(f"  Fresh {model_size} model (starting at epoch 0)")
    else:
        print(f"  From epoch: {start_epoch - 1}, prior games: {total_games_prior:,}")

    # Initial evaluation
    print("\nInitial evaluation...")
    initial_win_rate = evaluate_vs_random(model, n_games=100, device=device)['team0_win_rate']
    print(f"  vs Random: {initial_win_rate:.1%}")

    # W&B setup
    use_wandb = wandb_enabled and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        import wandb
        # Profile mode: always fresh run, never resume
        if profile_mode or not wandb_run_id:
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            if profile_mode:
                actual_run_name = f"b200-profiling-{timestamp}"
                tags = ["profiling", "b200", "phase-timing"]
            else:
                actual_run_name = run_name or f"selfplay-b200-{timestamp}"
                tags = ["selfplay", "alphazero", "modal", "b200"]
            wandb.init(
                project="zeb-mcts",
                name=actual_run_name,
                tags=tags,
                config={
                    'mode': 'profiling' if profile_mode else 'selfplay',
                    'profile_mode': profile_mode,
                    'initial_win_rate': initial_win_rate,
                    'start_epoch': start_epoch,
                    'epochs': epochs,
                    'games_per_epoch': games_per_epoch,
                    'n_simulations': n_simulations,
                    'n_parallel_games': n_parallel_games,
                    'max_mcts_nodes': max_mcts_nodes,
                    'batch_size': batch_size,
                    'lr': lr,
                    'temperature': temperature,
                    'replay_buffer_size': replay_buffer_size,
                    'training_steps': training_steps,
                    'model_config': model_config,
                    'total_params': total_params,
                    'gpu': gpu_name,
                },
            )
            print(f"W&B run: {wandb.run.url}")
        else:
            wandb.init(project="zeb-mcts", id=wandb_run_id, resume="must")
            print(f"W&B run resumed: {wandb.run.url}")

    if freeze_data:
        print(f"\n=== MEMORIZATION EXPERIMENT (frozen data) ===")
    else:
        print(f"\n=== Self-Play Training ===")
    print(f"Epochs: {start_epoch} to {start_epoch + epochs - 1}")
    if not freeze_data:
        print(f"Games/epoch: {games_per_epoch}, MCTS sims: {n_simulations}")
        print(f"Parallel games: {n_parallel_games}, Max nodes: {max_mcts_nodes}")
    print(f"Replay buffer: {replay_buffer_size:,}, Training steps: {training_steps}")
    print()

    # Create self-play pipeline
    print("Creating self-play pipeline...")
    t0 = time.time()
    pipeline = create_selfplay_pipeline(
        model=model,
        device=device,
        n_parallel_games=n_parallel_games,
        n_simulations=n_simulations,
        max_mcts_nodes=max_mcts_nodes,
        temperature=temperature,
    )
    print(f"Pipeline created in {time.time() - t0:.1f}s")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Load replay buffer from checkpoint if present
    saved_buffer = ckpt.get('replay_buffer', [])
    replay_buffer = GPUReplayBuffer.from_list(
        saved_buffer,
        capacity=replay_buffer_size,
        device=torch.device(device),
    )
    print(f"Replay buffer: {len(replay_buffer):,}/{replay_buffer_size:,} examples")

    # Output dir for checkpoints
    output_dir = Path(ZEB_MOUNT_PATH) / (run_name or "selfplay-run")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    start_time = time.time()
    total_games = total_games_prior
    best_win_rate = initial_win_rate
    final_epoch = start_epoch

    epoch_times = []

    # Profile mode: helper to wait until next time boundary
    def wait_until_boundary(boundary_seconds: float, label: str):
        """Wait until elapsed time hits the next boundary (e.g., 60s, 90s)."""
        elapsed = time.time() - start_time
        next_boundary = ((elapsed // boundary_seconds) + 1) * boundary_seconds
        wait_time = next_boundary - elapsed
        if wait_time > 0:
            print(f"  [PROFILE] {label}: waiting {wait_time:.1f}s until t={next_boundary:.0f}s")
            if use_wandb:
                wandb.log({'profile/waiting': 1, 'profile/wait_until': next_boundary})
            time.sleep(wait_time)
            if use_wandb:
                wandb.log({'profile/waiting': 0})

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.time()
        final_epoch = epoch

        # === GENERATION PHASE ===
        if freeze_data:
            # Skip generation for memorization experiment
            gen_time = 0.0
            n_examples = len(replay_buffer)
            games_per_sec = 0.0
        else:
            if profile_mode and use_wandb:
                elapsed = time.time() - start_time
                wandb.log({'profile/phase': 1, 'profile/phase_name': 'gen_start', 'profile/elapsed_s': elapsed})
                print(f"  [PROFILE] GEN START at t={elapsed:.1f}s")

            model.eval()
            t0 = time.time()
            examples = pipeline.generate_games_gpu(n_games=games_per_epoch)
            gen_time = time.time() - t0

            if profile_mode and use_wandb:
                elapsed = time.time() - start_time
                wandb.log({'profile/phase': 2, 'profile/phase_name': 'gen_end', 'profile/elapsed_s': elapsed})
                print(f"  [PROFILE] GEN END at t={elapsed:.1f}s (took {gen_time:.1f}s)")

            total_games += games_per_epoch
            n_examples = examples.n_examples
            games_per_sec = games_per_epoch / gen_time

        # Add to GPU replay buffer (skip if frozen)
        if not freeze_data:
            replay_buffer.add_batch(examples)

        # Profile mode: wait until 60s boundary before training
        if profile_mode:
            wait_until_boundary(60.0, "pre-train")

        # === TRAINING PHASE ===
        if profile_mode and use_wandb:
            elapsed = time.time() - start_time
            wandb.log({'profile/phase': 3, 'profile/phase_name': 'train_start', 'profile/elapsed_s': elapsed})
            print(f"  [PROFILE] TRAIN START at t={elapsed:.1f}s")

        model.train()
        t1 = time.time()

        if training_steps > 0 and len(replay_buffer) >= batch_size:
            # AlphaZero-style: fixed number of training steps from buffer
            metrics = pipeline.train_n_steps_from_buffer(
                model=model,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                n_steps=training_steps,
                batch_size=batch_size,
            )
        else:
            # Single pass over current examples
            metrics = pipeline.train_epoch_gpu(
                model=model,
                optimizer=optimizer,
                examples=examples,
                batch_size=batch_size,
            )
        train_time = time.time() - t1

        if profile_mode and use_wandb:
            elapsed = time.time() - start_time
            wandb.log({'profile/phase': 4, 'profile/phase_name': 'train_end', 'profile/elapsed_s': elapsed})
            print(f"  [PROFILE] TRAIN END at t={elapsed:.1f}s (took {train_time:.1f}s)")

        # Update pipeline's model reference
        pipeline.set_model(model)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        # Log
        epoch_str = f"Epoch {epoch:4d}: policy={metrics['policy_loss']:.4f}, value={metrics['value_loss']:.4f}"
        if freeze_data:
            epoch_str += f" (train={train_time:.1f}s) [FROZEN]"
        else:
            epoch_str += f" (gen={gen_time:.1f}s, train={train_time:.1f}s, {games_per_sec:.1f} g/s)"
            epoch_str += f" [buffer: {len(replay_buffer):,}]"
        print(epoch_str)

        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/policy_loss': metrics['policy_loss'],
                'train/value_loss': metrics['value_loss'],
                'train/total_loss': metrics['policy_loss'] + metrics['value_loss'],
                'perf/gen_time_s': gen_time,
                'perf/train_time_s': train_time,
                'perf/epoch_time_s': epoch_time,
                'perf/games_per_sec': games_per_sec,
                'perf/examples': n_examples,
                'stats/total_games': total_games,
                'stats/replay_buffer_size': len(replay_buffer),
            })

        # === EVALUATION PHASE ===
        if (epoch + 1) % eval_every == 0:
            if profile_mode and use_wandb:
                elapsed = time.time() - start_time
                wandb.log({'profile/phase': 5, 'profile/phase_name': 'eval_start', 'profile/elapsed_s': elapsed})
                print(f"  [PROFILE] EVAL START at t={elapsed:.1f}s")

            model.eval()
            win_rate = evaluate_vs_random(model, n_games=100, device=device)['team0_win_rate']

            if profile_mode and use_wandb:
                elapsed = time.time() - start_time
                wandb.log({'profile/phase': 6, 'profile/phase_name': 'eval_end', 'profile/elapsed_s': elapsed})
                print(f"  [PROFILE] EVAL END at t={elapsed:.1f}s")

            print(f"         -> vs Random: {win_rate:.1%}")
            if use_wandb:
                wandb.log({'eval/vs_random_win_rate': win_rate, 'epoch': epoch})
            if win_rate > best_win_rate:
                best_win_rate = win_rate

        # Profile mode: wait until 90s boundary (halfway to next epoch's gen at 120s)
        if profile_mode:
            wait_until_boundary(60.0, "post-eval")

        # Save checkpoint to volume
        if (epoch + 1) % save_every == 0:
            ckpt_path = output_dir / f"selfplay-epoch{epoch:04d}.pt"
            ckpt_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_config': model_config,
                'training_config': {
                    'mode': 'selfplay',
                    'games_per_epoch': games_per_epoch,
                    'n_simulations': n_simulations,
                    'lr': lr,
                    'replay_buffer_size': replay_buffer_size,
                    'training_steps': training_steps,
                },
                'total_games': total_games,
                'wandb_run_id': wandb.run.id if use_wandb else None,
                'replay_buffer': replay_buffer.to_list(),
            }
            torch.save(ckpt_data, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path.name}")
            zeb_volume.commit()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    model.eval()
    final_win_rate = evaluate_vs_random(model, n_games=200, device=device)['team0_win_rate']
    print(f"vs Random: {final_win_rate:.1%}")
    print(f"Improvement: {initial_win_rate:.1%} -> {final_win_rate:.1%} ({final_win_rate - initial_win_rate:+.1%})")

    total_time = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0

    print(f"\n=== Performance Summary ===")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Avg epoch time: {avg_epoch_time:.1f}s")
    print(f"Total games: {total_games:,}")
    print(f"Model queries: {pipeline.total_model_queries:,}")

    if use_wandb:
        wandb.log({
            'final/vs_random_win_rate': final_win_rate,
            'final/improvement': final_win_rate - initial_win_rate,
            'final/total_time_s': total_time,
            'final/avg_epoch_time_s': avg_epoch_time,
        })
        wandb.finish()

    # Serialize final checkpoint to return
    final_ckpt = {
        'epoch': final_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': model_config,
        'training_config': {
            'mode': 'selfplay',
            'games_per_epoch': games_per_epoch,
            'n_simulations': n_simulations,
            'lr': lr,
            'replay_buffer_size': replay_buffer_size,
            'training_steps': training_steps,
        },
        'total_games': total_games,
        'replay_buffer': replay_buffer.to_list(),
    }
    out_buffer = io.BytesIO()
    torch.save(final_ckpt, out_buffer)
    final_checkpoint_bytes = out_buffer.getvalue()

    zeb_volume.commit()

    return {
        'run_name': run_name or "selfplay-run",
        'start_epoch': start_epoch,
        'final_epoch': final_epoch,
        'epochs_trained': epochs,
        'total_games': total_games,
        'initial_win_rate': initial_win_rate,
        'final_win_rate': final_win_rate,
        'best_win_rate': best_win_rate,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60,
        'avg_epoch_time_s': avg_epoch_time,
        'avg_games_per_sec': (epochs * games_per_epoch) / total_time if total_time > 0 else 0,
        'model_queries': pipeline.total_model_queries,
        'gpu': gpu_name,
        'final_checkpoint_bytes': final_checkpoint_bytes,
    }


@app.local_entrypoint(name="zeb_train")
def zeb_train(
    checkpoint: str,
    epochs: int = 10,
    games_per_epoch: int = 256,
    n_simulations: int = 200,
    n_parallel_games: int = 256,
    max_mcts_nodes: int = 256,
    batch_size: int = 64,
    lr: float = 1e-4,
    replay_buffer_size: int = 50000,
    training_steps: int = 1000,
    eval_every: int = 1,
    save_every: int = 10,
    run_name: str | None = None,
    no_wandb: bool = False,
    save_checkpoint: str | None = None,
    profile_mode: bool = False,
    freeze_data: bool = False,
    model_size: str | None = None,
):
    """Train Zeb via self-play MCTS on Modal B200.

    Usage:
        # Benchmark 10 epochs (bead t42-19vw)
        modal run forge/modal_app.py::zeb_train \\
            --checkpoint forge/zeb/checkpoints/selfplay-epoch1847.pt \\
            --epochs 10 \\
            --run-name "b200-benchmark"

        # Profile mode: adds deliberate gaps between phases for GPU profiling
        modal run forge/modal_app.py::zeb_train \\
            --checkpoint forge/zeb/checkpoints/selfplay-epoch1845.pt \\
            --epochs 3 \\
            --profile-mode

        # Production run
        modal run forge/modal_app.py::zeb_train \\
            --checkpoint forge/zeb/checkpoints/selfplay-epoch1847.pt \\
            --epochs 100 \\
            --games-per-epoch 512 \\
            --save-checkpoint forge/zeb/checkpoints/selfplay-b200.pt
    """
    from pathlib import Path
    from rich.console import Console
    console = Console()

    # Load checkpoint
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        console.print(f"[red]Checkpoint not found: {checkpoint}[/red]")
        return

    if profile_mode:
        console.print(f"[bold yellow]Zeb PROFILING MODE on Modal B200[/bold yellow]")
    elif freeze_data:
        console.print(f"[bold yellow]MEMORIZATION EXPERIMENT on Modal B200[/bold yellow]")
    else:
        console.print(f"[bold]Zeb Self-Play Training on Modal B200[/bold]")
    console.print(f"  Checkpoint: {checkpoint}")
    if model_size:
        console.print(f"  [yellow]Model: {model_size} (fresh random weights)[/yellow]")
    console.print(f"  Epochs: {epochs}")
    if not freeze_data:
        console.print(f"  Games/epoch: {games_per_epoch}")
        console.print(f"  MCTS sims: {n_simulations}")
        console.print(f"  Parallel games: {n_parallel_games}")
    console.print(f"  Replay buffer: {replay_buffer_size:,}")
    console.print(f"  Training steps/epoch: {training_steps}")
    console.print(f"  W&B: {'disabled' if no_wandb else 'enabled'}")
    if profile_mode:
        console.print(f"  [yellow]Profile mode: deliberate gaps between phases[/yellow]")
    if freeze_data:
        console.print(f"  [yellow]Frozen data: training on replay buffer only[/yellow]")
    console.print()

    # Read checkpoint bytes
    checkpoint_bytes = ckpt_path.read_bytes()
    ckpt_size_mb = len(checkpoint_bytes) / 1024 / 1024
    console.print(f"Uploading checkpoint ({ckpt_size_mb:.1f} MB)...")

    result = train_zeb.remote(
        checkpoint_bytes=checkpoint_bytes,
        epochs=epochs,
        games_per_epoch=games_per_epoch,
        n_simulations=n_simulations,
        n_parallel_games=n_parallel_games,
        max_mcts_nodes=max_mcts_nodes,
        batch_size=batch_size,
        lr=lr,
        replay_buffer_size=replay_buffer_size,
        training_steps=training_steps,
        eval_every=eval_every,
        save_every=save_every,
        run_name=run_name,
        wandb_enabled=not no_wandb,
        profile_mode=profile_mode,
        freeze_data=freeze_data,
        model_size=model_size,
    )

    console.print(f"\n[bold green]Training Complete![/bold green]")
    console.print(f"  GPU: {result['gpu']}")
    console.print(f"  Epochs: {result['start_epoch']} -> {result['final_epoch']}")
    console.print(f"  Total games: {result['total_games']:,}")
    console.print(f"  Time: {result['total_time_minutes']:.1f} min")
    console.print(f"  Avg epoch: {result['avg_epoch_time_s']:.1f}s")
    console.print(f"  Games/sec: {result['avg_games_per_sec']:.1f}")
    console.print(f"  Win rate: {result['initial_win_rate']:.1%} -> {result['final_win_rate']:.1%}")

    # Compare to 3050 Ti baseline
    baseline_epoch_time = 98.0  # seconds
    speedup = baseline_epoch_time / result['avg_epoch_time_s'] if result['avg_epoch_time_s'] > 0 else 0
    console.print(f"\n  [bold]Speedup vs 3050 Ti: {speedup:.1f}x[/bold]")

    # Save final checkpoint if requested
    if save_checkpoint and 'final_checkpoint_bytes' in result:
        out_path = Path(save_checkpoint)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(result['final_checkpoint_bytes'])
        console.print(f"\n  Saved: {save_checkpoint}")


# =============================================================================
# GPU MCTS Training (t42-ssb9)
# =============================================================================
#
# AlphaZero-style MCTS training using GPU-native pipeline on B200.
# Uses oracle for leaf evaluation instead of neural network rollouts.


@app.function(
    image=forge_image,
    gpu=GPU_B200,
    volumes={ZEB_MOUNT_PATH: zeb_volume},
    timeout=14400,  # 4 hours max
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_gpu_mcts(
    epochs: int = 100,
    games_per_epoch: int = 100,
    n_simulations: int = 50,
    n_parallel_games: int = 2048,
    max_mcts_nodes: int = 1024,
    batch_size: int = 64,
    lr: float = 1e-3,
    temperature: float = 1.0,
    model_size: str = "medium",
    eval_every: int = 5,
    keep_checkpoints: int = 5,
    seed: int = 42,
    run_name: str | None = None,
    resume_from: str | None = None,
) -> dict:
    """Train Zeb via GPU-native MCTS on Modal B200.

    AlphaZero-style training with oracle leaf evaluation:
    - Deals generated on GPU
    - MCTS tree operations on GPU
    - Oracle evaluation (GPU tensors)
    - Training with zero CPU<->GPU copies

    Checkpoints saved every epoch for easy resume after interruption.

    Args:
        epochs: Number of training epochs (total, not additional)
        games_per_epoch: Games to generate per epoch
        n_simulations: MCTS simulations per move
        n_parallel_games: Concurrent MCTS games (high for B200)
        max_mcts_nodes: Max nodes per MCTS tree
        batch_size: Training batch size
        lr: Learning rate
        temperature: Action sampling temperature
        model_size: 'small', 'medium', or 'large'
        eval_every: Evaluate vs random every N epochs
        keep_checkpoints: Keep last N checkpoints (older ones deleted)
        seed: Random seed
        run_name: Run name (required for resume, auto-generated otherwise)
        resume_from: Run name to resume from (loads latest checkpoint)

    Returns:
        Dict with training results
    """
    import sys
    sys.path.insert(0, "/root")

    import os
    import time
    import random
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import torch

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # W&B setup
    use_wandb = bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb:
        import wandb

    from forge.zeb.gpu_training_pipeline import create_gpu_pipeline
    from forge.zeb.model import ZebModel, get_model_config
    from forge.zeb.run_mcts_training import evaluate_vs_random

    device = "cuda"
    print(f"[{torch.cuda.get_device_name(0)}] Starting GPU MCTS training")
    print(f"  Epochs: {epochs}, Games/epoch: {games_per_epoch}")
    print(f"  MCTS sims: {n_simulations}, Parallel games: {n_parallel_games}")
    print(f"  Model size: {model_size}")

    # Handle resume
    start_epoch = 0
    wandb_run_id = None
    total_games_prior = 0
    total_oracle_prior = 0

    if resume_from:
        run_name = resume_from
        print(f"  Resuming from: {resume_from}")

    # Run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        run_name = f"gpu-mcts-{model_size}-{n_simulations}sim-{timestamp}"

    # Output dir
    output_dir = Path(ZEB_MOUNT_PATH) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Initialize GPU pipeline
    print("Creating GPU training pipeline...")
    t0 = time.time()
    pipeline = create_gpu_pipeline(
        oracle_device=device,
        n_parallel_games=n_parallel_games,
        n_simulations=n_simulations,
        max_mcts_nodes=max_mcts_nodes,
        temperature=temperature,
    )
    print(f"Pipeline created in {time.time() - t0:.1f}s")

    # Model
    model_config = get_model_config(model_size)
    model = ZebModel(**model_config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    total_params = sum(p.numel() for p in model.parameters())
    model_size_str = f"{total_params / 1_000_000:.1f}M" if total_params >= 1_000_000 else f"{total_params // 1000}k"
    print(f"ZebModel ({model_size}): {total_params:,} parameters")

    # Load checkpoint if resuming
    if resume_from:
        # Find latest checkpoint
        ckpts = sorted(output_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if ckpts:
            latest_ckpt = ckpts[-1]
            print(f"Loading checkpoint: {latest_ckpt}")
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            wandb_run_id = ckpt.get("wandb_run_id")
            total_games_prior = ckpt.get("total_games", 0)
            total_oracle_prior = ckpt.get("total_oracle_queries", 0)
            print(f"  Resuming from epoch {start_epoch}")
            print(f"  Prior games: {total_games_prior:,}, Prior oracle queries: {total_oracle_prior:,}")
        else:
            print(f"  No checkpoints found in {output_dir}, starting fresh")

    # W&B init
    if use_wandb:
        if wandb_run_id:
            # Resume existing run
            wandb.init(
                project="zeb-mcts",
                id=wandb_run_id,
                resume="must",
            )
            print(f"W&B run resumed: {wandb.run.url}")
        else:
            # New run
            wandb.init(
                project="zeb-mcts",
                name=run_name,
                tags=["gpu-mcts", "modal", "b200", model_size, "oracle"],
                config={
                    "epochs": epochs,
                    "games_per_epoch": games_per_epoch,
                    "n_simulations": n_simulations,
                    "n_parallel_games": n_parallel_games,
                    "max_mcts_nodes": max_mcts_nodes,
                    "batch_size": batch_size,
                    "lr": lr,
                    "temperature": temperature,
                    "model_size": model_size,
                    "total_params": total_params,
                    "pipeline": "gpu-native",
                    "gpu": torch.cuda.get_device_name(0),
                },
            )
            print(f"W&B run: {wandb.run.url}")

    # Helper to save checkpoint
    def save_checkpoint(epoch: int, win_rate: float | None = None):
        ckpt_path = output_dir / f"epoch_{epoch:04d}.pt"
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model_config,
            "training_config": {
                "n_simulations": n_simulations,
                "n_parallel_games": n_parallel_games,
                "games_per_epoch": games_per_epoch,
                "batch_size": batch_size,
                "lr": lr,
            },
            "total_games": total_games_prior + pipeline.total_games_generated,
            "total_oracle_queries": total_oracle_prior + pipeline.total_oracle_queries,
            "wandb_run_id": wandb.run.id if use_wandb else None,
        }
        if win_rate is not None:
            ckpt_data["win_rate"] = win_rate
        torch.save(ckpt_data, ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path.name}")

        # Cleanup old checkpoints (keep last N)
        ckpts = sorted(output_dir.glob("epoch_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        for old_ckpt in ckpts[:-keep_checkpoints]:
            old_ckpt.unlink()
            print(f"  Removed old checkpoint: {old_ckpt.name}")

        # Commit to volume so checkpoint persists even if interrupted
        zeb_volume.commit()

    # Initial evaluation (skip if resuming)
    if start_epoch == 0:
        print("\nInitial evaluation...")
        win_rate = evaluate_vs_random(model, n_games=100, device=device)
        print(f"  vs Random: {win_rate:.1%}")
        if use_wandb:
            wandb.log({"eval/vs_random_win_rate": win_rate, "epoch": -1})
        best_win_rate = win_rate
    else:
        best_win_rate = 0.0  # Will be updated on first eval

    # Training loop
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Track oracle queries
        oracle_queries_before = pipeline.total_oracle_queries

        # Generate games with GPU-native MCTS
        examples = pipeline.generate_games_gpu(n_games=games_per_epoch)
        gen_time = time.time() - t0

        # Stats
        oracle_queries = pipeline.total_oracle_queries - oracle_queries_before
        n_examples = examples.n_examples
        games_per_sec = games_per_epoch / gen_time

        # Train
        t1 = time.time()
        metrics = pipeline.train_epoch_gpu(
            model=model,
            optimizer=optimizer,
            examples=examples,
            batch_size=batch_size,
        )
        train_time = time.time() - t1

        # Log
        epoch_str = f"Epoch {epoch:3d}: policy_loss={metrics['policy_loss']:.4f}, value_loss={metrics['value_loss']:.4f}"
        epoch_str += f" (gen={gen_time:.1f}s, train={train_time:.2f}s, {games_per_sec:.2f} games/s)"
        epoch_str += f" [oracle: {oracle_queries:,}]"
        print(epoch_str)

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/policy_loss": metrics["policy_loss"],
                "train/value_loss": metrics["value_loss"],
                "train/total_loss": metrics["policy_loss"] + metrics["value_loss"],
                "perf/gen_time_s": gen_time,
                "perf/train_time_s": train_time,
                "perf/games_per_sec": games_per_sec,
                "perf/examples": n_examples,
                "perf/oracle_queries": oracle_queries,
                "stats/total_games_generated": total_games_prior + pipeline.total_games_generated,
                "stats/total_oracle_queries": total_oracle_prior + pipeline.total_oracle_queries,
            })

        # Periodic evaluation
        win_rate = None
        if (epoch + 1) % eval_every == 0:
            win_rate = evaluate_vs_random(model, n_games=100, device=device)
            print(f"         -> vs Random: {win_rate:.1%}")
            if use_wandb:
                wandb.log({"eval/vs_random_win_rate": win_rate, "epoch": epoch})
            if win_rate > best_win_rate:
                best_win_rate = win_rate

        # Save checkpoint every epoch
        save_checkpoint(epoch, win_rate)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_win_rate = evaluate_vs_random(model, n_games=200, device=device)
    print(f"vs Random: {final_win_rate:.1%}")

    # Stats
    total_games = total_games_prior + pipeline.total_games_generated
    total_oracle = total_oracle_prior + pipeline.total_oracle_queries
    print(f"\n=== Cumulative Stats ===")
    print(f"Total games generated: {total_games:,}")
    print(f"Total oracle queries: {total_oracle:,}")

    if use_wandb:
        wandb.log({"final/vs_random_win_rate": final_win_rate})
        wandb.finish()

    total_time = time.time() - start_time
    return {
        "run_name": run_name,
        "epochs": epochs,
        "start_epoch": start_epoch,
        "total_games": total_games,
        "total_oracle_queries": total_oracle,
        "final_vs_random": final_win_rate,
        "best_vs_random": best_win_rate,
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60,
        "games_per_sec": pipeline.total_games_generated / total_time if total_time > 0 else 0,
        "gpu": torch.cuda.get_device_name(0),
    }


@app.local_entrypoint(name="gpu_mcts_train")
def gpu_mcts_train(
    epochs: int = 100,
    games_per_epoch: int = 100,
    n_simulations: int = 50,
    n_parallel_games: int = 2048,
    max_mcts_nodes: int = 1024,
    batch_size: int = 64,
    lr: float = 1e-3,
    model_size: str = "medium",
    eval_every: int = 5,
    keep_checkpoints: int = 5,
    resume_from: str | None = None,
):
    """Train Zeb via GPU MCTS on Modal B200.

    Usage:
        # Fresh training run
        modal run forge/modal_app.py::gpu_mcts_train

        # More epochs/games
        modal run forge/modal_app.py::gpu_mcts_train --epochs 200 --games-per-epoch 200

        # Resume interrupted run (use exact run_name from previous run)
        modal run forge/modal_app.py::gpu_mcts_train --resume-from gpu-mcts-medium-50sim-202602021500 --epochs 100
    """
    from rich.console import Console
    console = Console()

    total_games = epochs * games_per_epoch

    console.print("[bold]GPU MCTS Training on Modal B200[/bold]")
    if resume_from:
        console.print(f"  [yellow]RESUMING from: {resume_from}[/yellow]")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Games/epoch: {games_per_epoch}")
    console.print(f"  Total games: {total_games:,}")
    console.print(f"  MCTS sims: {n_simulations}")
    console.print(f"  Parallel games: {n_parallel_games}")
    console.print(f"  Max MCTS nodes: {max_mcts_nodes}")
    console.print(f"  Model: {model_size}")
    console.print(f"  Keep checkpoints: {keep_checkpoints}")
    console.print()

    result = train_gpu_mcts.remote(
        epochs=epochs,
        games_per_epoch=games_per_epoch,
        n_simulations=n_simulations,
        n_parallel_games=n_parallel_games,
        max_mcts_nodes=max_mcts_nodes,
        batch_size=batch_size,
        lr=lr,
        model_size=model_size,
        eval_every=eval_every,
        keep_checkpoints=keep_checkpoints,
        resume_from=resume_from,
    )

    console.print(f"\n[bold green]Training Complete![/bold green]")
    console.print(f"  Run: {result['run_name']}")
    console.print(f"  GPU: {result['gpu']}")
    if result['start_epoch'] > 0:
        console.print(f"  Resumed from epoch: {result['start_epoch']}")
    console.print(f"  Total games: {result['total_games']:,}")
    console.print(f"  Oracle queries: {result['total_oracle_queries']:,}")
    console.print(f"  Time: {result['total_time_minutes']:.1f} minutes")
    console.print(f"  Games/sec: {result['games_per_sec']:.2f}")
    console.print(f"  Final vs Random: {result['final_vs_random']:.1%}")
    console.print(f"  Best vs Random: {result['best_vs_random']:.1%}")


@app.local_entrypoint(name="train")
def train(
    epochs: int = 10,
    batch_size: int = 32768,
    lr: float = 3e-4,
    embed_dim: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    ff_dim: int = 128,
    shuffle_hands: bool = True,
    loss_mode: str = "policy",
    wandb_group: str | None = None,
    compile: bool = True,
    limit_train_batches: int | None = None,
):
    """Train a model on Modal GPU.

    Usage:
        modal run forge/modal_app.py::train --epochs 10
        modal run forge/modal_app.py::train --epochs 10 --wandb-group slot-bias-fix
        modal run forge/modal_app.py::train --no-shuffle-hands  # Compare without shuffle
        modal run forge/modal_app.py::train --no-compile --limit-train-batches 100  # Quick perf test
    """
    from rich.console import Console
    console = Console()

    console.print("[bold]Training on Modal GPU[/bold]")
    console.print(f"  Epochs: {epochs}")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Architecture: {n_layers}L-{n_heads}H-d{embed_dim}-ff{ff_dim}")
    console.print(f"  Shuffle hands: {shuffle_hands}")
    console.print(f"  Loss mode: {loss_mode}")
    console.print(f"  Compile: {compile}")
    if limit_train_batches:
        console.print(f"  Limit batches: {limit_train_batches}")
    console.print()

    result = train_model.remote(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        embed_dim=embed_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ff_dim=ff_dim,
        shuffle_hands=shuffle_hands,
        loss_mode=loss_mode,
        wandb_group=wandb_group,
        compile=compile,
        limit_train_batches=limit_train_batches,
    )

    console.print(f"\n[green]Training complete![/green]")
    console.print(f"  Run: {result['run_name']}")
    console.print(f"  Epochs: {result['epochs_trained']}")
    console.print(f"  Best Q-gap: {result['best_q_gap']:.4f}")
    console.print(f"  Time: {result['total_time_seconds'] / 60:.1f} minutes")
    console.print(f"  Checkpoint: {result['best_checkpoint']}")


# =============================================================================
# Eval Matrix (pairwise model comparison on T4)
# =============================================================================
#
# Run a full pairwise eval matrix between any combination of E[Q] and Zeb
# players on a cheap T4 GPU. Results returned as a formatted table + JSON.
#
# Usage:
#     modal run forge/modal_app.py::eval-matrix \
#         --players "eq:n=100;zeb:source=hf,weights_name=large-belief.pt;zeb:source=hf,weights_name=large.pt;zeb:source=hf" \
#         --n-games 1000 --batch-size 50


@app.function(image=eval_image, gpu="T4", timeout=3600)
def eval_matrix_remote(players: list[str], n_games: int, batch_size: int) -> dict:
    """Run pairwise eval matrix on GPU. Returns formatted table and JSON."""
    import sys
    sys.path.insert(0, "/root")
    import os
    os.chdir("/root")

    from forge.zeb.eval.players import parse_player_spec
    from forge.zeb.eval.engine import MatchConfig, run_match
    from forge.zeb.eval.results import format_matrix

    specs = [parse_player_spec(p) for p in players]
    names = [s.display_name for s in specs]
    n = len(specs)
    model_cache: dict = {}

    print(f"Eval matrix: {n} players, {n_games} games per matchup")
    for i, name in enumerate(names):
        print(f"  [{i}] {name}")

    # Run all-pairs (results[i][j] = MatchResult for player i vs player j)
    results = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            print(f"\n--- {names[i]} vs {names[j]} ---")
            config = MatchConfig(
                spec_a=specs[i],
                spec_b=specs[j],
                n_games=n_games,
                batch_size=batch_size,
                model_cache=model_cache,
                quiet=False,
            )
            results[i][j] = run_match(config)
            wr = results[i][j].team_a_win_rate
            print(f"  -> {names[i]} win rate: {wr:.1%}")

    matrix_text = format_matrix(results, names)
    matrix_json = format_matrix(results, names, json_mode=True)

    return {"matrix_text": matrix_text, "json": matrix_json, "names": names}


@app.local_entrypoint(name="eval-matrix")
def eval_matrix_entry(players: str, n_games: int = 1000, batch_size: int = 50):
    """Run pairwise eval matrix on Modal T4.

    Usage:
        modal run forge/modal_app.py::eval-matrix \\
            --players "eq:n=100;zeb:source=hf,weights_name=large-belief.pt;zeb:source=hf,weights_name=large.pt;zeb:source=hf" \\
            --n-games 1000 --batch-size 50
    """
    import json as json_mod
    import wandb
    from forge.zeb.eval.players import parse_player_spec
    from forge.zeb.eval.results import compute_elo_ratings, format_elo
    from forge.zeb.hf import DEFAULT_REPO, get_remote_training_state

    player_list = [p.strip() for p in players.split(";") if p.strip()]
    specs = [parse_player_spec(p) for p in player_list]

    print(f"Eval matrix: {len(player_list)} players, {n_games} games/matchup")
    for i, p in enumerate(player_list):
        print(f"  [{i}] {p}")
    print()

    result = eval_matrix_remote.remote(player_list, n_games, batch_size)

    print("\n" + result["matrix_text"])
    print("\n" + result["json"])

    # --- Parse JSON results into wins matrix ---
    names = result["names"]
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}
    wins = [[0] * n for _ in range(n)]

    matchups = json_mod.loads(result["json"])
    for key, data in matchups.items():
        a_name, b_name = data["team_a"], data["team_b"]
        i, j = name_to_idx[a_name], name_to_idx[b_name]
        wins[i][j] = data["team_a_wins"]
        wins[j][i] = data["team_b_wins"]

    # --- Compute Elo ratings ---
    # Anchor on E[Q] player if present, otherwise first player
    eq_anchor = None
    for spec, name in zip(specs, names):
        if spec.kind == 'eq':
            eq_anchor = name
            break

    ratings = compute_elo_ratings(names, wins, anchor=eq_anchor)
    print("\n" + format_elo(ratings))

    # --- Fetch HF training state for zeb players ---
    model_steps: dict[str, dict] = {}
    for spec, name in zip(specs, names):
        if spec.kind == 'zeb' and spec.params.get('source') == 'hf':
            repo_id = spec.params.get('repo_id', DEFAULT_REPO)
            weights_name = spec.params.get('weights_name', 'model.pt')
            state = get_remote_training_state(repo_id, weights_name)
            if state:
                model_steps[name] = {
                    'step': int(state.get('step', 0)),
                    'total_games': int(state.get('total_games', 0)),
                }
                print(f"  {name}: step={model_steps[name]['step']}, "
                      f"games={model_steps[name]['total_games']}")

    # Primary model = highest step (actively training)
    training_step = max(
        (info['step'] for info in model_steps.values()), default=0
    )

    # --- W&B logging ---
    wandb.init(
        project="zeb-eval",
        config={
            "players": player_list,
            "n_games": n_games,
            "batch_size": batch_size,
            "model_steps": model_steps,
        },
    )
    wandb.define_metric("elo/*", step_metric="training_step")

    log_data: dict = {"training_step": training_step}
    for name, elo in ratings.items():
        log_data[f"elo/{name}"] = elo
    wandb.log(log_data)

    # Pairwise results table
    columns = ["player_a", "player_b", "a_wins", "b_wins", "n_games", "a_win_rate"]
    table = wandb.Table(columns=columns)
    for key, data in matchups.items():
        total = data["team_a_wins"] + data["team_b_wins"]
        wr = data["team_a_wins"] / total if total > 0 else 0.0
        table.add_data(
            data["team_a"], data["team_b"],
            data["team_a_wins"], data["team_b_wins"],
            data["n_games"], round(wr, 4),
        )
    wandb.log({"pairwise_results": table})

    wandb.finish()
    print("\nW&B run logged to project: zeb-eval")
