"""Modal cloud training for Zeb self-play."""
import modal
from pathlib import Path

# Import from parent modal_app
app = modal.App("zeb-training")

# Reuse forge image
forge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "lightning>=2.0",
        "numpy>=1.26,<2",
        "rich",
        "wandb>=0.16",
    )
    .add_local_dir(local_path="forge", remote_path="/root/forge")
)

# Volume for checkpoints
ZEB_VOLUME = "zeb-training-data"
ZEB_MOUNT = "/zeb_data"
zeb_volume = modal.Volume.from_name(ZEB_VOLUME, create_if_missing=True)


@app.function(
    image=forge_image,
    gpu="A100-40GB",
    volumes={ZEB_MOUNT: zeb_volume},
    timeout=14400,  # 4 hours
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_zeb_cloud(
    epochs: int = 100,
    games_per_epoch: int = 5000,
    batch_size: int = 2048,
    model_size: str = 'medium',
    temperature: float = 1.0,
    lr: float = 3e-4,
    seed: int = 42,
    wandb_project: str = 'zeb-training',
) -> dict:
    """Train Zeb on cloud GPU."""
    import sys
    sys.path.insert(0, "/root")

    import torch
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import WandbLogger

    from forge.zeb.model import get_model_config
    from forge.zeb.module import ZebLightningModule
    from forge.zeb.self_play import play_games_batched, trajectories_to_batch
    from forge.zeb.evaluate import evaluate_vs_random, evaluate_vs_heuristic

    L.seed_everything(seed)
    torch.set_float32_matmul_precision("high")

    # Model
    config = get_model_config(model_size)
    module = ZebLightningModule(**config, lr=lr)

    # Compile for speed
    module.model = torch.compile(module.model)

    # Logging
    logger = WandbLogger(project=wandb_project, name=f"zeb-{model_size}")

    # Output dir
    output_dir = Path(ZEB_MOUNT) / "runs" / f"zeb-{model_size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Custom training loop (self-play generates data)
    device = 'cuda'
    module.to(device)

    optimizer = torch.optim.AdamW(
        module.parameters(),
        lr=lr,
        weight_decay=0.01,
    )

    for epoch in range(epochs):
        # Generate training data via self-play
        trajectories = play_games_batched(
            module.model,
            n_games=games_per_epoch,
            temperature=temperature,
            device=device,
            base_seed=epoch * games_per_epoch,
        )

        batch_data = trajectories_to_batch(trajectories)
        dataset = torch.utils.data.TensorDataset(
            *[t.to(device) for t in batch_data]
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        # Training
        module.train()
        epoch_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            loss = module.training_step(batch, 0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate periodically
        if epoch % 10 == 0:
            vs_random = evaluate_vs_random(module.model, n_games=100, device=device)
            vs_heuristic = evaluate_vs_heuristic(module.model, n_games=100, device=device)

            print(f"Epoch {epoch}: loss={epoch_loss:.3f}, "
                  f"vs_random={vs_random['team0_win_rate']:.1%}, "
                  f"vs_heuristic={vs_heuristic['team0_win_rate']:.1%}")

            logger.log_metrics({
                'epoch': epoch,
                'loss': epoch_loss,
                'vs_random_win_rate': vs_random['team0_win_rate'],
                'vs_heuristic_win_rate': vs_heuristic['team0_win_rate'],
            })

        # Save checkpoint
        if epoch % 20 == 0:
            ckpt_path = output_dir / f"epoch-{epoch}.pt"
            torch.save(module.state_dict(), ckpt_path)

    # Final save
    final_path = output_dir / "final.pt"
    torch.save(module.state_dict(), final_path)
    zeb_volume.commit()

    return {
        'final_checkpoint': str(final_path),
        'epochs': epochs,
        'games_played': epochs * games_per_epoch,
    }


@app.local_entrypoint()
def main(
    epochs: int = 100,
    games_per_epoch: int = 5000,
    model_size: str = 'medium',
):
    result = train_zeb_cloud.remote(
        epochs=epochs,
        games_per_epoch=games_per_epoch,
        model_size=model_size,
    )
    print(result)
