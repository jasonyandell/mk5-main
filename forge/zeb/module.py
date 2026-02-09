"""Zeb PyTorch Lightning module for REINFORCE training."""
import torch
import torch.nn.functional as F
import lightning as L
from torch import Tensor
from typing import Dict, Any, Tuple
import random
import numpy as np

from .model import ZebModel


class ZebLightningModule(L.LightningModule):
    """Lightning wrapper for Zeb self-play training."""

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        entropy_weight: float = 0.1,
        value_weight: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self._logged_initial_entropy = False

        self.model = ZebModel(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

    def forward(
        self,
        tokens: Tensor,
        mask: Tensor,
        hand_indices: Tensor,
        hand_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        return self.model(tokens, mask, hand_indices, hand_mask)

    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """REINFORCE with baseline subtraction.

        Batch contains:
            tokens: [B, seq, 8]
            mask: [B, seq]
            hand_indices: [B, 7]
            hand_mask: [B, 7]
            actions: [B] chosen action indices
            outcomes: [B] game outcomes in [-1, 1]
        """
        tokens, mask, hand_indices, hand_mask, actions, outcomes = batch

        # Forward pass
        policy, value, _belief = self(tokens, mask, hand_indices, hand_mask)

        # Mask illegal actions
        policy = policy.masked_fill(~hand_mask, float('-inf'))

        # Policy loss: REINFORCE with value baseline
        advantage = outcomes - value.detach()  # Don't backprop through value for policy
        log_probs = F.log_softmax(policy, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(advantage * action_log_probs).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(value, outcomes)

        # Entropy bonus for exploration
        probs = F.softmax(policy, dim=-1)
        # Only compute entropy over legal actions
        probs_masked = probs * hand_mask
        probs_masked = probs_masked / probs_masked.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        log_probs_masked = torch.log(probs_masked + 1e-8)
        entropy = -(probs_masked * log_probs_masked).sum(dim=-1).mean()

        # Total loss
        loss = (
            policy_loss
            + self.hparams.value_weight * value_loss
            - self.hparams.entropy_weight * entropy
        )

        # Logging
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/policy_loss', policy_loss, sync_dist=True)
        self.log('train/value_loss', value_loss, sync_dist=True)
        self.log('train/entropy', entropy, sync_dist=True)
        self.log('train/advantage_mean', advantage.mean(), sync_dist=True)
        self.log('train/advantage_std', advantage.std(), sync_dist=True)

        # Log initial entropy to verify healthy starting point
        if not self._logged_initial_entropy:
            self._logged_initial_entropy = True
            initial_entropy_value = entropy.item() if hasattr(entropy, 'item') else float(entropy)
            self.log('train/initial_entropy', initial_entropy_value, sync_dist=True)
            print(f"[Zeb] Initial entropy: {initial_entropy_value:.3f} (healthy range: 0.5-1.5)")
            # Also log to W&B summary for visibility (single-point metrics need this)
            try:
                import wandb
                if wandb.run is not None:
                    wandb.run.summary['initial_entropy'] = initial_entropy_value
            except ImportError:
                pass

        return loss

    def validation_step(self, batch: Tuple, batch_idx: int) -> None:
        """Validation metrics."""
        tokens, mask, hand_indices, hand_mask, actions, outcomes = batch

        policy, value, _belief = self(tokens, mask, hand_indices, hand_mask)
        policy = policy.masked_fill(~hand_mask, float('-inf'))

        # Value prediction error
        value_mse = F.mse_loss(value, outcomes)

        # Policy accuracy (did we pick same action?)
        pred_actions = policy.argmax(dim=-1)
        accuracy = (pred_actions == actions).float().mean()

        self.log('val/value_mse', value_mse, sync_dist=True, prog_bar=True)
        self.log('val/action_accuracy', accuracy, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        # Cosine annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    # RNG state preservation for reproducibility
    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        checkpoint['rng_state'] = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        if torch.cuda.is_available():
            checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state_all()

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            np.random.set_state(checkpoint['rng_state']['numpy'])
            random.setstate(checkpoint['rng_state']['python'])
            if torch.cuda.is_available() and 'cuda' in checkpoint['rng_state']:
                torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
