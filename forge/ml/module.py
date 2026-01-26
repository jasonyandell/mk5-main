"""Lightning module for Domino transformer training."""

import random
from typing import Any, Dict, Optional, Tuple

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .metrics import (
    compute_accuracy,
    compute_blunder_rate,
    compute_per_declaration_regret,
    compute_per_slot_accuracy,
    compute_per_slot_regret,
    compute_per_slot_tie_rate,
    compute_qgaps_per_sample,
    compute_regret_stats,
)


class DominoTransformer(nn.Module):
    """
    The actual model architecture for domino play prediction.

    Extracted from train_pretokenized.py - keeps architecture code
    separate from Lightning wrapper.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Feature embeddings for the token representation
        self.high_pip_embed = nn.Embedding(7, embed_dim // 6)
        self.low_pip_embed = nn.Embedding(7, embed_dim // 6)
        self.is_double_embed = nn.Embedding(2, embed_dim // 12)
        self.count_value_embed = nn.Embedding(3, embed_dim // 12)
        self.trump_rank_embed = nn.Embedding(8, embed_dim // 6)
        self.player_id_embed = nn.Embedding(4, embed_dim // 12)
        self.is_current_embed = nn.Embedding(2, embed_dim // 12)
        self.is_partner_embed = nn.Embedding(2, embed_dim // 12)
        self.is_remaining_embed = nn.Embedding(2, embed_dim // 12)
        self.token_type_embed = nn.Embedding(8, embed_dim // 6)
        self.decl_embed = nn.Embedding(10, embed_dim // 6)
        self.leader_embed = nn.Embedding(4, embed_dim // 12)

        self.input_proj = nn.Linear(self._calc_input_dim(), embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(embed_dim, 1)  # Action logits
        self.value_head = nn.Linear(embed_dim, 1)   # State value prediction

    def _calc_input_dim(self) -> int:
        """Calculate total dimension of concatenated embeddings."""
        return (
            self.embed_dim // 6 * 4 +
            self.embed_dim // 12 * 6 +
            self.embed_dim // 6 +
            self.embed_dim // 12
        )

    def forward(self, tokens: Tensor, mask: Tensor, current_player: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            tokens: Input token tensor, shape (batch, seq_len, 12)
            mask: Attention mask, shape (batch, seq_len)
            current_player: Current player index, shape (batch,)

        Returns:
            Tuple of:
                logits: Action logits, shape (batch, 7)
                value: State value prediction, shape (batch,)
        """
        device = tokens.device

        # Build embeddings from token features
        embeds = [
            self.high_pip_embed(tokens[:, :, 0]),
            self.low_pip_embed(tokens[:, :, 1]),
            self.is_double_embed(tokens[:, :, 2]),
            self.count_value_embed(tokens[:, :, 3]),
            self.trump_rank_embed(tokens[:, :, 4]),
            self.player_id_embed(tokens[:, :, 5]),
            self.is_current_embed(tokens[:, :, 6]),
            self.is_partner_embed(tokens[:, :, 7]),
            self.is_remaining_embed(tokens[:, :, 8]),
            self.token_type_embed(tokens[:, :, 9]),
            self.decl_embed(tokens[:, :, 10]),
            self.leader_embed(tokens[:, :, 11]),
        ]

        x = torch.cat(embeds, dim=-1)
        x = self.input_proj(x)

        # Apply transformer with attention mask
        attn_mask = (mask == 0)
        x = self.transformer(x, src_key_padding_mask=attn_mask)

        # Extract hand representations for current player
        # Player's 7 dominoes start at index 1 + player_id * 7
        start_indices = 1 + current_player * 7
        offsets = torch.arange(7, device=device)
        gather_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
        gather_indices = gather_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        hand_repr = torch.gather(x, dim=1, index=gather_indices)
        logits = self.output_proj(hand_repr).squeeze(-1)

        # Value prediction: mean pool over valid tokens
        # Use context token (always present at position 0) for value
        value = self.value_head(x[:, 0, :]).squeeze(-1)

        return logits, value


class DominoLightningModule(L.LightningModule):
    """
    Lightning wrapper for DominoTransformer.

    Separates research code (model) from engineering code (training).
    Handles loss computation, logging, optimizer configuration, and
    checkpoint RNG state preservation.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        ff_dim: int = 128,
        dropout: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        temperature: float = 3.0,
        soft_weight: float = 0.7,
        value_weight: float = 0.5,
        loss_mode: str = 'policy',
    ):
        """
        Initialize the Lightning module.

        Args:
            embed_dim: Transformer embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            ff_dim: Feed-forward dimension
            dropout: Dropout rate
            lr: Learning rate
            weight_decay: AdamW weight decay
            temperature: Temperature for soft target distribution
            soft_weight: Weight for soft loss (1 - soft_weight = hard loss weight)
            value_weight: Weight for value head loss
            loss_mode: 'policy' (cross-entropy) or 'qvalue' (MSE on Q-values)
        """
        super().__init__()
        if loss_mode not in ('policy', 'qvalue'):
            raise ValueError(f"loss_mode must be 'policy' or 'qvalue', got {loss_mode}")
        self.save_hyperparameters()  # Auto-save all args
        self.model = DominoTransformer(embed_dim, n_heads, n_layers, ff_dim, dropout)

    def forward(self, tokens: Tensor, mask: Tensor, current_player: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through the model. Returns (logits, value)."""
        return self.model(tokens, mask, current_player)

    def _compute_loss(
        self,
        logits: Tensor,
        value: Tensor,
        targets: Tensor,
        legal: Tensor,
        qvals: Tensor,
        teams: Tensor,
        oracle_values: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute combined action + value loss.

        Dispatches to policy loss (cross-entropy) or Q-value loss (MSE) based on loss_mode.

        Returns:
            Tuple of (total_loss, action_loss, value_loss) for logging
        """
        if self.hparams.loss_mode == 'qvalue':
            return self._compute_qvalue_loss(logits, value, legal, qvals, teams, oracle_values)
        else:
            return self._compute_policy_loss(logits, value, targets, legal, qvals, teams, oracle_values)

    def _compute_policy_loss(
        self,
        logits: Tensor,
        value: Tensor,
        targets: Tensor,
        legal: Tensor,
        qvals: Tensor,
        teams: Tensor,
        oracle_values: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Policy loss: cross-entropy + soft distillation.

        Action loss: hard (cross-entropy) + soft (distillation from Q-values)
        Value loss: MSE between predicted and oracle state value

        Returns:
            Tuple of (total_loss, action_loss, value_loss) for logging
        """
        logits_masked = logits.masked_fill(legal == 0, float('-inf'))

        # Hard loss (cross-entropy with oracle target)
        hard_loss = F.cross_entropy(logits_masked, targets)

        # Soft loss (distillation from Q-values)
        team_sign = torch.where(teams == 0, 1.0, -1.0).unsqueeze(1)
        q_for_soft = qvals * team_sign
        q_masked = torch.where(
            legal > 0,
            q_for_soft,
            torch.tensor(float('-inf'), device=logits.device)
        )
        soft_targets = F.softmax(q_masked / self.hparams.temperature, dim=-1)
        soft_targets = soft_targets.clamp(min=1e-8)
        soft_targets = soft_targets / soft_targets.sum(dim=-1, keepdim=True)

        log_probs = F.log_softmax(logits_masked, dim=-1)
        log_probs_safe = log_probs.masked_fill(legal == 0, 0.0)
        soft_loss = -(soft_targets * log_probs_safe).sum(dim=-1).mean()

        # Combined action loss
        action_loss = (1 - self.hparams.soft_weight) * hard_loss + self.hparams.soft_weight * soft_loss

        # Value loss: MSE between predicted and oracle value
        # Oracle value is from team 0's perspective, adjust for current team
        # Normalize to [-1, 1] range (divide by 42) so loss scale matches action loss
        team_sign_value = torch.where(teams == 0, 1.0, -1.0)
        target_value = (oracle_values * team_sign_value) / 42.0
        value_loss = F.mse_loss(value, target_value)

        # Total loss
        total_loss = action_loss + self.hparams.value_weight * value_loss
        return total_loss, action_loss, value_loss

    def _compute_qvalue_loss(
        self,
        predicted_q: Tensor,
        value: Tensor,
        legal: Tensor,
        qvals: Tensor,
        teams: Tensor,
        oracle_values: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Q-value loss: MSE between predicted and oracle Q-values.

        Model outputs are interpreted as Q-values (expected points), not logits.
        Loss is computed only on legal actions.

        Returns:
            Tuple of (total_loss, q_loss, value_loss) for logging
        """
        # Flip Q-values for current player's perspective
        # Oracle qvals are from team 0's perspective
        team_sign = torch.where(teams == 0, 1.0, -1.0).unsqueeze(1)
        target_q = qvals * team_sign

        # Normalize to [-1, 1] for stable training (divide by 42)
        target_q_norm = target_q / 42.0
        predicted_q_norm = predicted_q / 42.0

        # Mask for legal actions only
        legal_mask = legal > 0

        # MSE loss on legal actions only
        if legal_mask.any():
            q_loss = F.mse_loss(
                predicted_q_norm[legal_mask],
                target_q_norm[legal_mask]
            )
        else:
            q_loss = torch.tensor(0.0, device=predicted_q.device)

        # Value loss: same as policy mode
        team_sign_value = torch.where(teams == 0, 1.0, -1.0)
        target_value = (oracle_values * team_sign_value) / 42.0
        value_loss = F.mse_loss(value, target_value)

        # Total loss
        total_loss = q_loss + self.hparams.value_weight * value_loss
        return total_loss, q_loss, value_loss

    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """Training step."""
        tokens, masks, players, targets, legal, qvals, teams, values = batch
        logits, value = self(tokens, masks, players)
        loss, action_loss, value_loss = self._compute_loss(
            logits, value, targets, legal, qvals, teams, values
        )
        acc = compute_accuracy(logits, targets, legal)

        # Structured logging with prefixes (sync_dist for multi-GPU)
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/action_loss', action_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/value_loss', value_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def on_validation_start(self) -> None:
        """Initialize accumulators for epoch-level metrics."""
        self._val_gaps = []
        self._val_targets = []
        self._val_decl_ids = []
        self._val_logits = []
        self._val_qvals = []
        self._val_legal = []
        self._val_teams = []

    def validation_step(self, batch: Tuple, batch_idx: int) -> None:
        """Validation step with comprehensive metrics."""
        tokens, masks, players, targets, legal, qvals, teams, values = batch
        logits, value = self(tokens, masks, players)

        # Compute all metrics
        loss, action_loss, value_loss = self._compute_loss(
            logits, value, targets, legal, qvals, teams, values
        )
        acc = compute_accuracy(logits, targets, legal)
        gaps = compute_qgaps_per_sample(logits, qvals, legal, teams)
        q_gap = gaps.mean()
        blunder_rate = compute_blunder_rate(gaps)

        # Regret statistics
        regret_stats = compute_regret_stats(gaps)
        self.log('val/regret_mean', regret_stats['mean'], sync_dist=True)
        self.log('val/regret_p99', regret_stats['p99'], sync_dist=True)
        self.log('val/regret_max', regret_stats['max'], sync_dist=True)
        self.log('val/zero_regret_rate', regret_stats['zero_rate'], sync_dist=True)

        # Value prediction error (MAE in points)
        # Model predicts normalized [-1,1], multiply by 42 to get points
        team_sign_value = torch.where(teams == 0, 1.0, -1.0)
        target_value_normalized = (values * team_sign_value) / 42.0
        value_mae = ((value - target_value_normalized) * 42.0).abs().mean()

        # CRITICAL: sync_dist=True for multi-GPU
        self.log('val/loss', loss, sync_dist=True, prog_bar=True)
        self.log('val/action_loss', action_loss, sync_dist=True)
        self.log('val/value_loss', value_loss, sync_dist=True)
        self.log('val/value_mae', value_mae, sync_dist=True, prog_bar=True)
        self.log('val/accuracy', acc, sync_dist=True, prog_bar=True)
        self.log('val/q_gap', q_gap, sync_dist=True, prog_bar=True)
        self.log('val/blunder_rate', blunder_rate, sync_dist=True)

        # Accumulate for epoch-level metrics
        self._val_gaps.append(gaps.detach())
        self._val_targets.append(targets.detach())
        self._val_logits.append(logits.detach())
        self._val_qvals.append(qvals.detach())
        self._val_legal.append(legal.detach())
        self._val_teams.append(teams.detach())

        # Extract declaration ID from context token (position 0, feature 10)
        decl_ids = tokens[:, 0, 10]
        self._val_decl_ids.append(decl_ids.detach())

        # Q-value calibration metrics (for qvalue loss mode)
        if self.hparams.loss_mode == 'qvalue':
            team_sign = torch.where(teams == 0, 1.0, -1.0).unsqueeze(1)
            target_q = qvals * team_sign
            legal_mask = legal > 0

            # MAE between predicted and actual Q-values (in points)
            if legal_mask.any():
                q_mae = (logits[legal_mask] - target_q[legal_mask]).abs().mean()
                # Calibration: mean predicted vs mean actual
                mean_pred_q = logits[legal_mask].mean()
                mean_actual_q = target_q[legal_mask].mean()
                q_calibration_error = (mean_pred_q - mean_actual_q).abs()
            else:
                q_mae = torch.tensor(0.0, device=logits.device)
                q_calibration_error = torch.tensor(0.0, device=logits.device)

            self.log('val/q_mae', q_mae, sync_dist=True, prog_bar=True)
            self.log('val/q_calibration_error', q_calibration_error, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        """Compute epoch-level validation metrics."""
        if not self._val_gaps:
            return

        # Concatenate accumulated tensors
        gaps = torch.cat(self._val_gaps)
        targets = torch.cat(self._val_targets)
        decl_ids = torch.cat(self._val_decl_ids)
        logits = torch.cat(self._val_logits)
        qvals = torch.cat(self._val_qvals)
        legal = torch.cat(self._val_legal)
        teams = torch.cat(self._val_teams)

        # Per-slot accuracy
        slot_accs = compute_per_slot_accuracy(logits, targets, legal)
        for slot in range(7):
            if not torch.isnan(slot_accs[slot]):
                self.log(f'val/slot_acc_{slot}', slot_accs[slot], sync_dist=True)

        # Per-slot regret
        slot_regrets = compute_per_slot_regret(gaps, targets)
        for slot in range(7):
            if not torch.isnan(slot_regrets[slot]):
                self.log(f'val/slot_regret_{slot}', slot_regrets[slot], sync_dist=True)

        # Per-declaration regret
        decl_regrets = compute_per_declaration_regret(gaps, decl_ids)
        for key, val in decl_regrets.items():
            self.log(f'val/regret_{key}', val, sync_dist=True)

        # Per-slot tie rate (for detecting slot bias in tie-breaking)
        tie_rates = compute_per_slot_tie_rate(logits, qvals, legal, teams)
        for slot in range(7):
            self.log(f'val/tie_rate_slot_{slot}', tie_rates[slot], sync_dist=True)

        # Log warnings for potential issues at epoch 1+
        if self.current_epoch >= 1:
            # Check for slot accuracy imbalance
            valid_accs = slot_accs[~torch.isnan(slot_accs)]
            if len(valid_accs) > 0 and valid_accs.min() < 0.9 * valid_accs.max():
                print(f"Warning: Slot accuracy imbalance detected. "
                      f"Min: {valid_accs.min():.3f}, Max: {valid_accs.max():.3f}")

            # Check for slot 0 tie-breaking bias
            if tie_rates.sum() > 0 and tie_rates[0] > 0.2:  # More than ~20% when uniform would be ~14%
                print(f"Warning: Possible slot 0 bias in tie-breaking. "
                      f"Slot 0 tie rate: {tie_rates[0]:.3f} (expected ~0.143)")

        # Clear accumulators
        self._val_gaps = []
        self._val_targets = []
        self._val_decl_ids = []
        self._val_logits = []
        self._val_qvals = []
        self._val_legal = []
        self._val_teams = []

    def test_step(self, batch: Tuple, batch_idx: int) -> None:
        """Test step - same as validation."""
        tokens, masks, players, targets, legal, qvals, teams, values = batch
        logits, value = self(tokens, masks, players)

        # Compute all metrics
        loss, action_loss, value_loss = self._compute_loss(
            logits, value, targets, legal, qvals, teams, values
        )
        acc = compute_accuracy(logits, targets, legal)
        gaps = compute_qgaps_per_sample(logits, qvals, legal, teams)
        q_gap = gaps.mean()
        blunder_rate = compute_blunder_rate(gaps)

        # Value prediction error (MAE in points)
        # Model predicts normalized [-1,1], multiply by 42 to get points
        team_sign_value = torch.where(teams == 0, 1.0, -1.0)
        target_value_normalized = (values * team_sign_value) / 42.0
        value_mae = ((value - target_value_normalized) * 42.0).abs().mean()

        # Log with test/ prefix
        self.log('test/loss', loss, sync_dist=True)
        self.log('test/action_loss', action_loss, sync_dist=True)
        self.log('test/value_loss', value_loss, sync_dist=True)
        self.log('test/value_mae', value_mae, sync_dist=True)
        self.log('test/accuracy', acc, sync_dist=True)
        self.log('test/q_gap', q_gap, sync_dist=True)
        self.log('test/blunder_rate', blunder_rate, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # Handle case where trainer might not have max_epochs set yet
        max_epochs = getattr(self.trainer, 'max_epochs', 10) or 10

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs
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
        """Save RNG state for exact reproducibility."""
        checkpoint['rng_state'] = {
            'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'python': random.getstate(),
        }
        if torch.cuda.is_available():
            checkpoint['rng_state']['cuda'] = torch.cuda.get_rng_state_all()

    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """Restore RNG state for exact reproducibility."""
        if 'rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['rng_state']['torch'])
            np.random.set_state(checkpoint['rng_state']['numpy'])
            random.setstate(checkpoint['rng_state']['python'])
            if torch.cuda.is_available() and 'cuda' in checkpoint['rng_state']:
                torch.cuda.set_rng_state_all(checkpoint['rng_state']['cuda'])
