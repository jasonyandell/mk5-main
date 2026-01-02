"""Model loading and batched inference for bidding simulation.

Handles:
- Loading trained checkpoint
- torch.compile for faster inference
- Batched forward passes with inference_mode
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

from forge.ml.module import DominoTransformer


DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "models" / "domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt"


class PolicyModel:
    """Compiled policy model for fast batched inference."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: str | torch.device | None = None,
        compile_model: bool = True,
    ):
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint. Uses default if None.
            device: Device to use. Auto-selects if None.
            compile_model: Whether to use torch.compile for speed.
        """
        if checkpoint_path is None:
            checkpoint_path = DEFAULT_CHECKPOINT

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        # Load checkpoint directly to avoid on_load_checkpoint RNG restoration
        checkpoint = torch.load(str(checkpoint_path), map_location=self.device, weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})

        # Create model with same hyperparameters
        self.model = DominoTransformer(
            embed_dim=hparams.get("embed_dim", 64),
            n_heads=hparams.get("n_heads", 4),
            n_layers=hparams.get("n_layers", 2),
            ff_dim=hparams.get("ff_dim", 128),
            dropout=hparams.get("dropout", 0.1),
        )

        # Load model weights (strip 'model.' prefix from state dict)
        state_dict = checkpoint["state_dict"]
        model_state = {k.replace("model.", ""): v for k, v in state_dict.items() if k.startswith("model.")}
        self.model.load_state_dict(model_state)

        self.model.eval()
        self.model.to(self.device)

        # Compile for faster inference
        if compile_model and hasattr(torch, "compile"):
            self.model = torch.compile(
                self.model,
                mode="reduce-overhead",
                fullgraph=True,
            )

        self._warmed_up = False

    def warmup(self, batch_size: int) -> None:
        """Warmup compiled model with exact batch size to avoid recompilation.

        Args:
            batch_size: The batch size that will be used for inference.
        """
        if self._warmed_up:
            return

        with torch.inference_mode():
            dummy_tokens = torch.zeros(batch_size, 32, 12, dtype=torch.long, device=self.device)
            dummy_mask = torch.ones(batch_size, 32, dtype=torch.float32, device=self.device)
            dummy_player = torch.zeros(batch_size, dtype=torch.long, device=self.device)

            # Run a few warmup passes
            for _ in range(3):
                self.model(dummy_tokens, dummy_mask, dummy_player)

        self._warmed_up = True

    @torch.inference_mode()
    def forward(
        self,
        tokens: Tensor,
        mask: Tensor,
        current_player: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Run batched forward pass.

        Args:
            tokens: (batch, 32, 12) int64 token features
            mask: (batch, 32) float32 attention mask
            current_player: (batch,) int64 current player index

        Returns:
            Tuple of:
                logits: (batch, 7) action logits
                value: (batch,) state value predictions
        """
        logits, value = self.model(tokens, mask, current_player)
        return logits, value

    @torch.inference_mode()
    def get_action_probs(
        self,
        tokens: Tensor,
        mask: Tensor,
        current_player: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        """Get action probabilities with legal masking.

        Args:
            tokens: (batch, 32, 12) int64 token features
            mask: (batch, 32) float32 attention mask
            current_player: (batch,) int64 current player index
            legal_mask: (batch, 7) bool legal action mask

        Returns:
            probs: (batch, 7) action probabilities (0 for illegal)
        """
        logits, _ = self.forward(tokens, mask, current_player)

        # Mask illegal actions
        logits = logits.masked_fill(~legal_mask, float("-inf"))

        # Softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        return probs

    @torch.inference_mode()
    def sample_actions(
        self,
        tokens: Tensor,
        mask: Tensor,
        current_player: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        """Sample actions from policy distribution.

        Args:
            tokens: (batch, 32, 12) int64 token features
            mask: (batch, 32) float32 attention mask
            current_player: (batch,) int64 current player index
            legal_mask: (batch, 7) bool legal action mask

        Returns:
            actions: (batch,) int64 sampled local action indices
        """
        probs = self.get_action_probs(tokens, mask, current_player, legal_mask)
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return actions

    @torch.inference_mode()
    def greedy_actions(
        self,
        tokens: Tensor,
        mask: Tensor,
        current_player: Tensor,
        legal_mask: Tensor,
    ) -> Tensor:
        """Get greedy (argmax) actions.

        Args:
            tokens: (batch, 32, 12) int64 token features
            mask: (batch, 32) float32 attention mask
            current_player: (batch,) int64 current player index
            legal_mask: (batch, 7) bool legal action mask

        Returns:
            actions: (batch,) int64 greedy local action indices
        """
        logits, _ = self.forward(tokens, mask, current_player)
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        actions = logits.argmax(dim=-1)
        return actions
