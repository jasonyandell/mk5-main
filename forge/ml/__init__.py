"""forge.ml - Machine learning module for domino transformer training."""

from .data import DominoDataModule, DominoDataset
from .metrics import compute_accuracy, compute_blunder_rate, compute_qgap, compute_qgaps_per_sample
from .module import DominoLightningModule, DominoTransformer
from .tokenize import get_split, process_shard, tokenize_shards

__all__ = [
    # Core classes
    "DominoTransformer",
    "DominoLightningModule",
    "DominoDataModule",
    "DominoDataset",
    # Tokenization
    "tokenize_shards",
    "process_shard",
    "get_split",
    # Metrics
    "compute_qgap",
    "compute_qgaps_per_sample",
    "compute_blunder_rate",
    "compute_accuracy",
]
