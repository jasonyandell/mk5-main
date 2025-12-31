"""Data pipeline for Domino training with Lightning DataModule."""

from pathlib import Path
from typing import Optional, Tuple

import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class DominoDataset(Dataset):
    """
    Loads pre-tokenized numpy arrays with memory mapping.

    Expected files in each split directory:
        tokens.npy, masks.npy, players.npy, targets.npy, legal.npy, qvals.npy, teams.npy, values.npy

    Uses memory-mapped files to handle large datasets without loading
    everything into RAM.
    """

    def __init__(self, data_path: str, split: str):
        """
        Initialize dataset for a specific split.

        Args:
            data_path: Base directory containing split subdirectories
            split: One of 'train', 'val', or 'test'
        """
        split_dir = Path(data_path) / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Memory-mapped for large datasets
        self.tokens = np.load(split_dir / 'tokens.npy', mmap_mode='r')
        self.masks = np.load(split_dir / 'masks.npy', mmap_mode='r')
        self.players = np.load(split_dir / 'players.npy', mmap_mode='r')
        self.targets = np.load(split_dir / 'targets.npy', mmap_mode='r')
        self.legal = np.load(split_dir / 'legal.npy', mmap_mode='r')
        self.qvals = np.load(split_dir / 'qvals.npy', mmap_mode='r')
        self.teams = np.load(split_dir / 'teams.npy', mmap_mode='r')
        self.values = np.load(split_dir / 'values.npy', mmap_mode='r')

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Get a single sample.

        Returns tuple of:
            tokens: (seq_len, feature_dim) int64
            masks: (seq_len,) float32
            players: () int64 - current player index
            targets: () int64 - oracle best action
            legal: (7,) float32 - legal action mask
            qvals: (7,) float32 - oracle Q-values
            teams: () int64 - team assignment (0 or 1)
            values: () float32 - oracle state value V
        """
        # Copy from mmap for PyTorch compatibility
        # np.array() forces a copy from memmap, which is needed for PyTorch
        tokens = torch.from_numpy(np.array(self.tokens[idx], dtype=np.int64))
        masks = torch.from_numpy(np.array(self.masks[idx], dtype=np.float32))
        players = torch.tensor(int(self.players[idx]), dtype=torch.long)
        targets = torch.tensor(int(self.targets[idx]), dtype=torch.long)
        legal = torch.from_numpy(np.array(self.legal[idx], dtype=np.float32))
        qvals = torch.from_numpy(np.array(self.qvals[idx], dtype=np.float32))
        teams = torch.tensor(int(self.teams[idx]), dtype=torch.long)
        values = torch.tensor(float(self.values[idx]), dtype=torch.float32)

        return tokens, masks, players, targets, legal, qvals, teams, values


class DominoDataModule(L.LightningDataModule):
    """
    Data pipeline for Domino training.

    Follows Lightning best practices:
    - prepare_data() for download/processing (single process)
    - setup() for dataset creation (per-GPU)

    Expected directory structure:
        data_path/
            train/
                tokens.npy, masks.npy, targets.npy, legal.npy, qvals.npy, teams.npy, players.npy, values.npy
            val/
                ...
            test/
                ...
    """

    def __init__(
        self,
        data_path: str = 'data/tokenized',  # Canonical location
        batch_size: int = 512,
        num_workers: int = 8,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
    ):
        """
        Initialize DataModule.

        Args:
            data_path: Path to tokenized data directory
            batch_size: Batch size for all dataloaders
            num_workers: Number of dataloader workers (8 = half of 16 threads)
            pin_memory: Whether to pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch per worker
        """
        super().__init__()
        self.save_hyperparameters()

        # Will be set in setup()
        self.train_dataset: Optional[DominoDataset] = None
        self.val_dataset: Optional[DominoDataset] = None
        self.test_dataset: Optional[DominoDataset] = None

    def prepare_data(self) -> None:
        """
        Called once. Use for download/tokenization.

        Verifies that required data exists.
        """
        data_path = Path(self.hparams.data_path)
        if not (data_path / 'train').exists():
            raise FileNotFoundError(f'No train data at {data_path}')
        if not (data_path / 'val').exists():
            raise FileNotFoundError(f'No val data at {data_path} (required for training)')

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Called on each GPU. Create datasets here.

        Args:
            stage: One of 'fit', 'validate', 'test', or None (all)
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = DominoDataset(self.hparams.data_path, 'train')
            self.val_dataset = DominoDataset(self.hparams.data_path, 'val')

        if stage == 'validate':
            if self.val_dataset is None:
                self.val_dataset = DominoDataset(self.hparams.data_path, 'val')

        if stage == 'test' or stage is None:
            test_path = Path(self.hparams.data_path) / 'test'
            if test_path.exists():
                self.test_dataset = DominoDataset(self.hparams.data_path, 'test')

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,  # Keep workers alive
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.num_workers > 0,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
        )
