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

    def __init__(self, data_path: str, split: str, mmap: bool = False, shuffle_hands: bool = False):
        """
        Initialize dataset for a specific split.

        Args:
            data_path: Base directory containing split subdirectories
            split: One of 'train', 'val', or 'test'
            mmap: If True, use memory-mapped files (slow random access but low RAM).
                  If False, load fully into RAM (fast but needs ~5GB for full dataset).
            shuffle_hands: If True, randomly shuffle each player's 7-slot hand during
                           __getitem__ to provide data augmentation and fix slot 0 bias.
                           Each epoch sees different permutations.
        """
        split_dir = Path(data_path) / split

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        self.shuffle_hands = shuffle_hands

        # Load data - mmap for huge datasets, RAM for speed
        mmap_mode = 'r' if mmap else None
        self.tokens = np.load(split_dir / 'tokens.npy', mmap_mode=mmap_mode)
        self.masks = np.load(split_dir / 'masks.npy', mmap_mode=mmap_mode)
        self.players = np.load(split_dir / 'players.npy', mmap_mode=mmap_mode)
        self.targets = np.load(split_dir / 'targets.npy', mmap_mode=mmap_mode)
        self.legal = np.load(split_dir / 'legal.npy', mmap_mode=mmap_mode)
        self.qvals = np.load(split_dir / 'qvals.npy', mmap_mode=mmap_mode)
        self.teams = np.load(split_dir / 'teams.npy', mmap_mode=mmap_mode)
        self.values = np.load(split_dir / 'values.npy', mmap_mode=mmap_mode)

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
        tokens = np.array(self.tokens[idx], dtype=np.int64)
        masks = np.array(self.masks[idx], dtype=np.float32)
        players = int(self.players[idx])
        targets = int(self.targets[idx])
        legal = np.array(self.legal[idx], dtype=np.float32)
        qvals = np.array(self.qvals[idx], dtype=np.float32)
        teams = int(self.teams[idx])
        values = float(self.values[idx])

        if self.shuffle_hands:
            # Use idx as seed component for reproducibility within epoch.
            # Different epochs will have different DataLoader shuffle order,
            # so each sample sees different permutations across epochs.
            rng = np.random.default_rng(idx)

            # Generate 4 permutations (one per player's 7-slot hand)
            perms = [rng.permutation(7) for _ in range(4)]

            # Shuffle all 4 hand token blocks
            # Token layout: position 0 = context, positions 1-7 = player 0's hand,
            # 8-14 = player 1, 15-21 = player 2, 22-28 = player 3
            for p in range(4):
                start = 1 + p * 7
                tokens[start:start + 7] = tokens[start:start + 7][perms[p]]

            # Apply current player's permutation to qvals and legal
            cp = players
            qvals = qvals[perms[cp]]
            legal = legal[perms[cp]]

            # Map target through inverse permutation
            # If target was slot i, after shuffling it's at position inv_perm[i]
            inv_perm = np.argsort(perms[cp])
            targets = int(inv_perm[targets])

        return (
            torch.from_numpy(tokens),
            torch.from_numpy(masks),
            torch.tensor(players, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.from_numpy(legal),
            torch.from_numpy(qvals),
            torch.tensor(teams, dtype=torch.long),
            torch.tensor(values, dtype=torch.float32),
        )


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
        num_workers: int = 16,
        pin_memory: bool = True,
        prefetch_factor: int = 8,
        mmap: bool = False,
        shuffle_hands: bool = True,
    ):
        """
        Initialize DataModule.

        Args:
            data_path: Path to tokenized data directory
            batch_size: Batch size for all dataloaders
            num_workers: Number of dataloader workers (8 = half of 16 threads)
            pin_memory: Whether to pin memory for faster GPU transfer
            prefetch_factor: Number of batches to prefetch per worker
            mmap: If True, use memory-mapped files (low RAM, slow). Default False (load to RAM).
            shuffle_hands: If True, shuffle hand slots during training (data augmentation).
                           Fixes slot 0 positional bias. Default True.
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
        mmap = self.hparams.mmap
        shuffle_hands = self.hparams.shuffle_hands
        if stage == 'fit' or stage is None:
            # Shuffle hands for both training and validation
            self.train_dataset = DominoDataset(
                self.hparams.data_path, 'train', mmap=mmap, shuffle_hands=shuffle_hands
            )
            self.val_dataset = DominoDataset(
                self.hparams.data_path, 'val', mmap=mmap, shuffle_hands=shuffle_hands
            )

        if stage == 'validate':
            if self.val_dataset is None:
                self.val_dataset = DominoDataset(self.hparams.data_path, 'val', mmap=mmap)

        if stage == 'test' or stage is None:
            test_path = Path(self.hparams.data_path) / 'test'
            if test_path.exists():
                self.test_dataset = DominoDataset(self.hparams.data_path, 'test', mmap=mmap)

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
