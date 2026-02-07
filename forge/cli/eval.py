#!/usr/bin/env python3
"""Evaluate checkpoint on test set."""
import argparse

import lightning as L

from forge.ml.data import DominoDataModule
from forge.ml.module import DominoLightningModule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/tokenized', help='Path to tokenized data')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    args = parser.parse_args()

    # Load model from checkpoint (hyperparameters auto-restored)
    # weights_only=False needed for numpy arrays in checkpoint (trusted local checkpoints)
    model = DominoLightningModule.load_from_checkpoint(args.checkpoint, weights_only=False)
    data = DominoDataModule(args.data, batch_size=args.batch_size)

    trainer = L.Trainer(accelerator='auto', devices='auto')

    if args.split == 'test':
        trainer.test(model, data)
    else:
        trainer.validate(model, data)


if __name__ == '__main__':
    main()
