"""Tests for learner eval-aux source routing and staleness controls."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from forge.zeb.gpu_training_pipeline import GPUReplayBuffer, TrainingExamples
from forge.zeb.learner.run import (
    IngestStats,
    SOURCE_EVAL_AUX,
    SOURCE_SELFPLAY,
    TrainWeights,
    _eval_aux_keep_weight,
    _route_batch_into_replay,
    _source_from_metadata,
    train_n_steps_source_aware,
)


def _make_batch(*, source: str, model_step: int, n: int = 8) -> TrainingExamples:
    obs = torch.zeros((n, 36, 8), dtype=torch.int32)
    masks = torch.zeros((n, 36), dtype=torch.bool)
    masks[:, :8] = True
    hand_idx = torch.arange(1, 8, dtype=torch.int64).unsqueeze(0).expand(n, -1).clone()
    hand_masks = torch.ones((n, 7), dtype=torch.bool)
    policy = torch.zeros((n, 7), dtype=torch.float32)
    policy[:, 0] = 1.0
    value = torch.zeros(n, dtype=torch.float32)
    belief_targets = torch.zeros((n, 28), dtype=torch.int64)
    belief_mask = torch.ones((n, 28), dtype=torch.bool)
    return TrainingExamples(
        observations=obs,
        masks=masks,
        hand_indices=hand_idx,
        hand_masks=hand_masks,
        policy_targets=policy,
        value_targets=value,
        belief_targets=belief_targets,
        belief_mask=belief_mask,
        metadata={
            'source': source,
            'model_step': model_step,
            'n_games': 2,
        },
    )


def _make_buffers() -> tuple[GPUReplayBuffer, GPUReplayBuffer]:
    selfplay = GPUReplayBuffer(capacity=2000, device=torch.device('cpu'), belief=True)
    eval_aux = GPUReplayBuffer(capacity=2000, device=torch.device('cpu'), belief=True)
    return selfplay, eval_aux


def _make_args(**overrides) -> argparse.Namespace:
    defaults = {
        'eval_aux_enabled': True,
        'eval_aux_max_model_lag': 400,
        'eval_aux_lag_half_life': 0,
        'eval_aux_min_keep_weight': 0.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_source_defaults_to_selfplay_when_missing():
    assert _source_from_metadata(None) == SOURCE_SELFPLAY
    assert _source_from_metadata({}) == SOURCE_SELFPLAY


def test_eval_aux_keep_weight_respects_floor():
    assert _eval_aux_keep_weight(lag_steps=0, half_life=200, min_weight=0.1) == 1.0
    w = _eval_aux_keep_weight(lag_steps=10_000, half_life=200, min_weight=0.1)
    assert abs(w - 0.1) < 1e-9


def test_kill_switch_skips_eval_aux_batch():
    selfplay_buffer, eval_aux_buffer = _make_buffers()
    batch = _make_batch(source=SOURCE_EVAL_AUX, model_step=100)
    stats = IngestStats()

    added_examples, added_games = _route_batch_into_replay(
        batch=batch,
        cycle=150,
        args=_make_args(eval_aux_enabled=False),
        device='cpu',
        selfplay_buffer=selfplay_buffer,
        eval_aux_buffer=eval_aux_buffer,
        stats=stats,
    )

    assert added_examples == 0
    assert added_games == 0
    assert len(eval_aux_buffer) == 0
    assert stats.eval_examples_skipped_disabled == batch.n_examples


def test_stale_eval_aux_batch_is_filtered():
    selfplay_buffer, eval_aux_buffer = _make_buffers()
    batch = _make_batch(source=SOURCE_EVAL_AUX, model_step=10)
    stats = IngestStats()

    added_examples, _ = _route_batch_into_replay(
        batch=batch,
        cycle=1_000,
        args=_make_args(eval_aux_max_model_lag=50),
        device='cpu',
        selfplay_buffer=selfplay_buffer,
        eval_aux_buffer=eval_aux_buffer,
        stats=stats,
    )

    assert added_examples == 0
    assert len(eval_aux_buffer) == 0
    assert stats.eval_examples_skipped_stale == batch.n_examples


def test_eval_aux_downweight_drops_some_examples():
    selfplay_buffer, eval_aux_buffer = _make_buffers()
    batch = _make_batch(source=SOURCE_EVAL_AUX, model_step=0, n=1024)
    stats = IngestStats()

    torch.manual_seed(0)
    added_examples, _ = _route_batch_into_replay(
        batch=batch,
        cycle=1,
        args=_make_args(eval_aux_lag_half_life=1, eval_aux_min_keep_weight=0.0),
        device='cpu',
        selfplay_buffer=selfplay_buffer,
        eval_aux_buffer=eval_aux_buffer,
        stats=stats,
    )

    assert 300 <= added_examples <= 700
    assert stats.eval_examples_dropped_downweight == batch.n_examples - added_examples
    assert len(eval_aux_buffer) == added_examples


def test_selfplay_batch_routes_to_selfplay_buffer():
    selfplay_buffer, eval_aux_buffer = _make_buffers()
    batch = _make_batch(source=SOURCE_SELFPLAY, model_step=100)
    stats = IngestStats()

    added_examples, added_games = _route_batch_into_replay(
        batch=batch,
        cycle=120,
        args=_make_args(),
        device='cpu',
        selfplay_buffer=selfplay_buffer,
        eval_aux_buffer=eval_aux_buffer,
        stats=stats,
    )

    assert added_examples == batch.n_examples
    assert added_games == 2
    assert len(selfplay_buffer) == batch.n_examples
    assert stats.selfplay_examples == batch.n_examples
    assert len(eval_aux_buffer) == 0


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 16)
        self.policy_head = nn.Linear(16, 7)
        self.value_head = nn.Sequential(nn.Linear(16, 1), nn.Tanh())
        self.belief_head = nn.Linear(16, 28 * 3)
        self.has_belief_head = True

    def forward(self, tokens, mask, hand_indices, hand_mask):
        pooled = tokens.float().mean(dim=(1, 2), keepdim=True)
        h = torch.tanh(self.encoder(pooled).squeeze(1))
        policy = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        belief = self.belief_head(h).view(-1, 28, 3)
        return policy, value, belief


def test_source_aware_training_runs_with_eval_aux_policy_zero():
    selfplay_buffer, eval_aux_buffer = _make_buffers()
    selfplay_buffer.add_batch(_make_batch(source=SOURCE_SELFPLAY, model_step=100, n=64))
    eval_aux_buffer.add_batch(_make_batch(source=SOURCE_EVAL_AUX, model_step=98, n=64))

    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_n_steps_source_aware(
        model,
        optimizer,
        selfplay_buffer,
        eval_aux_buffer,
        n_steps=3,
        batch_size=16,
        eval_aux_batch_fraction=0.5,
        selfplay_weights=TrainWeights(policy=1.0, value=1.0, belief=0.5),
        eval_aux_weights=TrainWeights(policy=0.0, value=1.0, belief=0.5),
    )

    assert metrics['eval_aux_examples_per_step'] > 0
    assert metrics['selfplay_examples_per_step'] > 0
    assert 'eval_aux_value_loss' in metrics
    assert metrics['objective_loss'] >= 0.0
