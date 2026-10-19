# SPDX-FileCopyrightText: 2026 Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""A Monte Carlo value baseline on frozen, clean information-state features.

The head predicts undiscounted terminal success. It never supplies rewards or
changes the actor's inputs. Old predictions precede fitting to the current
rollouts; actor advantages stay fixed throughout the clipped PPO update.
"""
import math

import numpy as np
import torch
from torch import nn


@torch.inference_mode()
def actor_features(reference, x, length, batch, device):
    """Only the original causal token prefix enters the frozen reference."""
    gathered = []
    for start in range(0, len(x), batch):
        tokens = torch.as_tensor(x[start:start+batch], device=device)
        position = torch.as_tensor(length[start:start+batch], device=device)-1
        rows = torch.arange(len(tokens), device=device)
        result = {}
        def capture(module, arguments, output):
            result['features'] = output[rows, position].detach()
        handle = reference.backbone.norm.register_forward_hook(capture)
        try:
            reference.backbone(tokens)
        finally:
            handle.remove()
        gathered.append(result['features'].cpu().numpy())
    features = np.concatenate(gathered)
    if not np.isfinite(features).all():
        raise ValueError('nonfinite value features')
    return features


def monte_carlo_advantages(reward, old_value):
    """Complete binary returns minus pre-update values; batch normalization.

    Seven equally weighted decisions per hand give the same denominator as the
    existing actor loss. No within-root centering or bootstrapped target is used.
    Normalization is a finite-sample PPO convention, not an unbiasedness claim.
    """
    reward = np.asarray(reward)
    value = np.asarray(old_value, dtype=np.float32)
    if reward.shape != value.shape or reward.ndim != 1 or not len(reward):
        raise ValueError('one reward and old value per decision required')
    if not np.isin(reward, (-1, 1)).all() or not np.isfinite(value).all():
        raise ValueError('invalid verified reward or old value')
    returns = ((reward+1)/2).astype(np.float32)
    raw = returns-value
    mean = float(raw.mean(dtype=np.float64))
    std = float(raw.astype(np.float64).std())
    normalized = ((raw.astype(np.float64)-mean)/max(std, 1e-8)).astype(np.float32)
    return returns, raw, normalized, {'raw_mean': mean, 'raw_std': std,
                                    'zero_fraction': float(np.mean(normalized == 0))}


class ValueBaseline:
    """Small separate critic; the shared KL-reference representation is frozen."""
    def __init__(self, width, recipe, device):
        if (set(recipe) != {'representation', 'estimator', 'lr', 'steps', 'clip_norm'}
            or recipe['representation'] != 'frozen_reference'
            or recipe['estimator'] != 'monte_carlo_batch_normalized'
            or type(recipe['steps']) is not int or recipe['steps'] < 1
            or any(not math.isfinite(recipe[k]) or recipe[k] <= 0 for k in ('lr', 'clip_norm'))):
            raise ValueError('invalid declared value-baseline recipe')
        self.recipe, self.device = recipe, device
        self.head = nn.Linear(width, 1).to(device)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, .5)
        self.optimizer = torch.optim.Adam(self.head.parameters(), lr=recipe['lr'])
        self.steps = 0

    def state(self):
        return {'model': self.head.state_dict(), 'optimizer': self.optimizer.state_dict(), 'steps': self.steps}

    def restore(self, state):
        if set(state) != {'model', 'optimizer', 'steps'}:
            raise ValueError('invalid value-baseline checkpoint')
        self.head.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.steps = state['steps']

    @torch.no_grad()
    def attach(self, data, reference, batch):
        features = actor_features(reference, data['x'], data['length'], batch, self.device)
        values = self.head(torch.as_tensor(features, device=self.device)).squeeze(-1).cpu().numpy()
        returns, raw, advantage, stats = monte_carlo_advantages(data['reward'], values)
        data.update(group_advantage=data['advantage'].copy(), value_features=features,
                    old_value=values, returns=returns, raw_advantage=raw, advantage=advantage)
        return stats

    def update(self, data):
        features = torch.as_tensor(data['value_features'], device=self.device)
        targets = torch.as_tensor(data['returns'], device=self.device)
        with torch.no_grad():
            before = self.head(features).squeeze(-1)
            old = torch.as_tensor(data['old_value'], device=self.device)
            if not torch.equal(before, old):
                raise ValueError('value baseline changed after advantage capture')
            loss_before = float(((before-targets)**2).mean())
        for _ in range(self.recipe['steps']):
            self.optimizer.zero_grad(set_to_none=True)
            prediction = self.head(features).squeeze(-1)
            loss = ((prediction-targets)**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), self.recipe['clip_norm'], error_if_nonfinite=True)
            self.optimizer.step()
            self.steps += 1
        with torch.no_grad():
            after = self.head(features).squeeze(-1)
            if not torch.isfinite(after).all():
                raise ValueError('nonfinite fitted value')
            loss_after = float(((after-targets)**2).mean())
        return {'mse_before': loss_before, 'mse_after': loss_after, 'optimizer_steps': self.steps,
                'parameters': sum(p.numel() for p in self.head.parameters())}, after.cpu().numpy()
