# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Cooperative stopping and complete random state at optimizer boundaries."""
import math
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch


class StopRequest:
    """Signals request a checkpoint; they never interrupt an optimizer write.

    SIGKILL cannot be handled. A stop file remains in place until its owner
    removes it, so a restart cannot accidentally ignore a persistent request.
    """
    def __init__(self, stop_file=None, seconds=None):
        self.path = Path(stop_file) if stop_file else None
        self.deadline = time.monotonic() + seconds if seconds is not None else None
        self.reason = None
        self.handlers = {}

    def _signal(self, number, frame):
        self.reason = self.reason or signal.Signals(number).name

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.handlers[number] = signal.signal(number, self._signal)
        return self

    def __exit__(self, *exc):
        for number, handler in self.handlers.items():
            signal.signal(number, handler)

    def requested(self):
        if self.path is not None and self.path.exists():
            self.reason = self.reason or 'stop-file'
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.reason = self.reason or 'time-budget'
        return self.reason is not None


def capture_rng(device):
    numpy = np.random.get_state()
    state = {'python': random.getstate(),
             'numpy': (numpy[0], numpy[1].tolist(), *numpy[2:]),
             'torch': torch.get_rng_state()}
    if torch.device(device).type == 'mps':
        state['mps'] = torch.mps.get_rng_state()
    if torch.device(device).type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'])
    if 'mps' in state:
        torch.mps.set_rng_state(state['mps'])
    if 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])


def learning_rate(schedule, step):
    """One-based update rate. The fixed horizon is part of resume identity."""
    if schedule['kind'] == 'constant':
        return schedule['peak']
    fraction = min(1., max(0., (step - 1) / max(1, schedule['steps'] - 1)))
    return schedule['minimum'] + .5 * (schedule['peak'] - schedule['minimum']) * (1 + math.cos(math.pi * fraction))
