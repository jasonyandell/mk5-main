# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Content identity for dense inference weights, independent of checkpoint history.

Callers must include architecture, tokenizer and inference mode in metadata and
pin runtime/source separately. Equal weights alone do not prove equal behavior
under different execution code, hardware, batching or random streams.
"""
import hashlib
import json
import sys

import torch


def state_identity(state, metadata):
    """Hash names, dtypes, shapes and exact tensor bytes without mutating state."""
    if not state or not all(isinstance(name, str) for name in state):
        raise ValueError('nonempty state with string names required')
    tensors = {}
    for name, value in sorted(state.items()):
        if (not isinstance(name, str) or not isinstance(value, torch.Tensor)
                or value.layout != torch.strided or value.is_quantized):
            raise ValueError('named dense unquantized tensors required')
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        tensors[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                         'sha256': hashlib.sha256(raw.numpy().tobytes()).hexdigest()}
    payload = {'schema': 'lem-inference-state-v1', 'byteorder': sys.byteorder,
               'metadata': metadata, 'tensors': tensors}
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    # Return a normalized snapshot, not references to mutable caller metadata.
    return {**json.loads(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}
