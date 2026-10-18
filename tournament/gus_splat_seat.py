# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Joint Gus checkpoints in the fixed-contract ladder; original pins stay valid."""
import re

from .gus_seat import LADDER, ROOT, SOURCE_FILES, verify_sources
from .participants import Participant


def source_hashes():
    from hashlib import sha256
    paths = (*SOURCE_FILES, 'tournament/gus_splat_seat.py',
             'lem/observer/splat/model.py', 'lem/observer/splat/play.py',
             'tournament/seats.py', 'tournament/faceoff.py')
    return {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def participant(name, revision, sha256, commit, head='tile'):
    if (not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name)
            or not re.fullmatch(r'[0-9a-f]{40}', revision)
            or not re.fullmatch(r'[0-9a-f]{64}', sha256) or head not in ('tile', 'splat')):
        raise ValueError('named joint release, immutable revision, checksum and valid head required')
    return Participant(f'gus-{name}-bid30-{head}-{sha256[:8]}', 'gus-splat',
        {'repo': 'jasonyandell/gus', 'revision': revision, 'file': 'gus.pt',
         'sha256': sha256, 'source_commit': commit, 'source_files': source_hashes()},
        {'ladders': [LADDER], 'bidder': 'Bid30Bidder', 'declarations': list(range(7)),
         'device': 'cpu', 'dtype': 'float32', 'torch_threads': 1, 'head': head,
         'budget': 'one forward pass; legal argmax; lowest tile id breaks exact ties'},
        ('torch', 'gus-observer'),
        'Frozen joint tile/splat model; own original seven and public bid30/pip/history only. '
        'Two copies per partnership; head choice is explicit; common Bid30Bidder.')


def make_seat(p, artifacts):
    from hashlib import sha256
    import torch
    from arena.bidders import Bid30Bidder
    from lem.observer.splat.play import SplatPlay
    from .seats import ArenaSeat

    verify_sources(p.pin)
    checkpoint = artifacts / 'gus.pt'
    if sha256(checkpoint.read_bytes()).hexdigest() != p.pin['sha256']:
        raise RuntimeError('cached joint Gus checkpoint differs from immutable pin')
    if (p.config['device'], p.config['dtype'], p.config['torch_threads']) != ('cpu', 'float32', 1):
        raise ValueError('joint Gus tournament inference requires CPU float32, one thread')
    torch.set_num_threads(1)
    return ArenaSeat(Bid30Bidder(), SplatPlay(checkpoint, 'cpu', p.config['head']))
