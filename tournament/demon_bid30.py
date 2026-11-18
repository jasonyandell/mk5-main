# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
# STRATEGY-FUSION FENCE.
# The Demon: a fenced tournament opponent; measuring stick only.
"""Pinned Demon cardplay under the common Bid30Bidder and paired deal seeds."""
from dataclasses import replace
from hashlib import sha256
import re
import subprocess

from .gus_seat import LADDER, ROOT, source_hashes
from .participants import DEMON_BAIT40_REUSE_N100


def adapter_hashes():
    sources = source_hashes()
    for name in ('tournament/demon_bid30.py', 'tournament/demon_fenced_seat.py',
                 'tournament/seats.py', 'tournament/faceoff.py'):
        sources[name] = sha256((ROOT / name).read_bytes()).hexdigest()
    return sources


def participant(commit):
    pinned = DEMON_BAIT40_REUSE_N100
    return replace(pinned, id=pinned.id + '-bid30-' + commit[:8],
        pin={**pinned.pin, 'bid30_source_commit': commit, 'source_files': adapter_hashes()},
        config={**pinned.config, 'ladders': [LADDER], 'common_bidder': 'Bid30Bidder',
                'seed_key': 'paired-deal-v1'},
        needs=(*pinned.needs, 'gus-observer'),
        note='Pinned 100-world root-reuse cardplay; fixed common Bid30Bidder replaces the '
             'native auction in this separate ladder. Fenced opponent; no values used as training targets.')


def reader(commit):
    """The public-play reader uses the same bidder, referee and seed identity as the control."""
    baseline = participant(commit)
    native = 'e946b151ca6f3cc1fdd1f9356ea3268bfc27695a'
    trees = {'silas': '4bdc2bec03573daae709442b4c65019d2be2c3e5',
             'scenario/core': '93db4d7c9ba188f6f46b84f37f59edd805d7859a',
             'tournament/demon': '4c558df127be9ceec1ed80439276ce47e08e0abd'}
    return replace(baseline, id=f'demon-drop30-reader-n80-{native[:8]}-{commit[:8]}',
        pin={**baseline.pin, 'commit': native, 'trees': trees},
        config={**baseline.config, 'worlds': 80, 'play_model': 'drop30-reader-v1'},
        needs=(*baseline.needs, 'demon-drop30-reader-v1'),
        note='Fenced drop-30 opponent. Sample hands using the common declaration and public '
             'tile choices, then solve up to 80 weighted draws. Score complete games to seven.')


def source_commit():
    """Pin the last change to the adapter or its dependencies, including the common bidder."""
    files = tuple(adapter_hashes())
    commit = subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', *files],
                                     cwd=ROOT, text=True).strip()
    if subprocess.run(['git', 'diff', '--quiet', commit, '--', *files], cwd=ROOT).returncode:
        raise RuntimeError('commit the bid30 sources before pinning a player')
    return commit


def game_key(game):
    """Couple a fixed opponent's draws across candidate names and claim dates."""
    match = re.fullmatch(r'.*-([0-9a-f]{16})/(\d+)/([01])', game)
    if not match:
        raise ValueError('paired Demon seed needs a tournament game identity')
    seed, pair, team = match.groups()
    return f'{LADDER}/paired-deal-v1/{seed}/{int(pair)}/{team}'
