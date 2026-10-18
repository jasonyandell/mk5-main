# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
# STRATEGY-FUSION FENCE.
# The Demon: a fenced tournament opponent; measuring stick only.
"""The unchanged pinned Demon cardplay with the ladder's common bid30 auction."""
from dataclasses import replace
from hashlib import sha256
import re

from .gus_seat import LADDER, ROOT, source_hashes
from .participants import DEMON_BAIT40_REUSE_N100


def participant(commit):
    pinned = DEMON_BAIT40_REUSE_N100
    sources = source_hashes()
    for name in ('tournament/demon_bid30.py', 'tournament/demon_fenced_seat.py',
                 'tournament/seats.py', 'tournament/faceoff.py'):
        sources[name] = sha256((ROOT / name).read_bytes()).hexdigest()
    return replace(pinned, id=pinned.id + '-bid30',
        pin={**pinned.pin, 'bid30_source_commit': commit, 'source_files': sources},
        config={**pinned.config, 'ladders': [LADDER], 'common_bidder': 'Bid30Bidder',
                'seed_key': 'paired-deal-v1'},
        needs=(*pinned.needs, 'gus-observer'),
        note='Pinned 100-world root-reuse cardplay; fixed common Bid30Bidder replaces the '
             'native auction in this separate ladder. Fenced opponent; no values used as training targets.')


def game_key(game):
    """Couple a fixed opponent's draws across candidate names and claim dates."""
    match = re.fullmatch(r'.*-([0-9a-f]{16})/(\d+)/([01])', game)
    if not match:
        raise ValueError('paired Demon seed needs a tournament game identity')
    seed, pair, team = match.groups()
    return f'{LADDER}/paired-deal-v1/{seed}/{int(pair)}/{team}'
