# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""One fixed duplicate pair using the existing arena and independent replay audit."""
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from arena.bidders import Bid30Bidder
from arena.engine import ArenaConfig
from arena.match import run_match
from arena.walt_play import parse_walt_spec
from ..h2h import paired_summary, verify_replays
from ..play import ObserverPlay
from ..population import atomic_json
from .play import SplatPlay


def run(args):
    torch.set_num_threads(1)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    candidate=SplatPlay(args.candidate,'cpu',args.head)
    if args.opponent.startswith('walt:'):
        opponent=parse_walt_spec(args.opponent[5:])
        identity={'policy':args.opponent}
        for part in args.opponent[5:].split(','):
            if part.startswith('F='):
                identity['field_sha256']=hashlib.sha256(Path(part[2:]).read_bytes()).hexdigest()
    else:
        opponent=ObserverPlay(args.opponent,'cpu');identity=opponent.identity
    excluded={tuple(map(int,row)) for row in np.load(args.exclude_hands,allow_pickle=False)}
    atomic_json(out/'manifest.json',{'candidate':candidate.identity,'opponent':identity,
        'seed':args.seed,'games':2,'contract':'common Bid30Bidder; pip declaration; first-to-seven',
        'thinking':'one forward pass or pinned deterministic Walt budget; CPU float32',
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'exclusion_sha256':hashlib.sha256(Path(args.exclude_hands).read_bytes()).hexdigest()})
    result=run_match(bid_a=Bid30Bidder(),bid_b=Bid30Bidder(),play_a=candidate,play_b=opponent,
        n_games=2,cfg=ArenaConfig(base_seed=args.seed),fast_batching=True,
        label_a='gus-splat',label_b=args.opponent,verbose=True)
    with (out/'games.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as f:
            for g in result.games:f.write((json.dumps(asdict(g),sort_keys=True)+'\n').encode())
    report={**paired_summary(result.games),'audit':verify_replays(result.games,excluded),'elapsed_s':result.elapsed_s}
    atomic_json(out/'report.json',report);print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',required=True);p.add_argument('--opponent',required=True)
    p.add_argument('--head',choices=('splat','tile'),default='splat')
    p.add_argument('--exclude-hands',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',required=True)
    run(p.parse_args())
