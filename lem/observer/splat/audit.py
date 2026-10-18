# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Verify the completed formation trajectory and expose one literal test input/output."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data import label, VERSION
from ..population import atomic_json, load_model
from .data import Corpus, digest
from .model import TASK
from .play import SplatPlay


def run(root, baseline):
    root=Path(root);training=root/'training'
    report=json.loads((training/'report.json').read_text())
    args=report['arguments'];steps=report['steps'];interval=args['eval_every']
    expected=sorted({s for s in (250,500,1000,steps,*range(interval,steps+1,interval)) if s<=steps})
    actual=sorted(int(p.stem.split('-')[1]) for p in training.glob('step-*.pt'))
    if actual!=expected:raise ValueError('formation checkpoint gaps or unexpected steps')
    initial=torch.load(training/'initial.pt',weights_only=True,map_location='cpu')
    baseline_model,_=load_model(baseline,'cpu')
    checks=[]
    for step in [0,*expected]:
        p=training/('initial.pt' if step==0 else f'step-{step:06d}.pt')
        saved=torch.load(p,weights_only=True,map_location='cpu')
        if saved['step']!=step or saved['task']!=TASK or saved['tokenizer']!=VERSION:
            raise ValueError('checkpoint identity mismatch')
        if saved['data_sha256']!=report['data_sha256'] or saved['config']!=report['config']:
            raise ValueError('checkpoint data/model changed')
        if not all(torch.isfinite(v).all() for v in saved['model'].values()):
            raise ValueError('nonfinite saved parameters')
        if step and (not saved['optimizer']['state'] or torch.equal(saved['sampler_state'],initial['sampler_state'])):
            raise ValueError('missing optimizer or unchanged sampler')
        checks.append({'step':step,'sha256':digest(p),'bytes':p.stat().st_size})
    best=torch.load(training/'best.pt',weights_only=True,map_location='cpu')
    minimum=min(report['history'],key=lambda r:r['validation']['joint_loss'])
    if best['step']!=minimum['step'] or best['step']!=report['best_step']:
        raise ValueError('best selection differs from registered criterion')
    if digest(training/'latest.pt')!=digest(training/f'step-{steps:06d}.pt'):
        # torch serialization container names can differ: compare tensors/state instead.
        final=torch.load(training/'latest.pt',weights_only=True,map_location='cpu')
        if final['step']!=steps or any(not torch.equal(v,final['model'][k]) for k,v in saved['model'].items()):
            raise ValueError('final model differs from final formation checkpoint')
    c=Corpus(args['cache'])
    # Fixed selection: first test view with a labelled own discretionary choice.
    example=None
    for j,shard in enumerate(c.shards):
        for row in np.flatnonzero(shard['split']==2):
            choices=np.flatnonzero((shard['lookup'][row]>0).sum(-1)>1)
            if not len(choices):continue
            own=int(choices[0]);position=int(shard['own_positions'][row,own])
            lookup=shard['lookup'][row,own];lookup=lookup[lookup>0]
            ids=shard['candidates'][lookup].tolist()
            tokens=shard['x'][row,:position+1].astype(np.int64).tolist()
            player=SplatPlay(training/'best.pt','cpu')
            with torch.inference_mode():
                tile,splat=player.model(torch.tensor([tokens]),torch.tensor([[position]]))
                p=splat[0,0].softmax(-1);base=baseline_model(torch.tensor([tokens]))[0,-1]
            make=p[:,30:].sum(-1);sign=1 if (tokens[2]-28)%2==0 else -1
            example={'selection':'first source, first test viewer with a labelled own choice; first such choice',
                'seed':int(shard['seed'][row//4]),'viewer':int(row%4),'public_plays':(position-13)//2,
                'tokens':tokens,'decoded_tokens':[label(t) for t in tokens],
                'legal_tile_ids':ids,'predicted_points_histograms':{str(i):p[i].tolist() for i in ids},
                'predicted_make_probability':{label(i):float(make[i]) for i in ids},
                'teacher_make_probability':{label(i):float(shard['histograms'][li,30:].sum()) for i,li in zip(ids,lookup)},
                'splat_selected_tile':label(max(ids,key=lambda t:float(sign*make[t]))),
                'joint_tile_selected_tile':label(max(ids,key=lambda t:float(tile[0,-1,t]))),
                'g0_selected_tile':label(max(ids,key=lambda t:float(base[t])))}
            break
        if example is not None:break
    atomic_json(root/'example.json',example)
    audit={'task':TASK,'steps':steps,'formation_checkpoints_including_initial':len(checks),
        'best_step':best['step'],'all_parameters_finite':True,'optimizer_and_sampler_preserved':True,
        'validation_selection_verified':True,'source_order_input_audit':'input-audit.json',
        'data_sha256':c.sha256,'checkpoints':checks,'best_sha256':digest(training/'best.pt'),
        'final_sha256':digest(training/'latest.pt')}
    atomic_json(root/'trajectory-audit.json',audit);print(audit,flush=True)
    return audit


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--baseline',required=True)
    torch.set_num_threads(2)
    a=p.parse_args();run(a.root,a.baseline)
