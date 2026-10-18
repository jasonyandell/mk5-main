# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Paired whole-deal diagnostic comparison; this reused test is not confirmation."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from ..data import QUERIES
from ..population import atomic_json, load_model
from .data import Corpus
from .play import SplatPlay
from .train import DeviceCorpus, diagnostics


def interval(sums, numerator, denominator, draws=2000):
    """Ratio of totals, resampling physical deals (all four views together)."""
    n=sums[numerator];d=sums[denominator]
    rng=np.random.default_rng(42)
    estimates=[]
    for _ in range(draws):
        ix=rng.integers(len(n),size=len(n))
        estimates.append(float(n[ix].sum()/d[ix].sum()))
    return {'estimate':float(n.sum()/d.sum()),'ci95':np.quantile(estimates,[.025,.975]).tolist()}


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    c=Corpus(args.cache);d=DeviceCorpus(c,args.device)
    candidate=SplatPlay(args.candidate,args.device)
    baseline,saved=load_model(args.baseline,args.device)
    rows={}; indices=d.splits['test']
    for lo in range(0,len(indices),256):
        ix=indices[lo:lo+256];b=d.batch(ix)
        tile,splat=candidate.model(b['x'],b['own_positions'])
        base=baseline(b['x'])[:,QUERIES].softmax(-1).cpu().numpy()
        p=tile[:,QUERIES].softmax(-1).cpu().numpy();y=b['targets'].cpu().numpy()
        pos=((b['own_positions']-13)//2).cpu().numpy();batch=np.arange(len(ix))[:,None]
        diag=diagnostics(c,ix)
        own_legal=(np.take_along_axis(diag['legal'],pos,1)[...,None]&(1<<np.arange(28)))!=0
        consequential=(own_legal.sum(-1)>1)&~np.take_along_axis(diag['settled'],pos,1)
        targets=b['histograms'].cpu().numpy();tiles=b['candidates'].cpu().numpy()
        available=targets.sum(-1)>0
        consequential &= available.any(-1)
        q=splat.gather(-2,b['candidates'][...,None].expand(*b['candidates'].shape,43)).softmax(-1).cpu().numpy()
        truth=targets[...,30:].sum(-1);forecast=q[...,30:].sum(-1)
        sign=np.where((b['x'][:,2].cpu().numpy()-28)%2==0,1.,-1.)[:,None,None]
        best=np.max(np.where(available,truth*sign,-np.inf),axis=-1)
        def append(key,value):rows.setdefault(key,[]).append(value.astype(np.float64))
        append('tokens',np.full(len(ix),28));append('choices',consequential.sum(-1))
        for name,probability in (('g0',base),('joint_tile',p)):
            append(name+'_tile_hits',(probability.argmax(-1)==y).sum(-1))
            append(name+'_tile_nll',-np.log(np.take_along_axis(probability,y[...,None],-1)[...,0].clip(1e-30)).sum(-1))
            own_prob=probability[batch,pos]
            chosen=np.where(own_legal,own_prob,-np.inf).argmax(-1)
            append(name+'_own_legal_hits',((chosen==y[batch,pos])*consequential).sum(-1))
            candidate_scores=np.take_along_axis(own_prob,tiles,-1)
            choice=np.where(available,candidate_scores,-np.inf).argmax(-1)
            achieved=np.take_along_axis(truth*sign,choice[...,None],-1)[...,0]
            append(name+'_regret',np.where(consequential,best-achieved,0).sum(-1))
        choice=np.where(available,forecast*sign,-np.inf).argmax(-1)
        achieved=np.take_along_axis(truth*sign,choice[...,None],-1)[...,0]
        regret=np.where(consequential,best-achieved,0)
        append('splat_regret',regret.sum(-1))
        append('splat_ties',((regret<=1e-6)*consequential).sum(-1))
        count=available.sum(-1).clip(1)
        append('splat_make_mae',(((np.abs(truth-forecast)*available).sum(-1)/count)*consequential).sum(-1))
        kl=(targets*(np.log(targets.clip(1e-30))-np.log(q.clip(1e-30)))).sum(-1)
        append('splat_kl',(((kl*available).sum(-1)/count)*consequential).sum(-1))
        uniform=(truth*sign*available).sum(-1)/count
        append('uniform_regret',np.where(consequential,best-uniform,0).sum(-1))
    # Split keeps all four contiguous viewer rows. Assert rather than assume it.
    if not np.array_equal(indices.reshape(-1,4)%4,np.tile(np.arange(4),(len(indices)//4,1))):
        raise ValueError('test rows do not group into four physical-deal views')
    sums={k:np.concatenate(v).reshape(-1,4).sum(-1) for k,v in rows.items()}
    sums['delta_regret_g0']=sums['splat_regret']-sums['g0_regret']
    sums['delta_regret_joint_tile']=sums['splat_regret']-sums['joint_tile_regret']
    sums['delta_tile_hits']=sums['joint_tile_tile_hits']-sums['g0_tile_hits']
    np.savez_compressed(out/'per-deal.npz',**sums)
    stats={}
    for name in ('g0','joint_tile'):
        for metric,denom in (('tile_hits','tokens'),('tile_nll','tokens'),('own_legal_hits','choices'),('regret','choices')):
            stats[f'{name}_{metric}']=interval(sums,f'{name}_{metric}',denom)
    for key in ('splat_regret','splat_ties','splat_make_mae','splat_kl','uniform_regret','delta_regret_g0','delta_regret_joint_tile'):
        stats[key]=interval(sums,key,'choices')
    stats['delta_tile_hits']=interval(sums,'delta_tile_hits','tokens')
    report={'scope':'previously used test deals; comparative diagnostic, not fresh strength confirmation',
        'physical_deals':len(indices)//4,'views':len(indices),'consequential_choices':int(sums['choices'].sum()),
        'uncertainty':'2000 physical-deal percentile bootstrap draws, seed42; ratio of total sums',
        'candidate':candidate.identity,'baseline_sha256':hashlib.sha256(Path(args.baseline).read_bytes()).hexdigest(),
        'data_sha256':c.sha256,'metrics':stats}
    atomic_json(out/'report.json',report)
    print(report,flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',action='append',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--baseline',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',choices=('mps','cpu','cuda'),default='mps')
    run(p.parse_args())
