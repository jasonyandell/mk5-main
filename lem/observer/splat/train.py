# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Joint pretraining or explicit continuation; cooperative exact recovery."""
import argparse
from dataclasses import asdict
import json
import random
from pathlib import Path
import subprocess
import time
import uuid

import numpy as np
import torch

from ..data import QUERIES, VERSION
from ..model import Config
from ..population import atomic_json, atomic_save, digest
from .data import Corpus
from .model import SplatObserver, TASK, joint_loss
from .recovery import StopRequest, capture_rng, restore_rng, learning_rate


class DeviceCorpus:
    """Compact immutable labels on the accelerator; no Python examples per step."""
    def __init__(self, corpus, device):
        self.corpus, self.device = corpus, device
        self.arrays = {}
        for key in ('x','targets','own_positions'):
            a = np.concatenate([s[key] for s in corpus.shards])
            self.arrays[key] = torch.tensor(a,dtype=torch.long,device=device)
        offsets = np.r_[0,np.cumsum([len(s['histograms']) for s in corpus.shards])]
        lookup = np.concatenate([np.asarray(s['lookup'],np.int64)+offsets[j] for j,s in enumerate(corpus.shards)])
        self.lookup = torch.tensor(lookup,dtype=torch.long,device=device)
        for key in ('histograms','candidates'):
            a = np.concatenate([s[key] for s in corpus.shards])
            self.arrays[key] = torch.tensor(a,device=device)
        self.splits = {k:np.flatnonzero(corpus.split==i) for i,k in enumerate(('train','validation','test'))}
        if any(not len(v) for v in self.splits.values()):
            raise ValueError('need deals in every split')

    def batch(self, indices):
        ix = torch.as_tensor(indices,dtype=torch.long,device=self.device)
        return {**{k:self.arrays[k][ix] for k in ('x','targets','own_positions')},
                **{k:self.arrays[k][self.lookup[ix]] for k in ('histograms','candidates')}}


def diagnostics(corpus, indices):
    """Referee labels for metrics only. None are passed to the backbone."""
    result={k:np.empty((len(indices),28),np.uint32 if k!='settled' else bool) for k in ('possible','legal','settled')}
    for j,s in enumerate(corpus.shards):
        mask=(indices>=corpus.offsets[j])&(indices<corpus.offsets[j+1])
        ix=indices[mask]-corpus.offsets[j]
        result['possible'][mask]=s['possible'][ix]
        for k in ('legal','settled'):
            result[k][mask]=s[k][ix//4]
    return result


@torch.inference_mode()
def evaluate(model, data, indices, batch_size=256):
    model.eval()
    totals={}
    per_deal=[]
    for lo in range(0,len(indices),batch_size):
        ix=indices[lo:lo+batch_size]
        b=data.batch(ix)
        tile,splat=model(b['x'],b['own_positions'])
        p=tile[:,QUERIES].softmax(-1).cpu().numpy()
        y=b['targets'].cpu().numpy()
        pos=((b['own_positions']-13)//2).cpu().numpy()
        nll=-np.log(np.take_along_axis(p,y[...,None],-1)[...,0].clip(1e-30))
        own=np.zeros(y.shape,bool);own[np.arange(len(y))[:,None],pos]=True
        diag=diagnostics(data.corpus,ix)
        legal_count=np.bitwise_count(diag['legal'])
        possible=(diag['possible'][...,None]&(1<<np.arange(28)))!=0
        impossible=(p*~possible).sum(-1)
        groups={'all':np.ones_like(own),'own':own,'others':~own,
                'own_consequential':own&(legal_count>1)&~diag['settled']}
        def add(name, values, mask):
            a=totals.setdefault(name,[0.,0])
            a[0]+=float(values[mask].sum());a[1]+=int(mask.sum())
        for name,mask in groups.items():
            add(f'tile_{name}_nll',nll,mask)
            add(f'tile_{name}_top1',p.argmax(-1)==y,mask)
            add(f'tile_{name}_impossible_mass',impossible,mask)
        q=splat.gather(-2,b['candidates'][...,None].expand(*b['candidates'].shape,43)).softmax(-1).cpu().numpy()
        target=b['histograms'].cpu().numpy()
        mask=target.sum(-1)>0; count=mask.sum(-1); have=count>0
        ce=-(target*np.log(q.clip(1e-30))).sum(-1)
        entropy=-(target*np.log(target.clip(1e-30))).sum(-1)
        pc=(ce*mask).sum(-1)/count.clip(1)
        pe=(entropy*mask).sum(-1)/count.clip(1)
        truth=target[...,30:].sum(-1); estimate=q[...,30:].sum(-1)
        sign=np.where((b['x'][:,2].cpu().numpy()-28)%2==0,1.,-1.)[:,None,None]
        scores=np.where(mask,estimate*sign,-np.inf)
        chosen=scores.argmax(-1)
        chosen_value=np.take_along_axis(truth*sign,chosen[...,None],-1)[...,0]
        best=np.max(np.where(mask,truth*sign,-np.inf),axis=-1)
        regret=np.where(have,best-chosen_value,0.)
        uniform_regret=np.where(have,best-(truth*sign*mask).sum(-1)/count.clip(1),0.)
        # Teacher ties are exact up to float32 cache transport.
        for name,position_mask in {'all':have,'consequential':have&~np.take_along_axis(diag['settled'],pos,1)}.items():
            add(f'splat_{name}_ce',pc,position_mask)
            add(f'splat_{name}_entropy',pe,position_mask)
            add(f'splat_{name}_kl',pc-pe,position_mask)
            add(f'splat_{name}_teacher_regret',regret,position_mask)
            add(f'splat_{name}_uniform_regret',uniform_regret,position_mask)
            add(f'splat_{name}_tie_aware_accuracy',regret<=1e-6,position_mask)
            add(f'splat_{name}_make_mae',(np.abs(truth-estimate)*mask).sum(-1)/count.clip(1),position_mask)
        per_deal.extend(np.column_stack((ix,regret.sum(-1),have.sum(-1))).tolist())
    result={k:v/n for k,(v,n) in totals.items() if n}
    result['joint_loss']=result['tile_all_nll']+result['splat_all_ce']
    result['views']=len(indices)
    result['labelled_own_positions']=totals['splat_all_ce'][1]
    return result


def save(path,model,optimizer,sampler,step,best,history,args,data,schedule,identity):
    atomic_save(path,{'task':TASK,'tokenizer':VERSION,'config':asdict(model.config),
        'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'sampler_state':sampler.get_state(),'torch_rng_state':torch.get_rng_state(),
        'rng_state':capture_rng(args.device),'schedule':schedule,'run_identity':identity,
        'step':step,'best_loss':best,'history':history,'arguments':vars(args),
        'data_sha256':data.corpus.sha256})


DEFAULTS = dict(schedule='constant',min_lr=.0001,fork_from=None,
                stop_file=None,stop_after_seconds=None)


def run(args):
    for key,value in DEFAULTS.items():
        if not hasattr(args,key):setattr(args,key,value)
    if args.resume and args.fork_from:
        # The recorded fork source remains an argument when resuming that fork.
        if not (Path(args.out)/'latest.pt').exists():
            raise ValueError('resume requires the existing continuation checkpoint')
    if min(args.steps,args.eval_every,args.batch)<1 or args.lr<=0 or not 0<=args.min_lr<=args.lr:
        raise ValueError('positive budgets/rate and 0 <= min_lr <= lr required')
    if args.stop_after_seconds is not None and args.stop_after_seconds<=0:
        raise ValueError('stop-after-seconds must be positive')
    with StopRequest(args.stop_file,args.stop_after_seconds) as stop:
        return _run(args,stop)


def _run(args,stop):
    torch.set_num_threads(2);torch.manual_seed(args.seed)
    random.seed(args.seed);np.random.seed(args.seed)
    out=Path(args.out)
    if out.exists() and not args.resume:
        raise ValueError('run exists; choose new out or resume')
    out.mkdir(parents=True,exist_ok=True)
    corpus=Corpus(args.cache)
    data=DeviceCorpus(corpus,args.device)
    model=SplatObserver(Config(width=args.width,layers=2,heads=2)).to(args.device)
    optimizer=torch.optim.AdamW([
        {'params':[p for p in model.parameters() if p.ndim>=2],'weight_decay':.1},
        {'params':[p for p in model.parameters() if p.ndim<2],'weight_decay':0.}],lr=args.lr,betas=(.9,.95))
    sampler=torch.Generator().manual_seed(args.seed+1)
    schedule=dict(kind=args.schedule,peak=args.lr,minimum=args.min_lr,steps=args.steps)
    history=[];start=0;best=float('inf');saved=None;parent=None
    if args.resume or args.fork_from:
        source=out/'latest.pt' if args.resume else Path(args.fork_from)
        saved=torch.load(source,weights_only=True,map_location='cpu')
        if saved['task']!=TASK or saved['tokenizer']!=VERSION or saved['data_sha256']!=corpus.sha256 or saved['config']!=asdict(model.config):
            raise ValueError('resume/fork task/data/model mismatch')
        if args.resume:
            for k in ('seed','lr','batch','eval_every','device','width','schedule','min_lr','fork_from',
                      'wandb_project','wandb_entity','wandb_name'):
                if saved['arguments'].get(k,DEFAULTS.get(k))!=vars(args)[k]:
                    raise ValueError(f'resume settings differ: {k}')
            if args.schedule!='constant' and saved['arguments']['steps']!=args.steps:
                raise ValueError('resume settings differ: schedule horizon')
            start,best,history=saved['step'],saved['best_loss'],saved['history']
            if args.steps<start:raise ValueError('cannot rewind a training trajectory')
        else:
            parent=dict(path=str(source.resolve()),sha256=digest(source),step=saved['step'],
                        optimizer_and_sampler_inherited=True)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        sampler.set_state(saved['sampler_state'])
    about={'task':TASK,'tokenizer':VERSION,'data_sha256':corpus.sha256,'sources':corpus.manifests,
           'config':asdict(model.config),'parameters':sum(p.numel() for p in model.parameters()),
           'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
           'source_files':{str(p):digest(p) for p in (Path(__file__),Path(__file__).with_name('recovery.py'))},
           'split_deals':{k:len(v)//4 for k,v in data.splits.items()},
           'recipe':'tile CE + mean position/mean legal candidate soft histogram CE, weights 1:1',
           'selection':'minimum full validation joint loss; fixed update budget',
           'epoch_definition':'updates * batch / training_views; sampling with replacement',
           'arguments':vars(args),'schedule':schedule,'parent':parent}
    if args.resume:
        about=json.loads((out/'data.json').read_text())
        if about.get('parent') and digest(args.fork_from)!=about['parent']['sha256']:
            raise ValueError('fork source bytes changed')
        if (out/'report.json').exists():
            report=json.loads((out/'report.json').read_text())
            if report['steps']==args.steps:return report
    else:
        atomic_json(out/'data.json',about)
    tracker=None;identity=None
    receipt=out/'wandb-run.json'
    if args.wandb_project:
        import wandb
        identity=json.loads(receipt.read_text()) if args.resume and receipt.exists() else None
        if args.resume and saved.get('run_identity') and identity!=saved['run_identity']:
            raise ValueError('resume W&B identity mismatch')
        if identity is None:
            identity=dict(id=uuid.uuid4().hex[:8],project=args.wandb_project,entity=args.wandb_entity)
            atomic_json(receipt,identity)
        tracker=wandb.init(project=identity['project'],entity=identity['entity'],name=args.wandb_name,
            id=identity['id'],resume='allow' if args.resume else 'never',
            dir=str(out),config=about,tags=['gus','splat','pretraining','clean-neuron'])
        identity=dict(id=tracker.id,url=tracker.url,entity=tracker.entity,project=tracker.project)
        atomic_json(receipt,identity)
        tracker.define_metric('epoch_equivalent');tracker.define_metric('*',step_metric='epoch_equivalent')
        print('LIVE_RUN '+tracker.url,flush=True)
    # Tracking setup may use RNGs. The optimizer trajectory starts after setup.
    if saved is not None:
        if 'rng_state' in saved:restore_rng(saved['rng_state'])
        else:torch.set_rng_state(saved['torch_rng_state'])
    initial_rng=capture_rng(args.device)
    started=time.monotonic();step=start;paused=False
    def checkpoint(path):
        save(path,model,optimizer,sampler,step,best,history,args,data,schedule,identity)
    def pause():
        nonlocal paused
        checkpoint(out/'latest.pt')
        checkpoint(out/f'pause-{step:06d}.pt')
        result=dict(status='paused',step=step,planned_steps=args.steps,reason=stop.reason,
                    checkpoint='latest.pt',sha256=digest(out/'latest.pt'),
                    segment_seconds=time.monotonic()-started)
        atomic_json(out/'status.json',result)
        paused=True
        if tracker:tracker.summary.update({'training_status':'paused','paused_step':step})
        print(json.dumps(result),flush=True)
        return result
    try:
        if not args.resume:
            checkpoint(out/'initial.pt');checkpoint(out/'latest.pt')
        if stop.requested():return pause()
        if not (out/'initial.json').exists():
            initial=evaluate(model,data,data.splits['validation'])
            atomic_json(out/'initial.json',initial)
            # A continuation can keep its starting model if every update worsens it.
            if args.fork_from:
                best=initial['joint_loss'];checkpoint(out/'best.pt')
            print(json.dumps({'initial_validation':initial,'parameters':about['parameters']}),flush=True)
            restore_rng(initial_rng)
        if stop.requested():return pause()
        atomic_json(out/'status.json',dict(status='running',step=step,planned_steps=args.steps))
        train=torch.tensor(data.splits['train'],device=args.device)
        for step in range(start+1,args.steps+1):
            if stop.requested():
                step-=1
                return pause()
            model.train()
            lr=learning_rate(schedule,step)
            for group in optimizer.param_groups:group['lr']=lr
            ix=train[torch.randint(len(train),(args.batch,),generator=sampler).to(args.device)]
            b=data.batch(ix)
            tile,splat=model(b['x'],b['own_positions'])
            loss,tl,sl=joint_loss(tile[:,QUERIES],b['targets'],splat,b['histograms'],b['candidates'])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            early=step in (250,500,1000)
            if early or step%args.eval_every==0 or step==args.steps:
                validation=evaluate(model,data,data.splits['validation'])
                probe=evaluate(model,data,data.splits['train'][:1024])
                row={'step':step,'epoch_equivalent':step*args.batch/len(train),'learning_rate':lr,
                     'train_tile_ce':float(tl.detach().cpu()),'train_splat_ce':float(sl.detach().cpu()),
                     'validation':validation,'train_probe':probe,'segment_seconds':time.monotonic()-started}
                if not np.isfinite(validation['joint_loss']):
                    raise ValueError('nonfinite validation loss')
                history.append(row)
                improved=validation['joint_loss']<best;best=min(best,validation['joint_loss'])
                checkpoint(out/'latest.pt');checkpoint(out/f'step-{step:06d}.pt')
                if improved:checkpoint(out/'best.pt')
                atomic_json(out/'progress.json',row)
                print(json.dumps(row),flush=True)
                if tracker:
                    tracker.log({'training_step':step,'epoch_equivalent':row['epoch_equivalent'],
                         'learning_rate':lr,'train_tile_ce':row['train_tile_ce'],'train_splat_ce':row['train_splat_ce'],
                         **{f'validation_{k}':v for k,v in validation.items()},
                         **{f'train_probe_{k}':v for k,v in probe.items()}})
            if stop.requested():return pause()
        final=evaluate(model,data,data.splits['test'])
        saved=torch.load(out/'best.pt',weights_only=True,map_location='cpu')
        model.load_state_dict(saved['model'])
        test=evaluate(model,data,data.splits['test'])
        report={**about,'best_step':saved['step'],'steps':args.steps,'test':test,'final_test':final,'history':history,
                'seconds':time.monotonic()-started}
        atomic_json(out/'report.json',report)
        atomic_json(out/'status.json',dict(status='complete',step=step,planned_steps=args.steps))
        if tracker:tracker.summary.update({'training_status':'complete','best_step':saved['step'],**{f'best_test_{k}':v for k,v in test.items()}})
        return report
    finally:
        if tracker:tracker.finish(exit_code=0 if paused or (out/'report.json').exists() else 1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',action='append',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=200000)
    p.add_argument('--eval-every',type=int,default=2500)
    p.add_argument('--batch',type=int,default=64)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',choices=('mps','cuda','cpu'),default='mps')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--fork-from',help='new continuation from full checkpoint, retaining optimizer and sampler')
    p.add_argument('--schedule',choices=('constant','cosine'),default='constant')
    p.add_argument('--min-lr',type=float,default=.0001)
    p.add_argument('--stop-file',help='pause at an optimizer boundary while this path exists')
    p.add_argument('--stop-after-seconds',type=float,help='cooperative segment budget; use below the outer cpu-slot cap')
    p.add_argument('--wandb-project');p.add_argument('--wandb-entity');p.add_argument('--wandb-name')
    args=p.parse_args()
    if min(args.steps,args.eval_every,args.batch)<1:p.error('positive training budgets required')
    run(args)


if __name__=='__main__':main()
