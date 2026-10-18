# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Publish a separate Gus splat revision and its allowlisted formation evidence."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import torch

from ..data import label, VERSION
from ..population import atomic_json
from .data import digest
from .model import TASK
from .play import SplatPlay

MODEL_REPO='jasonyandell/gus'
DATA_REPO='jasonyandell/texas-42-zeb-research-2026-10'
ARCHIVE='lem/gus-splat-pretraining-2026-10-18'
BRANCH='splat-v1'
TAG='splat-v1-20261018'


def publish_data(root, initial=False):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
    root=Path(root)
    files=list(sorted((root/'archive').glob('cache-*.tar.gz')))
    files += [root/f'cache-{i}'/'manifest.json' for i in range(7)]
    files += [root/'input-audit.json',root/'h2h-plan.json',root/'exclusions.json',root/'excluded-hands.npy']
    files += [root/'training'/name for name in ('data.json','initial.json','initial.pt','wandb-run.json')]
    if not initial:
        files += list(sorted((root/'training').glob('step-*.pt')))
        files += [root/'training'/name for name in ('best.pt','latest.pt','report.json','progress.json')]
        files += [root/'diagnostics'/name for name in ('report.json','per-deal.npz')]
        files += [root/'h2h'/name for name in ('attempts.json','summary.json')]
        for pair in sorted((root/'h2h').glob('*-pair-*')):
            files += [p for p in pair.iterdir() if p.is_file() and p.name in ('manifest.json','report.json','games.jsonl.gz')]
        files += list((root/'h2h').glob('*.log'))
        for name in ('training.log','progress.png','progress.pdf','example.json'):
            if (root/name).exists():files.append(root/name)
    if len(list((root/'archive').glob('cache-*.tar.gz')))!=7 or not all(p.is_file() for p in files):
        raise ValueError('incomplete evidence package')
    size=sum(p.stat().st_size for p in files)
    if size>2_000_000_000:raise ValueError('study exceeds the documented 2GB upload review threshold')
    for path in files:
        if path.suffix in ('.json','.log') and re.search(r'hf_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY',path.read_text()):
            raise ValueError('credential-shaped text in evidence')
    manifest={'files':{str(p.relative_to(root)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in files},
        'bytes':size,'license':'CC0-1.0','initial_only':initial,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    manifest_path=root/('initial-publication-manifest.json' if initial else 'publication-manifest.json')
    atomic_json(manifest_path,manifest);files.append(manifest_path)
    api=HfApi()
    receipt=root/('initial-publication.json' if initial else 'data-publication.json')
    if receipt.exists():return json.loads(receipt.read_text())
    # One commit avoids the per-file commit quota that interrupted the first pilot.
    commit=api.create_commit(DATA_REPO,repo_type='dataset',commit_message='Gus splat: '+('compiled lawful labels and untrained twin' if initial else 'complete formation, forecast comparison and short H2H'),
        operations=[CommitOperationAdd(path_in_repo=f'{ARCHIVE}/{p.relative_to(root)}',path_or_fileobj=str(p)) for p in files])
    atomic_json(receipt,{'revision':commit.oid,'bytes':size,'files':len(files),'verification':'pending'})
    verified=[]
    samples=[manifest_path,root/'input-audit.json',root/'training'/'initial.pt']
    if not initial:samples += [root/'training'/'best.pt',root/'diagnostics'/'report.json',root/'h2h'/'summary.json']
    for p in samples:
        remote=Path(hf_hub_download(DATA_REPO,f'{ARCHIVE}/{p.relative_to(root)}',repo_type='dataset',
                    revision=commit.oid,token=False,force_download=True,local_dir=root/'verify-data'))
        if digest(remote)!=digest(p):raise ValueError('fresh data download differs')
        verified.append(str(p.relative_to(root)))
    result={'revision':commit.oid,'bytes':size,'files':len(files),'anonymous_verified_samples':verified,
            'verification':'listed small evidence and checkpoint downloads; bulk cache archives separately checksummed',
            'url':f'https://huggingface.co/datasets/{DATA_REPO}/tree/{commit.oid}/{ARCHIVE}'}
    atomic_json(receipt,result);return result


def publish_model(root):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download, ModelCard
    from safetensors.torch import save_file,load_file
    root=Path(root);stage=root/'model-release';stage.mkdir(exist_ok=True)
    checkpoint=root/'training'/'best.pt'
    player=SplatPlay(checkpoint,'cpu');model=player.model
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    training=json.loads((root/'training'/'report.json').read_text())
    comparison=json.loads((root/'diagnostics'/'report.json').read_text())
    games=json.loads((root/'h2h'/'summary.json').read_text())
    archive=json.loads((root/'data-publication.json').read_text())
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    save_file(state,str(stage/'model.safetensors'),metadata={'format':'pt','name':'Gus','task':TASK})
    torch.save({k:saved[k] for k in ('task','config','model','step','tokenizer','data_sha256')},stage/'gus.pt')
    atomic_json(stage/'config.json',{'task':TASK,**saved['config']})
    atomic_json(stage/'tokenizer.json',{'encoding':VERSION,'input_tokens':[{'id':i,'label':label(i)} for i in range(46)],
        'context':68,'tile_outputs':28,'splat_output_shape':[28,43],
        'outcomes':'bidder-team points 0..42, marginal weighted lawful make30 continuation',
        'prefix':'BOS BID30 relative_bidder DECL HAND own_7_sorted_tiles PLAY; relative_actor/tile pairs; end at actor',
        'candidate_identity':'output axis only; never an input token'})
    atomic_json(stage/'provenance.json',{'source_commit':source,'training_source_commit':training['source_commit'],
        'source_checkpoint_sha256':digest(checkpoint),'training':training,'archive':archive,
        'parameters':sum(p.numel() for p in model.parameters()),'selected_step':saved['step']})
    atomic_json(stage/'evaluation.json',{'diagnostics':comparison,'h2h':games})
    example={'tokens':[42,45,28,38,43,0,2,5,9,14,20,27,44,28],'legal_tile_ids':[0,2,5,9,14,20,27]}
    torch.set_num_threads(2)
    with torch.inference_mode():
        tile,splat=model(torch.tensor([example['tokens']]),torch.tensor([[13]]))
        make=splat[0,0].softmax(-1)[:,30:].sum(-1)
    example.update(decoded_tokens=[label(t) for t in example['tokens']],
        expected_tile_logits=tile[0,-1].tolist(),expected_make_probabilities=make.tolist(),
        expected_legal_choice=max(example['legal_tile_ids'],key=lambda t:float(make[t])))
    atomic_json(stage/'example.json',example)
    atomic_json(stage/'export-verification.json',{'all_tensors_match_source':all(torch.equal(v,saved['model'][k]) for k,v in state.items()),
        'source_checkpoint_sha256':digest(checkpoint),'parameters':sum(v.numel() for v in state.values())})
    (stage/'LICENSE').write_text(Path('LICENSES/CC0-1.0.txt').read_text())
    card=f'''---
license: cc0-1.0
library_name: pytorch
model_name: Gus splat v1
tags: [texas-42, dominoes, nanogpt, causal-transformer, interpretability, custom-code]
---
# Gus: same clean view, richer consequences

185,652 parameters: the original two-layer, width-64, two-head causal backbone
plus a 28 × 43 outcome head. Fresh pretraining on 500,000 lawful recorded deals,
200,000 AdamW updates, approximately eight equivalent passes. Selected step:
{saved['step']}, by full-validation joint loss. This is a separately versioned
experiment; it does not replace the established baseline or assert promotion.

## Exactly what goes in and out

The original seven own tiles, fixed bid 30, public bidder/declaration, relative
actors and public play history. Same 46 tokens and 68 positions. No other hands,
Walt values, candidate query, evaluate token, legal list, or match score is an
input. The tile head predicts observed plays; at one's own actor position the
splat head predicts bidder-team points 0–42 separately for each candidate tile.
Only legal candidate rows with teacher labels were trained. Invalid candidate
rows are not meaningful forecasts.

The targets marginalize lawful Walt splats. They describe the named make-30
teacher and its random legal finish after settlement, not optimal points play.
They discard cross-action scenario pairing. For play, bidder partners maximize
probability of 30–42 points; defenders maximize its complement. Both apply the
same legal-output mask. This model does not bid or play Nello.

## Measurements

Full held-out diagnostic tile accuracy: {training['test']['tile_all_top1']:.2%}.
Consequential teacher-relative make-probability MAE:
{training['test']['splat_consequential_make_mae']:.4f}. The reused test set measures
forecast fit, not independent playing strength. `evaluation.json` contains
physical-deal intervals and all eight scheduled short faceoffs against G0/Zeb,
including any incomplete pair. A small pilot is not a general win-rate claim.
The neurons' meanings remain ungraded.

## Load the exact model

Use the source at [mk5 commit {source[:12]}](https://github.com/jasonyandell/mk5-main/tree/{source})
and the pinned Hub revision `{TAG}`:

```python
from huggingface_hub import hf_hub_download
from lem.observer.splat.play import SplatPlay
checkpoint = hf_hub_download("{MODEL_REPO}", "gus.pt", revision="{TAG}")
player = SplatPlay(checkpoint, device="cpu", head="splat")
# player.choose(arena_states, bid_values) returns legal hand-slot indices.
# head="tile" instead uses this same model's next-tile output.
```

For raw outputs instantiate `SplatObserver(Config(**saved["config"]))` using
`gus.pt`, load `saved["model"]`, and call `model(tokens, positions)`.
`positions` selects output locations; it is not embedded as an extra input.
This is custom PyTorch code, not a Transformers AutoModel. The inference export
omits optimizer/RNG state; full resumable formation and exact compiled targets
are in the [research archive]({archive['url']}).

Weights/data: CC0-1.0, Jason Yandell. nanoGPT-derived implementation remains MIT,
Andrej Karpathy and Jason Yandell; retain `lem/observer/LICENSE.nanoGPT`.
The historical oracle-era Gus is a different lineage.
'''
    (stage/'README.md').write_text(card);ModelCard.load(stage/'README.md').validate()
    for p in stage.iterdir():
        if p.suffix in ('.md','.json') and re.search(r'hf_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY',p.read_text()):
            raise ValueError('credential-shaped release text')
    (stage/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(stage.iterdir()) if p.name!='SHA256SUMS'))
    receipt=root/'model-publication.json';api=HfApi()
    if receipt.exists():
        result=json.loads(receipt.read_text());revision=result['revision']
    else:
        api.create_branch(MODEL_REPO,branch=BRANCH)
        commit=api.create_commit(MODEL_REPO,revision=BRANCH,commit_message='Gus splat v1: joint pretraining with unchanged clean inputs',
            operations=[CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=str(p)) for p in sorted(stage.iterdir())])
        revision=commit.oid
        atomic_json(receipt,{'revision':revision,'verification':'pending'})
    api.create_tag(MODEL_REPO,tag=TAG,revision=revision,exist_ok=True)
    if api.model_info(MODEL_REPO,revision=TAG).sha!=revision:raise ValueError('tag resolves to different bytes')
    verified=[]
    for p in sorted(stage.iterdir()):
        remote=Path(hf_hub_download(MODEL_REPO,p.name,revision=revision,token=False,force_download=True,local_dir=root/'verify-model'))
        if digest(remote)!=digest(p):raise ValueError('fresh model file differs: '+p.name)
        verified.append(p.name)
    restored=load_file(str(root/'verify-model'/'model.safetensors'))
    if any(not torch.equal(v,restored[k]) for k,v in state.items()):raise ValueError('safe-format tensor mismatch')
    fresh=SplatPlay(root/'verify-model'/'gus.pt','cpu')
    with torch.inference_mode():
        t,s=fresh.model(torch.tensor([example['tokens']]),torch.tensor([[13]]))
    torch.testing.assert_close(t,tile,atol=0,rtol=0);torch.testing.assert_close(s,splat,atol=0,rtol=0)
    result={'revision':revision,'tag':TAG,'url':f'https://huggingface.co/{MODEL_REPO}/tree/{TAG}',
        'fresh_anonymous_verified_files':verified,'tensor_and_both_head_parity':True,'main_branch_unchanged':True}
    atomic_json(receipt,result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--phase',choices=('initial','data','model'),required=True)
    a=p.parse_args()
    print(publish_model(a.root) if a.phase=='model' else publish_data(a.root,a.phase=='initial'),flush=True)
