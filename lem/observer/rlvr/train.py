# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""A one-answer GRPO curriculum and its matched supervised comparator.

Both optimize the existing next-tile policy. The exercise's two legal outputs
are gathered after the ordinary clean forward pass; no answer key enters it.
"""
import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from ..data import VERSION
from ..population import atomic_json, atomic_save, digest
from ..splat.model import TASK
from ..splat.play import SplatPlay
from ..splat.recovery import StopRequest, capture_rng, restore_rng
from .data import load


class Exercises:
    def __init__(self, rows, device):
        self.rows = rows
        length = max(len(r['tokens']) for r in rows)
        x = torch.zeros((len(rows), length), dtype=torch.long)
        for i, r in enumerate(rows):
            x[i, :len(r['tokens'])] = torch.tensor(r['tokens'])
        self.x = x.to(device)
        self.pos = torch.tensor([len(r['tokens']) - 1 for r in rows], device=device)
        self.legal = torch.tensor([r['question']['certificate']['legal'] for r in rows], device=device)
        self.best = torch.tensor([[a in r['question']['certificate']['best'] for a in r['question']['certificate']['legal']]
                                 for r in rows], dtype=torch.float32, device=device)
        self.q = [[Fraction(r['question']['certificate']['values'][str(a)])
                   for a in r['question']['certificate']['legal']] for r in rows]

    def logp(self, model, index):
        index = index.to(self.x.device)
        logits = model.backbone(self.x[index])
        chosen = logits[torch.arange(len(index), device=self.x.device), self.pos[index]]
        return chosen.gather(1, self.legal[index]).log_softmax(-1)


def advantages(rewards):
    centered = rewards - rewards.mean(-1, keepdim=True)
    std = rewards.std(-1, correction=0, keepdim=True)
    return centered / std.clamp_min(1e-8)


def objective(logp, ref, mode, targets, actions=None, old=None, adv=None, clip=.2, beta=.02):
    kl = (logp.exp() * (logp - ref)).sum(-1).mean()
    if mode == 'sft':
        target = targets / targets.sum(-1, keepdim=True)
        policy = -(target * logp).sum(-1).mean()
        clip_fraction = logp.new_tensor(0.)
    else:
        ratio = (logp.gather(1, actions) - old.gather(1, actions)).exp()
        policy = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
        clip_fraction = ((ratio - 1).abs() > clip).float().mean()
    return policy + beta * kl, {'policy_loss': float(policy.detach()), 'kl': float(kl.detach()),
                              'clip_fraction': float(clip_fraction.detach())}


@torch.inference_mode()
def predictions(model, data):
    return torch.cat([data.logp(model, torch.arange(i, min(i + 128, len(data.rows))))
                      for i in range(0, len(data.rows), 128)]).cpu()


def metrics(logp, data):
    p = logp.exp().numpy(); actions = p.argmax(-1)
    exact_regrets = [max(q) - q[a] for q, a in zip(data.q, actions)]
    n = len(actions)
    accuracy = sum(int(data.best[i, a]) for i, a in enumerate(actions)) / n
    regret = sum(exact_regrets, Fraction(0)) / n
    return {'accuracy': accuracy, 'regret': float(regret), 'regret_exact': str(regret),
            'return': float(sum((q[a] for q, a in zip(data.q, actions)), Fraction(0)) / n),
            'policy_return': float(np.mean([sum(float(v) * z for v, z in zip(q, row)) for q, row in zip(data.q, p)])),
            'correct_probability': float((p * data.best.cpu().numpy()).sum(-1).mean()),
            'entropy': float(-(logp.exp() * logp).sum(-1).mean())}


def run(args):
    if min(args.rounds, args.batch, args.group, args.inner, args.eval_every) < 1 or args.lr <= 0 or args.beta < 0 or not 0 < args.clip < 1:
        raise ValueError('positive budgets/rate, nonnegative KL, and 0 < clip < 1 required')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    manifest, rows = load(args.data)
    train = Exercises([r for r in rows if r['split'] == 'train'], args.device)
    validation = Exercises([r for r in rows if r['split'] == 'validation'], args.device)
    model = SplatPlay(args.initial, args.device, 'tile').model
    reference = SplatPlay(args.initial, args.device, 'tile').model
    for p in reference.parameters():
        p.requires_grad_(False)
    for p in model.outcomes.parameters():
        p.requires_grad_(False)
    groups = [{'params': [p for p in model.parameters() if p.requires_grad and (p.ndim >= 2) == matrix],
               'weight_decay': .1 if matrix else 0.} for matrix in (True, False)]
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(.9, .95))
    sampler = torch.Generator().manual_seed(args.seed)
    answer_rng = torch.Generator().manual_seed(args.seed + 1)
    settings = {k: getattr(args, k) for k in ('mode', 'rounds', 'batch', 'group', 'inner', 'lr', 'beta', 'clip', 'seed', 'device', 'eval_every', 'wandb_project', 'wandb_entity')}
    settings.update(initial_sha256=digest(args.initial), data_sha256=manifest['questions_sha256'],
                    model_id=manifest['model_id'], torch_version=str(torch.__version__))
    saved = None; completed = 0; history = []; best = None; example = None
    if args.resume:
        saved = torch.load(out / 'latest.pt', map_location='cpu', weights_only=True)
        if saved['settings'] != settings:
            raise ValueError('resume identity changed; use a fresh output for a new recipe')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        sampler.set_state(saved['sampler']); answer_rng.set_state(saved['answer_rng'])
        completed, history, best, example = saved['round'], saved['history'], saved['best'], saved['example']
        if completed == args.rounds and (out / 'report.json').exists():
            report = json.loads((out / 'report.json').read_text())
            if report['settings'] != settings or report['latest_sha256'] != digest(out / 'latest.pt'):
                raise ValueError('completion receipt differs from its checkpoint')
            return report
    elif (out / 'latest.pt').exists():
        raise ValueError('output exists; use --resume')
    tracker = None; identity = None
    if args.wandb_project:
        import wandb
        receipt = out / 'wandb-run.json'
        identity = json.loads(receipt.read_text()) if receipt.exists() else {
            'id': uuid.uuid4().hex[:8], 'project': args.wandb_project, 'entity': args.wandb_entity}
        if saved and saved['run_identity'] != identity:
            raise ValueError('W&B identity changed')
        atomic_json(receipt, identity)
        tracker = wandb.init(project=identity['project'], entity=identity['entity'], id=identity['id'],
                             name='silas-rlvr-v1-' + args.mode, resume='allow', dir=str(out), config=settings,
                             tags=['gus', 'rlvr', 'silas', args.mode])
        identity = dict(id=tracker.id, project=tracker.project, entity=tracker.entity, url=tracker.url)
        atomic_json(receipt, identity)
        tracker.define_metric('round'); tracker.define_metric('*', step_metric='round')
        print('LIVE_RUN ' + tracker.url, flush=True)
    if saved:
        restore_rng(saved['rng_state'])
    else:
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    # Fixed reference distributions are a frozen prior, distinct from each
    # round's old policy, which generated that round's sampled answers.
    ref_train = predictions(reference, train).to(args.device)
    del reference

    def save(path):
        atomic_save(path, {'task': TASK, 'tokenizer': VERSION, 'config': asdict(model.config),
                          'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                          'sampler': sampler.get_state(), 'answer_rng': answer_rng.get_state(),
                          'rng_state': capture_rng(args.device), 'round': completed, 'step': completed * args.inner,
                          'data_sha256': manifest['questions_sha256'], 'settings': settings, 'best': best,
                          'history': history, 'example': example, 'run_identity': identity})

    def evaluate():
        nonlocal best
        report = {'round': completed, **metrics(predictions(model, validation), validation)}
        history.append(report)
        key = (Fraction(report['regret_exact']), -report['accuracy'], completed)
        if best is None or key < (Fraction(best['regret_exact']), -best['accuracy'], best['round']):
            best = report; save(out / 'best.pt')
        save(out / f'round-{completed:04d}.pt')
        atomic_json(out / 'history.json', history)
        if tracker:
            tracker.log({'round': completed, **{'validation/' + k: v for k, v in report.items() if isinstance(v, (int, float))}})
        print(report, flush=True)

    start = time.monotonic()
    with StopRequest(args.stop_file, args.seconds) as stop:
        if not saved:
            evaluate(); save(out / 'latest.pt')
        while completed < args.rounds and not stop.requested():
            index = torch.randint(len(train.rows), (args.batch,), generator=sampler)
            device_index = index.to(args.device)
            targets = train.best[device_index]
            with torch.no_grad():
                old = train.logp(model, index).detach()
                actions = torch.multinomial(old.cpu().exp(), args.group, replacement=True, generator=answer_rng).to(args.device)
                rewards = targets.gather(1, actions)
                adv = advantages(rewards)
            log = {}
            for _ in range(args.inner):
                logp = train.logp(model, index)
                loss, log = objective(logp, ref_train[device_index], args.mode, targets, actions, old, adv, args.clip, args.beta)
                optimizer.zero_grad(set_to_none=True); loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            completed += 1
            varying = rewards.std(-1, correction=0) > 0
            if example is None and bool(varying.any()) and args.mode == 'grpo':
                j = int(varying.nonzero()[0]); row = train.rows[int(index[j])]
                with torch.no_grad():
                    new = train.logp(model, index)[j].cpu().tolist()
                example = {'round': completed, 'row': row, 'sampled_tile_ids': train.legal[device_index][j, actions[j]].cpu().tolist(),
                           'rewards': rewards[j].cpu().tolist(), 'advantages': adv[j].cpu().tolist(),
                           'old_logp': old[j].cpu().tolist(), 'new_logp': new, 'reference_logp': ref_train[device_index][j].cpu().tolist()}
                atomic_json(out / 'worked-update.json', example)
            log.update(round=completed, grad_norm=float(grad_norm), mean_reward=float(rewards.mean()),
                       zero_variance_fraction=float((~varying).float().mean()))
            if tracker:
                tracker.log({'round': completed, **{'train/' + k: v for k, v in log.items() if k != 'round'}})
            if completed % args.eval_every == 0 or completed == args.rounds:
                evaluate()
            save(out / 'latest.pt')
        status = {'state': 'complete' if completed == args.rounds else 'paused', 'round': completed,
                  'reason': stop.reason, 'segment_seconds': time.monotonic() - start,
                  'best': best, 'latest_sha256': digest(out / 'latest.pt')}
        atomic_json(out / 'status.json', status)
        if status['state'] == 'complete':
            save(out / 'final.pt'); atomic_json(out / 'report.json', {**status, 'settings': settings, 'history': history})
    if tracker:
        tracker.finish()
    return status


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--initial', required=True); p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=('sft', 'grpo'), required=True); p.add_argument('--device', default='mps')
    p.add_argument('--rounds', type=int, default=256); p.add_argument('--batch', type=int, default=64)
    p.add_argument('--group', type=int, default=8); p.add_argument('--inner', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-5); p.add_argument('--beta', type=float, default=.02)
    p.add_argument('--clip', type=float, default=.2); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-every', type=int, default=32); p.add_argument('--seconds', type=float, default=240)
    p.add_argument('--stop-file'); p.add_argument('--resume', action='store_true')
    p.add_argument('--wandb-project'); p.add_argument('--wandb-entity')
    return p


if __name__ == '__main__':
    print(run(parser().parse_args()), flush=True)
