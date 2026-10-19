# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Full-hand GRPO/PPO: legal trajectories, verified outcomes, fixed seats.

Run through cpu-slot. The plan contains disjoint train/development/confirmation
physical deals and explicit frozen policies. Seeds identify deals and action
streams separately. A complete round is the recovery/commit boundary.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import platform
import random
import time
import uuid

import numpy as np
import torch

from forge.zeb.game import apply_action, current_player
from scenario.rules import mask_of, rules_for
from scenario.record import empty_record
from .data import VERSION
from .population import (atomic_json, atomic_save, collect, digest, features,
                         initial_state, log_distribution, seed_for, verify_episode)
from .reward import policies, save_episodes
from .rlvr.rehearsal import Rehearsal
from .rlvr.train import advantages
from .splat.model import TASK
from .splat.play import SplatPlay
from .splat.recovery import StopRequest, capture_rng, restore_rng
from .value import ValueBaseline

SCHEMA = 'gus-full-hand-outcomes-v6'


def publicly_settled(points, bidder, bid=30):
    """The binary contract outcome is fixed by public banked points alone.

    Remaining points are nonnegative and the hand totals 42. Once the bidder
    has 30 or defenders have 13, later legal plays cannot change make/set.
    """
    return points[bidder % 2] >= bid or points[1-bidder % 2] > 42-bid


def validate_plan(plan, replay):
    seen = set()
    for split in ('train', 'development', 'confirmation'):
        if not plan[split]:
            raise ValueError('each physical-deal split must be nonempty')
        for spec in plan[split]:
            key = tuple(mask_of(h) for h in spec['hands'])
            if key in seen:
                raise ValueError('physical-deal overlap')
            seen.add(key)
            learner = spec['learner']
            expected = ['zeb'] * 4
            expected[learner] = 'learner'; expected[(learner + 2) % 4] = 'g0'
            if spec['seats'] != expected:
                raise ValueError('fixed G0 partner and Zeb opponents required')
            tiles = [t for h in spec['hands'] for t in h]
            if sorted(tiles) != list(range(28)) or any(len(h) != 7 for h in spec['hands']):
                raise ValueError('invalid complete deal')
            if spec['decl'] not in range(7) or max(spec['bids']) != 30:
                raise ValueError('bid30/pip only')
    if seen & {tuple(map(int, h)) for h in replay.hands}:
        raise ValueError('rehearsal overlaps physical-deal panels')


def grouped_specs(roots, group, namespace):
    if group < 1:
        raise ValueError('positive group required')
    result = []
    for i, root in enumerate(roots):
        for draw in range(group):
            action_seed = seed_for(namespace, root['seed'], root['learner'], draw)
            result.append({**root, 'root_deal_seed': root['seed'], 'seed': action_seed,
                           'action_seed': seed_for(action_seed, root['learner'], 'action'),
                           'group': i, 'draw': draw})
    return result


def leave_one_out(rewards):
    """Other independent draws supply each hand's baseline; no normalization.

    Rewards are verified success indicators (0/1). At the sampling policy this
    action-independent baseline preserves the expected score-function gradient.
    Clipping and subsequent optimization are separate approximations.
    """
    if rewards.ndim != 2 or rewards.shape[1] < 2 or not torch.isfinite(rewards).all():
        raise ValueError('leave-one-out requires finite groups of at least two')
    group = rewards.shape[1]
    return rewards - (rewards.sum(-1, keepdim=True) - rewards) / (group - 1)


def trajectories(model, field, roots, group, namespace, device, sample=True, estimator='grpo'):
    data, episodes = collect(model.backbone, field, grouped_specs(roots, group, namespace), device, sample)
    for episode in episodes:
        episode['seed'] = episode['root_deal_seed']
    data['pre_settlement'] = audit(data, episodes)
    rewards = torch.tensor([(e['reward'] + 1) / 2 for e in episodes]).reshape(len(roots), group)
    if estimator not in ('grpo', 'rloo'):
        raise ValueError('unknown trajectory estimator')
    adv = (leave_one_out(rewards) if estimator == 'rloo' else advantages(rewards)).numpy().ravel()
    data['advantage'] = adv[data['hand']]
    return data, episodes, {'win_rate': float(rewards.mean()),
        'zero_variance_groups': float((rewards.std(-1, correction=0) == 0).float().mean()),
        'hands': len(episodes), 'decisions': len(data['action']),
        'post_settlement_fraction': float(1-data['pre_settlement'].mean())}


def audit(data, episodes):
    """Rebuild every actual decision from the replay, never from saved labels."""
    if len(data['action']) != 7 * len(episodes):
        raise ValueError('incomplete trajectory decisions')
    active = np.empty(len(data['action']), dtype=bool)
    for i, episode in enumerate(episodes):
        verify_episode(episode)
        state = initial_state(episode)
        record = empty_record(episode['bidder']); rules = rules_for(episode['decl'])
        ids = np.flatnonzero(data['hand'] == i)
        if len(ids) != 7:
            raise ValueError('one learner decision per trick required')
        at = 0
        for seat, tile in episode['plays']:
            if tuple(state.team_points) != record.banked:
                raise ValueError('public partial-score replay mismatch')
            if seat == episode['learner']:
                j = ids[at]; x, length, legal = features([state]); at += 1
                active[j] = not publicly_settled(record.banked, episode['bidder'])
                if (not np.array_equal(data['x'][j], x[0]) or data['length'][j] != length[0]
                    or not np.array_equal(data['legal'][j], legal[0]) or data['action'][j] != tile):
                    raise ValueError('learner input/output replay mismatch')
                p = data['old_probs'][j]
                if (not np.isfinite(p).all() or np.any(p < 0) or np.any(p[~legal[0]])
                    or not np.isclose(p.sum(), 1, atol=2e-6)
                    or not np.isclose(np.log(p[tile]), data['old_logp'][j], atol=2e-6)
                    or data['reward'][j] != episode['reward']):
                    raise ValueError('saved probability/reward mismatch')
            state = apply_action(state, state.hands[current_player(state)].index(tile))
            record = record.advance(tile, rules)
    if 'pre_settlement' in data and not np.array_equal(active, data['pre_settlement']):
        raise ValueError('public settlement mask replay mismatch')
    return active


def update(model, reference, optimizer, data, replay, replay_rng, recipe, device):
    tensors = {k: torch.as_tensor(v, device=device) for k, v in data.items()}
    def distribution(net, ids):
        return log_distribution(net.backbone, tensors['x'][ids], tensors['length'][ids], tensors['legal'][ids])
    n = len(data['action']); batch = recipe['batch']
    with torch.no_grad():
        before = torch.cat([distribution(model, slice(i, i+batch)) for i in range(0, n, batch)])
        error = float((before.gather(1, tensors['action'][:, None]).squeeze(1) - tensors['old_logp']).abs().max())
        if error > 5e-5:
            raise ValueError(f'old/new probability mismatch: {error}')
        ref = torch.cat([distribution(reference, slice(i, i+batch)) for i in range(0, n, batch)])
    metrics = []
    # Equal seven-decision trajectories: this mean equals mean over hands then turns.
    # Accumulate all microbatches before AdamW so controls have identical step counts.
    for _ in range(recipe['inner']):
        optimizer.zero_grad(set_to_none=True); totals = np.zeros(4)
        for start in range(0, n, batch):
            sl = slice(start, start+batch); lp = distribution(model, sl)
            ratio = (lp.gather(1, tensors['action'][sl, None]).squeeze(1) - tensors['old_logp'][sl]).exp()
            adv = tensors['advantage'][sl]
            if recipe['reward_horizon'] == 'unsettled':
                # Same seven-turn denominator; no renormalization by active length.
                adv = adv * tensors['pre_settlement'][sl]
            policy = -torch.minimum(ratio*adv, ratio.clamp(1-recipe['clip'], 1+recipe['clip'])*adv).mean()
            kl = (lp.exp() * (lp - ref[sl])).sum(-1).mean()
            entropy = -(lp.exp()*lp).sum(-1).mean()
            weight = len(lp)/n
            if recipe['mode'] in ('grpo', 'ppo', 'rloo'):
                ((policy + recipe['beta']*kl)*weight).backward()
            totals += weight*np.array([float(policy.detach()), float(kl.detach()), float(entropy.detach()),
                                      float(((ratio-1).abs() > recipe['clip']).float().mean().detach())])
        ix = torch.randint(len(replay.splits['train']), (recipe['replay_batch'],), generator=replay_rng).numpy()
        # Keep the same sampled diagnostics and RNG stream when imitation is
        # disabled, without creating zero gradients on the unused splat head.
        with torch.set_grad_enabled(recipe['replay_weight'] > 0):
            loss, tile, splat, _ = replay.loss(model, replay.splits['train'][ix])
        if recipe['replay_weight'] > 0:
            (recipe['replay_weight']*loss).backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        metrics.append(dict(zip(('policy_loss', 'reference_kl', 'entropy', 'clip_fraction'), map(float, totals)),
                            tile_ce=float(tile.detach()), splat_ce=float(splat.detach()), gradient_norm=float(norm)))
    with torch.no_grad():
        after = torch.cat([distribution(model, slice(i, i+batch)) for i in range(0, n, batch)])
    return {'initial_logp_error': error, 'updates': metrics}, {'before': before.cpu().numpy(),
                                                              'after': after.cpu().numpy(), 'reference': ref.cpu().numpy()}


def source_identity():
    root = Path(__file__).resolve().parents[2]
    paths = [p for folder in ('lem/observer', 'scenario', 'arena', 'forge/zeb')
             for p in (root / folder).rglob('*.py')]
    paths += [root / 'forge/oracle/rng.py']  # deal generator only
    import scenario_core
    native = Path(scenario_core.__file__).parent
    paths += list(native.glob('*.so'))
    return {str(p.relative_to(root)) if p.is_relative_to(root) else str(p): digest(p) for p in sorted(set(paths))}


def _run(args):
    plan_path = Path(args.plan); plan = json.loads(plan_path.read_text())
    recipe = dict(plan['recipe'], mode=args.mode)
    if recipe.get('reward_horizon') not in ('full', 'unsettled'):
        raise ValueError('declare reward_horizon as full or unsettled')
    if recipe.get('selection') not in ('retention', 'outcomes'):
        raise ValueError('declare selection as retention or outcomes')
    if (args.mode not in ('grpo', 'ppo', 'rloo', 'rehearsal') or min(recipe[k] for k in ('rounds', 'roots', 'group', 'inner', 'batch', 'replay_batch', 'eval_every')) < 1
        or recipe['group'] < 2 or recipe['lr'] <= 0 or not 0 < recipe['clip'] < 1 or recipe['beta'] < 0
        or recipe['replay_weight'] < 0 or (args.mode == 'rehearsal' and recipe['replay_weight'] == 0)
        or min(recipe['retention_nll'], recipe['retention_splat']) < 0):
        raise ValueError('invalid full-hand recipe')
    if len(plan['train']) != recipe['rounds']*recipe['roots']:
        raise ValueError('each round requires its registered fresh roots')
    for key in ('initial', 'reference', 'replay'):
        if key not in plan or key+'_sha256' not in plan:
            raise ValueError('explicit initial/reference/replay identities required')
        path = Path(plan[key]) / 'rehearsal.npz' if key == 'replay' else plan[key]
        if digest(path) != plan[key+'_sha256']:
            raise ValueError(f'{key} identity changed')
    torch.set_num_threads(1)
    random.seed(plan['seed']); np.random.seed(plan['seed']); torch.manual_seed(plan['seed'])
    model = SplatPlay(plan['initial'], args.device, 'tile').model
    reference = SplatPlay(plan['reference'], args.device, 'tile').model
    reference.requires_grad_(False)
    replay = Rehearsal(plan['replay'], args.device); validate_plan(plan, replay)
    field = policies(plan['field'], args.device)
    if plan.get('source') is not None and plan['source'] != source_identity():
        raise ValueError('registered source identity changed')
    settings = {'schema': SCHEMA, 'plan_sha256': digest(plan_path), 'recipe': recipe,
                'device': args.device, 'torch': str(torch.__version__), 'numpy': np.__version__,
                'python': platform.python_version(), 'source': source_identity()}
    groups = [{'params': [p for p in model.parameters() if (p.ndim >= 2) == matrix],
               'weight_decay': .1 if matrix else 0.} for matrix in (True, False)]
    optimizer = torch.optim.AdamW(groups, lr=recipe['lr'], betas=(.9, .95))
    if args.mode == 'ppo' and 'value' not in recipe:
        raise ValueError('explicit value-baseline recipe required for PPO')
    value = ValueBaseline(reference.config.width, recipe['value'], args.device) if args.mode == 'ppo' else None
    replay_rng = torch.Generator().manual_seed(plan['seed']+2)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    completed = 0; history = []; best = None; saved = None
    if args.resume:
        saved = torch.load(out/'latest.pt', map_location='cpu', weights_only=True)
        if saved['settings'] != settings:
            raise ValueError('resume identity changed')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        if value is not None:
            value.restore(saved['value'])
        replay_rng.set_state(saved['replay_rng'])
        completed, history, best = saved['round'], saved['history'], saved['best']
        if completed == recipe['rounds'] and (out/'report.json').exists():
            report = json.loads((out/'report.json').read_text())
            if (report['settings'] != settings or report['latest_sha256'] != digest(out/'latest.pt')
                or report['best_sha256'] != digest(out/'best.pt')):
                raise ValueError('completion checksum changed')
            return report
        # Repair an interrupted best-file write from the committed selection.
        selected_path = out / (f"formation-{best['round']:04d}.pt" if best['round'] else 'initial.pt')
        (out/'best.pt').write_bytes(selected_path.read_bytes())
    elif any(out.glob('*.pt')):
        raise ValueError('existing run requires --resume')
    tracker = None; identity = None
    if plan.get('wandb'):
        import wandb
        receipt = out/'wandb-run.json'
        identity = json.loads(receipt.read_text()) if receipt.exists() else dict(plan['wandb'], id=uuid.uuid4().hex[:8])
        if saved and saved['run_identity'] != identity:
            raise ValueError('tracking identity changed')
        atomic_json(receipt, identity)
        tracker = wandb.init(project=identity['project'], entity=identity['entity'], id=identity['id'],
            name=out.name, resume='allow', dir=str(out), config=settings, tags=['gus', 'full-hand', args.mode])
        identity['url'] = tracker.url; atomic_json(receipt, identity)
        tracker.define_metric('round'); tracker.define_metric('*', step_metric='round')
        print('LIVE_RUN '+tracker.url, flush=True)
    if saved:
        restore_rng(saved['rng_state'])
    else:
        random.seed(plan['seed']); np.random.seed(plan['seed']); torch.manual_seed(plan['seed'])
    reference_retention = replay.evaluate(reference)

    def save(path):
        atomic_save(path, {'task': TASK, 'tokenizer': VERSION, 'config': asdict(model.config),
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'replay_rng': replay_rng.get_state(),
            'value': value.state() if value is not None else None,
            'rng_state': capture_rng(args.device), 'round': completed, 'step': completed*recipe['inner'],
            'data_sha256': settings['plan_sha256'], 'settings': settings, 'history': history, 'best': best,
            'run_identity': identity})

    def evaluate():
        folder = out/f'eval-{completed:04d}'; folder.mkdir(exist_ok=True)
        scores = {}
        for sampled in (False, True):
            label = 'sampled' if sampled else 'greedy'
            _, episodes, stats = trajectories(model, field, plan['development'], 1, plan['namespace']+'/development', args.device, sampled)
            save_episodes(folder/(label+'.jsonl.gz'), episodes)
            scores[label] = stats['win_rate']
        retention = replay.evaluate(model)
        feasible = all(retention[k] <= reference_retention[k]+recipe[bound]
                       for k, bound in [('tile_nll','retention_nll'), ('splat_ce','retention_splat')])
        return dict(round=completed, **scores, retention=retention, retention_pass=feasible,
                    selection_eligible=recipe['selection'] == 'outcomes' or feasible)

    with StopRequest(args.stop_file, args.seconds) as stop:
        if not saved:
            best = evaluate()
            if not best['selection_eligible']:
                raise ValueError('initial policy violates reference retention bounds')
            save(out/'initial.pt'); save(out/'best.pt'); save(out/'latest.pt')
        while completed < recipe['rounds'] and not stop.requested():
            folder = out/f'round-{completed+1:04d}'; folder.mkdir(exist_ok=True)
            roots = plan['train'][completed*recipe['roots']:(completed+1)*recipe['roots']]
            started = time.monotonic()
            data, episodes, stats = trajectories(model, field, roots, recipe['group'],
                f"{plan['namespace']}/train/{completed}", args.device,
                estimator='rloo' if args.mode == 'rloo' else 'grpo')
            if value is not None:
                stats['value_advantage'] = value.attach(data, reference, recipe['batch'])
            collection_seconds = time.monotonic()-started
            save_episodes(folder/'hands.jsonl.gz', episodes)
            np.savez_compressed(folder/'decisions.npz', **data)
            before = time.monotonic()
            metrics, distributions = update(model, reference, optimizer, data, replay, replay_rng, recipe, args.device)
            if value is not None:
                metrics['value'], distributions['value_after'] = value.update(data)
            np.savez_compressed(folder/'update.npz', **distributions)
            # Wall clocks are descriptive receipts, never part of recovery or selection.
            atomic_json(folder/'timing.json', {'collection_seconds': collection_seconds, 'update_seconds': time.monotonic()-before})
            completed += 1
            row = dict(round=completed, **stats, optimizer=metrics)
            selected = False
            if completed % recipe['eval_every'] == 0 or completed == recipe['rounds']:
                row['development'] = evaluate()
                candidate = row['development']
                if candidate['selection_eligible'] and (candidate['greedy'], candidate['sampled']) > (best['greedy'], best['sampled']):
                    best = candidate; selected = True
            history.append(row)
            if selected:
                save(out/'best.pt')
            save(out/f'formation-{completed:04d}.pt'); save(out/'latest.pt')
            atomic_json(folder/'complete.json', {'decisions_sha256': digest(folder/'decisions.npz'),
                'hands_sha256': digest(folder/'hands.jsonl.gz'), 'checkpoint_sha256': digest(out/'latest.pt')})
            if tracker:
                tracker.log({'round': completed, 'train/win_rate': stats['win_rate'],
                    'train/zero_variance_groups': stats['zero_variance_groups'],
                    'train/post_settlement_fraction': stats['post_settlement_fraction'],
                    'train/reference_kl': metrics['updates'][-1]['reference_kl'],
                    **({'value/'+k: v for k,v in metrics['value'].items()} if value is not None else {}),
                    **({'value_advantage/'+k: v for k,v in stats['value_advantage'].items()} if value is not None else {}),
                    **({'dev/'+k: row['development'][k] for k in ('greedy','sampled')} if 'development' in row else {})})
            print(json.dumps(row), flush=True)
        state = 'complete' if completed == recipe['rounds'] else 'paused'
        save(out/'latest.pt')
        report = {'state': state, 'round': completed, 'reason': stop.reason, 'best': best,
                  'settings': settings, 'reference_retention': reference_retention,
                  'latest_sha256': digest(out/'latest.pt'), 'best_sha256': digest(out/'best.pt')}
        atomic_json(out/'status.json', report)
        if state == 'complete':
            atomic_json(out/'report.json', report)
    if tracker:
        tracker.finish()
    return report


def run(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    with (out/'.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('training output is already owned by another process') from exc
        return _run(args)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--out', required=True)
    p.add_argument('--mode', choices=('grpo','ppo','rloo','rehearsal'), required=True)
    p.add_argument('--device', choices=('cpu','mps','cuda'), default='mps')
    p.add_argument('--seconds', type=float); p.add_argument('--stop-file'); p.add_argument('--resume', action='store_true')
    return p


if __name__ == '__main__':
    print(json.dumps(run(parser().parse_args())), flush=True)
