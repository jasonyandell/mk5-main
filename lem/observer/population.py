# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Mixed-seat hands and policy-only PPO, with the observer's unchanged inputs.

The environment owns complete deals for lawful transitions and replay. Networks
receive only play.prefix; legality constrains outputs. Rewards are actual team
contract outcomes, never a per-world planner estimate.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from arena.auction import run_auction
from arena.bidders import Bid30Bidder
from forge.oracle.rng import deal_from_seed
from forge.zeb.game import apply_action, current_player, legal_actions, is_terminal
from forge.zeb.types import BidState, GamePhase, ZebGameState
from scenario.record import empty_record
from scenario.rules import mask_of, rules_for

from .data import VERSION
from .model import Config, Observer
from .play import prefix


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(*parts):
    raw = json.dumps(parts, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_save(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_model(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["tokenizer"] != VERSION:
        raise ValueError("input contract changed")
    model = Observer(Config(**saved["config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    return model, saved


def deepen(saved, layers, seed):
    """Append identity residual blocks; preserve old computations exactly."""
    old = Config(**saved["config"])
    if layers <= old.layers:
        raise ValueError("depth must increase")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = Observer(replace(old, layers=layers))
    state = model.state_dict()
    state.update(saved["model"])
    model.load_state_dict(state)
    with torch.no_grad():
        for block in model.blocks[old.layers:]:
            for projection in (block.attention.projection, block.projection):
                projection.weight.zero_()
                projection.bias.zero_()
    return {"config": asdict(model.config), "model": model.state_dict(), "step": 0,
            "tokenizer": VERSION, "data_sha256": saved["data_sha256"],
            "growth": {"from_layers": old.layers, "seed": seed,
                       "method": "append blocks with zero attention and MLP output projections"}}


def active_pool(generations, limit=5):
    if not generations or limit < 1 or len(set(generations)) != len(generations):
        raise ValueError("need distinct completed Gus generations")
    # Zeb is a separately pinned seat, never subject to Gus retention.
    return list(generations[-limit:])


def make_specs(count, namespace, pool, excluded):
    """Sample the entire table before play; reject overlap before any outcome."""
    if count < 1 or not pool or "zeb" in pool or "learner" in pool:
        raise ValueError("need hands and a named Gus pool")
    rng = np.random.default_rng(seed_for(namespace, "roster"))
    specs, used = [], set(excluded)
    bidder = Bid30Bidder()
    for i in range(count):
        learner, dealer = i % 4, ((i // 4) % 4 - 1) % 4
        attempt = 0
        while True:
            seed = seed_for(namespace, "deal", i, attempt)
            hands = tuple(tuple(h) for h in deal_from_seed(seed))
            key = tuple(mask_of(h) for h in hands)
            if key not in used:
                used.add(key)
                break
            attempt += 1
        auction = run_auction(hands, dealer, (bidder,) * 4, random.Random(seed))
        if auction.high_bid != 30 or auction.decl_id not in range(7):
            raise ValueError("population contract must remain bid30/pip")
        seats = ["zeb" if rng.random() < .25 else str(rng.choice(pool)) for _ in range(4)]
        seats[learner] = "learner"
        specs.append({"seed": seed, "hands": hands, "dealer": dealer,
                      "bidder": auction.winner, "bids": auction.bids,
                      "decl": auction.decl_id, "learner": learner, "seats": seats})
    return specs


def initial_state(spec):
    return ZebGameState(hands=tuple(tuple(h) for h in spec["hands"]), dealer=spec["dealer"],
                        phase=GamePhase.PLAYING,
                        bid_state=BidState(tuple(spec["bids"]), spec["bidder"], 30),
                        decl_id=spec["decl"], bidder=spec["bidder"], played=frozenset(),
                        play_history=(), current_trick=(), trick_leader=spec["bidder"],
                        team_points=(0, 0))


def features(states):
    x = np.zeros((len(states), 68), dtype=np.int64)
    length = np.empty(len(states), dtype=np.int64)
    legal = np.zeros((len(states), 28), dtype=bool)
    for i, state in enumerate(states):
        tokens = prefix(state, 30)
        length[i] = len(tokens)
        x[i, :len(tokens)] = tokens
        hand = state.hands[current_player(state)]
        legal[i, [hand[slot] for slot in legal_actions(state)]] = True
    return x, length, legal


def log_distribution(model, x, length, legal):
    if not bool(legal.any(-1).all()):
        raise ValueError("empty legal support")
    logits = model(x)[torch.arange(len(x), device=x.device), length - 1]
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("nonfinite model logits")
    # Finite masked log probabilities avoid 0 * -inf in entropy/KL.
    return logits.masked_fill(~legal, -1e9).log_softmax(-1)


def team_reward(points, bidder, learner, bid=30):
    winner = bidder % 2 if points[bidder % 2] >= bid else 1 - bidder % 2
    return 1.0 if learner % 2 == winner else -1.0


def verify_episode(episode):
    rules, rec = rules_for(episode["decl"]), empty_record(episode["bidder"])
    hands = tuple(mask_of(h) for h in episode["hands"])
    for seat, tile in episode["plays"]:
        if seat != rec.seat or not rules.legal(hands[seat] & ~rec.played, rec.led) & (1 << tile):
            raise ValueError("illegal population replay")
        rec = rec.advance(tile, rules)
    if not rec.finished or tuple(episode["points"]) != rec.banked or sum(rec.banked) != 42:
        raise ValueError("population/referee score mismatch")
    if team_reward(rec.banked, episode["bidder"], episode["learner"]) != episode["reward"]:
        raise ValueError("wrong team reward")


@torch.inference_mode()
def collect(model, policies, specs, device, sample=True):
    """One learner seat per hand, frozen other seats; full independent replay."""
    model.eval()
    states = [initial_state(s) for s in specs]
    rngs = [np.random.default_rng(seed_for(s["seed"], s["learner"], "action")) for s in specs]
    arrays = {"x": [], "length": [], "legal": [], "action": [], "old_logp": [],
              "old_probs": [], "hand": []}
    for _ in range(28):
        groups = {}
        for i, state in enumerate(states):
            name = specs[i]["seats"][current_player(state)]
            groups.setdefault(name, []).append(i)
        for name, ids in groups.items():
            batch = [states[i] for i in ids]
            if name == "learner":
                x, length, legal = features(batch)
                logp = log_distribution(model, torch.as_tensor(x, device=device),
                                        torch.as_tensor(length, device=device),
                                        torch.as_tensor(legal, device=device)).cpu().numpy()
                probabilities = np.exp(logp)
                choices = []
                for row, i in enumerate(ids):
                    p = probabilities[row].astype(np.float64)
                    p /= p.sum()
                    tile = int(rngs[i].choice(28, p=p)) if sample else int(p.argmax())
                    state = states[i]
                    choices.append(state.hands[current_player(state)].index(tile))
                    for key, value in {"x": x[row], "length": length[row], "legal": legal[row],
                                       "action": tile, "old_logp": logp[row, tile],
                                       "old_probs": probabilities[row], "hand": i}.items():
                        arrays[key].append(value)
            else:
                choices = policies[name].choose(batch, [30] * len(batch))
            if len(choices) != len(ids):
                raise ValueError("policy returned wrong batch size")
            for i, choice in zip(ids, choices):
                if choice not in legal_actions(states[i]):
                    raise ValueError("policy chose an illegal move")
                states[i] = apply_action(states[i], choice)
    episodes = []
    for spec, state in zip(specs, states):
        if not is_terminal(state):
            raise ValueError("incomplete hand")
        episode = {**spec, "plays": state.play_history, "points": state.team_points,
                   "reward": team_reward(state.team_points, spec["bidder"], spec["learner"])}
        verify_episode(episode)
        episodes.append(episode)
    data = {k: np.asarray(v) for k, v in arrays.items()}
    data["reward"] = np.array([episodes[i]["reward"] for i in data["hand"]], dtype=np.float32)
    if len(data["action"]) != 7 * len(specs):
        raise ValueError("each learner must have seven turns")
    return data, episodes


def clipped_loss(logp, actions, old_logp, advantage, clip=.2, entropy_weight=.01):
    chosen = logp.gather(1, actions[:, None]).squeeze(1)
    ratio = (chosen - old_logp).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-clip, 1+clip) * advantage)
    entropy = -(logp.exp() * logp).sum(-1)
    return -(surrogate + entropy_weight * entropy).mean(), ratio, entropy


def update(model, optimizer, data, baseline, seed, device, passes=4, batch=256):
    model.train()
    tensors = {k: torch.as_tensor(v, device=device) for k, v in data.items()}
    n = len(data["action"])
    sampler = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def diagnostics():
        kls, errors = [], []
        for start in range(0, n, batch):
            sl = slice(start, start+batch)
            lp = log_distribution(model, tensors["x"][sl], tensors["length"][sl], tensors["legal"][sl])
            old = tensors["old_probs"][sl]
            kls.append((old * (old.clamp_min(1e-30).log() - lp)).sum(-1))
            errors.append((lp.gather(1, tensors["action"][sl, None]).squeeze(1) - tensors["old_logp"][sl]).abs())
        return float(torch.cat(kls).mean()), float(torch.cat(errors).max())

    initial_kl, initial_error = diagnostics()
    if initial_error > 5e-5:
        raise ValueError(f"sampling/update probability mismatch: {initial_error}")
    metrics = []
    for epoch in range(passes):
        order = torch.randperm(n, generator=sampler).to(device)
        rows = []
        for start in range(0, n, batch):
            ids = order[start:start+batch]
            lp = log_distribution(model, tensors["x"][ids], tensors["length"][ids], tensors["legal"][ids])
            loss, ratio, entropy = clipped_loss(lp, tensors["action"][ids], tensors["old_logp"][ids],
                                               tensors["reward"][ids] - baseline)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            rows.append([len(ids), float(loss.detach()), float(entropy.mean().detach()),
                         float(((ratio-1).abs() > .2).float().mean().detach()), float(norm)])
        kl, _ = diagnostics()
        values = np.asarray(rows)
        averages = np.average(values[:, 1:], axis=0, weights=values[:, 0])
        metrics.append(dict(zip(("loss", "entropy", "clip_fraction", "gradient_norm"), map(float, averages))))
        metrics[-1].update(pass_index=epoch+1, kl=kl)
        if kl > .03:
            break
    return {"initial_kl": initial_kl, "initial_logp_error": initial_error,
            "passes": metrics, "passes_completed": len(metrics), "baseline": baseline}
