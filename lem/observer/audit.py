# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Audit archived learner inputs/outputs directly against complete public replays.

Run through cpu-slot. This reconstructs tokens independently of the live encoder
and audits completed generations only; it never changes model selection.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from scenario.record import empty_record
from scenario.rules import mask_of, rules_for
from .data import VERSION
from .population import atomic_json, digest, verify_episode


def audit_panel(root, name, baseline):
    root = Path(root)
    folder = root / name / "development"
    report = json.loads((folder / "report.json").read_text())
    assert report["specs_sha256"] == digest(root / "development-specs.json")
    specs = json.loads((root / "development-specs.json").read_text())
    with gzip.open(folder / "hands.jsonl.gz", "rt") as stream:
        episodes = [json.loads(line) for line in stream]
    assert len(episodes) == len(baseline) == len(specs) == report["hands"]
    changed, won, lost = 0, 0, 0
    wins = []
    for before, after, spec in zip(baseline, episodes, specs):
        verify_episode(before); verify_episode(after)
        assert all(before[k] == v and after[k] == v for k,v in spec.items())
        first = next((i for i,(a,b) in enumerate(zip(before["plays"],after["plays"])) if a != b), None)
        if first is not None:
            changed += 1
            # Only the learner changes. A frozen player's first divergence on
            # identical public history would invalidate the fixed-table claim.
            assert before["plays"][first][0] == after["plays"][first][0] == spec["learner"]
        won += after["reward"] > before["reward"]
        lost += after["reward"] < before["reward"]
        wins.append((after["reward"]+1)/2)
    assert wins == report["wins"] and float(np.mean(wins)) == report["team_win_rate"]
    assert (won-lost)/len(episodes) == report["vs_reference"]["win_rate_difference"]
    return {"hands_with_different_play":changed, "loss_to_win":won, "win_to_loss":lost,
            "all_first_divergences_belong_to_learner":True,
            "reported_wins_match_independent_replays":True}


def audit_generation(run, generation, expected_pool, excluded):
    run = Path(run)
    config = json.loads((run / "config.json").read_text())
    assert set(config["pool"]) == set(expected_pool), "wrong recent-five roster"
    assert digest(run / "specs.json") == config["specs_sha256"]
    for frozen in config["pool"].values():
        assert digest(frozen["path"]) == frozen["sha256"]
    assert digest(config["zeb_field"]) == config["zeb_field_sha256"]
    specs = json.loads((run / "specs.json").read_text())
    assert len(specs) == config["blocks"] * config["hands_per_block"]
    for spec in specs:
        deal = tuple(mask_of(h) for h in spec["hands"])
        assert deal not in excluded, "physical training deal overlap"
        excluded.add(deal)
        assert spec["seats"].count("learner") == 1
        assert spec["seats"][spec["learner"]] == "learner"
        assert set(spec["seats"]) <= set(expected_pool) | {"learner", "zeb"}
    previous = run / "initial.pt"
    previous_weights = torch.load(previous, map_location="cpu", weights_only=True)["model"]
    prior_return, prior_hands, updates, decisions = 0., 0, 0, 0
    for block in range(config["blocks"]):
        folder = run / f"block-{block:02d}"
        report = json.loads((folder / "complete.json").read_text())
        for file, key in [("checkpoint.pt","checkpoint_sha256"), ("decisions.npz","decisions_sha256"),
                          ("hands.jsonl.gz","hands_sha256")]:
            assert digest(folder / file) == report[key]
        with gzip.open(folder / "hands.jsonl.gz", "rt") as stream:
            episodes = [json.loads(line) for line in stream]
        lo = block * config["hands_per_block"]
        assert len(episodes) == config["hands_per_block"]
        data = dict(np.load(folder / "decisions.npz", allow_pickle=False))
        assert len(data["action"]) == len(episodes) * 7
        assert np.isfinite(data["old_probs"]).all() and (data["old_probs"] >= 0).all()
        assert np.allclose(data["old_probs"].sum(1), 1., atol=1e-6)
        assert (data["old_probs"][~data["legal"]] == 0).all()
        assert np.allclose(np.exp(data["old_logp"]), data["old_probs"][np.arange(len(data["action"])),data["action"]],atol=1e-7)
        for hand, episode in enumerate(episodes):
            assert all(episode[k] == v for k,v in specs[lo+hand].items())
            learner = episode["learner"]
            rows = np.flatnonzero(data["hand"] == hand)
            assert len(rows) == 7
            # Literal, independent reconstruction of the accepted input contract.
            tokens = [42,45,28+(episode["bidder"]-learner)%4,32+episode["decl"],43,
                      *sorted(episode["hands"][learner]),44]
            rules, rec = rules_for(episode["decl"]), empty_record(episode["bidder"])
            own = mask_of(episode["hands"][learner])
            own_turn = 0
            for actor, tile in episode["plays"]:
                assert actor == rec.seat
                legal = rules.legal(mask_of(episode["hands"][actor]) & ~rec.played, rec.led)
                assert legal & (1 << tile)
                if actor == learner:
                    row = rows[own_turn]; own_turn += 1
                    prefix = tokens + [28]
                    assert data["length"][row] == len(prefix)
                    assert np.array_equal(data["x"][row,:len(prefix)], prefix)
                    assert (data["x"][row,len(prefix):] == 0).all()
                    own_legal = rules.legal(own & ~rec.played, rec.led)
                    assert np.array_equal(data["legal"][row], [(own_legal >> t) & 1 for t in range(28)])
                    assert data["action"][row] == tile
                tokens += [28+(actor-learner)%4,tile]
                rec = rec.advance(tile,rules)
            assert rec.finished and list(rec.banked) == episode["points"] and sum(rec.banked) == 42
            bidder_team = episode["bidder"] % 2
            made = rec.banked[bidder_team] >= 30
            reward = 1. if made == (learner % 2 == bidder_team) else -1.
            assert episode["reward"] == reward and (data["reward"][rows] == reward).all()
        metrics = report["optimizer"]
        assert metrics["baseline"] == (prior_return / prior_hands if prior_hands else 0.)
        assert metrics["initial_logp_error"] <= 5e-5
        assert 1 <= metrics["passes_completed"] <= 4
        assert len(metrics["passes"]) == metrics["passes_completed"]
        assert all(p["kl"] <= .03 for p in metrics["passes"][:-1])
        assert metrics["passes_completed"] == 4 or metrics["passes"][-1]["kl"] > .03
        assert all(np.isfinite(v) for p in metrics["passes"] for v in p.values())
        updates += metrics["passes_completed"] * ((len(data["action"])+255)//256)
        checkpoint = folder / "checkpoint.pt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert saved["tokenizer"] == VERSION and saved["config"]["layers"] == generation["layers"]
        assert saved["config"]["width"] == 64 and saved["config"]["heads"] == 2
        assert saved["config"]["vocabulary"] == 46 and saved["config"]["context"] == 68
        assert saved["parent_checkpoint_sha256"] == digest(previous)
        prior_hands += len(episodes); prior_return += sum(e["reward"] for e in episodes)
        assert saved["prior_hands"] == prior_hands and saved["prior_return"] == prior_return
        assert saved["rl_updates"] == updates == report["rl_updates"]
        assert all(torch.isfinite(t).all() for t in saved["model"].values())
        assert any(not torch.equal(t,previous_weights[k]) for k,t in saved["model"].items()), "weights never updated"
        for group in saved["optimizer"]["param_groups"]:
            assert group["lr"] == config["lr"] and group["weight_decay"] == 0.
            assert tuple(group["betas"]) == (.9,.95)
        previous_weights = saved["model"]
        previous = checkpoint; decisions += len(data["action"])
    assert digest(previous) == generation["sha256"]
    publication = json.loads((run / "published.json").read_text())
    assert publication["anonymous_model_and_replay_verified"]
    return {"generation":generation["id"], "hands":prior_hands, "decisions":decisions,
            "checkpoint_sha256":generation["sha256"],
            "updates":updates, "layers":generation["layers"], "archive_revision":publication["archive_revision"],
            "exact_visible_inputs_legal_outputs_rewards_and_checkpoint_chain_verified":True}


def audit(root):
    root = Path(root).resolve()
    torch.set_num_threads(2)
    state = json.loads((root / "state.json").read_text())
    config = json.loads((root / "campaign.json").read_text())
    source = config["source_deals"]
    assert digest(source["path"]) == source["sha256"]
    excluded = {tuple(r["hands"]) for r in json.loads(Path(source["path"]).read_text())}
    for label in ("development", "confirmation"):
        panel = json.loads((root / f"{label}-specs.json").read_text())
        for spec in panel:
            deal = tuple(mask_of(h) for h in spec["hands"])
            assert deal not in excluded
            excluded.add(deal)
    generations = state["generations"]
    with gzip.open(root / "baseline/hands.jsonl.gz", "rt") as stream:
        baseline = [json.loads(line) for line in stream]
    results = []
    for i,generation in enumerate(generations[1:],start=1):
        expected = [g["id"] for g in generations[max(0,i-5):i]]
        results.append(audit_generation(root/generation["id"],generation,expected,excluded))
        results[-1]["development"] = audit_panel(root,generation["id"],baseline)
        print(f"AUDITED {generation['id']}",flush=True)
    report = {"generations":results, "completed_generation_count":len(results),
              "total_hands":sum(r["hands"] for r in results), "total_learner_decisions":sum(r["decisions"] for r in results),
              "all_completed_generations_verified":True, "source_sha256":digest(__file__)}
    atomic_json(root/"audit.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.root)),flush=True)
