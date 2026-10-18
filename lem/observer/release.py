# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Publish versioned Gus candidates and allowlisted, reproducible evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import torch

from .data import label
from .population import atomic_json, digest, load_model

MODEL_REPO = "jasonyandell/gus"
DATA_REPO = "jasonyandell/texas-42-zeb-research-2026-10"
ARCHIVE = "lem/gus-depth-campaign-2026-10-18"


def publish_generation(root, run, checkpoint):
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download, ModelCard
    from safetensors.torch import save_file, load_file
    root,run=Path(root),Path(run)
    receipt=run/"published.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    config=json.loads((run/"config.json").read_text())
    result=json.loads((run/"result.json").read_text())
    name=config["generation"]
    stage=run/"release";stage.mkdir(exist_ok=True)
    model,saved=load_model(checkpoint,"cpu")
    state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    save_file(state,str(stage/"model.safetensors"),metadata={"format":"pt","name":"Gus","generation":name})
    inference={k:saved[k] for k in ("config","model","step","tokenizer","data_sha256")}
    torch.save(inference,stage/"gus.pt")
    atomic_json(stage/"config.json",saved["config"])
    atomic_json(stage/"tokenizer.json",{"encoding":saved["tokenizer"],"input_tokens":[{"id":i,"label":label(i)} for i in range(46)],
                "tile_outputs":28,"context":68,"prefix":"BOS BID30 relative_bidder DECL HAND own_7_sorted_tiles PLAY",
                "history":"relative_actor tile pairs; end at next actor","own_actor_id":28,
                "unsupported_id":40,"note":"Game-token schema, not a text tokenizer."})
    atomic_json(stage/"provenance.json",{"generation":name,"source_checkpoint_sha256":digest(checkpoint),
                "training":config,"result":result,"parameters":sum(p.numel() for p in model.parameters()),
                "archive":f"https://huggingface.co/datasets/{DATA_REPO}/tree/main/{ARCHIVE}/{name}"})
    atomic_json(stage/"evaluation.json",{"development":result["development"],
                "smokes":{s:json.loads((run/f"smoke-{s}/report.json").read_text()) for s in ("parent","zeb")}})
    example={"tokens":[42,45,28,38,43,0,2,5,9,14,20,27,44,28],"legal_tile_ids":[0,2,5,9,14,20,27]}
    torch.set_num_threads(2)
    with torch.inference_mode():
        logits=model(torch.tensor([example["tokens"]]))[0,-1]
    example.update(expected_last_logits=logits.tolist(),expected_argmax_legal_tile=max(example["legal_tile_ids"],key=lambda t:float(logits[t])))
    atomic_json(stage/"example.json",example)
    atomic_json(stage/"export-verification.json",{"source_checkpoint_sha256":digest(checkpoint),
                "parameter_count":sum(p.numel() for p in model.parameters()),
                "all_exported_tensors_match_source":all(torch.equal(state[k],saved["model"][k]) for k in state)})
    (stage/"LICENSE").write_text(Path("LICENSES/CC0-1.0.txt").read_text())
    params=sum(p.numel() for p in model.parameters())
    card=f'''---
license: cc0-1.0
library_name: pytorch
model_name: Gus
tags: [texas-42, dominoes, nanogpt, causal-transformer, reinforcement-learning, interpretability, custom-code]
---
# Gus — population candidate {name}

**{params:,} parameters, {saved['config']['layers']} blocks, width 64, two attention heads.**
This is an exploratory reward-trained candidate in the Gus depth campaign.
Latest does not mean strongest. The initial pretrained model remains frozen at
[`pretrained-v0`](https://huggingface.co/{MODEL_REPO}/tree/pretrained-v0).

## Inputs and outputs

Unchanged original own seven tiles, public bid-30 contract/bidder/declaration,
relative seats and public actor/tile history. No hidden hands, player identities,
planner values, legal-action list or match score are inputs. There are 46 input
tokens, 68 context positions and 28 tile logits. Tile outputs are masked by the
acting seat's lawful legal moves when playing. No bidding, Nello, value or splat
head is present. Game-token definitions are in `tokenizer.json`.

## Learning and scope

Starts from {config['parent']['sha256']}. This generation collected 4,096 fresh
hands, eight batches of 512. One learner seat samples legal softmax; each other
seat is independently 25% pinned Zeb and 75% uniform among the recent-five Gus
pool. Frozen Gus seats use legal argmax. PPO uses only the learner's own actions,
actual team make/set return, a prior-batch scalar baseline, and no hidden critic.
Bid 30 and pip declarations throughout. Exact settings and model/field pins are
in `provenance.json`; full episodes and optimizer checkpoints are archived.

The development score is {100*result['development']['team_win_rate']:.2f}% on a
fixed 256-hand G0/Zeb mixed table. This panel is reused for adaptive selection;
it is exploratory and is not a general win rate or independent confirmation.
Two full-game smokes versus the parent and two versus Zeb check integration only.
Native-neuron meanings remain ungraded. The depth stopping rule is four
generations without a two-percentage-point development improvement.

[Live/training run]({result['wandb']}).
[Protocol](https://github.com/jasonyandell/mk5-main/blob/main/wiki/decisions/gus-population-iteration.md).

## Load

Use the repository's `lem.observer.model.Observer` and `Config` classes with
`config.json` and `model.safetensors`, or pass downloaded `gus.pt` to
`lem.observer.play.ObserverPlay`. The code and loading pattern are the same as
the [pretrained release](https://huggingface.co/{MODEL_REPO}/blob/pretrained-v0/README.md);
set the Hub revision to **`gus-{name}`** and read this release's example's
`expected_argmax_legal_tile` rather than assuming six-six. This is custom PyTorch
code, not a Transformers AutoModel or prose model. `gus.pt` is inference-only,
compatible with `torch.load(..., weights_only=True)`; optimizer states are in
the [research archive](https://huggingface.co/datasets/{DATA_REPO}/tree/main/{ARCHIVE}/{name}).

Weights/artifacts: CC0-1.0, Jason Yandell. The nanoGPT-derived implementation is
MIT, Andrej Karpathy and Jason Yandell; retain its license. `SHA256SUMS` covers
release files. The historical April oracle-era Gus is a separate model lineage.
'''
    (stage/"README.md").write_text(card)
    ModelCard.load(stage/"README.md").validate()
    for p in stage.iterdir():
        if p.suffix in (".md",".json"):
            if re.search(r"hf_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY",p.read_text()):
                raise ValueError("credential-shaped release content")
    (stage/"SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(stage.iterdir()) if p.name!="SHA256SUMS"))
    api=HfApi()
    phases=run/"publication-progress.json"
    progress=json.loads(phases.read_text()) if phases.exists() else {}
    if "model_revision" not in progress:
        commit=api.create_commit(repo_id=MODEL_REPO,repo_type="model",commit_message=f"Gus {name}: depth {saved['config']['layers']} population candidate",
                operations=[CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=str(p)) for p in sorted(stage.iterdir())])
        progress["model_revision"]=commit.oid;atomic_json(phases,progress)
    api.create_tag(MODEL_REPO,tag=f"gus-{name}",revision=progress["model_revision"],exist_ok=True)
    if api.model_info(MODEL_REPO,revision=f"gus-{name}").sha != progress["model_revision"]:
        raise ValueError("generation tag was already used for a different release")
    if "archive_revision" not in progress:
        files=[run/p for p in ["config.json","specs.json","scheduled-deals.json","initial.pt","ready.json","result.json","wandb-run.json"]]
        for directory in sorted(run.glob("block-*")):
            files += [directory/p for p in ["checkpoint.pt","decisions.npz","hands.jsonl.gz","complete.json"]]
        files += [run/"development"/p for p in ["report.json","hands.jsonl.gz"]]
        for opponent in ("parent","zeb"):
            files += [run/f"smoke-{opponent}"/p for p in ["manifest.json","report.json","games.jsonl.gz"]]
        total=sum(p.stat().st_size for p in files)
        if total > 2_000_000_000:
            raise ValueError("generation archive exceeds the 2GB per-upload review limit")
        common=[root/p for p in ["campaign.json","development-specs.json","confirmation-specs.json","prediction-deals.json","prediction.npz"]]
        common += [root / "baseline" / p for p in ["report.json", "hands.jsonl.gz"]]
        common += list(root.glob(f"{name}-*.log")) + list(root.glob(f"{name}-*-execution.json"))
        common += list(root.glob(f"prepare-{name}*"))
        operations=[CommitOperationAdd(path_in_repo=f"{ARCHIVE}/{name}/{p.relative_to(run)}",path_or_fileobj=str(p)) for p in files]
        operations += [CommitOperationAdd(path_in_repo=f"{ARCHIVE}/{p.relative_to(root)}",path_or_fileobj=str(p)) for p in common]
        commit=api.create_commit(repo_id=DATA_REPO,repo_type="dataset",commit_message=f"Gus {name}: complete formation and lawful population trajectories",
                                operations=operations)
        progress.update(archive_revision=commit.oid,archive_bytes=total);atomic_json(phases,progress)
    downloaded=Path(hf_hub_download(MODEL_REPO,"model.safetensors",revision=progress["model_revision"],token=False,local_dir=run/"verify-model"))
    if digest(downloaded) != digest(stage/"model.safetensors"):
        raise ValueError("published model hash mismatch")
    restored=load_file(str(downloaded))
    if any(not torch.equal(state[k],restored[k]) for k in state):
        raise ValueError("published tensors differ")
    sample=run/"block-00/hands.jsonl.gz"
    remote=Path(hf_hub_download(DATA_REPO,f"{ARCHIVE}/{name}/block-00/hands.jsonl.gz",repo_type="dataset",
                    revision=progress["archive_revision"],token=False,local_dir=run/"verify-evidence"))
    if digest(remote)!=digest(sample):
        raise ValueError("published replay hash mismatch")
    result={**progress,"model_url":f"https://huggingface.co/{MODEL_REPO}/tree/gus-{name}",
            "anonymous_model_and_replay_verified":True,"weights_sha256":digest(downloaded)}
    atomic_json(receipt,result)
    return result


def publish_final(root):
    """Publish the selected final-depth specimen after reserved confirmation.

    Selection is already fixed. Publishing must not select again on confirmation.
    The selected specimen may be an identity-grown initialization if no trained
    candidate at that depth surpassed the inherited development score.
    """
    from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download, ModelCard
    from safetensors.torch import save_file, load_file
    root = Path(root).resolve()
    receipt = root / "final-published.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    state = json.loads((root / "state.json").read_text())
    final = json.loads((root / "final.json").read_text())
    campaign = json.loads((root / "campaign.json").read_text())
    if (state["status"] != "complete" or state["depth_index"] != len(campaign["depths"])-1
            or state["stale"] < campaign["plateau_patience"]):
        raise ValueError("finish all registered depths and confirmation before publishing selection")
    checkpoint = Path(final["selected"]["path"])
    if digest(checkpoint) != final["selected"]["sha256"]:
        raise ValueError("selected checkpoint changed")
    if final["confirmation"]["checkpoint_sha256"] != digest(checkpoint):
        raise ValueError("confirmation did not evaluate selected checkpoint")
    if final["baseline"]["specs_sha256"] != final["confirmation"]["specs_sha256"]:
        raise ValueError("confirmation comparisons use different panels")
    if final["confirmation"]["hands"] != campaign["confirmation_hands"]:
        raise ValueError("confirmation panel is incomplete")
    model, saved = load_model(checkpoint, "cpu")
    parameters = sum(p.numel() for p in model.parameters())
    if not 709_000 <= parameters <= 800_000 or saved["config"]["layers"] != campaign["depths"][-1]:
        raise ValueError("selected model has not reached the requested size")
    # Encoding/license are unchanged; all weight-dependent files are replaced.
    stage = root / "final-release"
    last = root / state["generations"][-1]["id"] / "release"
    shutil.copytree(last, stage, dirs_exist_ok=True)
    tensors = {k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    save_file(tensors, str(stage / "model.safetensors"), metadata={"format":"pt", "name":"Gus"})
    torch.save({k:saved[k] for k in ("config","model","step","tokenizer","data_sha256")}, stage / "gus.pt")
    atomic_json(stage / "config.json", saved["config"])
    atomic_json(stage / "evaluation.json", final)
    atomic_json(stage / "provenance.json", {
        "selection": final["selected"], "parameters": parameters, "campaign": campaign,
        "generations": state["generations"], "selected_before_confirmation": True,
        "identity_growth_initialization": checkpoint.name == "initial.pt",
        "git_commit": subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip(),
    })
    example = json.loads((stage / "example.json").read_text())
    torch.set_num_threads(2)
    with torch.inference_mode():
        logits = model(torch.tensor([example["tokens"]]))[0,-1]
    example.update(expected_last_logits=logits.tolist(), expected_argmax_legal_tile=max(
        example["legal_tile_ids"], key=lambda t:float(logits[t])))
    atomic_json(stage / "example.json", example)
    atomic_json(stage / "export-verification.json", {
        "source_checkpoint_sha256": digest(checkpoint), "parameter_count": parameters,
        "all_exported_tensors_match_source": all(torch.equal(tensors[k], saved["model"][k]) for k in tensors),
    })
    confirmation = final["confirmation"]
    delta = confirmation["vs_reference"]
    lo, hi = delta["ci95"]
    tag = "campaign-selected-20261018"
    card = f'''---
license: cc0-1.0
library_name: pytorch
model_name: Gus
tags: [texas-42, dominoes, nanogpt, causal-transformer, reinforcement-learning, interpretability, custom-code]
---
# Gus — selected 757k depth-campaign specimen

**{parameters:,} parameters; {saved['config']['layers']} blocks, width 64, two heads.**
The registered population campaign finished at its final depth's operational plateau.
Selection used the fixed 256-hand development panel, before the fresh confirmation
panel was evaluated. The complete formation history remains archived.

## Confirmation and scope

On the reserved {confirmation['hands']}-hand G0/Zeb mixed-table panel, this model's
team won **{100*confirmation['team_win_rate']:.2f}%**, versus G0's
**{100*final['baseline']['team_win_rate']:.2f}%**. The paired difference is
**{100*delta['win_rate_difference']:+.2f} percentage points**, 95% whole-hand
bootstrap interval **[{100*lo:+.2f}, {100*hi:+.2f}]** (2,000 resamples, seed 42).
This is one candidate-controlled seat, fixed mixed partners/opponents, bid 30,
pip declarations and legal argmax. It is not a general tournament rating.
The separate full-game comparisons were two-game integration smokes only.

The stopping rule was four generations without a two-percentage-point development
gain, at depths 2, 4, 8, 12 and 15. Best development weights, including a depth's
identity-grown initialization, remained eligible. Identity-grown initialization
selected: **{checkpoint.name == 'initial.pt'}**. An operational plateau does not
prove a capacity ceiling. Neuron meanings remain for separate interpretation.

## Inputs, outputs and learning

The original own seven tiles, public contract, relative seats and public play
history are unchanged: 46 input tokens, 68 positions, 28 tile logits. There are no
hidden hands, identities, planner values, legal-action lists or match scores in
the input. Own legal moves constrain output selection. No bidding, Nello, value
or splat head is present.

Each generation collected 4,096 hands and used policy-only PPO on actual team
make/set returns. Other seats sampled 25% pinned Zeb and 75% uniform among the
latest five completed Gus generations. All generations and formation checkpoints
are retained, including the initial [pretrained-v0](https://huggingface.co/{MODEL_REPO}/tree/pretrained-v0).

## Load and reproduce

Pin Hub revision **`{tag}`**. Use `lem.observer.model.Observer(Config(**config))`
with `config.json` and `model.safetensors`, or pass `gus.pt` to
`lem.observer.play.ObserverPlay`. These are custom PyTorch classes, not a
Transformers AutoModel. `example.json` contains an exact input and expected CPU
float32 logits; `tokenizer.json` defines the game tokens. The adapter checkpoint
supports `torch.load(..., weights_only=True)`.

[Experiment and protocol](https://github.com/jasonyandell/mk5-main/blob/main/wiki/experiments/gus-population-depth-2026-10-18.md).
[Full formation and confirmation evidence](https://huggingface.co/datasets/{DATA_REPO}/tree/main/{ARCHIVE}).
Weights and generated artifacts: CC0-1.0, Jason Yandell. The nanoGPT-derived
implementation retains MIT attribution to Andrej Karpathy and Jason Yandell.
The April oracle-era Gus is a different model lineage.
'''
    (stage / "README.md").write_text(card)
    ModelCard.load(stage / "README.md").validate()
    for path in stage.iterdir():
        if path.suffix in (".json", ".md") and re.search(r"hf_[A-Za-z0-9]{20,}|BEGIN .*PRIVATE KEY",path.read_text()):
            raise ValueError("credential-shaped final release content")
    paths = sorted(p for p in stage.iterdir() if p.name != "SHA256SUMS")
    (stage / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in paths))
    api = HfApi()
    progress_path = root / "final-publication-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if "model_revision" not in progress:
        commit = api.create_commit(repo_id=MODEL_REPO, commit_message="Gus: selected 757k specimen and reserved confirmation",
            operations=[CommitOperationAdd(path_in_repo=p.name,path_or_fileobj=str(p)) for p in sorted(stage.iterdir())])
        progress["model_revision"] = commit.oid
        atomic_json(progress_path, progress)
    api.create_tag(MODEL_REPO,tag=tag,revision=progress["model_revision"],exist_ok=True)
    if api.model_info(MODEL_REPO,revision=tag).sha != progress["model_revision"]:
        raise ValueError("selected tag points at a different release")
    if "archive_revision" not in progress:
        files = [root/p for p in ["final.json","state.json","campaign.json"]]
        for label in ("confirmation-g0", "confirmation-selected"):
            files += [root/label/p for p in ("report.json", "hands.jsonl.gz")]
            files += list(root.glob(f"{label}*.log")) + list(root.glob(f"{label}*-execution.json"))
        files += [root / g["id"] / "published.json" for g in state["generations"][1:]]
        # Include the original baseline cost and the disposable GPU preflight.
        files += list(root.glob("baseline*.log")) + list(root.glob("baseline*-execution.json"))
        if sum(p.stat().st_size for p in files) > 2_000_000_000:
            raise ValueError("final evidence exceeds the 2GB per-upload review limit")
        operations = [CommitOperationAdd(path_in_repo=f"{ARCHIVE}/{p.relative_to(root)}",path_or_fileobj=str(p)) for p in files]
        preflight = root.parent / "preflight"
        for relative in ("config.json", "specs.json", "block-00/complete.json", "block-00/hands.jsonl.gz",
                         "block-00/decisions.npz", "block-00/checkpoint.pt"):
            path = preflight / relative
            if path.exists():
                operations.append(CommitOperationAdd(path_in_repo=f"{ARCHIVE}/preflight/{relative}",path_or_fileobj=str(path)))
        head = api.dataset_info(DATA_REPO).sha
        readme = Path(hf_hub_download(DATA_REPO,"README.md",repo_type="dataset",revision=head)).read_text()
        if ARCHIVE not in readme:
            readme += f"\n## Gus population depth campaign\n\n`{ARCHIVE}/`: complete PPO trajectories, formation checkpoints, fixed development and reserved confirmation panels, and publication receipts. [Protocol and findings](https://github.com/jasonyandell/mk5-main/blob/main/wiki/experiments/gus-population-depth-2026-10-18.md). Generated by the unchanged public-history/own-hand Gus policy and pinned Zeb; exact code, field and population hashes are retained per generation. Artifacts are CC0. Credentials, local W&B caches and third-party implementation files are excluded.\n"
            operations.append(CommitOperationAdd(path_in_repo="README.md",path_or_fileobj=readme.encode()))
        commit = api.create_commit(repo_id=DATA_REPO,repo_type="dataset",parent_commit=head,
            commit_message="Gus: reserved confirmation and completed depth campaign",operations=operations)
        progress["archive_revision"] = commit.oid
        atomic_json(progress_path,progress)
    # Verify every final release file through a new anonymous download.
    for path in stage.iterdir():
        restored = Path(hf_hub_download(MODEL_REPO,path.name,revision=progress["model_revision"],
                                       token=False,local_dir=root/"verify-final"))
        if digest(restored) != digest(path):
            raise ValueError(f"final release hash mismatch: {path.name}")
    restored_tensors = load_file(str(root/"verify-final/model.safetensors"))
    if any(not torch.equal(tensors[k], restored_tensors[k]) for k in tensors):
        raise ValueError("final exported weights differ from selected checkpoint")
    sample = Path(hf_hub_download(DATA_REPO,f"{ARCHIVE}/confirmation-selected/hands.jsonl.gz",
        repo_type="dataset",revision=progress["archive_revision"],token=False,local_dir=root/"verify-final-evidence"))
    if digest(sample) != digest(root/"confirmation-selected/hands.jsonl.gz"):
        raise ValueError("final confirmation replay hash mismatch")
    result = {**progress,"model_url":f"https://huggingface.co/{MODEL_REPO}/tree/{tag}",
              "all_release_files_anonymously_verified":True,"selected_sha256":digest(checkpoint),
              "parameters":parameters}
    atomic_json(receipt,result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Publish completed Gus campaign selection")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(publish_final(args.root)), flush=True)
