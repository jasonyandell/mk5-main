# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0
"""Final publication exports the preselected specimen, never the latest one."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from lem.observer.data import VERSION
from lem.observer.model import Config, Observer
from lem.observer.population import atomic_json, digest
from lem.observer.release import DATA_REPO, MODEL_REPO, publish_final


def test_final_release_roundtrip_selects_weights_before_confirmation(tmp_path, monkeypatch):
    import huggingface_hub as hub
    from safetensors.torch import load_file

    torch.set_num_threads(2)
    torch.manual_seed(123)
    model = Observer(Config(layers=15)).eval()
    checkpoint = tmp_path / "selected.pt"
    torch.save({"config":asdict(model.config), "model":model.state_dict(), "step":123,
                "tokenizer":VERSION, "data_sha256":"trajectories"}, checkpoint)
    selected = {"path":str(checkpoint), "sha256":digest(checkpoint)}
    state = {"status":"complete", "depth_index":4, "stale":4,
             "generations":[{"id":"g000"},{"id":"g001"}]}
    atomic_json(tmp_path/"state.json", state)
    atomic_json(tmp_path/"campaign.json", {"depths":[2,4,8,12,15], "plateau_patience":4, "confirmation_hands":512})
    panel = {"hands":512, "team_win_rate":.55, "specs_sha256":"same-panel",
             "checkpoint_sha256":selected["sha256"],
             "vs_reference":{"win_rate_difference":.05, "ci95":[.01,.09]}}
    atomic_json(tmp_path/"final.json", {"selected":selected, "confirmation":panel,
        "baseline":{**panel,"team_win_rate":.5}, "state":state})
    for label in ("confirmation-g0", "confirmation-selected"):
        folder=tmp_path/label;folder.mkdir()
        atomic_json(folder/"report.json", panel)
        (folder/"hands.jsonl.gz").write_bytes(b"opaque-replay-bytes")
    latest=tmp_path/"g001";stage=latest/"release";stage.mkdir(parents=True)
    atomic_json(latest/"published.json", {"model_revision":"latest"})
    atomic_json(stage/"example.json", {"tokens":[42,45,28],"legal_tile_ids":[0,1]})
    atomic_json(stage/"tokenizer.json", {"encoding":VERSION})
    (stage/"LICENSE").write_text("CC0-1.0")
    # Copying the latest release alone would publish stale values.
    (stage/"model.safetensors").write_bytes(b"wrong-latest-weights")
    (stage/"provenance.json").write_text('{"generation":"wrong-latest"}')

    commits={};tags={};calls=[]
    class FakeHub:
        def create_commit(self, **kw):
            revision=f"commit-{len(commits)}"
            contents={}
            for operation in kw["operations"]:
                source=operation.path_or_fileobj
                contents[operation.path_in_repo] = source if isinstance(source, bytes) else Path(source).read_bytes()
            commits[revision]=contents;calls.append(kw["repo_id"])
            return SimpleNamespace(oid=revision)

        def create_tag(self, repo_id, tag, revision, **kw):
            tags[tag]=revision

        def model_info(self, repo_id, revision):
            return SimpleNamespace(sha=tags[revision])

        def dataset_info(self, repo_id):
            return SimpleNamespace(sha="dataset-before")

    def download(repo_id, filename, revision, local_dir=None, **kw):
        folder=Path(local_dir) if local_dir else tmp_path/"fake-cache"
        path=folder/filename;path.parent.mkdir(parents=True,exist_ok=True)
        raw = b"---\nlicense: cc0-1.0\n---\nExisting research\n" if revision=="dataset-before" else commits[revision][filename]
        path.write_bytes(raw)
        return str(path)

    monkeypatch.setattr(hub,"HfApi",FakeHub)
    monkeypatch.setattr(hub,"hf_hub_download",download)
    monkeypatch.setattr(hub.ModelCard,"validate",lambda self:None)
    result=publish_final(tmp_path)
    assert calls == [MODEL_REPO, DATA_REPO]
    assert result["parameters"] == 757184
    exported=load_file(str(tmp_path/"verify-final/model.safetensors"))
    adapter=torch.load(tmp_path/"verify-final/gus.pt",weights_only=True)
    for name, value in model.state_dict().items():
        assert torch.equal(exported[name], value)
        assert torch.equal(adapter["model"][name], value)
    example=json.loads((tmp_path/"verify-final/example.json").read_text())
    with torch.inference_mode():
        logits=model(torch.tensor([example["tokens"]]))[0,-1]
    assert torch.equal(logits, torch.tensor(example["expected_last_logits"]))
    provenance=json.loads((tmp_path/"verify-final/provenance.json").read_text())
    assert provenance["selection"] == selected
    assert provenance["selected_before_confirmation"]
    assert publish_final(tmp_path) == result
    assert len(calls) == 2  # A completed retry cannot publish a second selection.

    (tmp_path/"final-published.json").unlink()
    state["status"]="training"
    atomic_json(tmp_path/"state.json",state)
    with pytest.raises(ValueError, match="finish all registered depths"):
        publish_final(tmp_path)
    assert len(calls) == 2
