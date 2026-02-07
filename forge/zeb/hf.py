"""HuggingFace Hub integration for model weight distribution.

Workers pull latest weights; the learner pushes after each training step.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from forge.zeb.model import ZebModel

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None
    hf_hub_download = None


def _require_hf():
    if HfApi is None:
        raise ImportError("pip install huggingface_hub")


def init_repo(repo_id: str, model_config: dict) -> str:
    """Create HF repo if it doesn't exist. Upload initial config.json.

    Returns the repo URL.
    """
    _require_hf()
    api = HfApi()
    url = api.create_repo(repo_id, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(model_config, f)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo='config.json',
            repo_id=repo_id,
            commit_message='initial config',
        )
    return str(url)


def push_weights(
    model: ZebModel,
    repo_id: str,
    step: int,
    total_games: int,
    extra_metadata: dict | None = None,
):
    """Push model state_dict + config.json + training_state.json to HF Hub."""
    _require_hf()
    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Model weights
        torch.save(model.state_dict(), tmp_dir / 'model.pt')

        # Training state (tiny JSON, checked first by pull_weights_if_new)
        state = {'step': step, 'total_games': total_games}
        if extra_metadata:
            state.update(extra_metadata)
        (tmp_dir / 'training_state.json').write_text(json.dumps(state))

        api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            commit_message=f"step {step}, {total_games} games",
        )


def pull_weights(repo_id: str, device: str = 'cpu') -> tuple[dict, dict]:
    """Pull latest weights from HF. Returns (state_dict, model_config).

    Uses HF cache -- only downloads if changed.
    """
    _require_hf()
    config_path = hf_hub_download(repo_id, 'config.json')
    weights_path = hf_hub_download(repo_id, 'model.pt')

    with open(config_path) as f:
        model_config = json.load(f)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    return state_dict, model_config


def get_remote_step(repo_id: str) -> int:
    """Read the current training step from HuggingFace."""
    _require_hf()
    state_path = hf_hub_download(repo_id, 'training_state.json')
    with open(state_path) as f:
        state = json.load(f)
    return state.get('step', 0)


def pull_weights_if_new(
    repo_id: str,
    current_step: int,
    device: str = 'cpu',
) -> tuple[dict, dict, int] | None:
    """Pull only if remote step > current_step.

    Returns (state_dict, model_config, new_step) or None if up-to-date.
    """
    _require_hf()
    state_path = hf_hub_download(repo_id, 'training_state.json')
    with open(state_path) as f:
        state = json.load(f)

    remote_step = state.get('step', 0)
    if remote_step <= current_step:
        return None

    state_dict, model_config = pull_weights(repo_id, device)
    return state_dict, model_config, remote_step
