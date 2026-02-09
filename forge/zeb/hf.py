"""HuggingFace Hub integration for weight and example distribution.

Weights: learner pushes after training; workers pull periodically.
Examples: workers upload .pt batches; learner downloads new ones.
  The examples repo IS the replay buffer — rebuilt from HF on restart.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import torch

from forge.zeb.model import ZebModel

try:
    from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download
except ImportError:
    CommitOperationDelete = None
    HfApi = None
    hf_hub_download = None


DEFAULT_REPO = 'jasonyandell/zeb-42'


def _namespace(weights_name: str) -> tuple[str, str]:
    """Derive config/state filenames from weights_name.

    'model.pt' → ('config.json', 'training_state.json')  # backward compat
    'large.pt' → ('large-config.json', 'large-state.json')
    """
    stem = weights_name.removesuffix('.pt')
    if stem == 'model':
        return 'config.json', 'training_state.json'
    return f'{stem}-config.json', f'{stem}-state.json'


def _require_hf():
    if HfApi is None:
        raise ImportError("pip install huggingface_hub")


def _is_transient(e: Exception) -> bool:
    """Check if an exception is a transient network error worth retrying."""
    # Check exception class names up the MRO
    type_names = ' '.join(type(e).__mro__.__class__.__name__ if hasattr(type(e).__mro__, '__class__') else
                          cls.__name__ for cls in type(e).__mro__)
    err_str = f"{type(e).__name__}: {e}"
    for keyword in ('timeout', 'connection', 'reset', 'broken', '502', '503', '429'):
        if keyword in err_str.lower() or keyword in type_names.lower():
            return True
    return False


def _retry(fn, max_retries=3, base_delay=5):
    """Retry a function on transient network errors with exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            if not _is_transient(e) or attempt == max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"  HF transient error (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {e}")
            print(f"  Retrying in {delay}s...")
            time.sleep(delay)


def init_repo(repo_id: str, model_config: dict, weights_name: str = 'model.pt') -> str:
    """Create HF repo if it doesn't exist. Upload initial config.

    Returns the repo URL.
    """
    _require_hf()
    api = HfApi()
    url = api.create_repo(repo_id, exist_ok=True)
    config_file, _ = _namespace(weights_name)
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(model_config, f)
        f.flush()
        api.upload_file(
            path_or_fileobj=f.name,
            path_in_repo=config_file,
            repo_id=repo_id,
            commit_message=f'initial config ({config_file})',
        )
    return str(url)


def push_weights(
    model: ZebModel,
    repo_id: str,
    step: int,
    total_games: int,
    extra_metadata: dict | None = None,
    weights_name: str = 'model.pt',
):
    """Push model state_dict + config + training state to HF Hub."""
    _require_hf()
    api = HfApi()
    _, state_file = _namespace(weights_name)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # Model weights
        torch.save(model.state_dict(), tmp_dir / weights_name)

        # Training state (tiny JSON, checked first by pull_weights_if_new)
        state = {'step': step, 'total_games': total_games}
        if extra_metadata:
            state.update(extra_metadata)
        (tmp_dir / state_file).write_text(json.dumps(state))

        _retry(lambda: api.upload_folder(
            folder_path=tmp,
            repo_id=repo_id,
            commit_message=f"step {step}, {total_games} games",
        ))


def pull_weights(repo_id: str, device: str = 'cpu', weights_name: str = 'model.pt') -> tuple[dict, dict]:
    """Pull latest weights from HF. Returns (state_dict, model_config).

    Uses HF cache -- only downloads if changed.
    """
    _require_hf()
    config_file, _ = _namespace(weights_name)
    config_path = _retry(lambda: hf_hub_download(repo_id, config_file))
    weights_path = _retry(lambda: hf_hub_download(repo_id, weights_name))

    with open(config_path) as f:
        model_config = json.load(f)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    return state_dict, model_config


def load_zeb_from_hf(
    repo_id: str = DEFAULT_REPO,
    device: str = 'cuda',
    weights_name: str = 'model.pt',
) -> ZebModel:
    """Load a ZebModel from HuggingFace Hub, ready for inference.

    Downloads config.json + weights (cached by HF Hub).
    """
    state_dict, config = pull_weights(repo_id, device, weights_name)
    model = ZebModel(**config)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def get_remote_step(repo_id: str, weights_name: str = 'model.pt') -> int:
    """Read the current training step from HuggingFace."""
    state = get_remote_training_state(repo_id, weights_name)
    if state is None:
        return 0
    return int(state.get('step', 0))


def get_remote_training_state(repo_id: str, weights_name: str = 'model.pt') -> dict | None:
    """Read training state JSON from HuggingFace.

    Returns:
        Parsed dict if present, else None (e.g., first-run bootstrap before
        the state file exists).
    """
    _require_hf()
    _, state_file = _namespace(weights_name)
    try:
        state_path = _retry(lambda: hf_hub_download(repo_id, state_file))
    except Exception:
        return None

    with open(state_path) as f:
        return json.load(f)


def pull_weights_if_new(
    repo_id: str,
    current_step: int,
    device: str = 'cpu',
    weights_name: str = 'model.pt',
) -> tuple[dict, dict, int] | None:
    """Pull only if remote step > current_step.

    Returns (state_dict, model_config, new_step) or None if up-to-date.
    """
    _require_hf()
    _, state_file = _namespace(weights_name)
    state_path = _retry(lambda: hf_hub_download(repo_id, state_file))
    with open(state_path) as f:
        state = json.load(f)

    remote_step = state.get('step', 0)
    if remote_step <= current_step:
        return None

    state_dict, model_config = pull_weights(repo_id, device, weights_name)
    return state_dict, model_config, remote_step


# --- Example exchange (separate repo) ---


def init_examples_repo(repo_id: str) -> str:
    """Create HF repo for training examples if needed. Returns repo URL."""
    _require_hf()
    api = HfApi()
    url = api.create_repo(repo_id, exist_ok=True)
    return str(url)


def upload_examples(
    repo_id: str,
    local_path: str | Path,
    remote_name: str,
    namespace: str | None = None,
) -> None:
    """Upload an example batch .pt file to the examples repo."""
    _require_hf()
    api = HfApi()
    path_in_repo = f"{namespace}/{remote_name}" if namespace else remote_name
    _retry(lambda: api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        commit_message=f"examples: {path_in_repo}",
    ))


def upload_examples_folder(
    repo_id: str,
    folder_path: str | Path,
    n_files: int = 0,
    namespace: str | None = None,
) -> None:
    """Upload all .pt files in a folder to the examples repo in a single commit."""
    _require_hf()
    api = HfApi()
    prefix = f"{namespace}/" if namespace else ""
    msg = f"examples: {prefix}{n_files} batches" if n_files else f"examples: {prefix}batch"
    _retry(lambda: api.upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        path_in_repo=namespace or "",
        commit_message=msg,
    ))


def list_remote_examples(repo_id: str, namespace: str | None = None) -> list[str]:
    """List pending .pt example files in the examples repo.

    If namespace is given, only returns files under that subdirectory.
    If namespace is None, only returns root-level .pt files (no subdirectory).
    """
    _require_hf()
    api = HfApi()
    all_files = _retry(lambda: api.list_repo_files(repo_id))
    if namespace:
        prefix = f"{namespace}/"
        return sorted(f for f in all_files if f.endswith('.pt') and f.startswith(prefix))
    # Root-level only: exclude files in subdirectories
    return sorted(f for f in all_files if f.endswith('.pt') and '/' not in f)


def download_example(repo_id: str, remote_name: str) -> Path:
    """Download an example file from HF. Returns local cache path.

    Uses HF cache — unique filenames mean no staleness issues.
    """
    _require_hf()
    path = _retry(lambda: hf_hub_download(repo_id, remote_name))
    return Path(path)


def _example_timestamp(filename: str) -> int:
    """Extract unix timestamp from example filename (worker-0_1770431227_abc.pt)."""
    basename = filename.rsplit('/', 1)[-1]
    parts = basename.split('_')
    try:
        return int(parts[-2])
    except (ValueError, IndexError):
        return 0


def prune_remote_examples(repo_id: str, keep: int, namespace: str | None = None) -> list[str]:
    """Delete oldest example files, keeping only the most recent `keep`.

    Returns list of deleted filenames.
    """
    files = list_remote_examples(repo_id, namespace)
    if len(files) <= keep:
        return []

    by_time = sorted(files, key=_example_timestamp)
    to_delete = by_time[:len(by_time) - keep]

    _require_hf()
    api = HfApi()
    operations = [CommitOperationDelete(path_in_repo=f) for f in to_delete]
    _retry(lambda: api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=f"prune: removed {len(to_delete)} old batches, keeping {keep}",
    ))
    return to_delete
