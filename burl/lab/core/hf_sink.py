"""Optional Hugging Face Hub sink for harness sessions.

Gated by ``BURL_HARNESS_HF_PUSH=1``. Default off — pushing during tests
or local exploration is the wrong default.

If ``huggingface_hub`` is not installed, ``push_session`` logs and returns
``False``; the caller never has to check the env var first.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_REPO_ID = "jasonyandell/burl-harness-sessions"


def push_session(
    session_dir: Path | str,
    repo_id: str = DEFAULT_REPO_ID,
) -> bool:
    """Push a session directory to the HF Hub (datasets repo).

    Returns ``True`` on success; ``False`` if disabled, the dependency is
    missing, or the push fails. Never raises — pushing is a side channel,
    not a blocker for the main harness loop.
    """
    if os.environ.get("BURL_HARNESS_HF_PUSH") != "1":
        return False

    session_path = Path(session_dir)
    if not session_path.exists():
        log.warning("[hf_sink] session dir does not exist: %s", session_path)
        return False

    try:
        from huggingface_hub import HfApi  # type: ignore[import-not-found]
    except ImportError:
        log.info("[hf_sink] huggingface_hub not installed; skipping push")
        return False

    try:
        api = HfApi()
        api.upload_folder(
            folder_path=str(session_path),
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=session_path.name,
        )
        log.info("[hf_sink] pushed %s to %s", session_path.name, repo_id)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("[hf_sink] push failed: %s", exc)
        return False


__all__ = ["push_session", "DEFAULT_REPO_ID"]
