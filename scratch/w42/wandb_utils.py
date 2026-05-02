"""Small W&B helpers for w42 scratch experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def add_wandb_args(
    parser: argparse.ArgumentParser,
    *,
    default_project: str = "w42",
    default_group: str | None = None,
) -> None:
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", default_project))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-group", default=os.environ.get("WANDB_GROUP", default_group))
    parser.add_argument("--wandb-name", default=os.environ.get("WANDB_NAME"))
    parser.add_argument(
        "--wandb-mode",
        choices=["auto", "online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "auto"),
    )


def _has_wandb_login() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc_path = Path.home() / ".netrc"
    if not netrc_path.exists():
        return False
    try:
        text = netrc_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "api.wandb.ai" in text or "wandb.ai" in text


def _flatten_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            out[f"{prefix}/{key}"] = float(value)
        elif isinstance(value, int | float):
            out[f"{prefix}/{key}"] = float(value)
    return out


class WandbRun:
    def __init__(
        self,
        *,
        enabled: bool,
        available: bool,
        mode: str,
        project: str | None,
        entity: str | None,
        group: str | None,
        name: str | None,
        run: Any = None,
        error: str | None = None,
    ):
        self.enabled = enabled
        self.available = available
        self.mode = mode
        self.project = project
        self.entity = entity
        self.group = group
        self.name = name
        self.run = run
        self.error = error

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_metric_groups(self, groups: dict[str, dict[str, Any]], *, step: int | None = None) -> None:
        flat: dict[str, float] = {}
        for prefix, metrics in groups.items():
            flat.update(_flatten_metrics(prefix, metrics))
        self.log(flat, step=step)

    def update_summary(self, values: dict[str, Any]) -> None:
        if self.run is not None:
            self.run.summary.update(values)

    def log_artifact_files(self, *, name: str, artifact_type: str, paths: list[Path]) -> None:
        if self.run is None:
            return
        import wandb

        artifact = wandb.Artifact(name=name, type=artifact_type)
        added = False
        for path in paths:
            if path.exists():
                artifact.add_file(str(path))
                added = True
        if added:
            self.run.log_artifact(artifact)

    def status(self) -> dict[str, Any] | str:
        if not self.enabled:
            return "not applicable"
        status: dict[str, Any] = {
            "enabled": True,
            "available": self.available,
            "mode": self.mode,
            "project": self.project,
            "entity": self.entity,
            "group": self.group,
            "name": self.name,
        }
        if self.error:
            status["error"] = self.error
        if self.run is not None:
            sync_dir = str(Path(self.run.dir).parent)
            status.update(
                {
                    "id": self.run.id,
                    "url": self.run.url,
                    "dir": self.run.dir,
                }
            )
            if self.mode == "offline":
                status["sync_command"] = f"python -m wandb sync {sync_dir}"
        return status

    def finish(self, exit_code: int = 0) -> None:
        if self.run is not None:
            self.run.finish(exit_code=exit_code)


def init_wandb(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    output_dir: Path,
    tags: list[str],
) -> WandbRun:
    if not getattr(args, "wandb", False):
        return WandbRun(
            enabled=False,
            available=False,
            mode="disabled",
            project=None,
            entity=None,
            group=None,
            name=None,
        )
    try:
        import wandb
    except ImportError as exc:
        return WandbRun(
            enabled=True,
            available=False,
            mode="unavailable",
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            error=f"wandb import failed: {exc}",
        )

    mode = args.wandb_mode
    if mode == "auto":
        mode = "online" if _has_wandb_login() else "offline"
    if mode == "disabled":
        return WandbRun(
            enabled=False,
            available=True,
            mode="disabled",
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
        )

    wandb_dir = output_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            config=config,
            tags=tags,
            mode=mode,
            dir=str(wandb_dir),
        )
    except Exception as exc:  # pragma: no cover - defensive for auth/network quirks.
        return WandbRun(
            enabled=True,
            available=True,
            mode=mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=args.wandb_name,
            error=f"wandb init failed: {exc}",
        )

    return WandbRun(
        enabled=True,
        available=True,
        mode=mode,
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        name=args.wandb_name,
        run=run,
    )
