"""Runtime-registered improvised tools.

Built for the Jason↔Claude↔Burl experiment loop: Burl proposes a tool in
chat, Claude writes the impl and POSTs it here, the tool is live for the
next turn — no restart, no edit to ``wax_museum/tools.py``.

Contract for the registered Python source: must define a callable

    def tool(ctx, **kwargs) -> dict

returning ``{"prose": str, "structured": Any}``. ``ctx`` is a
``burl.wax_museum.tools.WaxContext``. The function may import anything in
the project (``burl.*``, ``forge.*``).

Persistence: each tool round-trips to a ``<name>.py`` file under
``BURL_CHAT_TOOLS_LIBRARY`` (default ``burl/chat/server/tools_library/``).
The file carries a ``DESCRIPTION = "..."`` constant followed by the
original source. The registry is rehydrated from this directory on import,
so tools survive server restarts and can be hand-curated by checking the
files into git.

This is a local dev workbench. ``exec`` is intentional. Do not deploy.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class ImprovisedTool:
    name: str
    description: str
    python_src: str
    impl: Callable[..., dict]


_REGISTRY: dict[str, ImprovisedTool] = {}


def _library_path() -> Path:
    override = os.environ.get("BURL_CHAT_TOOLS_LIBRARY")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).parent / "tools_library"


def _file_for(name: str) -> Path:
    return _library_path() / f"{name}.py"


def _serialize(name: str, description: str, python_src: str) -> str:
    """Render the on-disk form: DESCRIPTION constant followed by source."""
    body = python_src if python_src.endswith("\n") else python_src + "\n"
    return f'DESCRIPTION = {description!r}\n\n{body}'


def _persist(t: ImprovisedTool) -> None:
    lib = _library_path()
    lib.mkdir(parents=True, exist_ok=True)
    path = _file_for(t.name)
    path.write_text(_serialize(t.name, t.description, t.python_src))
    log.info("[improvised] persisted %s → %s", t.name, path)


def _delete_persisted(name: str) -> None:
    path = _file_for(name)
    if path.exists():
        path.unlink()
        log.info("[improvised] removed persisted %s", name)


def _compile_tool(name: str, python_src: str) -> Callable[..., dict]:
    namespace: dict[str, Any] = {"__name__": f"improvised_{name}"}
    exec(compile(python_src, f"<improvised:{name}>", "exec"), namespace)
    impl = namespace.get("tool")
    if not callable(impl):
        raise ValueError("python_src must define a callable named `tool`")
    return impl


def register(name: str, description: str, python_src: str) -> ImprovisedTool:
    """Compile + register + persist. Replaces an existing entry of the same name."""
    if not name.isidentifier():
        raise ValueError(f"tool name must be a valid identifier: {name!r}")
    impl = _compile_tool(name, python_src)
    t = ImprovisedTool(
        name=name,
        description=description,
        python_src=python_src,
        impl=impl,
    )
    _REGISTRY[name] = t
    _persist(t)
    log.info("[improvised] registered %s (%d chars)", name, len(python_src))
    return t


def unregister(name: str) -> bool:
    existed = _REGISTRY.pop(name, None) is not None
    _delete_persisted(name)
    return existed


def get(name: str) -> ImprovisedTool | None:
    return _REGISTRY.get(name)


def has(name: str) -> bool:
    return name in _REGISTRY


def list_all() -> list[ImprovisedTool]:
    return list(_REGISTRY.values())


def declaration(t: ImprovisedTool) -> str:
    """Render the wax_museum tool declaration string Burl can parse."""
    desc = t.description.replace('"', "'")
    return (
        f'<|tool>declaration:{t.name}{{description:<|"|>{desc}<|"|>,'
        f'parameters:{{type:<|"|>OBJECT<|"|>}}}}<tool|>'
    )


def _strip_description_line(text: str) -> str:
    """Drop the leading ``DESCRIPTION = ...\\n\\n`` block so re-saves do not stack."""
    lines = text.split("\n")
    if lines and lines[0].startswith("DESCRIPTION ="):
        body_start = 1
        if body_start < len(lines) and lines[body_start] == "":
            body_start += 1
        return "\n".join(lines[body_start:])
    return text


def _load_one(path: Path) -> None:
    """Parse a persisted tool file: pull DESCRIPTION, treat the rest as source."""
    name = path.stem
    if not name.isidentifier():
        log.warning("[improvised] skipping non-identifier filename %s", path)
        return
    text = path.read_text()
    namespace: dict[str, Any] = {"__name__": f"improvised_loader_{name}"}
    try:
        exec(compile(text, str(path), "exec"), namespace)
    except Exception:
        log.exception("[improvised] failed to load %s", path)
        return
    description = namespace.get("DESCRIPTION", "")
    impl = namespace.get("tool")
    if not callable(impl):
        log.warning("[improvised] %s defines no callable `tool`; skipping", path)
        return
    _REGISTRY[name] = ImprovisedTool(
        name=name,
        description=str(description),
        python_src=_strip_description_line(text),
        impl=impl,
    )
    log.info("[improvised] loaded %s from disk", name)


def _hydrate_from_disk() -> None:
    lib = _library_path()
    if not lib.exists():
        return
    for path in sorted(lib.glob("*.py")):
        if path.name.startswith("_"):
            continue
        _load_one(path)


_hydrate_from_disk()
