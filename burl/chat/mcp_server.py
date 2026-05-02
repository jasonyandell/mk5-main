"""MCP server for the Burl chat workbench — Claude's eyes and hands.

Lets Claude (running inside Claude Code) drive the experiment loop without
copy-paste:

  - read what Burl just said (read_chat_state)
  - inspect the decision under examination (inspect_decision)
  - register a hot tool from Python source (register_improvised_tool)
  - exec an existing tool to see its output (run_tool)

This is a thin proxy over the FastAPI endpoints already serving the web
workbench. Same backend, different transport.

Wire it into Claude Code with:

    claude mcp add burl-chat \
        /Users/jason/code/mk5-main/.venv/bin/python \
        -m burl.chat.mcp_server

The FastAPI server must be running (default http://127.0.0.1:8001).
Override via BURL_CHAT_API env var.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from mcp.server.fastmcp import FastMCP

API_BASE = os.environ.get("BURL_CHAT_API", "http://127.0.0.1:8001").rstrip("/")

mcp = FastMCP("burl-chat")


def _get(path: str) -> Any:
    with urllib.request.urlopen(API_BASE + path, timeout=120) as r:
        return json.loads(r.read())


def _post(path: str, body: dict) -> Any:
    req = urllib.request.Request(
        API_BASE + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def _delete(path: str) -> Any:
    req = urllib.request.Request(API_BASE + path, method="DELETE")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


@mcp.tool()
def list_harvests() -> list[dict]:
    """Available harvest dirs under scratch/belief_trajectory_rollout/, newest
    first. Each entry has ``{name, n_decisions, buckets: {bucket_name: count}}``.
    """
    return _get("/api/harvests")


@mcp.tool()
def list_decisions(
    harvest: str, bucket: str | None = None, limit: int = 40,
) -> list[dict]:
    """List decisions in a harvest. Filter by bucket (e.g.
    ``BURL_BREAKS_CONSENSUS``) when looking for experiment candidates."""
    params = []
    if bucket:
        params.append(f"bucket={urllib.parse.quote(bucket)}")
    params.append(f"limit={int(limit)}")
    return _get(f"/api/harvests/{harvest}/decisions?{'&'.join(params)}")


@mcp.tool()
def inspect_decision(harvest: str, global_idx: int) -> dict:
    """Full decision payload: meta (seed, declaration, narrator, oracle pick,
    bucket), reconstructed messages, typed display segments, and raw events.
    """
    return _get(f"/api/harvests/{harvest}/decisions/{int(global_idx)}")


@mcp.tool()
def list_tools() -> dict:
    """All tools currently visible to Burl, base + improvised registry."""
    return _get("/api/tools")


@mcp.tool()
def list_improvised_tools() -> dict:
    """Hot-registered improvised tools and their wax_museum declarations."""
    return _get("/api/improvised_tools")


@mcp.tool()
def register_improvised_tool(
    name: str, description: str, python_src: str,
) -> dict:
    """Hot-register a tool. ``python_src`` must define ``def tool(ctx, **kw)``
    returning ``{"prose": str, "structured": Any}``. ``ctx`` is a
    ``burl.wax_museum.tools.WaxContext`` (``ctx.game_state``, ``ctx.me_abs``,
    ``ctx.oracle``).

    Replaces an existing entry of the same name. Returns the tool's
    wax_museum declaration string Burl can parse.
    """
    return _post(
        "/api/improvised_tools",
        {"name": name, "description": description, "python_src": python_src},
    )


@mcp.tool()
def unregister_improvised_tool(name: str) -> dict:
    """Drop a tool from the live registry."""
    return _delete(f"/api/improvised_tools/{name}")


@mcp.tool()
def run_tool(
    harvest: str, global_idx: int, tool: str, args: dict | None = None,
) -> dict:
    """Exec a tool against the live game state of a harvest decision. Same
    code path Burl hits when it makes a tool call. Use this to smoke-test an
    improvised tool's output before advertising it in chat."""
    return _post(
        "/api/tools/run",
        {
            "harvest": harvest,
            "global_idx": int(global_idx),
            "tool": tool,
            "args": args or {},
        },
    )


@mcp.tool()
def read_chat_state() -> dict:
    """Latest snapshot the user shared from the web UI ('share with Claude'
    button). Contains ``{decision: {harvest, meta}, segments: [...]}`` —
    segments include the most recent assistant prose, tool calls, etc.

    Returns ``{"shared": false}`` if nothing has been shared yet.
    """
    try:
        return _get("/api/chat/shared")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"shared": False, "reason": "no chat shared yet"}
        raise


@mcp.tool()
def health() -> dict:
    """Confirm the FastAPI workbench is up and which model + adapter are loaded."""
    return _get("/api/health")


if __name__ == "__main__":
    mcp.run()
