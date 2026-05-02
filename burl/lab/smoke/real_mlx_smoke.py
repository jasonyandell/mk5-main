"""End-to-end real-MLX smoke against one harvested decision.

Runs the full burl.lab spine in-process — no HTTP — against decision_1
(``BURL_BREAKS_CONSENSUS``) of harvest ``harvest_batched_20260425_072910``.

Steps (all in-process):

1. Build a fresh session under ``scratch/burl-lab-sessions/<sid>/`` and
   journal the pre_game configuration moves: ``SystemSet`` for the bare
   "you are Burl" prompt (no protocol section — render.py composes that
   live from advertised ToolSpecs), ``ToolAdded`` ×3 for the base tools,
   ``AdvertisedSet`` to advertise all three, then ``UserText`` for the
   harvested decision's user prompt.  Cap with ``PhaseExit("pre_game")``
   + ``PhaseEnter("in_run")``.
2. Build a real ``WaxContext`` via the same ``_replay_state(seed,
   declaration, play_history)`` recipe burl/chat uses, with
   ``narrator_seat`` from the decision's ``meta`` event and the play
   history parsed from ``visible history:`` in ``prompt_user``.
3. Instantiate ``MlxEngine`` (default Gemma 4 E2B BF16, no adapter) and
   call ``drive(state, registry, engine, ctx=ctx)``.  Stream every Move
   to stdout, journal each via ``transcript.append``.
4. Stop the loop when drive returns (clean ``EngineDone``, harness-
   synthesised ``EngineCommit``, or ``EngineError``).  Print a
   one-line summary: phase, last 6 Move kinds, commit final, oracle
   ``bot_play`` from the harvest meta.

Run:

    python -u -m burl.lab.smoke.real_mlx_smoke

Env knobs (all optional):
    BURL_HARNESS_HARVEST_ROOT   default: scratch/belief_trajectory_rollout
    BURL_HARNESS_SESSION_ROOT   default: scratch/burl-lab-sessions
    BURL_HARNESS_MODEL_REPO     default: mlx-community/gemma-4-e2b-it-bf16
    BURL_HARNESS_ADAPTER_PATH   default: unset
    BURL_HARNESS_HARVEST_NAME   default: harvest_batched_20260425_072910
    BURL_HARNESS_DECISION_IDX   default: 1
    BURL_HARNESS_MAX_TOKENS     default: 2048

Cost: first run pulls ~5GB of Gemma 4 weights and loads the E[Q] oracle
(~2GB). Subsequent runs ~5–15s of model load + ~10–60s of inference.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("burl.lab.smoke")

# Defaults — overridable via env.
DEFAULT_HARVEST_ROOT = Path("/Users/jason/code/mk5-main/scratch/belief_trajectory_rollout")
DEFAULT_SESSION_ROOT = Path("/Users/jason/code/mk5-main/scratch/burl-lab-sessions")
DEFAULT_HARVEST_NAME = "harvest_batched_20260425_072910"
DEFAULT_DECISION_IDX = 1


# --------------------------------------------------------------------------- #
# Decision lookup + WaxContext construction (mirrors burl/chat/server/tools_runner.py)
# --------------------------------------------------------------------------- #

_VISIBLE_HISTORY_RE = re.compile(r"seat(\d+):(\d+)\(")
_VISIBLE_LINE_RE = re.compile(r"visible history:\s*(.+)")


def _parse_play_history(prompt_user: str) -> list[tuple[int, int]]:
    m = _VISIBLE_LINE_RE.search(prompt_user)
    if not m:
        return []
    body = m.group(1).strip()
    if body.startswith("("):
        return []
    return [(int(s), int(d)) for s, d in _VISIBLE_HISTORY_RE.findall(body)]


def _resolve_decision(harvest_root: Path, harvest: str, idx: int) -> tuple[Path, dict, str, str]:
    """Return (dec_dir, meta_event, prompt_user_content, prompt_system_content)."""
    idx_path = harvest_root / harvest / "corpus_index.jsonl"
    if not idx_path.exists():
        raise FileNotFoundError(idx_path)

    matched: dict | None = None
    with idx_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("global_idx") == idx:
                matched = row
                break
    if matched is None:
        raise KeyError(f"global_idx {idx} not found in {harvest}")

    transcript_rel = matched.get("transcript_path", "")
    dec_dir = harvest_root / harvest / transcript_rel.split("/transcript")[0]
    events_path = dec_dir / "events.jsonl"

    meta: dict | None = None
    user_content: str = ""
    system_content: str = ""
    with events_path.open() as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = e.get("kind")
            if kind == "meta":
                meta = e
            elif kind == "prompt_user":
                user_content = e.get("content", "")
            elif kind == "prompt_system":
                system_content = e.get("content", "")
            if meta and user_content:
                break
    if meta is None or not user_content:
        raise RuntimeError(f"missing meta or prompt_user in {events_path}")
    return dec_dir, meta, user_content, system_content


def _build_wax_ctx(meta: dict, prompt_user: str) -> Any:
    """Reconstruct WaxContext from a harvested decision's meta + prompt_user."""
    from burl.eval.decision_dataset import _replay_state
    from burl.tools.eq_distribution import load_eq_oracle
    from burl.wax_museum.tools import WaxContext

    play_history = _parse_play_history(prompt_user)
    state = _replay_state(
        int(meta["seed"]),
        int(meta["declaration"]),
        play_history,
    )
    log.info("[smoke] loading E[Q] oracle (one-time)")
    oracle = load_eq_oracle()
    return WaxContext(
        game_state=state,
        me_abs=int(meta["narrator_seat"]),
        oracle=oracle,
    )


# --------------------------------------------------------------------------- #
# Session journaling
# --------------------------------------------------------------------------- #

# Bare system prompt — render_system() will append the Decision Protocol
# section composed from each advertised ToolSpec's protocol_phrase. This is
# deliberately MUCH shorter than the harvested prompt_system: the point of
# burl/lab/ is that tool advertisement is rendered, not hand-written.
_BASE_SYSTEM = (
    "You are Burl, a Texas 42 dominoes agent. Pick the next play. "
    "You have tools that describe the game state; call them as needed. "
    "When you are ready to commit, call commit_play with the integer "
    "domino_id from your hand. That ends the decision."
)


def _new_session_dir(session_root: Path) -> Path:
    sid = uuid.uuid4().hex[:12]
    d = session_root / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bootstrap_session(session_dir: Path, registry, user_prompt: str) -> Any:
    """Journal pre_game config + transition to in_run. Returns the folded State."""
    from burl.lab.core.transcript import (
        AdvertisedSet,
        PhaseEnter,
        PhaseExit,
        Stamp,
        SystemSet,
        ToolAdded,
        UserText,
        append,
        fold,
        replay,
    )

    def _stamp(t_ms: int) -> Stamp:
        return Stamp(t_wall_ms=t_ms, t_mono_ns=t_ms * 1_000_000)

    # Start phase.
    append(session_dir, PhaseEnter(stamp=_stamp(0), phase="pre_game"))
    # Set bare system text.
    append(session_dir, SystemSet(stamp=_stamp(1), text=_BASE_SYSTEM))
    # Activate + advertise all three base tools.
    names = [s.name for s in registry.active()]
    for t_ms, name in enumerate(names, start=2):
        append(session_dir, ToolAdded(stamp=_stamp(t_ms), name=name))
    append(session_dir, AdvertisedSet(stamp=_stamp(10), names=list(names)))
    # Seed the user message (decision prompt).
    append(session_dir, UserText(stamp=_stamp(11), text=user_prompt))
    # Transition to in_run.
    append(session_dir, PhaseExit(stamp=_stamp(12), phase="pre_game"))
    append(session_dir, PhaseEnter(stamp=_stamp(13), phase="in_run"))

    moves = list(replay(session_dir))
    state = fold(moves, session_dir=session_dir, registry=registry)
    return state


# --------------------------------------------------------------------------- #
# Move pretty-print (for stdout streaming)
# --------------------------------------------------------------------------- #


def _short_move(mv: Any) -> str:
    name = type(mv).__name__
    if name == "EngineToken":
        text = getattr(mv, "text", "")
        return f"EngineToken {text!r}"
    if name == "EngineToolCall":
        return f"EngineToolCall name={mv.name!r} args={mv.args} call_id={mv.call_id[:8]}…"
    if name == "ToolResult":
        prose = (mv.evidence or {}).get("prose", "")
        head = prose.replace("\n", " ⏎ ")[:120]
        nt = mv.next_tools or []
        return f"ToolResult name={mv.name!r} prose='{head}…' next={nt}"
    if name == "EngineCommit":
        return f"EngineCommit final={mv.final}"
    if name == "EngineDone":
        return f"EngineDone reason={mv.reason!r}"
    if name == "EngineError":
        msg = (mv.message or "")[:200]
        return f"EngineError message={msg!r} during={mv.during}"
    if name == "EngineStart":
        return f"EngineStart hash={mv.messages_hash} n_messages={mv.n_messages} n_tools={mv.n_tools}"
    return f"{name} {mv}"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


async def _run() -> int:
    import asyncio  # noqa: F401  (re-import for clarity in sync main)

    from burl.lab.core.drive import drive
    from burl.lab.core.tool import Registry
    from burl.lab.core.transcript import append, fold, replay
    from burl.lab.tools import BELIEF_TRAJECTORY, COMMIT_PLAY, EXPLORE_GAME

    # Resolve env / defaults.
    harvest_root = Path(os.environ.get("BURL_HARNESS_HARVEST_ROOT", str(DEFAULT_HARVEST_ROOT)))
    session_root = Path(os.environ.get("BURL_HARNESS_SESSION_ROOT", str(DEFAULT_SESSION_ROOT)))
    harvest_name = os.environ.get("BURL_HARNESS_HARVEST_NAME", DEFAULT_HARVEST_NAME)
    decision_idx = int(os.environ.get("BURL_HARNESS_DECISION_IDX", str(DEFAULT_DECISION_IDX)))
    max_tokens = int(os.environ.get("BURL_HARNESS_MAX_TOKENS", "2048"))

    # 1. Resolve the decision.
    dec_dir, meta, user_prompt, _orig_system = _resolve_decision(
        harvest_root, harvest_name, decision_idx
    )
    log.info(
        "[smoke] decision %s/%s seed=%s decl=%s seat=%s legal=%s bot=%s",
        harvest_name,
        decision_idx,
        meta.get("seed"),
        meta.get("declaration"),
        meta.get("narrator_seat"),
        meta.get("legal_plays"),
        meta.get("bot_play"),
    )

    # 2. Build the registry with the three base ToolSpecs.
    registry = Registry()
    registry.add(BELIEF_TRAJECTORY)
    registry.add(EXPLORE_GAME)
    registry.add(COMMIT_PLAY)
    log.info("[smoke] registry: %s", [s.name for s in registry.active()])

    # 3. Build the WaxContext (heavy: oracle load).
    ctx = _build_wax_ctx(meta, user_prompt)
    log.info(
        "[smoke] WaxContext built me_abs=%s game_state=%s",
        ctx.me_abs,
        type(ctx.game_state).__name__,
    )

    # 4. Bootstrap session + transition to in_run.
    session_dir = _new_session_dir(session_root)
    log.info("[smoke] session dir: %s", session_dir)
    state = _bootstrap_session(session_dir, registry, user_prompt)
    log.info(
        "[smoke] State after bootstrap: phase=%s active_tools=%s advertised=%s",
        state.phase,
        list(state.active_tools),
        list(state.advertised),
    )

    # Print the rendered system prompt so we can eyeball it live.
    from burl.lab.core.render import render_messages

    rendered = render_messages(state, registry)
    sys_msg = next((m for m in rendered if m["role"] == "system"), None)
    if sys_msg is not None:
        head = sys_msg["content"]
        log.info("[smoke] rendered system prompt (%d chars):", len(head))
        for line in head.splitlines():
            print("    | " + line)

    # 5. Instantiate engine.
    log.info("[smoke] loading MlxEngine...")
    t0 = time.monotonic()
    from burl.lab.core.engine import MlxEngine

    engine = MlxEngine()
    log.info(
        "[smoke] engine ready in %.1fs: model=%s adapter=%s",
        time.monotonic() - t0,
        engine.model_repo,
        getattr(engine, "adapter_path", None),
    )

    # 6. Drive!  Stream Moves, journal each, print short one-line summary.
    log.info("[smoke] drive() starting...")
    t_drive_start = time.monotonic()
    n_moves = 0
    last_kind = ""
    async for mv in drive(state, registry, engine, max_tokens=max_tokens, ctx=ctx):
        append(session_dir, mv)
        n_moves += 1
        last_kind = type(mv).__name__
        # Coalesce EngineToken spam — print one line per chunk but truncated.
        if last_kind == "EngineToken":
            text = getattr(mv, "text", "")
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            print()  # newline after token stream
            print(f"  [{n_moves:03d}] {_short_move(mv)}")
    drive_wall = time.monotonic() - t_drive_start
    print()  # ensure trailing newline
    log.info("[smoke] drive() returned after %.1fs (%d Moves)", drive_wall, n_moves)

    # 6b. Post-drive phase transition.  The server normally does this; for
    # the smoke we mirror it so the journal tail matches what real /api/move
    # would produce.  Re-fold, ask the current phase whether the last
    # emitted Move triggers a transition, and journal PhaseExit/PhaseEnter.
    from burl.lab.phases import PHASES
    from burl.lab.core.transcript import PhaseEnter, PhaseExit
    from burl.lab.core.transcript import now_stamp

    last_emitted = None
    for m in reversed(list(replay(session_dir))):
        if type(m).__name__ != "PhaseEnter":  # any non-phase Move
            last_emitted = m
            break
    if last_emitted is not None:
        post_state = fold(list(replay(session_dir)), session_dir=session_dir, registry=registry)
        post_phase = PHASES.get(post_state.phase or "")
        if post_phase is not None:
            _, post_next = await post_phase.handle(post_state, last_emitted, registry)
            if post_next and post_next != post_state.phase:
                exit_mv = PhaseExit(stamp=now_stamp(post_state), phase=post_state.phase)
                enter_mv = PhaseEnter(stamp=now_stamp(post_state), phase=post_next)
                append(session_dir, exit_mv)
                append(session_dir, enter_mv)
                log.info(
                    "[smoke] post-drive transition: %s -> %s (journaled)",
                    post_state.phase, post_next,
                )

    # 7. Re-fold the journal post-drive and report final state + journal tail.
    final_state = fold(list(replay(session_dir)), session_dir=session_dir, registry=registry)
    tail_kinds = [type(m).__name__ for m in list(replay(session_dir))[-8:]]
    log.info("[smoke] final phase: %s", final_state.phase)
    log.info("[smoke] journal tail (last 8 kinds): %s", tail_kinds)

    # If there's a commit, surface it.
    commits = [m for m in replay(session_dir) if type(m).__name__ == "EngineCommit"]
    if commits:
        log.info("[smoke] EngineCommit.final = %s", commits[-1].final)
        log.info("[smoke] harvest's bot_play (oracle pick) was %s", meta.get("bot_play"))
    else:
        log.info("[smoke] NO EngineCommit reached — model didn't call commit_play")

    return 0


def main() -> int:
    import asyncio

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
