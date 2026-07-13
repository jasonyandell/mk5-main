#!/usr/bin/env python3
"""Unified monitor for Zeb experiment learner + self-play + eval-aux fleets.

Example:
  python forge/zeb/vast/monitor_experiment.py --exp lb-v-eq-3740
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
BATCH_RE = re.compile(
    r"\[(?P<worker>[^\]]+)\]\s+batch\s+(?P<batch>\d+):\s+.*?,\s+"
    r"(?P<gps>[0-9]+(?:\.[0-9]+)?)\s+games/s,\s+"
    r"(?:eq_win=(?P<eq_win>[0-9]+(?:\.[0-9]+)?%),\s+)?"
    r"step=(?P<step>\d+),\s+total_games=(?P<games>\d+)"
)


@dataclass
class CmdResult:
    rc: int
    out: str
    err: str


def _run(cmd: list[str], timeout: int = 30) -> CmdResult:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CmdResult(p.returncode, p.stdout or "", p.stderr or "")
    except Exception as exc:  # pragma: no cover - defensive path
        return CmdResult(1, "", f"{type(exc).__name__}: {exc}")


def _clean(text: str) -> str:
    return ANSI_RE.sub("", text.replace("\r", ""))


def _human_age(seconds: float | int | None) -> str:
    if seconds is None:
        return "?"
    s = int(seconds)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def _weights_state_filename(exp: str) -> str:
    if exp == "model":
        return "training_state.json"
    return f"{exp}-state.json"


def _find_learner(exp: str) -> list[dict[str, Any]]:
    res = _run(["ps", "-eo", "pid,etime,cmd"], timeout=10)
    if res.rc != 0:
        return []
    learners: list[dict[str, Any]] = []
    marker = f"--weights-name {exp}"
    for raw in res.out.splitlines():
        line = raw.strip()
        if "forge.zeb.learner.run" not in line or marker not in line:
            continue
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid = int(parts[0])
        etime = parts[1]
        cmd = parts[2]
        stdout_link = None
        try:
            stdout_link = os.readlink(f"/proc/{pid}/fd/1")
        except OSError:
            stdout_link = None
        learners.append({"pid": pid, "etime": etime, "cmd": cmd, "stdout": stdout_link})
    return learners


def _tail(path: str, n: int) -> list[str]:
    if n <= 0:
        return []
    res = _run(["tail", "-n", str(n), path], timeout=5)
    if res.rc != 0:
        return []
    return [ln for ln in _clean(res.out).splitlines() if ln.strip()]


def _load_instances() -> list[dict[str, Any]]:
    res = _run(["vastai", "show", "instances", "--raw"], timeout=30)
    if res.rc != 0:
        raise RuntimeError(f"vastai show instances failed: {res.err.strip()}")
    return json.loads(res.out)


def _fleet_instances(instances: list[dict[str, Any]], fleet: str) -> list[dict[str, Any]]:
    pref = f"{fleet}-"
    return [i for i in instances if (i.get("label") or "").startswith(pref)]


def _parse_worker_log(instance_id: int, tail_lines: int) -> dict[str, Any]:
    res = _run(["vastai", "logs", str(instance_id), "--tail", str(tail_lines)], timeout=40)
    text = _clean((res.out or "") + ("\n" + res.err if res.err else ""))
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    batches: list[dict[str, Any]] = []
    uploads: list[str] = []
    errors: list[str] = []
    keywords: list[str] = []

    err_re = re.compile(
        r"(No module named|can't open file|Traceback|Exception|ERROR|RuntimeError|CUDA out of memory)",
        re.IGNORECASE,
    )
    kw_re = re.compile(r"(batch\s+\d+|Upload complete|Uploading \d+ batches|eq_win=|No new weights)", re.IGNORECASE)

    for ln in lines:
        m = BATCH_RE.search(ln)
        if m:
            batches.append(
                {
                    "worker": m.group("worker"),
                    "batch": int(m.group("batch")),
                    "gps": float(m.group("gps")),
                    "step": int(m.group("step")),
                    "games": int(m.group("games")),
                    "eq_win": m.group("eq_win"),
                    "line": ln,
                }
            )
        if "Upload complete" in ln or ("Uploading" in ln and "batches" in ln):
            uploads.append(ln)
        if err_re.search(ln):
            errors.append(ln)
        if kw_re.search(ln):
            keywords.append(ln)

    return {
        "rc": res.rc,
        "last_batch": batches[-1] if batches else None,
        "upload_count": len(uploads),
        "errors": errors[-4:],
        "keywords": keywords[-6:],
        "raw_tail": lines[-20:],
    }


def _print_fleet(name: str, fleet_instances: list[dict[str, Any]], tail_lines: int) -> tuple[int, list[float], int]:
    now = time.time()
    print(f"{name}:")
    if not fleet_instances:
        print("  instances: 0")
        return 0, [], 0

    running = 0
    gps_values: list[float] = []
    with_batch = 0

    for inst in sorted(fleet_instances, key=lambda x: x.get("label") or ""):
        iid = inst.get("id")
        label = inst.get("label", "?")
        gpu = inst.get("gpu_name", "?")
        status = inst.get("actual_status", "unknown")
        start = inst.get("start_date")
        age_s = (now - start) if start else None
        dph = float(inst.get("dph_total") or 0.0)
        if status == "running":
            running += 1
        ssh = "-"
        if inst.get("ssh_host") and inst.get("ssh_port"):
            ssh = f"ssh -p {inst['ssh_port']} root@{inst['ssh_host']}"
        print(
            f"  - {label} [{iid}] status={status} age={_human_age(age_s)} "
            f"gpu={gpu} cost=${dph:.3f}/hr"
        )
        print(f"    ssh: {ssh}")

        parsed = _parse_worker_log(int(iid), tail_lines=tail_lines)
        lb = parsed["last_batch"]
        if lb:
            with_batch += 1
            gps_values.append(lb["gps"])
            eq = f", eq_win={lb['eq_win']}" if lb.get("eq_win") else ""
            print(
                f"    last_batch: #{lb['batch']} gps={lb['gps']:.2f} step={lb['step']} "
                f"games={lb['games']}{eq}"
            )
        else:
            print("    last_batch: <none>")

        if parsed["upload_count"]:
            print(f"    uploads_seen: {parsed['upload_count']}")
        if parsed["errors"]:
            for err in parsed["errors"]:
                print(f"    error: {err}")
        elif parsed["keywords"]:
            for k in parsed["keywords"][-2:]:
                print(f"    note: {k}")
    return running, gps_values, with_batch


def _hf_state(exp: str, weights_repo: str, examples_repo: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as exc:
        out["error"] = f"huggingface_hub unavailable: {exc}"
        return out

    state_name = _weights_state_filename(exp)
    try:
        path = hf_hub_download(repo_id=weights_repo, filename=state_name, force_download=True)
        out["state"] = json.loads(Path(path).read_text())
    except Exception as exc:
        out["state_error"] = str(exc)

    try:
        api = HfApi()
        files = api.list_repo_files(examples_repo)
        prefix = "" if exp == "model" else f"{exp}/"
        scoped = [f for f in files if f.startswith(prefix)]
        out["example_files"] = len(scoped)
    except Exception as exc:
        out["examples_error"] = str(exc)
    return out


def _wandb_summary(path: str, run_id: str | None) -> dict[str, Any]:
    if not run_id:
        return {"error": "no wandb_run_id in HF state"}
    try:
        import wandb
    except Exception as exc:
        return {"error": f"wandb unavailable: {exc}"}

    try:
        run = wandb.Api().run(f"{path}/{run_id}")
        summary = run.summary
        keys = [
            "stats/total_games",
            "stats/replay_buffer_selfplay",
            "stats/replay_buffer_eval_aux",
            "stats/selfplay_mix_ratio",
            "stats/eval_aux_mix_ratio",
            "stats/eval_aux_zeb_win_rate_running",
            "ingest/eval_aux_seen_examples",
            "ingest/eval_aux_kept_examples",
        ]
        return {"state": run.state, "url": run.url, "metrics": {k: summary.get(k) for k in keys}}
    except Exception as exc:
        return {"error": str(exc)}


def _origin_check() -> dict[str, Any]:
    out: dict[str, Any] = {}
    res = _run(
        [
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            "origin/forge",
            "forge/zeb/worker",
        ],
        timeout=10,
    )
    if res.rc != 0:
        out["error"] = res.err.strip() or "git ls-tree failed"
        return out
    files = [ln.strip() for ln in res.out.splitlines() if ln.strip()]
    out["has_run_eval_aux"] = "forge/zeb/worker/run_eval_aux.py" in files
    out["files"] = files
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Monitor Zeb learner + worker fleets")
    p.add_argument("--exp", required=True, help="Experiment weights-name namespace (e.g. lb-v-eq-3740)")
    p.add_argument("--weights-repo", default="jasonyandell/zeb-42")
    p.add_argument("--examples-repo", default="jasonyandell/zeb-42-examples")
    p.add_argument("--fleet-selfplay", default=None, help="Defaults to zeb-<exp>-selfplay")
    p.add_argument("--fleet-evalaux", default=None, help="Defaults to zeb-<exp>-evalaux")
    p.add_argument("--wandb-path", default="jasonyandell-forge42/zeb-mcts")
    p.add_argument("--tail-lines", type=int, default=120)
    p.add_argument("--learner-tail", type=int, default=6)
    p.add_argument("--check-origin", action="store_true", help="Check if origin/forge has eval aux worker file")
    args = p.parse_args()

    fleet_self = args.fleet_selfplay or f"zeb-{args.exp}-selfplay"
    fleet_eval = args.fleet_evalaux or f"zeb-{args.exp}-evalaux"

    print(f"timestamp: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"experiment: {args.exp}")
    print(f"weights_repo: {args.weights_repo}")
    print(f"examples_repo: {args.examples_repo}")
    print("")

    # Learner
    learners = _find_learner(args.exp)
    print("Learner:")
    if not learners:
        print("  status: DOWN")
    else:
        print(f"  status: UP ({len(learners)} process)")
        for l in learners:
            print(f"  - pid={l['pid']} etime={l['etime']}")
            print(f"    stdout={l.get('stdout') or '<unknown>'}")
            if l.get("stdout"):
                tail_lines = _tail(l["stdout"], args.learner_tail)
                if tail_lines:
                    for ln in tail_lines[-3:]:
                        print(f"    log: {ln}")
    print("")

    # Fleets
    try:
        instances = _load_instances()
    except Exception as exc:
        print(f"Fleet status error: {exc}")
        return 2

    self_instances = _fleet_instances(instances, fleet_self)
    eval_instances = _fleet_instances(instances, fleet_eval)

    self_running, self_gps, self_with_batch = _print_fleet("Self-play fleet", self_instances, args.tail_lines)
    print("")
    eval_running, eval_gps, eval_with_batch = _print_fleet("Eval-aux fleet", eval_instances, args.tail_lines)
    print("")

    # HF + W&B
    hf = _hf_state(args.exp, args.weights_repo, args.examples_repo)
    print("HF:")
    if "state" in hf:
        print(f"  state: {hf['state']}")
    if "example_files" in hf:
        print(f"  example_files: {hf['example_files']}")
    if "state_error" in hf:
        print(f"  state_error: {hf['state_error']}")
    if "examples_error" in hf:
        print(f"  examples_error: {hf['examples_error']}")
    print("")

    run_id = None
    if "state" in hf and isinstance(hf["state"], dict):
        run_id = hf["state"].get("wandb_run_id")
    wb = _wandb_summary(args.wandb_path, run_id)
    print("W&B:")
    if "error" in wb:
        print(f"  error: {wb['error']}")
    else:
        print(f"  run_state: {wb.get('state')}")
        print(f"  run_url: {wb.get('url')}")
        for k, v in (wb.get("metrics") or {}).items():
            print(f"  {k}: {v}")
    print("")

    if args.check_origin:
        oc = _origin_check()
        print("Origin check:")
        if "error" in oc:
            print(f"  error: {oc['error']}")
        else:
            print(f"  origin_has_run_eval_aux.py: {oc['has_run_eval_aux']}")
            if not oc["has_run_eval_aux"]:
                print("  warning: origin/forge is missing forge/zeb/worker/run_eval_aux.py")
        print("")

    # Summary verdict
    issues: list[str] = []
    if not learners:
        issues.append("learner_down")
    if self_running < 1:
        issues.append("no_selfplay_running")
    if self_with_batch < 1:
        issues.append("selfplay_not_producing")
    if eval_running < 1:
        issues.append("no_evalaux_running")
    if eval_with_batch < 1:
        issues.append("evalaux_not_producing")
    if eval_gps:
        eval_avg = sum(eval_gps) / len(eval_gps)
    else:
        eval_avg = 0.0
    self_avg = (sum(self_gps) / len(self_gps)) if self_gps else 0.0

    print("Summary:")
    print(f"  selfplay_running={self_running} selfplay_avg_gps={self_avg:.2f}")
    print(f"  evalaux_running={eval_running} evalaux_avg_gps={eval_avg:.2f}")
    if issues:
        print(f"  status=DEGRADED issues={','.join(issues)}")
        return 1
    print("  status=HEALTHY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
