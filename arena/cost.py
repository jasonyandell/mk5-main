# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""What a run cost, in more than wall seconds (#173).

Wall time depends on how crowded the machine is; CPU time much less, and the
instructions a process retired hardly at all. Every harness that reports wall
time can report these beside it:

    start = cost.snapshot()
    ...                                   # the work
    meta.update(cost.since(start))        # wall_s, cpu_s, cores, instructions, cycles

`cpu_s` is user + system of this process and of every child it has reaped
(`getrusage`), so a pool of worker processes is counted once they are joined.
`instructions` and `cycles` are macOS only (`proc_pid_rusage`, RUSAGE_INFO_V4),
this process alone, every thread; elsewhere they are absent, never zero.
`pid_cost(pid)` reads another process, live or exited but not yet reaped, and `wait(proc)`
reaps a child and returns what it cost.
Everything here is stdlib, costs microseconds, and degrades to fewer keys.
"""
from __future__ import annotations

import ctypes
import os
import resource
import select
import sys
import time
from dataclasses import dataclass

_V4_FIELDS = (
    "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
    "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime",
    "child_user_time", "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
    "child_pageins", "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
    "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
    "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
    "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time",
    "logical_writes", "lifetime_max_phys_footprint", "instructions", "cycles",
    "billed_energy", "serviced_energy", "interval_max_phys_footprint", "runnable_time")
# The kernel writes the whole struct rusage_info_v4 (<sys/resource.h>): every field above,
# in order, or it writes past the buffer.


class _RusageV4(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in _V4_FIELDS]


def _darwin():
    """(proc_pid_rusage, seconds per mach tick), or None off macOS."""
    if sys.platform != "darwin":
        return None
    try:
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        tb = (ctypes.c_uint32 * 2)()
        ctypes.CDLL("/usr/lib/libSystem.dylib").mach_timebase_info(tb)
        return lib.proc_pid_rusage, tb[0] / tb[1] / 1e9
    except (OSError, AttributeError):
        return None


_DARWIN = _darwin()


def _rusage_v4(pid: int):
    if _DARWIN is None:
        return None
    info = _RusageV4()
    return info if _DARWIN[0](pid, 4, ctypes.byref(info)) == 0 else None


@dataclass(frozen=True)
class Snapshot:
    wall: float
    cpu_self: float
    cpu_children: float
    instructions: int | None
    cycles: int | None


def snapshot() -> Snapshot:
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    v4 = _rusage_v4(os.getpid())
    return Snapshot(time.monotonic(), me.ru_utime + me.ru_stime, kids.ru_utime + kids.ru_stime,
                    v4.instructions if v4 else None, v4.cycles if v4 else None)


def since(start: Snapshot) -> dict:
    """The cost of the work between `start` and now, rounded for a record."""
    end = snapshot()
    wall = end.wall - start.wall
    cpu = (end.cpu_self - start.cpu_self) + (end.cpu_children - start.cpu_children)
    out = {"wall_s": round(wall, 3), "cpu_s": round(cpu, 3),
           "cpu_self_s": round(end.cpu_self - start.cpu_self, 3),
           "cpu_children_s": round(end.cpu_children - start.cpu_children, 3),
           "cores": round(cpu / wall, 2) if wall > 0 else None}
    if end.instructions is not None and start.instructions is not None:
        out["instructions"] = end.instructions - start.instructions
        out["cycles"] = end.cycles - start.cycles
    return out


def pid_cost(pid: int) -> dict | None:
    """CPU seconds (and, on macOS, instructions and cycles) of one process so far,
    every thread, not its children. Works on a zombie on macOS; None if unreadable."""
    v4 = _rusage_v4(pid)
    if v4 is not None:
        tick = _DARWIN[1]
        return {"cpu_s": round((v4.user_time + v4.system_time) * tick, 3),
                "instructions": v4.instructions, "cycles": v4.cycles}
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return {"cpu_s": round((int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"), 3)}
    except (OSError, ValueError, IndexError):
        return None


def wait(proc, timeout: float | None = None) -> dict | None:
    """Wait for a `subprocess.Popen` child to exit, at most `timeout` seconds, and reap it.
    Returns {"exit", "cpu_s"} (user + sys, the children it reaped included) and, on macOS,
    the child's own "self_instructions" and "self_cycles", read while it is a zombie, before
    the reap. None on timeout, the child still running."""
    end = None if timeout is None else time.monotonic() + timeout
    kq = None
    if hasattr(select, "kqueue"):
        kq = select.kqueue()
        try:
            kq.control([select.kevent(proc.pid, select.KQ_FILTER_PROC,
                                      select.KQ_EV_ADD | select.KQ_EV_ONESHOT, select.KQ_NOTE_EXIT)], 0, 0)
        except OSError:  # it has already exited and been reaped
            kq.close()
            kq = None
    try:
        while True:
            left = 1.0 if end is None else max(0.0, min(1.0, end - time.monotonic()))
            if kq is not None:
                kq.control(None, 1, left)
            own = pid_cost(proc.pid)
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                proc.returncode = os.waitstatus_to_exitcode(status)
                out = {"exit": proc.returncode, "cpu_s": round(usage.ru_utime + usage.ru_stime, 3)}
                if own and "instructions" in own:
                    out["self_instructions"], out["self_cycles"] = own["instructions"], own["cycles"]
                return out
            if end is not None and time.monotonic() >= end:
                return None
            if kq is None:
                time.sleep(0.02)
    except ChildProcessError:  # reaped elsewhere: the exit is known, the cost is not
        return {"exit": proc.poll()}
    finally:
        if kq is not None:
            kq.close()
