# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""arena.cost (#173): CPU seconds and, on macOS, instructions, beside wall time."""
from __future__ import annotations

import ctypes
import subprocess
import sys

import pytest

from arena import cost

BURN = "s = 0\nfor i in range(2 * 10**6): s += i\n"


def test_since_counts_this_process_and_its_reaped_children():
    start = cost.snapshot()
    subprocess.run([sys.executable, "-c", BURN], check=True)
    exec(BURN, {})
    spent = cost.since(start)
    assert spent["cpu_children_s"] > 0.02
    assert spent["cpu_s"] > spent["cpu_children_s"]
    assert spent["wall_s"] > 0 and spent["cores"] > 0
    if sys.platform == "darwin":
        assert spent["instructions"] > 10**7 and spent["cycles"] > 0
    else:
        assert "instructions" not in spent


def test_wait_reaps_and_reads_the_child_before_the_reap():
    p = subprocess.Popen([sys.executable, "-c", BURN + "raise SystemExit(3)"])
    spent = cost.wait(p, 30)
    assert spent["exit"] == 3 and p.returncode == 3
    assert spent["cpu_s"] > 0.02
    if sys.platform == "darwin":
        assert spent["self_instructions"] > 10**7


def test_wait_times_out_on_a_running_child():
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        assert cost.wait(p, 0.2) is None
        assert p.returncode is None
    finally:
        p.kill()
        assert cost.wait(p)["exit"] < 0


@pytest.mark.skipif(sys.platform != "darwin", reason="proc_pid_rusage is macOS")
def test_the_struct_is_the_kernels_size():
    # struct rusage_info_v4: a 16-byte uuid and 35 uint64s. Smaller and the kernel
    # writes past the buffer.
    assert ctypes.sizeof(cost._RusageV4) == 16 + 35 * 8
