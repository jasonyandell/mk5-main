# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""scripts/cpu-slot: the fixed three-slot mode and its fairness (#169), and the load mode
that admits by busy cores (#173). Every test uses its own temporary slot directory
(MK5_CPU_SLOT_DIR); the real /tmp/mk5-cpu-slots is never touched. The load-mode tests set
the busy cores through MK5_CPU_SLOT_BUSY_FILE, never the real machine's. Runs under plain unittest."""

import json

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "cpu-slot"
# A job: record start/end times, hold the slot for `hold` seconds.
JOB = "import sys,time; t=time.time(); open(sys.argv[1],'w').write(str(t)); time.sleep(float(sys.argv[2]))"


class Slots:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name) / "slots"
        self.procs = []
        self.addCleanup(self.reap)

    def reap(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()
            for f in (p.stdout, p.stderr):
                if f:
                    f.close()

    env = {"MK5_CPU_SLOTS": "3"}  # the fixed mode, as before #173

    def slot(self, *flags, name, hold, cwd=None, cmd=None, env=None):
        """Launch cpu-slot; the job drops a timestamp in `name` when it starts."""
        env = {**os.environ, "MK5_CPU_SLOT_DIR": str(self.dir), **self.env, **(env or {})}
        job = cmd or [sys.executable, "-c", JOB, str(Path(self.tmp.name) / name), str(hold)]
        p = subprocess.Popen([sys.executable, str(SCRIPT), *flags, "--", *job], env=env, cwd=cwd,
                             stderr=subprocess.PIPE, text=True)
        self.procs.append(p)
        return p

    def started(self, name, within=5.0):
        f = Path(self.tmp.name) / name
        end = time.time() + within
        while time.time() < end:
            if f.exists() and f.read_text():
                return float(f.read_text())
            time.sleep(0.02)
        return None

    def waiters(self):
        d = self.dir / "waiters"
        return sorted(d.iterdir()) if d.exists() else []

    def wait_waiters(self, n):
        end = time.time() + 5
        while time.time() < end and len(self.waiters()) < n:
            time.sleep(0.02)
        self.assertGreaterEqual(len(self.waiters()), n)


class FixedModeTest(Slots, unittest.TestCase):
    """--slots 3: the old script's rules, unchanged."""

    def test_cap_with_a_waiter_present(self):
        a1 = self.slot("--owner", "a", name="a1", hold=1.2)
        a2 = self.slot("--owner", "a", name="a2", hold=1.2)
        self.assertIsNotNone(self.started("a1"))
        self.assertIsNotNone(self.started("a2"))
        b1 = self.slot("--owner", "b", name="b1", hold=0.5)
        self.assertIsNotNone(self.started("b1"))  # all 3 slots now busy
        a3 = self.slot("--owner", "a", name="a3", hold=0.1)  # nobody else waiting yet, but full
        self.wait_waiters(1)
        c1 = self.slot("--owner", "c", name="c1", hold=0.1)
        self.wait_waiters(2)
        # b1 frees a slot first. a3 holds two already and c is waiting: c must get it.
        tc, ta3 = self.started("c1", 6), self.started("a3", 6)
        self.assertIsNotNone(tc)
        self.assertIsNotNone(ta3)
        self.assertLess(tc, ta3)
        a1.wait(), a2.wait(), a3.wait(), b1.wait(), c1.wait()
        self.assertGreaterEqual(ta3 - tc, 0.09)  # a3 only once c1 was done
        self.assertEqual(self.waiters(), [])

    def test_third_slot_taken_when_nobody_waits(self):
        t0 = time.time()
        self.slot("--owner", "a", name="a1", hold=1.5)
        self.slot("--owner", "a", name="a2", hold=1.5)
        self.slot("--owner", "a", name="a3", hold=1.5)
        for n in ("a1", "a2", "a3"):
            self.assertLess(self.started(n, 1.0) - t0, 1.0)

    def test_exclusive_holds_everything_and_blocks_newcomers(self):
        probe = ("import fcntl,os,sys,time\n"
                 "busy=0\n"
                 "for i in range(3):\n"
                 "    fd=os.open(os.path.join(sys.argv[1],f'slot-{i}'),os.O_RDWR)\n"
                 "    try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                 "    except BlockingIOError: busy+=1\n"
                 "open(sys.argv[2],'w').write(str(time.time()))\n"
                 "time.sleep(0.6)\n"
                 "open(sys.argv[3],'w').write(str(busy))\n")
        b1 = self.slot("--owner", "b", name="b1", hold=0.8)
        self.assertIsNotNone(self.started("b1"))
        tmp = self.tmp.name
        e = self.slot("--exclusive", "wall-clock confirmation", "--owner", "e", name="e", hold=0,
                      cmd=[sys.executable, "-c", probe, str(self.dir), f"{tmp}/e", f"{tmp}/e.busy"])
        self.wait_waiters(1)
        c1 = self.slot("--owner", "c", name="c1", hold=0.1)  # a free slot exists, but E is queued
        time.sleep(0.3)
        self.assertFalse((Path(tmp) / "c1").exists())
        te, tc = self.started("e", 6), self.started("c1", 6)
        self.assertIsNotNone(te)
        self.assertIsNotNone(tc)
        b1.wait(), e.wait(), c1.wait()
        self.assertGreater(te, self.started("b1") + 0.7)  # after b1 left
        self.assertGreater(tc, te + 0.5)  # after E released
        self.assertEqual((Path(tmp) / "e.busy").read_text(), "3")  # E held all three
        self.assertIn("wall-clock confirmation", e.stderr.read())

    def test_exclusive_needs_a_reason(self):
        for reason in ("", "   "):
            p = self.slot("--exclusive", reason, name="x", hold=0)
            p.wait()
            self.assertEqual(p.returncode, 2)
            self.assertIn("reason", p.stderr.read())

    def test_killed_waiter_does_not_block(self):
        b1 = self.slot("--owner", "b", name="b1", hold=0.8)
        b2 = self.slot("--owner", "b", name="b2", hold=0.8)
        b3 = self.slot("--owner", "b", name="b3", hold=0.8)
        for n in ("b1", "b2", "b3"):
            self.assertIsNotNone(self.started(n))
        e = self.slot("--exclusive", "will be killed", "--owner", "e", name="e", hold=0)
        self.wait_waiters(1)
        e.send_signal(signal.SIGKILL)
        e.wait()
        self.assertEqual(len(self.waiters()), 1)  # the dead entry is still on disk...
        t = time.time()
        c1 = self.slot("--owner", "c", name="c1", hold=0.1)  # ...but must not block c
        tc = self.started("c1", 6)
        self.assertIsNotNone(tc)
        self.assertLess(tc - t, 2.0)
        c1.wait()
        self.assertEqual(self.waiters(), [])  # swept
        b1.wait(), b2.wait(), b3.wait()

    def test_owner_override_and_default(self):
        # Same cwd, different --owner tags are different owners: 2 + 1 slots, a waiter per tag.
        a1 = self.slot("--owner", "a", name="a1", hold=1.5)
        a2 = self.slot("--owner", "a", name="a2", hold=1.5)
        self.assertIsNotNone(self.started("a1"))
        self.assertIsNotNone(self.started("a2"))
        self.slot("--owner", "b", name="b1", hold=0.6)
        self.assertIsNotNone(self.started("b1"))
        a3 = self.slot("--owner", "a", name="a3", hold=0.1)
        self.wait_waiters(1)
        b2 = self.slot("--owner", "b", name="b2", hold=0.1)
        self.wait_waiters(2)
        self.assertLess(self.started("b2", 6), self.started("a3", 6))
        a3.wait(), b2.wait(), a1.wait(), a2.wait()
        # The default owner is the worktree root of the cwd, whatever subdirectory we are in.
        root = Path(self.tmp.name) / "tree"
        (root / "sub").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        p = self.slot(name="d", hold=0, cwd=root / "sub",
                      cmd=["sh", "-c", f"cat {self.dir}/slot-* > {self.tmp.name}/slotfiles"])
        p.wait()
        self.assertIn(f"owner {root.resolve()}", (Path(self.tmp.name) / "slotfiles").read_text())
        p = self.slot("--owner", "my tag", name="d", hold=0,
                      cmd=["sh", "-c", f"cat {self.dir}/slot-* > {self.tmp.name}/slotfiles"])
        p.wait()
        self.assertIn("owner my tag", (Path(self.tmp.name) / "slotfiles").read_text())

    def test_old_holder_unaffected(self):
        # A holder from the old script: just a flock on slot-0 with one line of text.
        self.dir.mkdir(parents=True)
        old = subprocess.Popen([sys.executable, "-c",
                                "import fcntl,os,sys,time\n"
                                f"fd=os.open('{self.dir}/slot-0',os.O_CREAT|os.O_RDWR)\n"
                                "fcntl.flock(fd,fcntl.LOCK_EX); os.write(fd,b'1 old job\\n'); print('held',flush=True); time.sleep(1.0)"],
                               stdout=subprocess.PIPE, text=True)
        self.procs.append(old)
        old.stdout.readline()
        p = self.slot("--owner", "a", name="a1", hold=0.1)
        self.assertIsNotNone(self.started("a1", 2))
        self.assertEqual(p.wait(), 0)


class LoadModeTest(Slots, unittest.TestCase):
    """The default mode: a job starts when busy cores + young jobs' threads + its own fit."""

    def setUp(self):
        super().setUp()
        self.busy = Path(self.tmp.name) / "busy"
        self.busy.write_text("0")
        self.env = {"MK5_CPU_SLOTS": "", "MK5_CPU_SLOT_BUSY_FILE": str(self.busy)}

    def test_admits_by_threads_until_the_cores_are_spoken_for(self):
        t0 = time.time()
        a = [self.slot("--cores", "10", "--threads", "3", "--owner", "a", name=f"a{i}", hold=1.5)
             for i in range(3)]
        starts = [self.started(f"a{i}", 2.0) for i in range(3)]
        self.assertTrue(all(t is not None and t - t0 < 1.5 for t in starts))
        a4 = self.slot("--cores", "10", "--threads", "3", "--owner", "a", name="a4", hold=0)
        t4 = self.started("a4", 6)
        self.assertIsNotNone(t4)
        self.assertGreater(t4, min(starts) + 1.4)  # 3+3+3+3 > 10: only once a job ended
        for p in a + [a4]:
            self.assertEqual(p.wait(), 0)

    def test_a_busy_machine_holds_new_jobs_and_frees_them_when_it_quiets(self):
        self.busy.write_text("9.5")
        a = self.slot("--cores", "10", "--threads", "1", "--owner", "a", name="a", hold=2.0)
        self.assertIsNotNone(self.started("a", 2))  # a lone job always runs, however busy
        b = self.slot("--cores", "10", "--threads", "1", "--owner", "b", name="b", hold=0)
        time.sleep(0.6)
        self.assertFalse((Path(self.tmp.name) / "b").exists())  # 9.5 + 1 > 10
        t = time.time()
        self.busy.write_text("4")
        tb = self.started("b", 3)
        self.assertIsNotNone(tb)
        self.assertGreaterEqual(tb, t)
        self.assertLess(tb, self.started("a") + 1.9)  # while a still runs
        a.wait(), b.wait()

    def test_a_waiting_owner_that_holds_less_goes_first(self):
        a1 = self.slot("--cores", "4", "--owner", "a", name="a1", hold=0.6)
        a2 = self.slot("--cores", "4", "--owner", "a", name="a2", hold=2.5)
        self.assertIsNotNone(self.started("a1"))
        self.assertIsNotNone(self.started("a2"))  # 2 + 2 threads: full
        a3 = self.slot("--cores", "4", "--owner", "a", name="a3", hold=0)
        self.wait_waiters(1)
        b1 = self.slot("--cores", "4", "--owner", "b", name="b1", hold=0)
        self.wait_waiters(2)
        tb, ta = self.started("b1", 6), self.started("a3", 6)
        self.assertIsNotNone(tb)
        self.assertIsNotNone(ta)
        self.assertLess(tb, ta)  # a1 left room for one job; b holds nothing, a holds 2
        for p in (a1, a2, a3, b1):
            self.assertEqual(p.wait(), 0)

    def test_old_script_holders_keep_their_lanes_and_their_exclusive_runs(self):
        self.dir.mkdir(parents=True)
        hold = ("import fcntl,os,sys,time\n"
                "fds=[os.open(f'{sys.argv[1]}/slot-{i}',os.O_CREAT|os.O_RDWR) for i in range(int(sys.argv[2]))]\n"
                "[fcntl.flock(fd,fcntl.LOCK_EX) for fd in fds]\n"
                "[os.write(fd,sys.argv[3].encode()) for fd in fds]\n"
                "print('held',flush=True); time.sleep(float(sys.argv[4]))")
        old = subprocess.Popen([sys.executable, "-c", hold, str(self.dir), "3",
                                "1 old job EXCLUSIVE: quiet please\nowner x\n", "1.0"],
                               stdout=subprocess.PIPE, text=True)
        self.procs.append(old)
        old.stdout.readline()
        t = time.time()
        p = self.slot("--cores", "8", "--owner", "a", name="a", hold=0)
        ta = self.started("a", 4)
        self.assertGreater(ta, t + 0.8)  # slot-3..7 were free, but the old exclusive run went first
        self.assertEqual(p.wait(), 0)
        # The load mode takes the highest free slot, leaving slot-0..2 to the old script.
        p = self.slot("--cores", "8", "--owner", "a", name="a", hold=0,
                      cmd=["sh", "-c", f"grep -l \"^$PPID \" {self.dir}/slot-* > {self.tmp.name}/mine"])
        self.assertEqual(p.wait(), 0)
        self.assertEqual((Path(self.tmp.name) / "mine").read_text().strip(), f"{self.dir}/slot-7")

    def test_exclusive_holds_every_core_slot(self):
        probe = ("import fcntl,os,sys\n"
                 "busy=0\n"
                 "for i in range(5):\n"
                 "    fd=os.open(os.path.join(sys.argv[1],f'slot-{i}'),os.O_RDWR)\n"
                 "    try: fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                 "    except BlockingIOError: busy+=1\n"
                 "open(sys.argv[2],'w').write(str(busy)+' '+os.environ['MK5_CPU_SLOT_EXCLUSIVE'])\n")
        e = self.slot("--cores", "5", "--exclusive", "timing", name="e", hold=0,
                      cmd=[sys.executable, "-c", probe, str(self.dir), f"{self.tmp.name}/e"])
        self.assertEqual(e.wait(), 0)
        self.assertEqual((Path(self.tmp.name) / "e").read_text(), "5 1")

    def test_cost_is_reported_and_appended(self):
        cost = Path(self.tmp.name) / "cost.jsonl"
        job = [sys.executable, "-c", "import os; s=sum(range(3*10**6)); print(os.environ['MK5_CPU_SLOT_EXCLUSIVE'])"]
        p = subprocess.Popen([sys.executable, str(SCRIPT), "--threads", "1", "--", *job],
                             env={**os.environ, "MK5_CPU_SLOT_DIR": str(self.dir), **self.env,
                                  "MK5_CPU_SLOT_COST": str(cost)},
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, err = p.communicate(timeout=30)
        self.assertEqual(p.returncode, 0)
        self.assertEqual(out.strip(), "0")
        self.assertIn("cpu-slot: cost wall", err)
        rec = json.loads(cost.read_text())
        self.assertEqual((rec["exit"], rec["threads"], rec["exclusive"]), (0, 1, False))
        self.assertGreater(rec["cpu_s"], 0)
        if sys.platform == "darwin":
            self.assertGreater(rec["self_instructions"], 10**7)

    def test_the_real_machine_is_read(self):
        # No stand-in: the second job waits for half a second of real CPU ticks, then fits in 64 cores.
        env = {"MK5_CPU_SLOT_BUSY_FILE": ""}
        a = self.slot("--cores", "64", name="a", hold=3, env=env)
        self.assertIsNotNone(self.started("a"))
        t = time.time()
        b = self.slot("--cores", "64", name="b", hold=0, env=env)
        tb = self.started("b", 4)
        self.assertIsNotNone(tb)
        self.assertLess(tb - t, 2.5)
        a.kill(), b.wait()

    def test_exit_status_and_timeout(self):
        p = self.slot("--timeout", "1", name="t", hold=5)
        self.assertEqual(p.wait(), 124)
        p = self.slot(name="x", hold=0, cmd=[sys.executable, "-c", "raise SystemExit(7)"])
        self.assertEqual(p.wait(), 7)


if __name__ == "__main__":
    unittest.main()
