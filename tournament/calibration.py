# SPDX-FileCopyrightText: Jason Yandell
# SPDX-License-Identifier: CC0-1.0

"""Calibrating the tournament's estimator (#191, part 2): how often does an A/A miss?

The tournament reports a faceoff as duplicate pairs (a pair is two games on the same
deals, seats swapped) and a 95% percentile pair bootstrap over them: resample the
pairs, 40,000 resamples, seed 666 (the rematch, wiki/experiments/
rematch-zeb-041-vs-demon-2026-10-15.md). Does that interval cover what it should?

**The participant.** Native Zeb 0.4.1 contested, the rematch's Zeb (`ZEB_041_CONTESTED`):
160 worlds, cutoff 0.8, a deterministic budget (FAIR CLOCKS), no clock at all.

**Why the same participant against itself is no A/A.** Its host sends the native player
`seed: 0` on every request whatever the config says, the arena deals a pair's two games
from the same seeds with the dealer starting at seat 0 in both, and Zeb reads nothing
of the game key. So in a duplicate pair the second game is the first move for move with
the teams relabeled: every pair splits 1-1 and the margin is 0 by symmetry, an interval
of zero width. (The pilot checks this on whole games.)

**The nearest honest A/A.** The same build at another planner seed: team A is the
rematch's Zeb whose native requests carry `seed: 1` (forecast and play alike), team B is
the rematch's Zeb exactly as the ladder seats it (seed 0). Same binary, same JavaScript,
same host bytes; a seed shim between the host and the binary rewrites the one field.
Two seeds are two deterministic policies, so their true difference delta(1,0) need not
be 0; holding the pair fixed over replicates on disjoint deals pins it, and the
replicates test whether the intervals cover it (the F arm of part 1,
wiki/experiments/estimator-calibration-2026-10-16.md).

Research only: nothing here is appended to the tournament's store or ladder, and the
seeded participant is never added to it.

    python -u -m tournament.calibration play --arm F3 --first P --count N --out DIR
    python -u -m tournament.calibration drive --arms F0,F1,... --out DIR [--workers 4]
    python -u -m tournament.calibration report --arms F0,F1,... --out DIR
    python -u tournament/calibration.py shim BINARY SEED      (the seed shim; stdlib only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAIRS = 256                   # a replicate: the rematch's size
RESAMPLES, BOOT_SEED = 40_000, 666
CLAIMED_UTC = "20261016T000000Z"  # names the specs; the games depend only on the faceoff seed


def shim(binary: str, seed: int) -> int:
    """Forward JSON lines to the native player, setting every request's `seed`."""
    child = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    for line in sys.stdin:
        request = json.loads(line)
        if "seed" in request:
            request["seed"] = seed
        child.stdin.write(json.dumps(request) + "\n")
        child.stdin.flush()
        reply = child.stdout.readline()
        if not reply:
            break
        sys.stdout.write(reply)
        sys.stdout.flush()
    child.stdin.close()
    return child.wait()


def faceoff_seed(arm: str) -> int:
    """Each arm's 64-bit faceoff seed, a hash of its name: disjoint deals per replicate."""
    return int(hashlib.sha256(f"issue-191/tournament/{arm}".encode()).hexdigest()[:16], 16)


def participants(arm: str) -> tuple:
    """(A, B) for an arm. F<r>: seed 1 against the ladder's Zeb. P<k> (timing pilot): the
    same. S<k> (symmetry check): the ladder's Zeb against itself. I<k> (shim identity):
    the shim at seed 0 against the ladder's Zeb."""
    from tournament.participants import ZEB_041_CONTESTED as base
    kind, seed = arm[0], {"F": 1, "P": 1, "I": 0}.get(arm[0])
    if kind == "S":
        return base, base
    if seed is None:
        raise ValueError(f"unknown arm {arm!r}")
    return seeded(seed), base


def seeded(seed: int):
    from tournament.participants import Participant, ZEB_041_CONTESTED as base
    return Participant(
        id=f"{base.id}-planner-seed{seed}", kind=base.kind, pin=base.pin,
        config={**base.config, "seed": seed}, needs=base.needs,
        note=(f"Research seat (#191), never in the store: {base.id} with every native request's "
              f"`seed` set to {seed} by the calibration's seed shim; same binary, JavaScript and host."))


def _shim_exe(out: Path, binary: Path, seed: int) -> Path:
    exe = out / "bin" / f"zeb-native-seed{seed}"
    text = (f"#!/bin/sh\nexec {sys.executable} -u {Path(__file__).resolve()} shim {binary} {seed}\n")
    if not exe.exists() or exe.read_text() != text:
        exe.parent.mkdir(parents=True, exist_ok=True)
        tmp = exe.with_name(exe.name + f".{os.getpid()}")
        tmp.write_text(text)
        tmp.chmod(0o755)
        os.replace(tmp, exe)
    return exe


def play(arm: str, first: int, count: int, out: Path) -> None:
    from tournament import faceoff, seats
    from tournament.artifacts import materialize
    from tournament.participants import ZEB_041_CONTESTED as base
    from tournament.zeb_native_seat import verify_binary

    a, b = participants(arm)
    spec = faceoff.FaceoffSpec("marks7", a.id, b.id, faceoff_seed(arm), PAIRS, CLAIMED_UTC)
    records = {a.id: a.to_record(), b.id: b.to_record()}
    ladder_seat = seats.seat

    def seat(p, game, device=None, *, historical=False):
        if p.id == base.id:
            return ladder_seat(p, game, device, historical=historical)  # the ladder's own path
        art = materialize(base)
        exe = _shim_exe(out, verify_binary(art), p.config["seed"])
        return seats.NativeZebSeat(seats.NodeHost("zeb", art, {**base.config, "nativeBinary": str(exe)},
                                                  script=seats.NATIVE_HOST), game)

    seats.seat = seat  # faceoff.play_game imports `seat` at call time (the rematch's pattern)
    d = out / arm
    d.mkdir(parents=True, exist_ok=True)
    for pair in range(first, min(first + count, PAIRS)):
        for team in (0, 1):
            path = d / f"pair-{pair:04d}-team-{team}.json"
            if path.exists() and json.loads(path.read_text())["done"]:
                continue
            if path.exists():  # cut off by the slot's cap: resumed at its last hand boundary
                with (d / "resumes.log").open("a") as log:
                    log.write(json.dumps({"pair": pair, "team": team, "utc": _utc()}) + "\n")
            t = time.time()
            g = faceoff.play_game(spec.__dict__, pair, team, str(path), records)
            print(json.dumps({"arm": arm, "pair": pair, "team": team, "a_won": g["winner_team"] == team,
                              "marks": g["marks"], "hands": len(g["hands"]),
                              "wall_s": round(time.time() - t, 1)}), flush=True)


def _utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _done(out: Path, arm: str, pair: int) -> bool:
    return all((p := out / arm / f"pair-{pair:04d}-team-{t}.json").exists()
               and json.loads(p.read_text())["done"] for t in (0, 1))


def drive(arms: list[str], out: Path, workers: int, pairs: int, batch: int) -> None:
    """Every batch of `batch` pairs through scripts/cpu-slot, `workers` at a time, pair-major:
    the replicates grow together, so a run stopped at a quarter (64, 128, 192 pairs) leaves
    every replicate the same size. Finished games are skipped, so `--pairs 128` extends 64."""
    import concurrent.futures as cf
    jobs = [(arm, f) for f in range(0, pairs, batch) for arm in arms]

    def run(job):
        arm, first = job
        for attempt in range(1, 9):
            if all(_done(out, arm, p) for p in range(first, min(first + batch, pairs))):
                return
            cmd = [str(ROOT / "scripts/cpu-slot"), "--threads", "1", "--owner", "issue-191-tournament", "--",
                   sys.executable, "-u", "-m", "tournament.calibration", "play", "--arm", arm,
                   "--first", str(first), "--count", str(min(batch, pairs - first)), "--out", str(out)]
            with (out / "driver-jobs.log").open("a") as log:
                r = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            with (out / "driver.log").open("a") as log:
                log.write(json.dumps({"utc": _utc(), "arm": arm, "first": first, "attempt": attempt,
                                      "exit": r.returncode}) + "\n")
        raise RuntimeError(f"job {job} failed 8 times")

    out.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in cf.as_completed([pool.submit(run, j) for j in jobs]):
            fut.result()


# ---------------------------------------------------------------- the estimator and the reads

def pair_rows(out: Path, arm: str, pairs: int = PAIRS):
    """Per pair: A's win share and A's marks margin a game (each the mean over the pair's two
    games, as the rematch's report.py computes them), plus hands and A/B seat CPU."""
    import numpy as np
    rows = np.zeros((pairs, 2))
    hands = 0
    cpu = {"a": 0.0, "b": 0.0}
    for i in range(pairs):
        for team in (0, 1):
            g = json.loads((out / arm / f"pair-{i:04d}-team-{team}.json").read_text())
            assert g["done"], (arm, i, team)
            rows[i, 0] += int(g["winner_team"] == team) / 2
            rows[i, 1] += (g["marks"][team] - g["marks"][1 - team]) / 2
            hands += len(g["hands"])
            for side in cpu:
                cpu[side] += g.get("cost", {}).get(f"{side}_cpu_s", 0.0)
    return rows, hands, cpu


def bootstrap(rows):
    """The tournament's interval: 95% percentile pair bootstrap, 40,000 resamples, seed 666."""
    import numpy as np
    n = len(rows)
    idx = np.random.default_rng(BOOT_SEED).integers(0, n, (RESAMPLES, n))
    b = rows[idx].mean(1)
    lo, hi = np.quantile(b, [0.025, 0.975], axis=0)
    return rows.mean(0), lo, hi


def r_range(k: int) -> tuple[float, float]:
    """95% range of SD/RMS(SE) for k calibrated normal means: sqrt(chi2_{k-1}/(k-1))."""
    from scipy.stats import chi2
    return tuple(float((chi2.ppf(q, k - 1) / (k - 1)) ** 0.5) for q in (0.025, 0.975))


def miss_threshold(k: int, alpha: float = 0.02) -> int:
    """Smallest m with P(Binomial(k, 0.05) >= m) < alpha."""
    from scipy.stats import binom
    return next(m for m in range(k + 1) if binom.sf(m - 1, k, 0.05) < alpha)


def report(arms: list[str], out: Path, pairs: int = PAIRS) -> dict:
    import numpy as np
    from scipy.stats import binom, kurtosis
    reps, all_rows = [], []
    for arm in arms:
        rows, hands, cpu = pair_rows(out, arm, pairs)
        mean, lo, hi = bootstrap(rows)
        reps.append({"arm": arm, "faceoff_seed": f"{faceoff_seed(arm):016x}", "hands": hands,
                     "cpu_s": cpu, "mean": mean.tolist(), "lo": lo.tolist(), "hi": hi.tolist()})
        all_rows.append(rows)
    pooled_rows = np.concatenate(all_rows)
    pm, plo, phi = bootstrap(pooled_rows)
    k = len(arms)
    lo_r, hi_r = r_range(k)
    threshold = miss_threshold(k)
    reads = {}
    for j, name in enumerate(("win_share", "marks_per_game")):
        means = np.array([r["mean"][j] for r in reps])
        se = np.array([(r["hi"][j] - r["lo"][j]) / 3.92 for r in reps])
        misses = [r["arm"] for r in reps if not r["lo"][j] <= pm[j] <= r["hi"][j]]
        misses_zero = [r["arm"] for r in reps
                       if not r["lo"][j] <= (0.5 if j == 0 else 0.0) <= r["hi"][j]]
        rf = float(means.std(ddof=1) / np.sqrt((se ** 2).mean()))
        reads[name] = {
            "pooled": {"estimate": float(pm[j]), "interval": [float(plo[j]), float(phi[j])]},
            "R_F": rf, "R_F_calibrated_95_range": [lo_r, hi_r],
            "undercovers_by_R_F": rf > hi_r,
            "sd_of_replicate_means": float(means.std(ddof=1)), "rms_se": float(np.sqrt((se ** 2).mean())),
            "misses_pooled_mean": misses, "miss_count": len(misses),
            "miss_p_value": float(binom.sf(len(misses) - 1, k, 0.05)) if misses else 1.0,
            "undercovers_by_misses": len(misses) >= threshold,
            "misses_chance_value": misses_zero,
        }
    m = pooled_rows[:, 1]
    wins = np.round(pooled_rows[:, 0] * 2).astype(int)
    tails = {"pairs": int(len(m)),
             "marks_margin_per_game_sd": float(m.std(ddof=1)),
             "excess_kurtosis_marks_margin": float(kurtosis(m)),
             "share_abs_margin_ge_3_per_game": float((np.abs(m) >= 3).mean()),
             "pair_wins_A_0_1_2": [int((wins == v).sum()) for v in (0, 1, 2)],
             "share_split_pairs": float((wins == 1).mean())}
    summary = {"schema": "issue-191-tournament-calibration/1", "pairs_per_replicate": pairs,
               "replicates": k, "bootstrap": {"resamples": RESAMPLES, "seed": BOOT_SEED, "interval": 0.95},
               "miss_threshold_p_lt_0.02": threshold, "reads": reads, "tails": tails, "per_replicate": reps}
    return summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("shim")
    s.add_argument("binary")
    s.add_argument("seed", type=int)
    s = sub.add_parser("play")
    s.add_argument("--arm", required=True)
    s.add_argument("--first", type=int, required=True)
    s.add_argument("--count", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    for name in ("drive", "report"):
        s = sub.add_parser(name)
        s.add_argument("--arms", required=True, help="comma-separated, e.g. F0,F1")
        s.add_argument("--out", type=Path, required=True)
        s.add_argument("--pairs", type=int, default=PAIRS)
        if name == "drive":
            s.add_argument("--workers", type=int, default=4)
            s.add_argument("--batch", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd == "shim":
        sys.exit(shim(a.binary, a.seed))
    if a.cmd == "play":
        play(a.arm, a.first, a.count, a.out.resolve())
    elif a.cmd == "drive":
        drive(a.arms.split(","), a.out.resolve(), a.workers, a.pairs, a.batch)
    else:
        summary = report(a.arms.split(","), a.out.resolve(), a.pairs)
        (a.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: v for k, v in summary.items() if k != "per_replicate"}, indent=2))


if __name__ == "__main__":
    main()
