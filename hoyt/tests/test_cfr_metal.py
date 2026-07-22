#!/usr/bin/env python
"""Metal CFR smoke, accuracy, and calibration-contract gates."""
from __future__ import annotations

import sys

import numpy as np

from hoyt import reference as ref
from hoyt import toys as T
from hoyt.cfr import cfr_solve
from hoyt.metalcal import (br_shortfall_upper, calibrated_gap_interval,
                           calibration_report, conformal_upper)
from hoyt.metalkernel import metal_available

_failures = []


def _gate(ok, name, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name}"
          f"{(' — ' + detail) if detail else ''}")
    if not ok:
        _failures.append(name)


def gate_M1():
    _gate(metal_available(), "M1 MLX Metal GPU is available")


def gate_M2():
    worst_v = worst_g = worst_p = 0.0
    for name in ("t2_decl_w3", "t2_def_w3", "t2_decl_w12", "t2_def_w12"):
        sub = T.get_toy(name).build()
        kw = dict(iters=60, br_every=20, impl=ref)
        cpu = cfr_solve(sub, T.payoff_points(), engine="fused", **kw)
        gpu = cfr_solve(sub, T.payoff_points(), engine="metal", **kw)
        worst_v = max(worst_v, abs(cpu.value - gpu.value))
        worst_g = max(worst_g, abs(cpu.gap - gpu.gap))
        for key in cpu.profile.entries:
            worst_p = max(worst_p, float(np.max(np.abs(
                cpu.profile.entries[key][1]
                - gpu.profile.entries[key][1]))))
    ok = worst_v <= 1e-5 and worst_g <= 1e-5 and worst_p <= 1e-5
    _gate(ok, "M2 float32 Metal agrees on calibration toys",
          f"value {worst_v:.2e}, gap {worst_g:.2e}, policy {worst_p:.2e}")


def gate_M3():
    errors = np.arange(1, 21, dtype=float)
    bar = conformal_upper(errors, 0.95)
    cal = [(i, {"cfr_reference_value": 0, "final_gap": 0.01},
            {"cfr_reference_value": i / 1000,
             "final_gap": 0.01 + i / 10000})
           for i in range(1, 21)]
    hold = [(100 + i,
             {"cfr_reference_value": 1, "final_gap": 0.02,
              "root_margin": 0.1, "root_argmax": 4},
             {"cfr_reference_value": 1.005, "final_gap": 0.021,
              "root_argmax": 4}) for i in range(4)]
    report = calibration_report(cal, hold, 0.95, 0.05)
    shortfall = br_shortfall_upper(np.arange(20) / 100,
                                   np.arange(20) / 100 - 0.01, 0.95)
    lo, hi = calibrated_gap_interval([0.02, 0.03, 0.01, 0.0],
                                     [0.001] * 4, shortfall)
    ok = bar == 20.0 and report["value"]["heldout_covered"] == 4 \
        and report["convergence"]["wrong_when_certain"] == 0 \
        and abs(shortfall - 0.01) < 1e-12 and lo < 0.03 < hi
    _gate(ok, "M3 conformal error-bar contract", f"rank bar {bar:g}")


def main():
    gate_M1()
    gate_M2()
    gate_M3()
    if _failures:
        print(f"\n{len(_failures)} gate(s) FAILED: {_failures}")
        return 1
    print("\nall Metal CFR gates green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
