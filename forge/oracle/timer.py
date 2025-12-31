from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class SeedTimer:
    """Timer for tracking per-shard generation phases.

    Tracks timing for each phase and can return a metrics dict for wandb logging.
    """

    seed: int
    decl_id: int
    t0: float = field(default_factory=perf_counter)
    last: float = field(default_factory=perf_counter)
    phase_times: dict[str, float] = field(default_factory=dict)
    state_count: int | None = None
    vram_peaks: dict[str, float] = field(default_factory=dict)

    def phase(self, name: str, extra: str = "") -> float:
        """Record a phase completion and return elapsed time."""
        now = perf_counter()
        dt = now - self.last
        self.last = now
        self.phase_times[name] = dt

        # Parse state count from enumerate phase
        if name == "enumerate" and "states=" in extra:
            self.state_count = int(extra.split("states=")[1].split()[0].replace(",", ""))

        msg = f"seed={self.seed} decl={self.decl_id} | {name} | {dt:.2f}s"
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)
        return dt

    def record_vram(self, phase: str, peak_gb: float) -> None:
        """Record peak VRAM usage for a phase."""
        self.vram_peaks[phase] = peak_gb

    def done(self, root_value: int) -> dict[str, Any]:
        """Finalize timing and return metrics dict for logging."""
        total = perf_counter() - self.t0
        print(f"seed={self.seed} decl={self.decl_id} | DONE | {total:.2f}s | root={root_value:+d}", flush=True)

        metrics: dict[str, Any] = {
            "shard/seed": self.seed,
            "shard/decl_id": self.decl_id,
            "shard/total_time": total,
            "shard/root_value": root_value,
        }

        if self.state_count is not None:
            metrics["shard/state_count"] = self.state_count

        # Add phase times
        for phase, dt in self.phase_times.items():
            metrics[f"shard/{phase}_time"] = dt

        # Add VRAM peaks
        for phase, peak in self.vram_peaks.items():
            metrics[f"gpu/vram_{phase}_gb"] = peak

        return metrics
