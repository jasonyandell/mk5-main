from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class SeedTimer:
    seed: int
    decl_id: int
    t0: float = field(default_factory=perf_counter)
    last: float = field(default_factory=perf_counter)

    def phase(self, name: str, extra: str = "") -> None:
        now = perf_counter()
        dt = now - self.last
        self.last = now
        msg = f"seed={self.seed} decl={self.decl_id} | {name} | {dt:.2f}s"
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)

    def done(self, root_value: int) -> None:
        total = perf_counter() - self.t0
        print(f"seed={self.seed} decl={self.decl_id} | DONE | {total:.2f}s | root={root_value:+d}", flush=True)
